/**
 * @file poisson_solver_jacobi_gpu.cu
 * @brief Jacobi Poisson solver — CUDA GPU backend
 *
 * Implements the poisson_solver_t interface on the GPU. Unlike the CPU/SIMD
 * Jacobi solvers (which expose a per-iteration `iterate` driven by the common
 * host solve loop), this backend implements `solve` directly: it uploads the
 * RHS and initial guess once, runs the full Jacobi iteration on-device with a
 * relative-residual convergence check, then downloads the result. Per-iteration
 * host/device transfers would dominate the runtime, so they are avoided.
 *
 * Boundary handling matches the interface default: Neumann (zero-gradient) on
 * every face, applied via the unified bc_apply_scalar_3d_gpu() kernels.
 *
 * Restrictions (return CFD_ERROR_UNSUPPORTED from init): no CUDA device present.
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/logging.h"
#include "cfd/solvers/poisson_solver.h"

#include "poisson_gpu_primitives.cuh"

#include <cmath>
#include <cuda_runtime.h>

/* Internal-header helpers (poisson_solver_compute_3d_bounds, inv_dz2, etc.) are
 * C-linkage inline functions; include under extern "C" since this is a .cu TU. */
extern "C" {
#include "../linear_solver_internal.h"
}

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz, size;
    size_t stride_z;
    int k_start, k_end;
    double inv_dx2, inv_dy2, inv_dz2;
    double factor, inv_factor;
    int block_x, block_y;

    double* d_x;        /* solution (in/out)            */
    double* d_x_temp;   /* Jacobi double-buffer scratch */
    double* d_rhs;      /* right-hand side              */
    double* d_scalar;   /* single-double reduction accumulator */
    cudaStream_t stream;
} poisson_jacobi_gpu_ctx;

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/* L2 norm of the Poisson residual rhs - A*x for the field d_field. Returns -1.0
 * on any CUDA failure so the caller can fall back to the fixed iteration cap. */
static double jac_residual_norm(poisson_jacobi_gpu_ctx* ctx, const double* d_field,
                                dim3 grid_dim, dim3 block) {
    size_t shmem = (size_t)block.x * block.y * sizeof(double);
    if (cudaMemsetAsync(ctx->d_scalar, 0, sizeof(double), ctx->stream) != cudaSuccess)
        return -1.0;
    lin_gpu_kernel_residual_sq<<<grid_dim, block, shmem, ctx->stream>>>(
        d_field, ctx->d_rhs, ctx->d_scalar, ctx->nx, ctx->ny,
        ctx->stride_z, ctx->k_start, ctx->k_end,
        ctx->inv_dx2, ctx->inv_dy2, ctx->inv_dz2, ctx->factor);
    if (cudaGetLastError() != cudaSuccess)
        return -1.0;
    double h_sumsq = 0.0;
    if (cudaMemcpyAsync(&h_sumsq, ctx->d_scalar, sizeof(double),
                        cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess)
        return -1.0;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess)
        return -1.0;
    return sqrt(h_sumsq);
}

/* ============================================================================
 * INTERFACE IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t jacobi_gpu_init(poisson_solver_t* solver,
                                    size_t nx, size_t ny, size_t nz,
                                    double dx, double dy, double dz,
                                    const poisson_solver_params_t* params) {
    (void)params;
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_jacobi_gpu_ctx* ctx =
        (poisson_jacobi_gpu_ctx*)calloc(1, sizeof(poisson_jacobi_gpu_ctx));
    if (!ctx) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU Jacobi solver context");
        return CFD_ERROR_NOMEM;
    }

    ctx->nx = nx;
    ctx->ny = ny;
    ctx->nz = nz;
    ctx->size = nx * ny * (nz > 0 ? nz : 1);
    ctx->inv_dx2 = 1.0 / (dx * dx);
    ctx->inv_dy2 = 1.0 / (dy * dy);
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    ctx->factor = 2.0 * (ctx->inv_dx2 + ctx->inv_dy2 + ctx->inv_dz2);
    ctx->inv_factor = 1.0 / ctx->factor;

    size_t sz, ks, ke;
    poisson_solver_compute_3d_bounds(nz, nx, ny, &sz, &ks, &ke);
    ctx->stride_z = sz;
    ctx->k_start = (int)ks;
    ctx->k_end = (int)ke - 1;  /* primitives use inclusive k_end */

    gpu_config_t cfg = gpu_config_default();
    ctx->block_x = cfg.block_size_x;
    ctx->block_y = cfg.block_size_y;

    size_t bytes = ctx->size * sizeof(double);
    bool ok = cudaMalloc(&ctx->d_x, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_x_temp, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_rhs, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_scalar, sizeof(double)) == cudaSuccess
           && cudaStreamCreate(&ctx->stream) == cudaSuccess;
    if (!ok) {
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_x_temp);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_scalar);
        free(ctx);
        cfd_set_error(CFD_ERROR_NOMEM, "GPU Jacobi: device allocation failed");
        return CFD_ERROR_NOMEM;
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void jacobi_gpu_destroy(poisson_solver_t* solver) {
    if (!solver || !solver->context)
        return;
    poisson_jacobi_gpu_ctx* ctx = (poisson_jacobi_gpu_ctx*)solver->context;
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_x_temp);
    cudaFree(ctx->d_rhs);
    cudaFree(ctx->d_scalar);
    free(ctx);
    solver->context = NULL;
}

static cfd_status_t jacobi_gpu_solve(poisson_solver_t* solver,
                                     double* x, double* x_temp, const double* rhs,
                                     poisson_solver_stats_t* stats) {
    (void)x_temp;  /* host scratch unused: device double-buffer is internal */
    if (!solver || !x || !rhs)
        return CFD_ERROR_INVALID;
    poisson_jacobi_gpu_ctx* ctx = (poisson_jacobi_gpu_ctx*)solver->context;
    if (!ctx)
        return CFD_ERROR_INVALID;

    const poisson_solver_params_t* p = &solver->params;
    size_t nx = ctx->nx, ny = ctx->ny, nz = ctx->nz;
    size_t bytes = ctx->size * sizeof(double);

    dim3 block((unsigned)ctx->block_x, (unsigned)ctx->block_y);
    dim3 grid_dim((unsigned)((nx - 2 + block.x - 1) / block.x),
                  (unsigned)((ny - 2 + block.y - 1) / block.y));

    if (cudaMemcpyAsync(ctx->d_x, x, bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess
        || cudaMemcpyAsync(ctx->d_rhs, rhs, bytes, cudaMemcpyHostToDevice, ctx->stream)
               != cudaSuccess) {
        return CFD_ERROR;
    }

    /* Enforce the Neumann BC on the (possibly warm-started) initial guess before
     * measuring r0: the residual stencil reads boundary-adjacent cells. */
    bc_apply_scalar_3d_gpu(ctx->d_x, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);
    double r0 = jac_residual_norm(ctx, ctx->d_x, grid_dim, block);

    const double RES_FLOOR = 1e-30;
    double tol_abs = p->absolute_tolerance;
    int can_check = std::isfinite(r0) && (r0 >= 0.0);
    double tol_target = can_check ? p->tolerance * r0 : 0.0;
    if (tol_target < tol_abs)
        tol_target = tol_abs;

    int max_iter = p->max_iterations;
    int check_every = (max_iter > 0 && max_iter < 20) ? max_iter : 20;

    double res = r0;
    int iter = 0;
    int converged = 0;

    if (can_check && r0 <= tol_abs) {
        converged = 1;  /* already converged */
    } else {
        double* src = ctx->d_x;
        double* dst = ctx->d_x_temp;
        for (iter = 0; iter < max_iter; iter++) {
            lin_gpu_kernel_jacobi<<<grid_dim, block, 0, ctx->stream>>>(
                src, dst, ctx->d_rhs, nx, ny, ctx->stride_z, ctx->k_start, ctx->k_end,
                ctx->inv_dx2, ctx->inv_dy2, ctx->inv_dz2, ctx->inv_factor);
            bc_apply_scalar_3d_gpu(dst, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);
            double* tmp = src;
            src = dst;
            dst = tmp;

            if (can_check && (iter + 1) % check_every == 0) {
                double rnorm = jac_residual_norm(ctx, src, grid_dim, block);
                if (rnorm < 0.0 || !std::isfinite(rnorm)) {
                    can_check = 0;  /* residual eval failed: run out the cap */
                } else {
                    res = rnorm;
                    if (rnorm <= tol_target || rnorm <= RES_FLOOR) {
                        converged = 1;
                        iter++;
                        break;
                    }
                }
            }
        }
        /* Make sure the latest iterate lives in ctx->d_x before download. */
        if (src != ctx->d_x) {
            if (cudaMemcpyAsync(ctx->d_x, src, bytes, cudaMemcpyDeviceToDevice, ctx->stream)
                != cudaSuccess)
                return CFD_ERROR;
        }
    }

    if (cudaMemcpyAsync(x, ctx->d_x, bytes, cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess)
        return CFD_ERROR;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess)
        return CFD_ERROR;

    if (stats) {
        stats->initial_residual = r0;
        stats->final_residual = res;
        stats->iterations = iter;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }
    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/* ============================================================================
 * FACTORY
 * ============================================================================ */

extern "C" poisson_solver_t* create_jacobi_gpu_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU Jacobi solver");
        return NULL;
    }

    solver->name = "jacobi_gpu";
    solver->description = "Jacobi iteration (CUDA GPU)";
    solver->method = POISSON_METHOD_JACOBI;
    solver->backend = POISSON_BACKEND_GPU;
    solver->params = poisson_solver_params_default();
    solver->params.max_iterations = 2000;  /* Jacobi needs more iterations */

    solver->init = jacobi_gpu_init;
    solver->destroy = jacobi_gpu_destroy;
    solver->solve = jacobi_gpu_solve;
    solver->iterate = NULL;   /* full solve is implemented directly */
    solver->apply_bc = NULL;  /* Neumann applied internally on-device */

    return solver;
}
