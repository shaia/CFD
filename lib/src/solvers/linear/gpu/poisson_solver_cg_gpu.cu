/**
 * @file poisson_solver_cg_gpu.cu
 * @brief Conjugate Gradient Poisson solver — CUDA GPU backend
 *
 * Implements the poisson_solver_t interface on the GPU using Conjugate Gradient.
 * Like the GPU Jacobi backend, it implements `solve` directly (upload once, run
 * the full iteration on-device, download once) to avoid per-iteration transfers.
 *
 * The solve loop is expressed entirely through the shared device primitives in
 * poisson_gpu_primitives.cuh (SPD matvec, dot, axpy, xpay) wrapped in thin host
 * helpers — it contains no raw kernel launches of its own, honoring the project's
 * algorithm-primitive separation rule.
 *
 * Sign convention matches the CPU CG reference: the plain Laplacian is negative
 * definite, so CG runs on the SPD operator A = -Laplacian with b = -rhs. The
 * residual norm ||r|| equals ||rhs - Laplacian(x)|| (the Poisson residual).
 *
 * Boundaries: r/p/Ap stay zero on the boundary (interior-only operators), and the
 * Neumann BC is applied to x only at the start and end of the solve — identical to
 * the CPU CG.
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/poisson_solver.h"

#include "poisson_gpu_primitives.cuh"

#include <cmath>
#include <cuda_runtime.h>

extern "C" {
#include "../linear_solver_internal.h"
}

#define CG_GPU_BREAKDOWN 1e-30

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz, size;
    size_t stride_z;
    int k_start, k_end;
    double inv_dx2, inv_dy2, inv_dz2, factor;
    int block_x, block_y;

    double* d_x;       /* solution (in/out) */
    double* d_rhs;     /* right-hand side   */
    double* d_r;       /* residual          */
    double* d_p;       /* search direction  */
    double* d_Ap;      /* A*p               */
    double* d_scalar;  /* reduction accumulator (single double) */
    cudaStream_t stream;
} poisson_cg_gpu_ctx;

/* ============================================================================
 * PRIMITIVE HOST WRAPPERS (one kernel launch each)
 * ============================================================================ */

static void cg_matvec(poisson_cg_gpu_ctx* c, const double* d_in, double* d_out,
                      dim3 grid, dim3 block) {
    lin_gpu_kernel_spd_laplacian<<<grid, block, 0, c->stream>>>(
        d_in, d_out, c->nx, c->ny, c->stride_z, c->k_start, c->k_end,
        c->inv_dx2, c->inv_dy2, c->inv_dz2, c->factor);
}

static void cg_residual(poisson_cg_gpu_ctx* c, const double* d_x, double* d_r,
                        dim3 grid, dim3 block) {
    lin_gpu_kernel_residual_vec<<<grid, block, 0, c->stream>>>(
        d_x, c->d_rhs, d_r, c->nx, c->ny, c->stride_z, c->k_start, c->k_end,
        c->inv_dx2, c->inv_dy2, c->inv_dz2, c->factor);
}

static void cg_axpy(poisson_cg_gpu_ctx* c, double alpha, const double* d_x, double* d_y,
                    dim3 grid, dim3 block) {
    lin_gpu_kernel_axpy<<<grid, block, 0, c->stream>>>(
        alpha, d_x, d_y, c->nx, c->ny, c->stride_z, c->k_start, c->k_end);
}

static void cg_xpay(poisson_cg_gpu_ctx* c, const double* d_x, double beta, double* d_y,
                    dim3 grid, dim3 block) {
    lin_gpu_kernel_xpay<<<grid, block, 0, c->stream>>>(
        d_x, beta, d_y, c->nx, c->ny, c->stride_z, c->k_start, c->k_end);
}

/* Interior dot product (a,b), written to *out. Returns false on CUDA failure so
 * the caller aborts the solve. */
static bool cg_dot(poisson_cg_gpu_ctx* c, const double* d_a, const double* d_b,
                   dim3 grid, dim3 block, double* out) {
    size_t shmem = (size_t)block.x * block.y * sizeof(double);
    if (cudaMemsetAsync(c->d_scalar, 0, sizeof(double), c->stream) != cudaSuccess)
        return false;
    lin_gpu_kernel_dot<<<grid, block, shmem, c->stream>>>(
        d_a, d_b, c->d_scalar, c->nx, c->ny, c->stride_z, c->k_start, c->k_end);
    double h = 0.0;
    if (cudaMemcpyAsync(&h, c->d_scalar, sizeof(double), cudaMemcpyDeviceToHost, c->stream)
        != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(c->stream) != cudaSuccess)
        return false;
    *out = h;
    return true;
}

/* ============================================================================
 * INTERFACE IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t cg_gpu_init(poisson_solver_t* solver,
                                size_t nx, size_t ny, size_t nz,
                                double dx, double dy, double dz,
                                const poisson_solver_params_t* params) {
    (void)params;
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_cg_gpu_ctx* ctx = (poisson_cg_gpu_ctx*)calloc(1, sizeof(poisson_cg_gpu_ctx));
    if (!ctx) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU CG solver context");
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
           && cudaMalloc(&ctx->d_rhs, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_r, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_p, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_Ap, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_scalar, sizeof(double)) == cudaSuccess
           && cudaStreamCreate(&ctx->stream) == cudaSuccess;
    if (!ok) {
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_r);
        cudaFree(ctx->d_p);
        cudaFree(ctx->d_Ap);
        cudaFree(ctx->d_scalar);
        free(ctx);
        cfd_set_error(CFD_ERROR_NOMEM, "GPU CG: device allocation failed");
        return CFD_ERROR_NOMEM;
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void cg_gpu_destroy(poisson_solver_t* solver) {
    if (!solver || !solver->context)
        return;
    poisson_cg_gpu_ctx* ctx = (poisson_cg_gpu_ctx*)solver->context;
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_rhs);
    cudaFree(ctx->d_r);
    cudaFree(ctx->d_p);
    cudaFree(ctx->d_Ap);
    cudaFree(ctx->d_scalar);
    free(ctx);
    solver->context = NULL;
}

static cfd_status_t cg_gpu_solve(poisson_solver_t* solver,
                                 double* x, double* x_temp, const double* rhs,
                                 poisson_solver_stats_t* stats) {
    (void)x_temp;  /* CG keeps all working vectors on-device */
    if (!solver || !x || !rhs)
        return CFD_ERROR_INVALID;
    poisson_cg_gpu_ctx* c = (poisson_cg_gpu_ctx*)solver->context;
    if (!c)
        return CFD_ERROR_INVALID;

    const poisson_solver_params_t* p = &solver->params;
    size_t nx = c->nx, ny = c->ny, nz = c->nz;
    size_t bytes = c->size * sizeof(double);

    dim3 block((unsigned)c->block_x, (unsigned)c->block_y);
    dim3 grid((unsigned)((nx - 2 + block.x - 1) / block.x),
              (unsigned)((ny - 2 + block.y - 1) / block.y));

    /* Upload x and rhs; zero the working vectors so their boundaries stay 0. */
    if (cudaMemcpyAsync(c->d_x, x, bytes, cudaMemcpyHostToDevice, c->stream) != cudaSuccess
        || cudaMemcpyAsync(c->d_rhs, rhs, bytes, cudaMemcpyHostToDevice, c->stream) != cudaSuccess
        || cudaMemsetAsync(c->d_r, 0, bytes, c->stream) != cudaSuccess
        || cudaMemsetAsync(c->d_p, 0, bytes, c->stream) != cudaSuccess
        || cudaMemsetAsync(c->d_Ap, 0, bytes, c->stream) != cudaSuccess) {
        return CFD_ERROR;
    }

    /* BC on the initial guess, then r0 = b - A x0, p0 = r0. */
    bc_apply_scalar_3d_gpu(c->d_x, nx, ny, nz, BC_TYPE_NEUMANN, c->stream);
    cg_residual(c, c->d_x, c->d_r, grid, block);
    if (cudaMemcpyAsync(c->d_p, c->d_r, bytes, cudaMemcpyDeviceToDevice, c->stream) != cudaSuccess)
        return CFD_ERROR;

    double rho = 0.0;
    if (!cg_dot(c, c->d_r, c->d_r, grid, block, &rho))
        return CFD_ERROR;
    double initial_res = sqrt(rho);

    double tol = p->tolerance * initial_res;
    if (tol < p->absolute_tolerance)
        tol = p->absolute_tolerance;

    double res = initial_res;
    int iter = 0;
    int converged = (initial_res < p->absolute_tolerance);

    while (!converged && iter < p->max_iterations) {
        cg_matvec(c, c->d_p, c->d_Ap, grid, block);          /* Ap = A p          */
        double p_dot_Ap = 0.0;
        if (!cg_dot(c, c->d_p, c->d_Ap, grid, block, &p_dot_Ap))
            return CFD_ERROR;
        if (fabs(p_dot_Ap) < CG_GPU_BREAKDOWN)
            break;  /* stagnation / singular direction */

        double alpha = rho / p_dot_Ap;
        cg_axpy(c, alpha, c->d_p, c->d_x, grid, block);      /* x += alpha p      */
        cg_axpy(c, -alpha, c->d_Ap, c->d_r, grid, block);    /* r -= alpha Ap     */

        double rho_new = 0.0;
        if (!cg_dot(c, c->d_r, c->d_r, grid, block, &rho_new))
            return CFD_ERROR;
        res = sqrt(rho_new);
        iter++;

        if (res < tol || res < p->absolute_tolerance) {
            converged = 1;
            break;
        }
        if (fabs(rho) < CG_GPU_BREAKDOWN)
            break;

        double beta = rho_new / rho;
        cg_xpay(c, c->d_r, beta, c->d_p, grid, block);       /* p = r + beta p    */
        rho = rho_new;
    }

    /* Final Neumann BC on the solution, then download. */
    bc_apply_scalar_3d_gpu(c->d_x, nx, ny, nz, BC_TYPE_NEUMANN, c->stream);
    if (cudaMemcpyAsync(x, c->d_x, bytes, cudaMemcpyDeviceToHost, c->stream) != cudaSuccess)
        return CFD_ERROR;
    if (cudaStreamSynchronize(c->stream) != cudaSuccess)
        return CFD_ERROR;

    if (stats) {
        stats->initial_residual = initial_res;
        stats->final_residual = res;
        stats->iterations = iter;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }
    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/* ============================================================================
 * FACTORY
 * ============================================================================ */

extern "C" poisson_solver_t* create_cg_gpu_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU CG solver");
        return NULL;
    }

    solver->name = "cg_gpu";
    solver->description = "Conjugate Gradient (CUDA GPU)";
    solver->method = POISSON_METHOD_CG;
    solver->backend = POISSON_BACKEND_GPU;
    solver->params = poisson_solver_params_default();

    solver->init = cg_gpu_init;
    solver->destroy = cg_gpu_destroy;
    solver->solve = cg_gpu_solve;
    solver->iterate = NULL;
    solver->apply_bc = NULL;

    return solver;
}
