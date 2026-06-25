/**
 * @file poisson_solver_cg_gpu.cu
 * @brief Conjugate Gradient Poisson solver — CUDA GPU backend
 *
 * Implements the poisson_solver_t interface on the GPU using Conjugate Gradient.
 * This backend is a thin host-buffer wrapper around the shared device-resident
 * CG core in poisson_cg_gpu_solve.cuh: upload x/rhs once, run the full iteration
 * on the GPU, download x once — avoiding per-iteration field transfers. The same
 * core is called directly by the GPU projection solver: the pressure/RHS fields
 * stay device-resident for the whole solve (no per-iteration field round-trip),
 * though each CG iteration's dot-product reductions still copy a single scalar
 * back to the host and sync.
 *
 * Sign convention matches the CPU CG reference: the plain Laplacian is negative
 * definite, so CG runs on the SPD operator A = -Laplacian with b = -rhs. The
 * residual norm ||r|| equals ||rhs - Laplacian(x)|| (the Poisson residual).
 *
 * Boundaries: r/p/Ap stay zero on the boundary (interior-only operators), and the
 * Neumann BC is applied to x only at the start and end of the solve — identical to
 * the CPU CG.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/poisson_solver.h"

#include "poisson_cg_gpu_solve.cuh"

#include <cstdlib>
#include <cuda_runtime.h>

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
    size_t bytes = c->size * sizeof(double);

    /* Upload x and rhs, then run the full CG on-device via the shared core. */
    if (cudaMemcpyAsync(c->d_x, x, bytes, cudaMemcpyHostToDevice, c->stream) != cudaSuccess
        || cudaMemcpyAsync(c->d_rhs, rhs, bytes, cudaMemcpyHostToDevice, c->stream) != cudaSuccess)
        return CFD_ERROR;

    cg_gpu_solve_params cgp = {p->tolerance, p->absolute_tolerance, p->max_iterations};
    cg_gpu_solve_result cgr = {0};
    cfd_status_t rc = cg_gpu_solve_device(
        c->d_x, c->d_rhs, c->d_r, c->d_p, c->d_Ap, c->d_scalar,
        c->nx, c->ny, c->nz, c->stride_z, c->k_start, c->k_end,
        c->inv_dx2, c->inv_dy2, c->inv_dz2, c->factor,
        c->block_x, c->block_y, c->stream, &cgp, &cgr);
    if (rc == CFD_ERROR || rc == CFD_ERROR_INVALID)
        return rc;  /* CUDA failure — solution undefined, skip the download */

    /* Download the solution (the core already synchronized the stream). */
    if (cudaMemcpyAsync(x, c->d_x, bytes, cudaMemcpyDeviceToHost, c->stream) != cudaSuccess)
        return CFD_ERROR;
    if (cudaStreamSynchronize(c->stream) != cudaSuccess)
        return CFD_ERROR;

    if (stats) {
        stats->initial_residual = cgr.initial_residual;
        stats->final_residual = cgr.final_residual;
        stats->iterations = cgr.iterations;
        stats->status = cgr.converged   ? POISSON_CONVERGED
                        : cgr.stagnated ? POISSON_STAGNATED
                                        : POISSON_MAX_ITER;
    }
    return rc;
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
