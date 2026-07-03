/**
 * @file poisson_solver_sor_gpu.cu
 * @brief Plain SOR Poisson solver — CUDA GPU backend (Block SOR)
 *
 * Implements the poisson_solver_t interface on the GPU. Like the Jacobi and
 * Red-Black GPU backends (and unlike the CPU/SIMD plain-SOR solvers, which expose
 * a per-iteration `iterate` driven by the common host solve loop), this backend
 * implements `solve` directly: it uploads the RHS and initial guess once, runs the
 * full iteration on-device with a relative-residual convergence check, then
 * downloads the result. Per-iteration host/device transfers would dominate.
 *
 * Plain lexicographic SOR is sequential (each cell depends on its already-updated
 * left/down neighbors), which the GPU cannot reproduce cheaply. This backend uses
 * the same "Block SOR" idea as the AVX2/NEON plain-SOR solvers — each thread does a
 * sequential Gauss-Seidel/SOR sweep over a small tile — but parallelizes the tiles
 * with a red-black *tile* coloring: each iteration launches the red tile pass then
 * the black tile pass (the launch boundary is the color sync). Because adjacency
 * flips tile parity, a tile's halo is never written by another thread in the same
 * pass, so the update is in-place, race-free, and (being a consistent ordering)
 * provably convergent for 0<omega<2 — the auto optimal-omega stays stable. A naive
 * double-buffered tile scheme leaves the tiles Jacobi-coupled and diverges at the
 * optimal SOR over-relaxation. At 1x1 tiles this degenerates to the cell-level
 * Red-Black SOR. See poisson_gpu_primitives.cuh
 * (lin_gpu_kernel_block_sor_tile_sweep) for the per-cell update.
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
#include <cstdlib>
#include <cuda_runtime.h>

/* Internal-header helpers (poisson_solver_compute_3d_bounds, resolve_omega, etc.)
 * are C-linkage inline functions; include under extern "C" since this is a .cu TU. */
extern "C" {
#include "../linear_solver_internal.h"
}

/* Tile dimensions swept sequentially by each thread. 8x8 keeps the stale-halo
 * fraction small enough that the auto optimal-omega remains stable (mirrors the
 * AVX2 Block SOR, whose only staleness is the intra-block left neighbor). */
#define SOR_GPU_TILE_W 8
#define SOR_GPU_TILE_H 8

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz, size;
    size_t stride_z;
    int k_start, k_end;
    double inv_dx2, inv_dy2, inv_dz2;
    double factor, inv_factor;
    double omega;
    int block_x, block_y;
    int tile_w, tile_h;

    double* d_x;        /* solution (in/out) — updated in place */
    double* d_rhs;      /* right-hand side                      */
    double* d_scalar;   /* single-double reduction accumulator  */
    cudaStream_t stream;
} poisson_sor_gpu_ctx;

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/* L2 norm of the Poisson residual rhs - A*x for the field d_field. Returns -1.0
 * on any CUDA failure so the caller can fall back to the fixed iteration cap. */
static double sor_residual_norm(poisson_sor_gpu_ctx* ctx, const double* d_field,
                                dim3 cell_grid, dim3 block) {
    size_t shmem = (size_t)block.x * block.y * sizeof(double);
    if (cudaMemsetAsync(ctx->d_scalar, 0, sizeof(double), ctx->stream) != cudaSuccess)
        return -1.0;
    lin_gpu_kernel_residual_sq<<<cell_grid, block, shmem, ctx->stream>>>(
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

static cfd_status_t sor_gpu_init(poisson_solver_t* solver,
                                 size_t nx, size_t ny, size_t nz,
                                 double dx, double dy, double dz,
                                 const poisson_solver_params_t* params) {
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_sor_gpu_ctx* ctx =
        (poisson_sor_gpu_ctx*)calloc(1, sizeof(poisson_sor_gpu_ctx));
    if (!ctx) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU SOR solver context");
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
    ctx->omega = poisson_solver_resolve_omega(
        params ? params->omega : 0.0, nx, ny, nz, dx, dy, dz);

    size_t sz, ks, ke;
    poisson_solver_compute_3d_bounds(nz, nx, ny, &sz, &ks, &ke);
    ctx->stride_z = sz;
    ctx->k_start = (int)ks;
    ctx->k_end = (int)ke - 1;  /* primitives use inclusive k_end */

    gpu_config_t cfg = gpu_config_default();
    ctx->block_x = cfg.block_size_x;
    ctx->block_y = cfg.block_size_y;
    ctx->tile_w = SOR_GPU_TILE_W;
    ctx->tile_h = SOR_GPU_TILE_H;

    size_t bytes = ctx->size * sizeof(double);
    bool ok = cudaMalloc(&ctx->d_x, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_rhs, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_scalar, sizeof(double)) == cudaSuccess
           && cudaStreamCreate(&ctx->stream) == cudaSuccess;
    if (!ok) {
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_scalar);
        free(ctx);
        cfd_set_error(CFD_ERROR_NOMEM, "GPU SOR: device allocation failed");
        return CFD_ERROR_NOMEM;
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_gpu_destroy(poisson_solver_t* solver) {
    if (!solver || !solver->context)
        return;
    poisson_sor_gpu_ctx* ctx = (poisson_sor_gpu_ctx*)solver->context;
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_rhs);
    cudaFree(ctx->d_scalar);
    free(ctx);
    solver->context = NULL;
}

static cfd_status_t sor_gpu_solve(poisson_solver_t* solver,
                                  double* x, double* x_temp, const double* rhs,
                                  poisson_solver_stats_t* stats) {
    (void)x_temp;  /* host scratch unused: the solve runs in-place on device */
    if (!solver || !x || !rhs)
        return CFD_ERROR_INVALID;
    poisson_sor_gpu_ctx* ctx = (poisson_sor_gpu_ctx*)solver->context;
    if (!ctx)
        return CFD_ERROR_INVALID;

    const poisson_solver_params_t* p = &solver->params;
    size_t nx = ctx->nx, ny = ctx->ny, nz = ctx->nz;
    size_t bytes = ctx->size * sizeof(double);

    dim3 block((unsigned)ctx->block_x, (unsigned)ctx->block_y);
    /* Sweep grid: one thread per tile. Residual/BC grid: one thread per cell. */
    unsigned n_tiles_x = ((unsigned)(nx - 2) + (unsigned)ctx->tile_w - 1) / (unsigned)ctx->tile_w;
    unsigned n_tiles_y = ((unsigned)(ny - 2) + (unsigned)ctx->tile_h - 1) / (unsigned)ctx->tile_h;
    dim3 sweep_grid((n_tiles_x + block.x - 1) / block.x,
                    (n_tiles_y + block.y - 1) / block.y);
    dim3 cell_grid((unsigned)((nx - 2 + block.x - 1) / block.x),
                   (unsigned)((ny - 2 + block.y - 1) / block.y));

    if (cudaMemcpyAsync(ctx->d_x, x, bytes, cudaMemcpyHostToDevice, ctx->stream) != cudaSuccess
        || cudaMemcpyAsync(ctx->d_rhs, rhs, bytes, cudaMemcpyHostToDevice, ctx->stream)
               != cudaSuccess) {
        return CFD_ERROR;
    }

    /* Enforce the Neumann BC on the (possibly warm-started) initial guess before
     * measuring r0: the residual stencil reads boundary-adjacent cells. */
    bc_apply_scalar_3d_gpu(ctx->d_x, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);
    double r0 = sor_residual_norm(ctx, ctx->d_x, cell_grid, block);

    const double RES_FLOOR = 1e-30;
    double tol_abs = p->absolute_tolerance;
    int can_check = std::isfinite(r0) && (r0 >= 0.0);
    double tol_target = can_check ? p->tolerance * r0 : 0.0;
    if (tol_target < tol_abs)
        tol_target = tol_abs;

    int max_iter = p->max_iterations;
    /* Each convergence check is a device-side reduction + stream sync, so never
     * poll more often than every CHECK_FLOOR iterations; honor a larger explicit
     * check_interval. A small max_iter still polls once at the final iteration. */
    const int CHECK_FLOOR = 20;
    int check_every = p->check_interval > CHECK_FLOOR ? p->check_interval : CHECK_FLOOR;
    if (max_iter > 0 && check_every > max_iter)
        check_every = max_iter;

    double res = r0;
    int iter = 0;
    int converged = 0;

    if (can_check && r0 <= tol_abs) {
        converged = 1;  /* already converged */
    } else {
        for (iter = 0; iter < max_iter; iter++) {
            /* Red tile pass, then black tile pass: the launch boundary is the color
             * sync. In-place, so no double buffer. BCs applied after both passes,
             * matching the CPU/Red-Black reference. */
            lin_gpu_kernel_block_sor_tile_sweep<<<sweep_grid, block, 0, ctx->stream>>>(
                ctx->d_x, ctx->d_rhs, /*color=*/0, ctx->omega, ctx->tile_w, ctx->tile_h,
                nx, ny, ctx->stride_z, ctx->k_start, ctx->k_end,
                ctx->inv_dx2, ctx->inv_dy2, ctx->inv_dz2, ctx->inv_factor);
            lin_gpu_kernel_block_sor_tile_sweep<<<sweep_grid, block, 0, ctx->stream>>>(
                ctx->d_x, ctx->d_rhs, /*color=*/1, ctx->omega, ctx->tile_w, ctx->tile_h,
                nx, ny, ctx->stride_z, ctx->k_start, ctx->k_end,
                ctx->inv_dx2, ctx->inv_dy2, ctx->inv_dz2, ctx->inv_factor);
            bc_apply_scalar_3d_gpu(ctx->d_x, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);

            if (can_check && (iter + 1) % check_every == 0) {
                double rnorm = sor_residual_norm(ctx, ctx->d_x, cell_grid, block);
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
    }

    if (cudaMemcpyAsync(x, ctx->d_x, bytes, cudaMemcpyDeviceToHost, ctx->stream) != cudaSuccess)
        return CFD_ERROR;
    if (cudaStreamSynchronize(ctx->stream) != cudaSuccess)
        return CFD_ERROR;

    if (stats) {
        /* sor_residual_norm() returns -1.0 when the device reduction fails;
         * normalize such failures to NaN so stats never report a nonsensical
         * negative residual. */
        stats->initial_residual = (std::isfinite(r0) && r0 >= 0.0) ? r0 : NAN;
        stats->final_residual = (std::isfinite(res) && res >= 0.0) ? res : NAN;
        stats->iterations = iter;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }
    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/* ============================================================================
 * FACTORY
 * ============================================================================ */

extern "C" poisson_solver_t* create_sor_gpu_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU SOR solver");
        return NULL;
    }

    solver->name = "sor_gpu";
    solver->description = "SOR iteration (CUDA GPU, Block SOR)";
    solver->method = POISSON_METHOD_SOR;
    solver->backend = POISSON_BACKEND_GPU;
    solver->params = poisson_solver_params_default();

    solver->init = sor_gpu_init;
    solver->destroy = sor_gpu_destroy;
    solver->solve = sor_gpu_solve;
    solver->iterate = NULL;   /* full solve is implemented directly */
    solver->apply_bc = NULL;  /* Neumann applied internally on-device */

    return solver;
}
