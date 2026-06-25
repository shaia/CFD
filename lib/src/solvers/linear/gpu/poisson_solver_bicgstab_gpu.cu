/**
 * @file poisson_solver_bicgstab_gpu.cu
 * @brief BiCGSTAB Poisson solver — CUDA GPU backend
 *
 * Implements the poisson_solver_t interface on the GPU using Biconjugate Gradient
 * Stabilized (van der Vorst, 1992). Like the GPU CG backend it is a host-buffer
 * wrapper: upload x/rhs once, run the full iteration on-device, download x once —
 * avoiding per-iteration field transfers. Each iteration's dot-product reductions
 * still copy a single scalar back to the host and sync (same as GPU CG).
 *
 * The whole solve loop is expressed via host wrappers over the shared device
 * primitives in poisson_gpu_primitives.cuh (SPD matvec, residual, dot, axpy) plus
 * device-to-device copies, honoring the project's algorithm-primitive separation
 * rule — no new kernels are introduced.
 *
 * Sign convention matches the CPU BiCGSTAB reference (linear_solver_bicgstab.c):
 * the plain Laplacian is negative definite, so the operator is A = -Laplacian with
 * b = -rhs. lin_gpu_kernel_spd_laplacian computes A*x and lin_gpu_kernel_residual_vec
 * computes r = b - A*x = Laplacian(x) - rhs. The residual norm ||r|| equals the
 * Poisson residual ||rhs - Laplacian(x)||.
 *
 * Boundaries: all working vectors are zeroed once (their boundaries stay zero under
 * the interior-only primitives), and the Neumann BC is applied to x only at the start
 * and end of the solve — identical to the CPU BiCGSTAB and the GPU CG.
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/poisson_solver.h"

#include "poisson_gpu_primitives.cuh"

#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

extern "C" {
#include "../linear_solver_internal.h"
}

#define BICGSTAB_GPU_BREAKDOWN BICGSTAB_BREAKDOWN_THRESHOLD

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz, size;
    size_t stride_z;
    int k_start, k_end;  /* k_end inclusive (primitive convention) */
    double inv_dx2, inv_dy2, inv_dz2, factor;
    int block_x, block_y;

    double* d_x;       /* solution (in/out)     */
    double* d_rhs;     /* right-hand side       */
    double* d_r;       /* residual              */
    double* d_r_hat;   /* shadow residual r_hat */
    double* d_p;       /* search direction      */
    double* d_v;       /* A*p                   */
    double* d_s;       /* intermediate residual */
    double* d_t;       /* A*s                   */
    double* d_scalar;  /* reduction accumulator (single double) */
    cudaStream_t stream;
} poisson_bicgstab_gpu_ctx;

/* ============================================================================
 * PRIMITIVE HOST WRAPPERS (one kernel launch each)
 *
 * Thin wrappers over the shared primitives in poisson_gpu_primitives.cuh so the
 * solve loop below reads as straight-line linear algebra with no raw launches.
 * ============================================================================ */

namespace bicgstab_gpu_detail {

struct dims {
    size_t nx, ny, stride_z;
    int k_start, k_end;
    double inv_dx2, inv_dy2, inv_dz2, factor;
    cudaStream_t stream;
    dim3 grid, block;
};

/* out = A*in, A = -Laplacian (SPD operator). */
static inline void matvec(const dims& d, const double* d_in, double* d_out) {
    lin_gpu_kernel_spd_laplacian<<<d.grid, d.block, 0, d.stream>>>(
        d_in, d_out, d.nx, d.ny, d.stride_z, d.k_start, d.k_end,
        d.inv_dx2, d.inv_dy2, d.inv_dz2, d.factor);
}

/* r = b - A*x = Laplacian(x) - rhs. */
static inline void residual(const dims& d, const double* d_x, const double* d_rhs,
                            double* d_r) {
    lin_gpu_kernel_residual_vec<<<d.grid, d.block, 0, d.stream>>>(
        d_x, d_rhs, d_r, d.nx, d.ny, d.stride_z, d.k_start, d.k_end,
        d.inv_dx2, d.inv_dy2, d.inv_dz2, d.factor);
}

/* y += alpha * x (interior only). */
static inline void axpy(const dims& d, double alpha, const double* d_x, double* d_y) {
    lin_gpu_kernel_axpy<<<d.grid, d.block, 0, d.stream>>>(
        alpha, d_x, d_y, d.nx, d.ny, d.stride_z, d.k_start, d.k_end);
}

/* y = x + beta * y (interior only). */
static inline void xpay(const dims& d, const double* d_x, double beta, double* d_y) {
    lin_gpu_kernel_xpay<<<d.grid, d.block, 0, d.stream>>>(
        d_x, beta, d_y, d.nx, d.ny, d.stride_z, d.k_start, d.k_end);
}

/* Interior dot product (a,b) -> *out. Returns false on a CUDA failure. d_scalar is
 * a single-double device accumulator. */
static inline bool dot(const dims& d, const double* d_a, const double* d_b,
                       double* d_scalar, double* out) {
    size_t shmem = (size_t)d.block.x * d.block.y * sizeof(double);
    if (cudaMemsetAsync(d_scalar, 0, sizeof(double), d.stream) != cudaSuccess)
        return false;
    lin_gpu_kernel_dot<<<d.grid, d.block, shmem, d.stream>>>(
        d_a, d_b, d_scalar, d.nx, d.ny, d.stride_z, d.k_start, d.k_end);
    double h = 0.0;
    if (cudaMemcpyAsync(&h, d_scalar, sizeof(double), cudaMemcpyDeviceToHost, d.stream)
        != cudaSuccess)
        return false;
    if (cudaStreamSynchronize(d.stream) != cudaSuccess)
        return false;
    *out = h;
    return true;
}

}  // namespace bicgstab_gpu_detail

/* ============================================================================
 * INTERFACE IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t bicgstab_gpu_init(poisson_solver_t* solver,
                                      size_t nx, size_t ny, size_t nz,
                                      double dx, double dy, double dz,
                                      const poisson_solver_params_t* params) {
    (void)params;
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_bicgstab_gpu_ctx* ctx =
        (poisson_bicgstab_gpu_ctx*)calloc(1, sizeof(poisson_bicgstab_gpu_ctx));
    if (!ctx) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU BiCGSTAB solver context");
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
           && cudaMalloc(&ctx->d_r_hat, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_p, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_v, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_s, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_t, bytes) == cudaSuccess
           && cudaMalloc(&ctx->d_scalar, sizeof(double)) == cudaSuccess
           && cudaStreamCreate(&ctx->stream) == cudaSuccess;
    if (!ok) {
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_r);
        cudaFree(ctx->d_r_hat);
        cudaFree(ctx->d_p);
        cudaFree(ctx->d_v);
        cudaFree(ctx->d_s);
        cudaFree(ctx->d_t);
        cudaFree(ctx->d_scalar);
        free(ctx);
        cfd_set_error(CFD_ERROR_NOMEM, "GPU BiCGSTAB: device allocation failed");
        return CFD_ERROR_NOMEM;
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void bicgstab_gpu_destroy(poisson_solver_t* solver) {
    if (!solver || !solver->context)
        return;
    poisson_bicgstab_gpu_ctx* ctx = (poisson_bicgstab_gpu_ctx*)solver->context;
    if (ctx->stream)
        cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_rhs);
    cudaFree(ctx->d_r);
    cudaFree(ctx->d_r_hat);
    cudaFree(ctx->d_p);
    cudaFree(ctx->d_v);
    cudaFree(ctx->d_s);
    cudaFree(ctx->d_t);
    cudaFree(ctx->d_scalar);
    free(ctx);
    solver->context = NULL;
}

static cfd_status_t bicgstab_gpu_solve(poisson_solver_t* solver,
                                       double* x, double* x_temp, const double* rhs,
                                       poisson_solver_stats_t* stats) {
    (void)x_temp;  /* BiCGSTAB keeps all working vectors on-device */
    if (!solver || !x || !rhs)
        return CFD_ERROR_INVALID;
    poisson_bicgstab_gpu_ctx* c = (poisson_bicgstab_gpu_ctx*)solver->context;
    if (!c)
        return CFD_ERROR_INVALID;

    const poisson_solver_params_t* p = &solver->params;
    size_t bytes = c->size * sizeof(double);
    cudaStream_t stream = c->stream;

    bicgstab_gpu_detail::dims d;
    d.nx = c->nx;
    d.ny = c->ny;
    d.stride_z = c->stride_z;
    d.k_start = c->k_start;
    d.k_end = c->k_end;
    d.inv_dx2 = c->inv_dx2;
    d.inv_dy2 = c->inv_dy2;
    d.inv_dz2 = c->inv_dz2;
    d.factor = c->factor;
    d.stream = stream;
    d.block = dim3((unsigned)c->block_x, (unsigned)c->block_y);
    d.grid = dim3((unsigned)((c->nx - 2 + d.block.x - 1) / d.block.x),
                  (unsigned)((c->ny - 2 + d.block.y - 1) / d.block.y));

    /* Upload x and rhs. */
    if (cudaMemcpyAsync(c->d_x, x, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess
        || cudaMemcpyAsync(c->d_rhs, rhs, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return CFD_ERROR;

    /* Zero the working vectors so their boundaries stay 0 (interior-only ops). */
    if (cudaMemsetAsync(c->d_r, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(c->d_r_hat, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(c->d_p, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(c->d_v, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(c->d_s, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(c->d_t, 0, bytes, stream) != cudaSuccess)
        return CFD_ERROR;

    /* BC on the initial guess, then r0 = b - A x0, r_hat = r0. */
    bc_apply_scalar_3d_gpu(c->d_x, c->nx, c->ny, c->nz, BC_TYPE_NEUMANN, stream);
    bicgstab_gpu_detail::residual(d, c->d_x, c->d_rhs, c->d_r);
    if (cudaMemcpyAsync(c->d_r_hat, c->d_r, bytes, cudaMemcpyDeviceToDevice, stream)
        != cudaSuccess)
        return CFD_ERROR;

    double rho = 1.0, alpha = 1.0, omega = 1.0;

    double r_dot_r = 0.0;
    if (!bicgstab_gpu_detail::dot(d, c->d_r, c->d_r, c->d_scalar, &r_dot_r))
        return CFD_ERROR;
    double initial_res = sqrt(r_dot_r);

    double tol = p->tolerance * initial_res;
    if (tol < p->absolute_tolerance)
        tol = p->absolute_tolerance;

    int iter = 0;
    int converged = (initial_res < p->absolute_tolerance);
    int stagnated = 0;
    double res = initial_res;

    while (!converged && iter < p->max_iterations) {
        /* rho_new = (r_hat, r) */
        double rho_new = 0.0;
        if (!bicgstab_gpu_detail::dot(d, c->d_r_hat, c->d_r, c->d_scalar, &rho_new))
            return CFD_ERROR;
        if (fabs(rho_new) < BICGSTAB_GPU_BREAKDOWN) {
            stagnated = 1;
            break;
        }

        /* beta = (rho_new / rho) * (alpha / omega) */
        double beta = (rho_new / rho) * (alpha / omega);

        /* p = r + beta * (p - omega * v)  ==  p -= omega*v;  p = r + beta*p */
        bicgstab_gpu_detail::axpy(d, -omega, c->d_v, c->d_p);
        bicgstab_gpu_detail::xpay(d, c->d_r, beta, c->d_p);

        /* v = A * p */
        bicgstab_gpu_detail::matvec(d, c->d_p, c->d_v);

        /* alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = 0.0;
        if (!bicgstab_gpu_detail::dot(d, c->d_r_hat, c->d_v, c->d_scalar, &r_hat_dot_v))
            return CFD_ERROR;
        if (fabs(r_hat_dot_v) < BICGSTAB_GPU_BREAKDOWN) {
            stagnated = 1;
            break;
        }
        alpha = rho_new / r_hat_dot_v;

        /* s = r - alpha * v */
        if (cudaMemcpyAsync(c->d_s, c->d_r, bytes, cudaMemcpyDeviceToDevice, stream)
            != cudaSuccess)
            return CFD_ERROR;
        bicgstab_gpu_detail::axpy(d, -alpha, c->d_v, c->d_s);

        /* Early convergence test on s. */
        double s_dot_s = 0.0;
        if (!bicgstab_gpu_detail::dot(d, c->d_s, c->d_s, c->d_scalar, &s_dot_s))
            return CFD_ERROR;
        double s_norm = sqrt(s_dot_s);
        if (s_norm < tol || s_norm < p->absolute_tolerance) {
            bicgstab_gpu_detail::axpy(d, alpha, c->d_p, c->d_x);  /* x += alpha p */
            res = s_norm;
            iter++;
            converged = 1;
            break;
        }

        /* t = A * s */
        bicgstab_gpu_detail::matvec(d, c->d_s, c->d_t);

        /* omega = (t, s) / (t, t) */
        double t_dot_s = 0.0, t_dot_t = 0.0;
        if (!bicgstab_gpu_detail::dot(d, c->d_t, c->d_s, c->d_scalar, &t_dot_s)
            || !bicgstab_gpu_detail::dot(d, c->d_t, c->d_t, c->d_scalar, &t_dot_t))
            return CFD_ERROR;
        if (fabs(t_dot_t) < BICGSTAB_GPU_BREAKDOWN) {
            bicgstab_gpu_detail::axpy(d, alpha, c->d_p, c->d_x);  /* salvage progress */
            res = s_norm;
            stagnated = 1;
            iter++;
            break;
        }
        omega = t_dot_s / t_dot_t;

        /* x = x + alpha * p + omega * s */
        bicgstab_gpu_detail::axpy(d, alpha, c->d_p, c->d_x);
        bicgstab_gpu_detail::axpy(d, omega, c->d_s, c->d_x);

        /* r = s - omega * t */
        if (cudaMemcpyAsync(c->d_r, c->d_s, bytes, cudaMemcpyDeviceToDevice, stream)
            != cudaSuccess)
            return CFD_ERROR;
        bicgstab_gpu_detail::axpy(d, -omega, c->d_t, c->d_r);

        rho = rho_new;

        /* Residual norm + convergence check. */
        double rr = 0.0;
        if (!bicgstab_gpu_detail::dot(d, c->d_r, c->d_r, c->d_scalar, &rr))
            return CFD_ERROR;
        res = sqrt(rr);
        iter++;

        if (res < tol || res < p->absolute_tolerance) {
            converged = 1;
            break;
        }

        /* omega breakdown would divide by zero next iteration. */
        if (fabs(omega) < BICGSTAB_GPU_BREAKDOWN) {
            stagnated = 1;
            break;
        }
    }

    /* Final Neumann BC on the solution, then download. */
    bc_apply_scalar_3d_gpu(c->d_x, c->nx, c->ny, c->nz, BC_TYPE_NEUMANN, stream);
    if (cudaMemcpyAsync(x, c->d_x, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        return CFD_ERROR;
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return CFD_ERROR;

    if (stats) {
        stats->initial_residual = initial_res;
        stats->final_residual = res;
        stats->iterations = iter;
        stats->status = converged   ? POISSON_CONVERGED
                        : stagnated ? POISSON_STAGNATED
                                    : POISSON_MAX_ITER;
    }
    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/* ============================================================================
 * FACTORY
 * ============================================================================ */

extern "C" poisson_solver_t* create_bicgstab_gpu_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate GPU BiCGSTAB solver");
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_GPU;
    solver->description = "BiCGSTAB (CUDA GPU)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_GPU;
    solver->params = poisson_solver_params_default();

    solver->init = bicgstab_gpu_init;
    solver->destroy = bicgstab_gpu_destroy;
    solver->solve = bicgstab_gpu_solve;
    solver->iterate = NULL;
    solver->apply_bc = NULL;

    return solver;
}
