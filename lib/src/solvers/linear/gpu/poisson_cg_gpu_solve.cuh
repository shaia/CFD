/**
 * @file poisson_cg_gpu_solve.cuh
 * @brief Device-resident Conjugate Gradient solve loop (shared core)
 *
 * Runs the full CG iteration entirely on the GPU, operating on caller-supplied
 * device pointers — it performs no host<->device transfers of the field data.
 * This lets two callers share one CG driver:
 *   - poisson_solver_cg_gpu.cu wraps it (upload once / download once) to expose
 *     the host-buffer poisson_solver_t::solve interface.
 *   - the GPU projection solver calls it directly on its on-device pressure and
 *     RHS buffers, avoiding a device->host->device round-trip per step.
 *
 * The loop is expressed purely through host wrappers around the shared device
 * primitives in poisson_gpu_primitives.cuh (SPD matvec, residual, dot, axpy,
 * xpay), honoring the project's algorithm-primitive separation rule.
 *
 * Sign convention matches the CPU CG reference: the plain Laplacian is negative
 * definite, so CG runs on the SPD operator A = -Laplacian with b = -rhs. The
 * residual norm ||r|| equals ||rhs - Laplacian(x)|| (the Poisson residual).
 *
 * Boundaries: r/p/Ap stay zero on the boundary (interior-only operators, scratch
 * is zeroed here), and the Neumann BC is applied to x at the start and end of the
 * solve — identical to the CPU CG.
 *
 * Declared `static` so each translation unit including this header gets its own
 * copy — no multiple-definition clashes at link time.
 */

#ifndef CFD_POISSON_CG_GPU_SOLVE_CUH
#define CFD_POISSON_CG_GPU_SOLVE_CUH

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"

#include "poisson_gpu_primitives.cuh"

#include <cmath>
#include <cuda_runtime.h>

#define CG_GPU_SOLVE_BREAKDOWN 1e-30

/* ============================================================================
 * PARAMETERS / RESULT
 * ============================================================================ */

typedef struct {
    double tolerance;          /* relative: stop when ||r|| <= tolerance * ||r0|| */
    double absolute_tolerance; /* absolute floor on ||r||                          */
    int max_iterations;
} cg_gpu_solve_params;

typedef struct {
    int iterations;
    int converged;
    int stagnated; /* broke down on a singular/near-zero direction before converging */
    double initial_residual;
    double final_residual;
} cg_gpu_solve_result;

/* ============================================================================
 * PRIMITIVE HOST WRAPPERS (one kernel launch each)
 *
 * Grouped in a small context-of-pointers-free form: each takes the device
 * pointers and grid metadata it needs explicitly, so the same wrappers serve
 * any caller's buffers/stream.
 * ============================================================================ */

namespace cg_gpu_detail {

struct dims {
    size_t nx, ny, stride_z;
    int k_start, k_end;
    double inv_dx2, inv_dy2, inv_dz2, factor;
    cudaStream_t stream;
};

static inline void matvec(const dims& d, const double* d_in, double* d_out,
                          dim3 grid, dim3 block) {
    lin_gpu_kernel_spd_laplacian<<<grid, block, 0, d.stream>>>(
        d_in, d_out, d.nx, d.ny, d.stride_z, d.k_start, d.k_end,
        d.inv_dx2, d.inv_dy2, d.inv_dz2, d.factor);
}

static inline void residual(const dims& d, const double* d_x, const double* d_rhs,
                            double* d_r, dim3 grid, dim3 block) {
    lin_gpu_kernel_residual_vec<<<grid, block, 0, d.stream>>>(
        d_x, d_rhs, d_r, d.nx, d.ny, d.stride_z, d.k_start, d.k_end,
        d.inv_dx2, d.inv_dy2, d.inv_dz2, d.factor);
}

static inline void axpy(const dims& d, double alpha, const double* d_x, double* d_y,
                        dim3 grid, dim3 block) {
    lin_gpu_kernel_axpy<<<grid, block, 0, d.stream>>>(
        alpha, d_x, d_y, d.nx, d.ny, d.stride_z, d.k_start, d.k_end);
}

static inline void xpay(const dims& d, const double* d_x, double beta, double* d_y,
                        dim3 grid, dim3 block) {
    lin_gpu_kernel_xpay<<<grid, block, 0, d.stream>>>(
        d_x, beta, d_y, d.nx, d.ny, d.stride_z, d.k_start, d.k_end);
}

/* Interior dot product (a,b), written to *out. Returns false on CUDA failure so
 * the caller aborts the solve. d_scalar is a single-double device accumulator. */
static inline bool dot(const dims& d, const double* d_a, const double* d_b,
                       double* d_scalar, dim3 grid, dim3 block, double* out) {
    size_t shmem = (size_t)block.x * block.y * sizeof(double);
    if (cudaMemsetAsync(d_scalar, 0, sizeof(double), d.stream) != cudaSuccess)
        return false;
    lin_gpu_kernel_dot<<<grid, block, shmem, d.stream>>>(
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

}  // namespace cg_gpu_detail

/* ============================================================================
 * DEVICE-RESIDENT CG SOLVE
 * ============================================================================ */

/**
 * Solve nabla^2 x = rhs on the GPU with Conjugate Gradient, entirely on-device.
 *
 * @param d_x       solution, in/out (initial guess on entry); Neumann BC applied
 *                  to its boundary at start and end.
 * @param d_rhs     right-hand side (const input).
 * @param d_r,d_p,d_Ap  caller-owned scratch, each field-sized (nx*ny*max(nz,1));
 *                  their boundaries are zeroed here and stay zero (interior ops).
 * @param d_scalar  single-double device accumulator for reductions.
 * @param out       optional; iteration count / residuals / converged flag.
 * @return CFD_SUCCESS on convergence, CFD_ERROR_MAX_ITER if the cap/breakdown was
 *         hit without reaching tolerance, CFD_ERROR on a CUDA failure.
 */
static cfd_status_t cg_gpu_solve_device(
    double* d_x, const double* d_rhs,
    double* d_r, double* d_p, double* d_Ap, double* d_scalar,
    size_t nx, size_t ny, size_t nz, size_t stride_z, int k_start, int k_end,
    double inv_dx2, double inv_dy2, double inv_dz2, double factor,
    int block_x, int block_y, cudaStream_t stream,
    const cg_gpu_solve_params* params, cg_gpu_solve_result* out) {
    if (!d_x || !d_rhs || !d_r || !d_p || !d_Ap || !d_scalar || !params)
        return CFD_ERROR_INVALID;

    cg_gpu_detail::dims d{nx, ny, stride_z, k_start, k_end,
                          inv_dx2, inv_dy2, inv_dz2, factor, stream};

    size_t size = nx * ny * (nz > 0 ? nz : 1);
    size_t bytes = size * sizeof(double);

    dim3 block((unsigned)block_x, (unsigned)block_y);
    dim3 grid((unsigned)((nx - 2 + block.x - 1) / block.x),
              (unsigned)((ny - 2 + block.y - 1) / block.y));

    /* Zero the working vectors so their boundaries stay 0 (interior-only ops). */
    if (cudaMemsetAsync(d_r, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(d_p, 0, bytes, stream) != cudaSuccess
        || cudaMemsetAsync(d_Ap, 0, bytes, stream) != cudaSuccess)
        return CFD_ERROR;

    /* BC on the initial guess, then r0 = b - A x0, p0 = r0. */
    bc_apply_scalar_3d_gpu(d_x, nx, ny, nz, BC_TYPE_NEUMANN, stream);
    cg_gpu_detail::residual(d, d_x, d_rhs, d_r, grid, block);
    if (cudaMemcpyAsync(d_p, d_r, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
        return CFD_ERROR;

    double rho = 0.0;
    if (!cg_gpu_detail::dot(d, d_r, d_r, d_scalar, grid, block, &rho))
        return CFD_ERROR;
    double initial_res = sqrt(rho);

    double tol = params->tolerance * initial_res;
    if (tol < params->absolute_tolerance)
        tol = params->absolute_tolerance;

    double res = initial_res;
    int iter = 0;
    int converged = (initial_res < params->absolute_tolerance);
    int stagnated = 0;

    while (!converged && iter < params->max_iterations) {
        cg_gpu_detail::matvec(d, d_p, d_Ap, grid, block);          /* Ap = A p      */
        double p_dot_Ap = 0.0;
        if (!cg_gpu_detail::dot(d, d_p, d_Ap, d_scalar, grid, block, &p_dot_Ap))
            return CFD_ERROR;
        if (fabs(p_dot_Ap) < CG_GPU_SOLVE_BREAKDOWN) {
            stagnated = 1;  /* singular direction */
            break;
        }

        double alpha = rho / p_dot_Ap;
        cg_gpu_detail::axpy(d, alpha, d_p, d_x, grid, block);      /* x += alpha p  */
        cg_gpu_detail::axpy(d, -alpha, d_Ap, d_r, grid, block);    /* r -= alpha Ap */

        double rho_new = 0.0;
        if (!cg_gpu_detail::dot(d, d_r, d_r, d_scalar, grid, block, &rho_new))
            return CFD_ERROR;
        res = sqrt(rho_new);
        iter++;

        if (res < tol || res < params->absolute_tolerance) {
            converged = 1;
            break;
        }
        if (fabs(rho) < CG_GPU_SOLVE_BREAKDOWN) {
            stagnated = 1;
            break;
        }

        double beta = rho_new / rho;
        cg_gpu_detail::xpay(d, d_r, beta, d_p, grid, block);       /* p = r + beta p */
        rho = rho_new;
    }

    /* Final Neumann BC on the solution. (No download — caller owns the buffer.) */
    bc_apply_scalar_3d_gpu(d_x, nx, ny, nz, BC_TYPE_NEUMANN, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        return CFD_ERROR;

    if (out) {
        out->iterations = iter;
        out->converged = converged;
        out->stagnated = stagnated;
        out->initial_residual = initial_res;
        out->final_residual = res;
    }
    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

#endif /* CFD_POISSON_CG_GPU_SOLVE_CUH */
