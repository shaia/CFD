/**
 * @file poisson_gpu_primitives.cuh
 * @brief Shared CUDA primitive kernels for the GPU linear-solver backend
 *
 * These are the device-side building blocks used by the GPU Poisson solvers
 * (Jacobi, CG). Keeping them in one header — rather than inline in each solver —
 * enforces the project's algorithm-primitive separation rule: the high-level
 * solve loops are expressed purely via host wrappers around these primitives and
 * contain no raw kernel launches of their own.
 *
 * All kernels operate on the constant-coefficient 5/7-point Laplacian on a
 * uniform structured grid, indexing interior cells [1, n-1) in each dimension.
 * The diagonal coefficient is `factor = 2*(1/dx^2 + 1/dy^2 + 1/dz^2)`; for a 2D
 * grid the caller passes inv_dz2 = 0 and stride_z = 0 so the z-terms vanish.
 *
 * Kernels are declared `static` so each translation unit including this header
 * gets its own copy — no multiple-definition clashes at link time.
 */

#ifndef CFD_POISSON_GPU_PRIMITIVES_CUH
#define CFD_POISSON_GPU_PRIMITIVES_CUH

#include "cfd/core/indexing.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* ============================================================================
 * DEVICE KERNELS
 * ============================================================================ */

/**
 * Jacobi smoother: one sweep of nabla^2 x = rhs.
 *
 * Reads x_old, writes x_new (double-buffered, fully parallel).
 *   x_new = (sum_neighbors - rhs) / factor
 * where sum_neighbors = (x[i+1]+x[i-1])/dx^2 + ... and factor is the diagonal.
 */
static __global__ void lin_gpu_kernel_jacobi(const double* __restrict__ x_old,
                                             double* __restrict__ x_new,
                                             const double* __restrict__ rhs,
                                             size_t nx, size_t ny,
                                             size_t stride_z, int k_start, int k_end,
                                             double inv_dx2, double inv_dy2, double inv_dz2,
                                             double inv_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            double sum = (x_old[idx + 1] + x_old[idx - 1]) * inv_dx2
                       + (x_old[idx + nx] + x_old[idx - nx]) * inv_dy2
                       + (x_old[idx + stride_z] + x_old[idx - stride_z]) * inv_dz2;
            x_new[idx] = (sum - rhs[idx]) * inv_factor;
        }
    }
}

/**
 * Apply the Laplacian operator: out = A*x where A is the 5/7-point Laplacian.
 *   out = sum_neighbors - factor*x
 * Used by CG (out = A*p). Interior-only; boundaries must be set by the caller's
 * BC pass before this kernel reads boundary-adjacent cells.
 */
static __global__ void lin_gpu_kernel_laplacian(const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t nx, size_t ny,
                                                size_t stride_z, int k_start, int k_end,
                                                double inv_dx2, double inv_dy2, double inv_dz2,
                                                double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            double sum = (x[idx + 1] + x[idx - 1]) * inv_dx2
                       + (x[idx + nx] + x[idx - nx]) * inv_dy2
                       + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2;
            out[idx] = sum - factor * x[idx];
        }
    }
}

/**
 * SPD operator A = -Laplacian:  out = factor*x - sum_neighbors.
 *
 * The plain Laplacian is negative-definite, so CG operates on A = -Laplacian,
 * which is symmetric positive-definite (on the space orthogonal to constants).
 * This matches the CPU CG convention (A = -nabla^2, b = -rhs). Interior-only;
 * boundary cells of p stay zero (set once by calloc/memset) so no BC pass is
 * needed between matvecs.
 */
static __global__ void lin_gpu_kernel_spd_laplacian(const double* __restrict__ x,
                                                    double* __restrict__ out,
                                                    size_t nx, size_t ny,
                                                    size_t stride_z, int k_start, int k_end,
                                                    double inv_dx2, double inv_dy2, double inv_dz2,
                                                    double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            double sum = (x[idx + 1] + x[idx - 1]) * inv_dx2
                       + (x[idx + nx] + x[idx - nx]) * inv_dy2
                       + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2;
            out[idx] = factor * x[idx] - sum;
        }
    }
}

/**
 * CG initial residual vector: r = b - A*x with A = -Laplacian, b = -rhs, i.e.
 *   r = Laplacian(x) - rhs = (sum_neighbors - factor*x) - rhs.
 * Boundary cells of x must be set (Neumann) before launch; r is written on the
 * interior only.
 */
static __global__ void lin_gpu_kernel_residual_vec(const double* __restrict__ x,
                                                   const double* __restrict__ rhs,
                                                   double* __restrict__ r,
                                                   size_t nx, size_t ny,
                                                   size_t stride_z, int k_start, int k_end,
                                                   double inv_dx2, double inv_dy2, double inv_dz2,
                                                   double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            double sum = (x[idx + 1] + x[idx - 1]) * inv_dx2
                       + (x[idx + nx] + x[idx - nx]) * inv_dy2
                       + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2;
            r[idx] = (sum - factor * x[idx]) - rhs[idx];
        }
    }
}

/**
 * Block reduction helper: reduce sdata[0..nthreads) into sdata[0].
 * Handles non-power-of-two thread counts (block dims are user-configurable).
 */
static __device__ inline void lin_gpu_block_reduce(double* sdata, int tid, unsigned int nthreads) {
    __syncthreads();
    unsigned int s = 1;
    while (s < nthreads)
        s <<= 1;
    for (s >>= 1; s > 0; s >>= 1) {
        if ((unsigned int)tid < s && (unsigned int)tid + s < nthreads)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

/**
 * Residual sum-of-squares: accumulates sum over the interior of
 *   r = rhs - (A*x),   A*x = sum_neighbors - factor*x
 * into a single device scalar via block reduction + atomicAdd. Caller takes
 * sqrt() of the result to get the L2 residual norm. out_sumsq must be zeroed
 * before launch. Shared memory: blockDim.x*blockDim.y doubles.
 */
static __global__ void lin_gpu_kernel_residual_sq(const double* __restrict__ x,
                                                  const double* __restrict__ rhs,
                                                  double* __restrict__ out_sumsq,
                                                  size_t nx, size_t ny,
                                                  size_t stride_z, int k_start, int k_end,
                                                  double inv_dx2, double inv_dy2, double inv_dz2,
                                                  double factor) {
    extern __shared__ double sdata[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double local = 0.0;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            double sum = (x[idx + 1] + x[idx - 1]) * inv_dx2
                       + (x[idx + nx] + x[idx - nx]) * inv_dy2
                       + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2;
            double r = rhs[idx] - (sum - factor * x[idx]);
            local += r * r;
        }
    }
    sdata[tid] = local;
    lin_gpu_block_reduce(sdata, tid, blockDim.x * blockDim.y);
    if (tid == 0)
        atomicAdd(out_sumsq, sdata[0]);
}

/**
 * Interior dot product: accumulates sum over the interior of a[idx]*b[idx]
 * into out_sum via block reduction + atomicAdd. out_sum must be zeroed before
 * launch. Used by CG for (r,r) and (p,Ap). Shared mem: blockDim.x*blockDim.y doubles.
 */
static __global__ void lin_gpu_kernel_dot(const double* __restrict__ a,
                                          const double* __restrict__ b,
                                          double* __restrict__ out_sum,
                                          size_t nx, size_t ny,
                                          size_t stride_z, int k_start, int k_end) {
    extern __shared__ double sdata[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    double local = 0.0;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            local += a[idx] * b[idx];
        }
    }
    sdata[tid] = local;
    lin_gpu_block_reduce(sdata, tid, blockDim.x * blockDim.y);
    if (tid == 0)
        atomicAdd(out_sum, sdata[0]);
}

/**
 * AXPY over the interior: y += alpha * x.
 */
static __global__ void lin_gpu_kernel_axpy(double alpha,
                                           const double* __restrict__ x,
                                           double* __restrict__ y,
                                           size_t nx, size_t ny,
                                           size_t stride_z, int k_start, int k_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            y[idx] += alpha * x[idx];
        }
    }
}

/**
 * XPAY over the interior: y = x + beta * y  (used by CG to update the search
 * direction p = r + beta*p).
 */
static __global__ void lin_gpu_kernel_xpay(const double* __restrict__ x,
                                           double beta,
                                           double* __restrict__ y,
                                           size_t nx, size_t ny,
                                           size_t stride_z, int k_start, int k_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < (int)nx - 1 && j < (int)ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);
            y[idx] = x[idx] + beta * y[idx];
        }
    }
}

#endif /* CFD_POISSON_GPU_PRIMITIVES_CUH */
