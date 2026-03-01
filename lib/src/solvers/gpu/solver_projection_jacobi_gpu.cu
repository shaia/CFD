/**
 * GPU-Accelerated Solver Implementation (CUDA) - Optimized
 *
 * Key optimizations:
 * - Jacobi iteration for Poisson (fully parallel, no red-black dependency)
 * - Reduced Poisson iterations (50 instead of 1000)
 * - Fixed predictor step logic
 * - Pointer swapping instead of memcpy for Poisson iterations
 * - Unified boundary conditions via bc_apply_*_gpu() functions
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "cfd/core/gpu_device.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>


#define MAX_VELOCITY 100.0

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            return CFD_ERROR;                                                \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            return;                                                          \
        }                                                                    \
    } while (0)

// Internal context implementation structure
struct gpu_solver_context_impl {
    size_t nx, ny, nz, size;
    size_t stride_z;          // nx*ny for 3D, 0 for 2D
    int k_start, k_end;      // z-loop bounds for interior points
    double* d_u;
    double* d_v;
    double* d_w;
    double* d_p;
    double* d_rho;
    double* d_u_new;
    double* d_v_new;
    double* d_w_new;
    double* d_p_new;
    double* d_rhs;
    double* d_residual;
    double* d_x;
    double* d_y;
    double* d_z;
    double* d_dx;
    double* d_dy;
    double* d_dz;
    // Boundary value storage for preserving caller-set BCs
    double* d_u_bc;  // Stores u boundary values from initial upload
    double* d_v_bc;  // Stores v boundary values from initial upload
    double* d_w_bc;  // Stores w boundary values from initial upload
    gpu_config_t config;
    gpu_solver_stats_t stats;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    int initialized;
    int memory_allocated;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

// NOTE: Boundary condition kernels have been moved to boundary_conditions_gpu.cu
// Use bc_apply_scalar_gpu() and bc_apply_velocity_gpu() from the unified BC layer.

__global__ void kernel_compute_divergence(const double* __restrict__ u,
                                          const double* __restrict__ v,
                                          const double* __restrict__ w,
                                          double* __restrict__ div,
                                          size_t nx, size_t ny,
                                          size_t stride_z, int k_start, int k_end,
                                          double inv_2dx, double inv_2dy, double inv_2dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            div[idx] = (u[idx + 1] - u[idx - 1]) * inv_2dx
                     + (v[idx + nx] - v[idx - nx]) * inv_2dy
                     + (w[idx + stride_z] - w[idx - stride_z]) * inv_2dz;
        }
    }
}

// Fast Jacobi iteration - fully parallel (7-point stencil for 3D)
__global__ void kernel_poisson_jacobi(const double* __restrict__ p_old, double* __restrict__ p_new,
                                      const double* __restrict__ rhs, size_t nx, size_t ny,
                                      size_t stride_z, int k_start, int k_end,
                                      double inv_dx2, double inv_dy2, double inv_dz2,
                                      double inv_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            double sum = (p_old[idx + 1] + p_old[idx - 1]) * inv_dx2
                       + (p_old[idx + nx] + p_old[idx - nx]) * inv_dy2
                       + (p_old[idx + stride_z] + p_old[idx - stride_z]) * inv_dz2;
            p_new[idx] = (sum - rhs[idx]) * inv_factor;
        }
    }
}

// Predictor step - compute u*, v*, w* without pressure
__global__ void kernel_predictor(const double* __restrict__ u, const double* __restrict__ v,
                                 const double* __restrict__ w,
                                 double* __restrict__ u_star, double* __restrict__ v_star,
                                 double* __restrict__ w_star,
                                 size_t nx, size_t ny,
                                 size_t stride_z, int k_start, int k_end,
                                 double dt, double nu,
                                 double inv_2dx, double inv_2dy, double inv_2dz,
                                 double inv_dx2, double inv_dy2, double inv_dz2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            double u_c = u[idx], v_c = v[idx], w_c = w[idx];

            // u-momentum
            double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
            double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
            double du_dz = (u[idx + stride_z] - u[idx - stride_z]) * inv_2dz;
            double d2u = (u[idx + 1] - 2.0 * u_c + u[idx - 1]) * inv_dx2
                       + (u[idx + nx] - 2.0 * u_c + u[idx - nx]) * inv_dy2
                       + (u[idx + stride_z] - 2.0 * u_c + u[idx - stride_z]) * inv_dz2;
            double u_new = u_c + dt * (-(u_c * du_dx + v_c * du_dy + w_c * du_dz) + nu * d2u);

            // v-momentum
            double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
            double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;
            double dv_dz = (v[idx + stride_z] - v[idx - stride_z]) * inv_2dz;
            double d2v = (v[idx + 1] - 2.0 * v_c + v[idx - 1]) * inv_dx2
                       + (v[idx + nx] - 2.0 * v_c + v[idx - nx]) * inv_dy2
                       + (v[idx + stride_z] - 2.0 * v_c + v[idx - stride_z]) * inv_dz2;
            double v_new = v_c + dt * (-(u_c * dv_dx + v_c * dv_dy + w_c * dv_dz) + nu * d2v);

            // w-momentum
            double dw_dx = (w[idx + 1] - w[idx - 1]) * inv_2dx;
            double dw_dy = (w[idx + nx] - w[idx - nx]) * inv_2dy;
            double dw_dz = (w[idx + stride_z] - w[idx - stride_z]) * inv_2dz;
            double d2w = (w[idx + 1] - 2.0 * w_c + w[idx - 1]) * inv_dx2
                       + (w[idx + nx] - 2.0 * w_c + w[idx - nx]) * inv_dy2
                       + (w[idx + stride_z] - 2.0 * w_c + w[idx - stride_z]) * inv_dz2;
            double w_new = w_c + dt * (-(u_c * dw_dx + v_c * dw_dy + w_c * dw_dz) + nu * d2w);

            u_star[idx] = ::fmax((double)-MAX_VELOCITY, ::fmin((double)MAX_VELOCITY, (double)u_new));
            v_star[idx] = ::fmax((double)-MAX_VELOCITY, ::fmin((double)MAX_VELOCITY, (double)v_new));
            w_star[idx] = ::fmax((double)-MAX_VELOCITY, ::fmin((double)MAX_VELOCITY, (double)w_new));
        }
    }
}

__global__ void kernel_velocity_rhs(const double* __restrict__ u, const double* __restrict__ v,
                                    const double* __restrict__ w,
                                    const double* __restrict__ p,
                                    double* __restrict__ u_rhs, double* __restrict__ v_rhs,
                                    double* __restrict__ w_rhs,
                                    size_t nx, size_t ny,
                                    size_t stride_z, int k_start, int k_end,
                                    double inv_2dx, double inv_2dy, double inv_2dz,
                                    double inv_dx2, double inv_dy2, double inv_dz2,
                                    double nu, double inv_rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            double u_c = u[idx], v_c = v[idx], w_c = w[idx];

            double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
            double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
            double du_dz = (u[idx + stride_z] - u[idx - stride_z]) * inv_2dz;
            double d2u = (u[idx + 1] - 2.0 * u_c + u[idx - 1]) * inv_dx2
                       + (u[idx + nx] - 2.0 * u_c + u[idx - nx]) * inv_dy2
                       + (u[idx + stride_z] - 2.0 * u_c + u[idx - stride_z]) * inv_dz2;

            double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
            double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;
            double dv_dz = (v[idx + stride_z] - v[idx - stride_z]) * inv_2dz;
            double d2v = (v[idx + 1] - 2.0 * v_c + v[idx - 1]) * inv_dx2
                       + (v[idx + nx] - 2.0 * v_c + v[idx - nx]) * inv_dy2
                       + (v[idx + stride_z] - 2.0 * v_c + v[idx - stride_z]) * inv_dz2;

            double dw_dx = (w[idx + 1] - w[idx - 1]) * inv_2dx;
            double dw_dy = (w[idx + nx] - w[idx - nx]) * inv_2dy;
            double dw_dz = (w[idx + stride_z] - w[idx - stride_z]) * inv_2dz;
            double d2w = (w[idx + 1] - 2.0 * w_c + w[idx - 1]) * inv_dx2
                       + (w[idx + nx] - 2.0 * w_c + w[idx - nx]) * inv_dy2
                       + (w[idx + stride_z] - 2.0 * w_c + w[idx - stride_z]) * inv_dz2;

            double dp_dx = (p[idx + 1] - p[idx - 1]) * inv_2dx;
            double dp_dy = (p[idx + nx] - p[idx - nx]) * inv_2dy;
            double dp_dz = (p[idx + stride_z] - p[idx - stride_z]) * inv_2dz;

            u_rhs[idx] = -(u_c * du_dx + v_c * du_dy + w_c * du_dz) + nu * d2u - inv_rho * dp_dx;
            v_rhs[idx] = -(u_c * dv_dx + v_c * dv_dy + w_c * dv_dz) + nu * d2v - inv_rho * dp_dy;
            w_rhs[idx] = -(u_c * dw_dx + v_c * dw_dy + w_c * dw_dz) + nu * d2w - inv_rho * dp_dz;
        }
    }
}

__global__ void kernel_velocity_update(double* __restrict__ u, double* __restrict__ v,
                                       double* __restrict__ w,
                                       const double* __restrict__ u_rhs,
                                       const double* __restrict__ v_rhs,
                                       const double* __restrict__ w_rhs,
                                       size_t nx, size_t ny,
                                       size_t stride_z, int k_start, int k_end,
                                       double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            u[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(u[idx] + dt * u_rhs[idx])));
            v[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(v[idx] + dt * v_rhs[idx])));
            w[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(w[idx] + dt * w_rhs[idx])));
        }
    }
}

__global__ void kernel_projection_correct(double* __restrict__ u, double* __restrict__ v,
                                          double* __restrict__ w,
                                          const double* __restrict__ u_star,
                                          const double* __restrict__ v_star,
                                          const double* __restrict__ w_star,
                                          const double* __restrict__ p, size_t nx, size_t ny,
                                          size_t stride_z, int k_start, int k_end,
                                          double dt_rho,
                                          double inv_2dx, double inv_2dy, double inv_2dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            double dp_dx = (p[idx + 1] - p[idx - 1]) * inv_2dx;
            double dp_dy = (p[idx + nx] - p[idx - nx]) * inv_2dy;
            double dp_dz = (p[idx + stride_z] - p[idx - stride_z]) * inv_2dz;
            u[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(u_star[idx] - dt_rho * dp_dx)));
            v[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(v_star[idx] - dt_rho * dp_dy)));
            w[idx] = ::fmax((double)-MAX_VELOCITY,
                            ::fmin((double)MAX_VELOCITY, (double)(w_star[idx] - dt_rho * dp_dz)));
        }
    }
}

__global__ void kernel_pressure_update(double* __restrict__ p, const double* __restrict__ div,
                                       size_t nx, size_t ny,
                                       size_t stride_z, int k_start, int k_end,
                                       double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            p[idx] -= factor * div[idx];
        }
    }
}

__global__ void kernel_scale_rhs(double* __restrict__ rhs, size_t nx, size_t ny,
                                  size_t stride_z, int k_start, int k_end,
                                  double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            rhs[idx] *= scale;
        }
    }
}

// Copy boundary values from stored BC arrays to velocity arrays
// This preserves caller-set boundary conditions (e.g., Dirichlet for lid-driven cavity)
// For 3D: copies per-plane x/y boundaries for all k, plus z-face boundaries
__global__ void kernel_copy_velocity_boundaries(double* __restrict__ u, double* __restrict__ v,
                                                 double* __restrict__ w,
                                                 const double* __restrict__ u_bc,
                                                 const double* __restrict__ v_bc,
                                                 const double* __restrict__ w_bc,
                                                 size_t nx, size_t ny, size_t nz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;

    // Per-plane bottom/top (j=0, j=ny-1): one thread per (i, k) pair
    if (tid < (int)(nx * nz)) {
        int k = tid / (int)nx;
        int i = tid % (int)nx;
        size_t base = k * plane;
        size_t bot = base + i;
        size_t top = base + IDX_2D(i, ny - 1, nx);
        u[bot] = u_bc[bot]; u[top] = u_bc[top];
        v[bot] = v_bc[bot]; v[top] = v_bc[top];
        if (w && w_bc) { w[bot] = w_bc[bot]; w[top] = w_bc[top]; }
    }

    // Per-plane left/right (i=0, i=nx-1): one thread per (j, k) pair
    if (tid < (int)(ny * nz)) {
        int k = tid / (int)ny;
        int j = tid % (int)ny;
        size_t base = k * plane;
        size_t left = base + IDX_2D(0, j, nx);
        size_t right = base + IDX_2D(nx - 1, j, nx);
        u[left] = u_bc[left]; u[right] = u_bc[right];
        v[left] = v_bc[left]; v[right] = v_bc[right];
        if (w && w_bc) { w[left] = w_bc[left]; w[right] = w_bc[right]; }
    }

    // Z-face boundaries (k=0, k=nz-1): one thread per (i, j) pair
    if (nz > 1 && tid < (int)plane) {
        size_t zbot = tid;
        size_t ztop = (nz - 1) * plane + tid;
        u[zbot] = u_bc[zbot]; u[ztop] = u_bc[ztop];
        v[zbot] = v_bc[zbot]; v[ztop] = v_bc[ztop];
        if (w && w_bc) { w[zbot] = w_bc[zbot]; w[ztop] = w_bc[ztop]; }
    }
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

gpu_config_t gpu_config_default(void) {
    gpu_config_t config;
    config.enable_gpu = 1;
    config.min_grid_size = 10000;
    config.min_steps = 10;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.poisson_max_iter = 50;  // Reduced from 1000
    config.poisson_tolerance = 1e-4;
    config.persistent_memory = 1;
    config.async_transfers = 1;
    config.sync_after_kernel = 0;
    config.verbose = 0;
    return config;
}

int gpu_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        return 1;
    }
    // Secondary check - try to get current device
    int device = -1;
    err = cudaGetDevice(&device);
    if (err == cudaSuccess && device >= 0) {
        return 1;
    }
    // Clear any CUDA errors
    cudaGetLastError();
    return 0;
}

int gpu_get_device_info(gpu_device_info_t* info, int max_devices) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
        return 0;
    int count = (device_count < max_devices) ? device_count : max_devices;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            info[i].device_id = i;
            snprintf(info[i].name, sizeof(info[i].name), "%s", prop.name);
            info[i].total_memory = prop.totalGlobalMem;
            info[i].compute_capability_major = prop.major;
            info[i].compute_capability_minor = prop.minor;
            info[i].multiprocessor_count = prop.multiProcessorCount;
            info[i].max_threads_per_block = prop.maxThreadsPerBlock;
            info[i].warp_size = prop.warpSize;
            info[i].is_available = 1;
            size_t free_mem, total_mem;
            cudaSetDevice(i);
            info[i].free_memory =
                (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) ? free_mem : 0;
        }
    }
    return count;
}

cfd_status_t gpu_select_device(int device_id) {
    return (cudaSetDevice(device_id) == cudaSuccess) ? CFD_SUCCESS : CFD_ERROR;
}

int gpu_should_use(const gpu_config_t* config, size_t nx, size_t ny, size_t nz, int num_steps) {
    if (!config || !config->enable_gpu)
        return 0;
    if (!gpu_is_available())
        return 0;
    if (nz == 2)
        return 0;  // nz==2 invalid: need nz==1 (2D) or nz>=3 (3D)
    if (nx * ny * nz < config->min_grid_size)
        return 0;
    if (num_steps < config->min_steps)
        return 0;
    return 1;
}

gpu_solver_context_t* gpu_solver_create(size_t nx, size_t ny, size_t nz, const gpu_config_t* config) {
    if (!gpu_is_available())
        return nullptr;
    if (nz == 2) {
        cfd_set_error(CFD_ERROR_INVALID, "GPU solver requires nz==1 (2D) or nz>=3 (3D), got nz==2");
        return nullptr;
    }
    struct gpu_solver_context_impl* ctx =
        (struct gpu_solver_context_impl*)cfd_calloc(1, sizeof(struct gpu_solver_context_impl));
    if (!ctx)
        return nullptr;
    ctx->nx = nx;
    ctx->ny = ny;
    ctx->nz = nz;
    ctx->size = nx * ny * nz;
    // Branch-free 3D bounds: when nz==1, stride_z=0 and k_start=k_end=0
    if (nz > 1) {
        ctx->stride_z = nx * ny;
        ctx->k_start = 1;
        ctx->k_end = (int)(nz - 2);
    } else {
        ctx->stride_z = 0;
        ctx->k_start = 0;
        ctx->k_end = 0;
    }
    ctx->config = config ? *config : gpu_config_default();
    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        cfd_free(ctx);
        return nullptr;
    }
    cudaEventCreate(&ctx->start_event);
    cudaEventCreate(&ctx->stop_event);
    size_t bytes = ctx->size * sizeof(double);
    size_t nz_coords = (nz > 1) ? nz : 0;
    size_t nz_spacing = (nz > 1) ? (nz - 1) : 0;
    if (cudaMalloc(&ctx->d_u, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_v, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_w, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_p, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_rho, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_u_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_v_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_w_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_p_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_rhs, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_residual, sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_x, nx * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_y, ny * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_dx, (nx - 1) * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_dy, (ny - 1) * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_u_bc, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_v_bc, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_w_bc, bytes) != cudaSuccess ||
        (nz_coords > 0 && cudaMalloc(&ctx->d_z, nz_coords * sizeof(double)) != cudaSuccess) ||
        (nz_spacing > 0 && cudaMalloc(&ctx->d_dz, nz_spacing * sizeof(double)) != cudaSuccess)) {
        gpu_solver_destroy((gpu_solver_context_t*)ctx);
        return nullptr;
    }
    ctx->memory_allocated = 1;
    ctx->initialized = 1;
    // 11 field-sized buffers: u,v,w,p,rho,u_new,v_new,w_new,p_new,rhs + u_bc,v_bc,w_bc = 13
    ctx->stats.memory_allocated =
        bytes * 13 + (nx + ny + nz_coords + nx - 1 + ny - 1 + nz_spacing + 1) * sizeof(double);
    return (gpu_solver_context_t*)ctx;
}

void gpu_solver_destroy(gpu_solver_context_t* ctx_void) {
    if (!ctx_void)
        return;
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    cudaFree(ctx->d_u);
    cudaFree(ctx->d_v);
    cudaFree(ctx->d_w);
    cudaFree(ctx->d_p);
    cudaFree(ctx->d_rho);
    cudaFree(ctx->d_u_new);
    cudaFree(ctx->d_v_new);
    cudaFree(ctx->d_w_new);
    cudaFree(ctx->d_p_new);
    cudaFree(ctx->d_rhs);
    cudaFree(ctx->d_residual);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_y);
    cudaFree(ctx->d_z);
    cudaFree(ctx->d_dx);
    cudaFree(ctx->d_dy);
    cudaFree(ctx->d_dz);
    cudaFree(ctx->d_u_bc);
    cudaFree(ctx->d_v_bc);
    cudaFree(ctx->d_w_bc);
    cudaEventDestroy(ctx->start_event);
    cudaEventDestroy(ctx->stop_event);
    cudaStreamDestroy(ctx->stream);
    cfd_free(ctx);
}

cfd_status_t gpu_solver_upload(gpu_solver_context_t* ctx_void, const flow_field* field) {
    if (!ctx_void || !field)
        return CFD_ERROR_INVALID;
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    size_t bytes = ctx->size * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_u, field->u, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_v, field->v, bytes, cudaMemcpyHostToDevice, ctx->stream));
    if (field->w)
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_w, field->w, bytes, cudaMemcpyHostToDevice, ctx->stream));
    else
        CUDA_CHECK(cudaMemsetAsync(ctx->d_w, 0, bytes, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_p, field->p, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_rho, field->rho, bytes, cudaMemcpyHostToDevice, ctx->stream));
    // Store initial boundary values for preserving caller-set BCs (e.g., lid-driven cavity)
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_u_bc, field->u, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_v_bc, field->v, bytes, cudaMemcpyHostToDevice, ctx->stream));
    if (field->w)
        CUDA_CHECK(cudaMemcpyAsync(ctx->d_w_bc, field->w, bytes, cudaMemcpyHostToDevice, ctx->stream));
    else
        CUDA_CHECK(cudaMemsetAsync(ctx->d_w_bc, 0, bytes, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return CFD_SUCCESS;
}

cfd_status_t gpu_solver_download(gpu_solver_context_t* ctx_void, flow_field* field) {
    if (!ctx_void || !field)
        return CFD_ERROR_INVALID;
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    size_t bytes = ctx->size * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(field->u, ctx->d_u, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(field->v, ctx->d_v, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    if (field->w)
        CUDA_CHECK(cudaMemcpyAsync(field->w, ctx->d_w, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(field->p, ctx->d_p, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return CFD_SUCCESS;
}

cfd_status_t gpu_solver_step(gpu_solver_context_t* ctx_void, const grid* grid,
                             const ns_solver_params_t* params, gpu_solver_stats_t* stats) {
    if (!ctx_void || !grid || !params)
        return CFD_ERROR_INVALID;
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    size_t nx = ctx->nx, ny = ctx->ny, nz = ctx->nz;
    size_t stride_z = ctx->stride_z;
    int k_start = ctx->k_start, k_end = ctx->k_end;
    double dx = grid->dx[0], dy = grid->dy[0], dt = params->dt, nu = params->mu;
    double inv_2dx = 0.5 / dx, inv_2dy = 0.5 / dy, inv_dx2 = 1.0 / (dx * dx),
           inv_dy2 = 1.0 / (dy * dy);
    double inv_2dz = (nz > 1) ? 0.5 / grid->dz[0] : 0.0;
    double inv_dz2 = (nz > 1) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;
    dim3 block(ctx->config.block_size_x, ctx->config.block_size_y);
    dim3 grid_dim((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y);

    cudaEventRecord(ctx->start_event, ctx->stream);
    kernel_velocity_rhs<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u, ctx->d_v, ctx->d_w, ctx->d_p,
        ctx->d_u_new, ctx->d_v_new, ctx->d_w_new,
        nx, ny, stride_z, k_start, k_end,
        inv_2dx, inv_2dy, inv_2dz, inv_dx2, inv_dy2, inv_dz2, nu, 1.0);
    kernel_velocity_update<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u, ctx->d_v, ctx->d_w,
        ctx->d_u_new, ctx->d_v_new, ctx->d_w_new,
        nx, ny, stride_z, k_start, k_end, dt);
    bc_apply_velocity_3d_gpu(ctx->d_u, ctx->d_v, ctx->d_w, nx, ny, nz,
                              BC_TYPE_NEUMANN, ctx->stream);
    kernel_compute_divergence<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u, ctx->d_v, ctx->d_w, ctx->d_rhs,
        nx, ny, stride_z, k_start, k_end, inv_2dx, inv_2dy, inv_2dz);

    double ndim = (nz > 1) ? 3.0 : 2.0;
    double p_relax = 0.1 * dt * (inv_dx2 + inv_dy2 + inv_dz2) / ndim;
    kernel_pressure_update<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_p, ctx->d_rhs, nx, ny, stride_z, k_start, k_end, p_relax);
    bc_apply_scalar_3d_gpu(ctx->d_p, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);
    cudaEventRecord(ctx->stop_event, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    float ms = 0;
    cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event);
    ctx->stats.kernel_time_ms += ms;
    ctx->stats.kernels_launched += 6;
    if (stats)
        *stats = ctx->stats;
    return CFD_SUCCESS;
}

gpu_solver_stats_t gpu_solver_get_stats(const gpu_solver_context_t* ctx_void) {
    if (!ctx_void) {
        gpu_solver_stats_t e = {0};
        return e;
    }
    const struct gpu_solver_context_impl* ctx = (const struct gpu_solver_context_impl*)ctx_void;
    return ctx->stats;
}

void gpu_solver_reset_stats(gpu_solver_context_t* ctx_void) {
    if (!ctx_void)
        return;
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    memset(&ctx->stats, 0, sizeof(gpu_solver_stats_t));
    ctx->stats.memory_allocated = ctx->size * sizeof(double) * 8;
}

cfd_status_t solve_navier_stokes_gpu(flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params, const gpu_config_t* config) {
    if (!field || !grid || !params)
        return CFD_ERROR_INVALID;
    gpu_config_t cfg = config ? *config : gpu_config_default();
    if (!gpu_should_use(&cfg, field->nx, field->ny, field->nz, params->max_iter))
        return CFD_ERROR;
    gpu_solver_context_t* ctx = gpu_solver_create(field->nx, field->ny, field->nz, &cfg);
    if (!ctx)
        return CFD_ERROR_NOMEM;
    if (gpu_solver_upload(ctx, field) != CFD_SUCCESS) {
        gpu_solver_destroy(ctx);
        return CFD_ERROR;
    }
    gpu_solver_stats_t stats;
    for (int iter = 0; iter < params->max_iter; iter++) {
        if (gpu_solver_step(ctx, grid, params, &stats) != CFD_SUCCESS)
            break;
    }
    gpu_solver_download(ctx, field);
    gpu_solver_destroy(ctx);
    return CFD_SUCCESS;
}

cfd_status_t solve_projection_method_gpu(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params, const gpu_config_t* config) {
    if (!field || !grid || !params)
        return CFD_ERROR_INVALID;
    gpu_config_t cfg = config ? *config : gpu_config_default();
    size_t nx = field->nx, ny = field->ny, nz = field->nz;
    if (!gpu_should_use(&cfg, nx, ny, nz, params->max_iter))
        return CFD_ERROR;

    double dx = grid->dx[0], dy = grid->dy[0], dt = params->dt, nu = params->mu;
    double inv_2dx = 0.5 / dx, inv_2dy = 0.5 / dy, inv_dx2 = 1.0 / (dx * dx),
           inv_dy2 = 1.0 / (dy * dy);
    double inv_2dz = (nz > 1) ? 0.5 / grid->dz[0] : 0.0;
    double inv_dz2 = (nz > 1) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;
    double factor = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);
    double inv_factor = 1.0 / factor;
    double rho = (field->rho[0] > 1e-10) ? field->rho[0] : 1.0;
    double dt_rho = dt / rho;

    gpu_solver_context_t* ctx_void = gpu_solver_create(nx, ny, nz, &cfg);
    if (!ctx_void)
        return CFD_ERROR_NOMEM;
    if (gpu_solver_upload(ctx_void, field) != CFD_SUCCESS) {
        gpu_solver_destroy(ctx_void);
        return CFD_ERROR;
    }
    struct gpu_solver_context_impl* ctx = (struct gpu_solver_context_impl*)ctx_void;
    size_t stride_z = ctx->stride_z;
    int k_start = ctx->k_start, k_end = ctx->k_end;

    dim3 block(cfg.block_size_x, cfg.block_size_y);
    dim3 grid_dim((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y);

    // 1D grid for boundary copy kernel
    size_t max_bc_dim;
    if (nz == 1) {
        max_bc_dim = (nx > ny) ? nx : ny;
    } else {
        max_bc_dim = nx * nz;
        if (ny * nz > max_bc_dim) max_bc_dim = ny * nz;
        if (nx * ny > max_bc_dim) max_bc_dim = nx * ny;
    }
    int bc_block = 256;
    int bc_grid = (int)((max_bc_dim + bc_block - 1) / bc_block);

    for (int iter = 0; iter < params->max_iter; iter++) {
        // Step 1: Predictor - compute u*, v*, w* without pressure
        kernel_predictor<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_u, ctx->d_v, ctx->d_w,
            ctx->d_u_new, ctx->d_v_new, ctx->d_w_new,
            nx, ny, stride_z, k_start, k_end,
            dt, nu, inv_2dx, inv_2dy, inv_2dz, inv_dx2, inv_dy2, inv_dz2);
        // Copy caller-set boundary values to u*/v*/w* (preserves lid-driven cavity BCs)
        kernel_copy_velocity_boundaries<<<bc_grid, bc_block, 0, ctx->stream>>>(
            ctx->d_u_new, ctx->d_v_new, ctx->d_w_new,
            ctx->d_u_bc, ctx->d_v_bc, ctx->d_w_bc, nx, ny, nz);

        // Step 2: Compute divergence of u*
        kernel_compute_divergence<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_u_new, ctx->d_v_new, ctx->d_w_new, ctx->d_rhs,
            nx, ny, stride_z, k_start, k_end, inv_2dx, inv_2dy, inv_2dz);
        kernel_scale_rhs<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_rhs, nx, ny, stride_z, k_start, k_end, 1.0 / dt);

        // Step 3: Solve Poisson with Jacobi (pointer swap, 7-point stencil)
        double* p_src = ctx->d_p;
        double* p_dst = ctx->d_p_new;
        for (int pi = 0; pi < cfg.poisson_max_iter; pi++) {
            kernel_poisson_jacobi<<<grid_dim, block, 0, ctx->stream>>>(
                p_src, p_dst, ctx->d_rhs, nx, ny,
                stride_z, k_start, k_end,
                inv_dx2, inv_dy2, inv_dz2, inv_factor);
            bc_apply_scalar_3d_gpu(p_dst, nx, ny, nz, BC_TYPE_NEUMANN, ctx->stream);
            double* tmp = p_src;
            p_src = p_dst;
            p_dst = tmp;
        }
        // Ensure result is in d_p
        if (p_src != ctx->d_p) {
            cudaMemcpyAsync(ctx->d_p, p_src, ctx->size * sizeof(double),
                            cudaMemcpyDeviceToDevice, ctx->stream);
        }

        // Step 4: Corrector - project velocity
        kernel_projection_correct<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_u, ctx->d_v, ctx->d_w,
            ctx->d_u_new, ctx->d_v_new, ctx->d_w_new,
            ctx->d_p, nx, ny, stride_z, k_start, k_end,
            dt_rho, inv_2dx, inv_2dy, inv_2dz);
        // Copy caller-set boundary values back to final velocity
        kernel_copy_velocity_boundaries<<<bc_grid, bc_block, 0, ctx->stream>>>(
            ctx->d_u, ctx->d_v, ctx->d_w,
            ctx->d_u_bc, ctx->d_v_bc, ctx->d_w_bc, nx, ny, nz);
    }

    cudaStreamSynchronize(ctx->stream);
    gpu_solver_download(ctx_void, field);
    gpu_solver_destroy(ctx_void);
    return CFD_SUCCESS;
}

}  // extern "C"
