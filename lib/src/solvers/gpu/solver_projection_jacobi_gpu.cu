/**
 * GPU-Accelerated Solver Implementation (CUDA) - Optimized
 *
 * Key optimizations:
 * - Jacobi iteration for Poisson (fully parallel, no red-black dependency)
 * - Reduced Poisson iterations (50 instead of 1000)
 * - Fixed predictor step logic
 * - Pointer swapping instead of memcpy for Poisson iterations
 */

#include "solver_gpu.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>


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

#define MAX_VELOCITY 100.0
#define BLOCK_SIZE   16

struct GPUSolverContext {
    size_t nx, ny, size;
    double* d_u;
    double* d_v;
    double* d_p;
    double* d_rho;
    double* d_u_new;
    double* d_v_new;
    double* d_p_new;
    double* d_rhs;
    double* d_residual;
    double* d_x;
    double* d_y;
    double* d_dx;
    double* d_dy;
    GPUConfig config;
    GPUSolverStats stats;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    int initialized;
    int memory_allocated;
};

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void kernel_apply_boundary_pressure(double* p, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        p[idx * nx] = p[idx * nx + 1];
        p[idx * nx + nx - 1] = p[idx * nx + nx - 2];
    }
    if (idx < nx) {
        p[idx] = p[nx + idx];
        p[(ny - 1) * nx + idx] = p[(ny - 2) * nx + idx];
    }
}

__global__ void kernel_apply_boundary_velocity(double* u, double* v, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        u[idx * nx] = u[idx * nx + 1];
        u[idx * nx + nx - 1] = u[idx * nx + nx - 2];
        v[idx * nx] = v[idx * nx + 1];
        v[idx * nx + nx - 1] = v[idx * nx + nx - 2];
    }
    if (idx < nx) {
        u[idx] = u[nx + idx];
        u[(ny - 1) * nx + idx] = u[(ny - 2) * nx + idx];
        v[idx] = v[nx + idx];
        v[(ny - 1) * nx + idx] = v[(ny - 2) * nx + idx];
    }
}

__global__ void kernel_compute_divergence(const double* __restrict__ u,
                                          const double* __restrict__ v, double* __restrict__ div,
                                          size_t nx, size_t ny, double inv_2dx, double inv_2dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        div[idx] = (u[idx + 1] - u[idx - 1]) * inv_2dx + (v[idx + nx] - v[idx - nx]) * inv_2dy;
    }
}

// Fast Jacobi iteration - fully parallel
__global__ void kernel_poisson_jacobi(const double* __restrict__ p_old, double* __restrict__ p_new,
                                      const double* __restrict__ rhs, size_t nx, size_t ny,
                                      double inv_dx2, double inv_dy2, double inv_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double sum = (p_old[idx + 1] + p_old[idx - 1]) * inv_dx2 +
                     (p_old[idx + nx] + p_old[idx - nx]) * inv_dy2;
        p_new[idx] = (rhs[idx] + sum) * inv_factor;
    }
}

// Red-Black SOR (kept for compatibility)
__global__ void kernel_poisson_sor_rb(double* __restrict__ p, const double* __restrict__ rhs,
                                      size_t nx, size_t ny, double inv_dx2, double inv_dy2,
                                      double omega, double inv_factor, int color,
                                      double* __restrict__ residual_out) {
    __shared__ double s_residual[BLOCK_SIZE * BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    double local_residual = 0.0;

    if (i < nx - 1 && j < ny - 1 && (i + j) % 2 == color) {
        size_t idx = j * nx + i;
        double p_c = p[idx];
        double sum = (p[idx + 1] + p[idx - 1]) * inv_dx2 + (p[idx + nx] + p[idx - nx]) * inv_dy2;
        double p_gs = (rhs[idx] + sum) * inv_factor;
        p[idx] = p_c + omega * (p_gs - p_c);
        if (residual_out) {
            double lap = (p[idx + 1] - 2.0 * p_c + p[idx - 1]) * inv_dx2 +
                         (p[idx + nx] - 2.0 * p_c + p[idx - nx]) * inv_dy2;
            local_residual = fabs(lap - rhs[idx]);
        }
    }

    if (residual_out) {
        s_residual[tid] = local_residual;
        __syncthreads();
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s)
                s_residual[tid] = fmax(s_residual[tid], s_residual[tid + s]);
            __syncthreads();
        }
        if (tid == 0)
            atomicMax((unsigned long long*)residual_out, __double_as_longlong(s_residual[0]));
    }
}

// Predictor step - compute u* without pressure
__global__ void kernel_predictor(const double* __restrict__ u, const double* __restrict__ v,
                                 double* __restrict__ u_star, double* __restrict__ v_star,
                                 size_t nx, size_t ny, double dt, double nu, double inv_2dx,
                                 double inv_2dy, double inv_dx2, double inv_dy2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double u_c = u[idx], v_c = v[idx];
        double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
        double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
        double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
        double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;
        double d2u = (u[idx + 1] - 2.0 * u_c + u[idx - 1]) * inv_dx2 +
                     (u[idx + nx] - 2.0 * u_c + u[idx - nx]) * inv_dy2;
        double d2v = (v[idx + 1] - 2.0 * v_c + v[idx - 1]) * inv_dx2 +
                     (v[idx + nx] - 2.0 * v_c + v[idx - nx]) * inv_dy2;
        double u_new = u_c + dt * (-(u_c * du_dx + v_c * du_dy) + nu * d2u);
        double v_new = v_c + dt * (-(u_c * dv_dx + v_c * dv_dy) + nu * d2v);
        u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_new));
        v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_new));
    }
}

__global__ void kernel_velocity_rhs(const double* __restrict__ u, const double* __restrict__ v,
                                    const double* __restrict__ p, double* __restrict__ u_rhs,
                                    double* __restrict__ v_rhs, size_t nx, size_t ny,
                                    double inv_2dx, double inv_2dy, double inv_dx2, double inv_dy2,
                                    double nu, double inv_rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double u_c = u[idx], v_c = v[idx];
        double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
        double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
        double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
        double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;
        double d2u = (u[idx + 1] - 2.0 * u_c + u[idx - 1]) * inv_dx2 +
                     (u[idx + nx] - 2.0 * u_c + u[idx - nx]) * inv_dy2;
        double d2v = (v[idx + 1] - 2.0 * v_c + v[idx - 1]) * inv_dx2 +
                     (v[idx + nx] - 2.0 * v_c + v[idx - nx]) * inv_dy2;
        double dp_dx = (p[idx + 1] - p[idx - 1]) * inv_2dx;
        double dp_dy = (p[idx + nx] - p[idx - nx]) * inv_2dy;
        u_rhs[idx] = -(u_c * du_dx + v_c * du_dy) + nu * d2u - inv_rho * dp_dx;
        v_rhs[idx] = -(u_c * dv_dx + v_c * dv_dy) + nu * d2v - inv_rho * dp_dy;
    }
}

__global__ void kernel_velocity_update(double* __restrict__ u, double* __restrict__ v,
                                       const double* __restrict__ u_rhs,
                                       const double* __restrict__ v_rhs, size_t nx, size_t ny,
                                       double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u[idx] + dt * u_rhs[idx]));
        v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v[idx] + dt * v_rhs[idx]));
    }
}

__global__ void kernel_projection_correct(double* __restrict__ u, double* __restrict__ v,
                                          const double* __restrict__ u_star,
                                          const double* __restrict__ v_star,
                                          const double* __restrict__ p, size_t nx, size_t ny,
                                          double dt_rho, double inv_2dx, double inv_2dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double dp_dx = (p[idx + 1] - p[idx - 1]) * inv_2dx;
        double dp_dy = (p[idx + nx] - p[idx - nx]) * inv_2dy;
        u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx] - dt_rho * dp_dx));
        v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx] - dt_rho * dp_dy));
    }
}

__global__ void kernel_pressure_update(double* __restrict__ p, const double* __restrict__ div,
                                       size_t nx, size_t ny, double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        p[j * nx + i] -= factor * div[j * nx + i];
    }
}

__global__ void kernel_scale_rhs(double* __restrict__ rhs, size_t nx, size_t ny, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1)
        rhs[j * nx + i] *= scale;
}

// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

GPUConfig gpu_config_default(void) {
    GPUConfig config;
    config.enable_gpu = 1;
    config.min_grid_size = 10000;
    config.min_steps = 10;
    config.block_size_x = BLOCK_SIZE;
    config.block_size_y = BLOCK_SIZE;
    config.poisson_max_iter = 25;  // Balance between speed and accuracy
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

int gpu_get_device_info(GPUDeviceInfo* info, int max_devices) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
        return 0;
    int count = (device_count < max_devices) ? device_count : max_devices;
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            info[i].device_id = i;
            strncpy(info[i].name, prop.name, 255);
            info[i].name[255] = '\0';
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

int gpu_should_use(const GPUConfig* config, size_t nx, size_t ny, int num_steps) {
    if (!config || !config->enable_gpu)
        return 0;
    if (!gpu_is_available())
        return 0;
    if (nx * ny < config->min_grid_size)
        return 0;
    if (num_steps < config->min_steps)
        return 0;
    return 1;
}

GPUSolverContext* gpu_solver_create(size_t nx, size_t ny, const GPUConfig* config) {
    if (!gpu_is_available())
        return nullptr;
    GPUSolverContext* ctx = (GPUSolverContext*)cfd_calloc(1, sizeof(GPUSolverContext));
    if (!ctx)
        return nullptr;
    ctx->nx = nx;
    ctx->ny = ny;
    ctx->size = nx * ny;
    ctx->config = config ? *config : gpu_config_default();
    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        cfd_free(ctx);
        return nullptr;
    }
    cudaEventCreate(&ctx->start_event);
    cudaEventCreate(&ctx->stop_event);
    size_t bytes = ctx->size * sizeof(double);
    if (cudaMalloc(&ctx->d_u, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_v, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_p, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_rho, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_u_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_v_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_p_new, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_rhs, bytes) != cudaSuccess ||
        cudaMalloc(&ctx->d_residual, sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_x, nx * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_y, ny * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_dx, (nx - 1) * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&ctx->d_dy, (ny - 1) * sizeof(double)) != cudaSuccess) {
        gpu_solver_destroy(ctx);
        return nullptr;
    }
    ctx->memory_allocated = 1;
    ctx->initialized = 1;
    ctx->stats.memory_allocated = bytes * 8 + (nx + ny + nx - 1 + ny - 1 + 1) * sizeof(double);
    return ctx;
}

void gpu_solver_destroy(GPUSolverContext* ctx) {
    if (!ctx)
        return;
    if (ctx->memory_allocated) {
        cudaFree(ctx->d_u);
        cudaFree(ctx->d_v);
        cudaFree(ctx->d_p);
        cudaFree(ctx->d_rho);
        cudaFree(ctx->d_u_new);
        cudaFree(ctx->d_v_new);
        cudaFree(ctx->d_p_new);
        cudaFree(ctx->d_rhs);
        cudaFree(ctx->d_residual);
        cudaFree(ctx->d_x);
        cudaFree(ctx->d_y);
        cudaFree(ctx->d_dx);
        cudaFree(ctx->d_dy);
    }
    cudaEventDestroy(ctx->start_event);
    cudaEventDestroy(ctx->stop_event);
    cudaStreamDestroy(ctx->stream);
    cfd_free(ctx);
}

cfd_status_t gpu_solver_upload(GPUSolverContext* ctx, const FlowField* field) {
    if (!ctx || !field)
        return CFD_ERROR_INVALID;
    size_t bytes = ctx->size * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_u, field->u, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_v, field->v, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_p, field->p, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_rho, field->rho, bytes, cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return CFD_SUCCESS;
}

cfd_status_t gpu_solver_download(GPUSolverContext* ctx, FlowField* field) {
    if (!ctx || !field)
        return CFD_ERROR_INVALID;
    size_t bytes = ctx->size * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(field->u, ctx->d_u, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(field->v, ctx->d_v, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(field->p, ctx->d_p, bytes, cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    return CFD_SUCCESS;
}

cfd_status_t gpu_solver_step(GPUSolverContext* ctx, const Grid* grid, const SolverParams* params,
                             GPUSolverStats* stats) {
    if (!ctx || !grid || !params)
        return CFD_ERROR_INVALID;
    size_t nx = ctx->nx, ny = ctx->ny;
    double dx = grid->dx[0], dy = grid->dy[0], dt = params->dt, nu = params->mu;
    double inv_2dx = 0.5 / dx, inv_2dy = 0.5 / dy, inv_dx2 = 1.0 / (dx * dx),
           inv_dy2 = 1.0 / (dy * dy);
    dim3 block(ctx->config.block_size_x, ctx->config.block_size_y);
    dim3 grid_dim((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y);
    int bb = ((int)fmax(nx, ny) + 255) / 256;

    cudaEventRecord(ctx->start_event, ctx->stream);
    kernel_velocity_rhs<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u, ctx->d_v, ctx->d_p, ctx->d_u_new, ctx->d_v_new, nx, ny, inv_2dx, inv_2dy, inv_dx2,
        inv_dy2, nu, 1.0);
    kernel_velocity_update<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, ctx->d_u_new,
                                                                ctx->d_v_new, nx, ny, dt);
    kernel_apply_boundary_velocity<<<bb, 256, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, nx, ny);
    kernel_compute_divergence<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, ctx->d_rhs,
                                                                   nx, ny, inv_2dx, inv_2dy);
    kernel_pressure_update<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_p, ctx->d_rhs, nx, ny,
                                                                0.1 * dt / (dx * dx));
    kernel_apply_boundary_pressure<<<bb, 256, 0, ctx->stream>>>(ctx->d_p, nx, ny);
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

GPUSolverStats gpu_solver_get_stats(const GPUSolverContext* ctx) {
    if (!ctx) {
        GPUSolverStats e = {0};
        return e;
    }
    return ctx->stats;
}

void gpu_solver_reset_stats(GPUSolverContext* ctx) {
    if (!ctx)
        return;
    memset(&ctx->stats, 0, sizeof(GPUSolverStats));
    ctx->stats.memory_allocated = ctx->size * sizeof(double) * 8;
}

cfd_status_t solve_navier_stokes_gpu(FlowField* field, const Grid* grid, const SolverParams* params,
                                     const GPUConfig* config) {
    if (!field || !grid || !params)
        return CFD_ERROR_INVALID;
    GPUConfig cfg = config ? *config : gpu_config_default();
    if (!gpu_should_use(&cfg, field->nx, field->ny, params->max_iter))
        return CFD_ERROR;
    GPUSolverContext* ctx = gpu_solver_create(field->nx, field->ny, &cfg);
    if (!ctx)
        return CFD_ERROR_NOMEM;
    if (gpu_solver_upload(ctx, field) != CFD_SUCCESS) {
        gpu_solver_destroy(ctx);
        return CFD_ERROR;
    }
    GPUSolverStats stats;
    for (int iter = 0; iter < params->max_iter; iter++) {
        if (gpu_solver_step(ctx, grid, params, &stats) != CFD_SUCCESS)
            break;
    }
    gpu_solver_download(ctx, field);
    gpu_solver_destroy(ctx);
    return CFD_SUCCESS;
}

cfd_status_t solve_projection_method_gpu(FlowField* field, const Grid* grid,
                                         const SolverParams* params, const GPUConfig* config) {
    if (!field || !grid || !params)
        return CFD_ERROR_INVALID;
    GPUConfig cfg = config ? *config : gpu_config_default();
    if (!gpu_should_use(&cfg, field->nx, field->ny, params->max_iter))
        return CFD_ERROR;

    size_t nx = field->nx, ny = field->ny;
    double dx = grid->dx[0], dy = grid->dy[0], dt = params->dt, nu = params->mu;
    double inv_2dx = 0.5 / dx, inv_2dy = 0.5 / dy, inv_dx2 = 1.0 / (dx * dx),
           inv_dy2 = 1.0 / (dy * dy);
    double factor = 2.0 * (inv_dx2 + inv_dy2);
    double inv_factor = 1.0 / factor;
    double rho = (field->rho[0] > 1e-10) ? field->rho[0] : 1.0;
    double dt_rho = dt / rho;

    GPUSolverContext* ctx = gpu_solver_create(nx, ny, &cfg);
    if (!ctx)
        return CFD_ERROR_NOMEM;
    if (gpu_solver_upload(ctx, field) != CFD_SUCCESS) {
        gpu_solver_destroy(ctx);
        return CFD_ERROR;
    }

    dim3 block(cfg.block_size_x, cfg.block_size_y);
    dim3 grid_dim((nx - 2 + block.x - 1) / block.x, (ny - 2 + block.y - 1) / block.y);
    int bb = ((int)fmax(nx, ny) + 255) / 256;

    for (int iter = 0; iter < params->max_iter; iter++) {
        // Step 1: Predictor - compute u* without pressure
        kernel_predictor<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, ctx->d_u_new,
                                                              ctx->d_v_new, nx, ny, dt, nu, inv_2dx,
                                                              inv_2dy, inv_dx2, inv_dy2);
        kernel_apply_boundary_velocity<<<bb, 256, 0, ctx->stream>>>(ctx->d_u_new, ctx->d_v_new, nx,
                                                                    ny);

        // Step 2: Compute divergence of u*
        kernel_compute_divergence<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_u_new, ctx->d_v_new, ctx->d_rhs, nx, ny, inv_2dx, inv_2dy);
        kernel_scale_rhs<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_rhs, nx, ny, rho / dt);

        // Step 3: Solve Poisson with Jacobi (pointer swap)
        double* p_src = ctx->d_p;
        double* p_dst = ctx->d_p_new;
        for (int pi = 0; pi < cfg.poisson_max_iter; pi++) {
            kernel_poisson_jacobi<<<grid_dim, block, 0, ctx->stream>>>(
                p_src, p_dst, ctx->d_rhs, nx, ny, inv_dx2, inv_dy2, inv_factor);
            kernel_apply_boundary_pressure<<<bb, 256, 0, ctx->stream>>>(p_dst, nx, ny);
            double* tmp = p_src;
            p_src = p_dst;
            p_dst = tmp;
        }
        // Ensure result is in d_p
        if (p_src != ctx->d_p) {
            cudaMemcpyAsync(ctx->d_p, p_src, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice,
                            ctx->stream);
        }

        // Step 4: Corrector - project velocity
        kernel_projection_correct<<<grid_dim, block, 0, ctx->stream>>>(
            ctx->d_u, ctx->d_v, ctx->d_u_new, ctx->d_v_new, ctx->d_p, nx, ny, dt_rho, inv_2dx,
            inv_2dy);
        kernel_apply_boundary_velocity<<<bb, 256, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, nx, ny);
    }

    cudaStreamSynchronize(ctx->stream);
    gpu_solver_download(ctx, field);
    gpu_solver_destroy(ctx);
    return CFD_SUCCESS;
}

}  // extern "C"
