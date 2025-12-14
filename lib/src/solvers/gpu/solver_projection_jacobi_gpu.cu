/**
 * GPU-Accelerated Solver Implementation (CUDA) - Optimized
 * Re-written from scratch for stability and correctness.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_gpu.h"
#include "cfd/solvers/solver_interface.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE   16
#define MAX_VELOCITY 100.0

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return CFD_ERROR;                                                                 \
        }                                                                                     \
    } while (0)

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

__global__ void kernel_apply_boundary_velocity(double* __restrict__ u, double* __restrict__ v,
                                               size_t nx, size_t ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        // Left Wall (x=0) - No-slip
        if (i == 1) {
            u[j * nx + 0] = 0.0;
            v[j * nx + 0] = -v[j * nx + 1];
        }
        // Right Wall (x=nx-1) - No-slip
        if (i == nx - 2) {
            u[j * nx + nx - 1] = 0.0;
            v[j * nx + nx - 1] = -v[j * nx + nx - 2];
        }
        // Bottom Wall (y=0) - No-slip
        if (j == 1) {
            u[0 * nx + i] = -u[1 * nx + i];
            v[0 * nx + i] = 0.0;
        }
        // Top Wall (y=ny-1) - Lid-driven cavity (u=1.0)
        if (j == ny - 2) {
            u[(ny - 1) * nx + i] = 1.0;
            v[(ny - 1) * nx + i] = 0.0;
        }
    }
}

__global__ void kernel_apply_boundary_pressure(double* __restrict__ p, size_t nx, size_t ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Neumann condition (dp/dn = 0) -> copy adjacent value
    if (i < nx - 1 && j < ny - 1) {
        // Left (x=0)
        if (i == 1)
            p[j * nx + 0] = p[j * nx + 1];
        // Right (x=nx-1)
        if (i == nx - 2)
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        // Bottom (y=0)
        if (j == 1)
            p[0 * nx + i] = p[1 * nx + i];
        // Top (y=ny-1)
        if (j == ny - 2)
            p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
    }
}

__global__ void kernel_predictor(const double* __restrict__ u, const double* __restrict__ v,
                                 double* __restrict__ u_star, double* __restrict__ v_star,
                                 size_t nx, size_t ny, double dt, double nu, double inv_2dx,
                                 double inv_2dy, double inv_dx2, double inv_dy2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double u_c = u[idx];
        double v_c = v[idx];

        // First derivatives (Central difference)
        double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
        double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
        double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
        double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;

        // Second derivatives (Laplacian)
        double d2u = (u[idx + 1] - 2.0 * u_c + u[idx - 1]) * inv_dx2 +
                     (u[idx + nx] - 2.0 * u_c + u[idx - nx]) * inv_dy2;
        double d2v = (v[idx + 1] - 2.0 * v_c + v[idx - 1]) * inv_dx2 +
                     (v[idx + nx] - 2.0 * v_c + v[idx - nx]) * inv_dy2;

        // Explicit Euler Step
        double u_new = u_c + dt * (-(u_c * du_dx + v_c * du_dy) + nu * d2u);
        double v_new = v_c + dt * (-(u_c * dv_dx + v_c * dv_dy) + nu * d2v);

        // Clamp for stability
        u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_new));
        v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_new));

        if (isnan(u_new) || isnan(v_new)) {
            printf("NaN in predictor at %d,%d: u_new=%f v_new=%f, u=%f v=%f\n", i, j, u_new, v_new,
                   u_c, v_c);
        }
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
        if (isnan(div[idx])) {
            printf("NaN in divergence at %d,%d\n", i, j);
        }
    }
}

__global__ void kernel_scale_rhs(double* __restrict__ rhs, size_t nx, size_t ny, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        rhs[j * nx + i] *= scale;
    }
}

__global__ void kernel_poisson_jacobi(const double* __restrict__ p_old, double* __restrict__ p_new,
                                      const double* __restrict__ rhs, size_t nx, size_t ny,
                                      double inv_dx2, double inv_dy2, double inv_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        size_t idx = j * nx + i;
        double sum = (p_old[idx + 1] + p_old[idx - 1]) * inv_dx2 +
                     (p_old[idx + nx] + p_old[idx - nx]) * inv_dy2;
        p_new[idx] = (sum - rhs[idx]) * inv_factor;

        if (isnan(p_new[idx])) {
            printf("NaN in poisson at %d,%d: rhs=%f sum=%f inv_factor=%f\n", i, j, rhs[idx], sum,
                   inv_factor);
        }
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

        u[idx] = u_star[idx] - dt_rho * dp_dx;
        v[idx] = v_star[idx] - dt_rho * dp_dy;
    }
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
    config.poisson_max_iter = 50;
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
    } else {
        return 0;
    }
}

int gpu_get_device_info(GPUDeviceInfo* info, int max_devices) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
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
            info[i].is_available = 1;
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
        return NULL;

    GPUSolverContext* ctx = (GPUSolverContext*)cfd_calloc(1, sizeof(GPUSolverContext));
    if (!ctx)
        return NULL;

    ctx->nx = nx;
    ctx->ny = ny;
    ctx->size = nx * ny;
    ctx->config = config ? *config : gpu_config_default();

    if (cudaStreamCreate(&ctx->stream) != cudaSuccess) {
        cfd_free(ctx);
        return NULL;
    }

    // Allocate Memory
    size_t bytes = ctx->size * sizeof(double);
    bool alloc_failed = false;

    if (cudaMalloc((void**)&ctx->d_u, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_v, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_p, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_rho, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_u_new, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_v_new, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_p_new, bytes) != cudaSuccess)
        alloc_failed = true;
    if (cudaMalloc((void**)&ctx->d_rhs, bytes) != cudaSuccess)
        alloc_failed = true;

    if (alloc_failed) {
        gpu_solver_destroy(ctx);
        return NULL;
    }

    // Zero-initialize buffers
    cudaMemset(ctx->d_u, 0, bytes);
    cudaMemset(ctx->d_v, 0, bytes);
    cudaMemset(ctx->d_p, 0, bytes);
    cudaMemset(ctx->d_rho, 0, bytes);
    cudaMemset(ctx->d_u_new, 0, bytes);
    cudaMemset(ctx->d_v_new, 0, bytes);
    cudaMemset(ctx->d_p_new, 0, bytes);
    cudaMemset(ctx->d_rhs, 0, bytes);

    ctx->memory_allocated = 1;
    ctx->initialized = 1;

    return ctx;
}

void gpu_solver_destroy(GPUSolverContext* ctx) {
    if (!ctx)
        return;

    if (ctx->d_u)
        cudaFree(ctx->d_u);
    if (ctx->d_v)
        cudaFree(ctx->d_v);
    if (ctx->d_p)
        cudaFree(ctx->d_p);
    if (ctx->d_rho)
        cudaFree(ctx->d_rho);
    if (ctx->d_u_new)
        cudaFree(ctx->d_u_new);
    if (ctx->d_v_new)
        cudaFree(ctx->d_v_new);
    if (ctx->d_p_new)
        cudaFree(ctx->d_p_new);
    if (ctx->d_rhs)
        cudaFree(ctx->d_rhs);

    if (ctx->stream)
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
    // Default rho
    if (field->rho) {
        CUDA_CHECK(
            cudaMemcpyAsync(ctx->d_rho, field->rho, bytes, cudaMemcpyHostToDevice, ctx->stream));
    } else {
        cudaMemsetAsync(ctx->d_rho, 0, bytes, ctx->stream);
    }
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

    size_t nx = ctx->nx;
    size_t ny = ctx->ny;
    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;

    double inv_2dx = 0.5 / dx;
    double inv_2dy = 0.5 / dy;
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);

    double factor = 2.0 * (inv_dx2 + inv_dy2);
    double inv_factor = 1.0 / factor;
    double rho = 1.0;
    double dt_rho = dt / rho;

    // Block/Grid Config - 2D Configuration
    dim3 block(ctx->config.block_size_x, ctx->config.block_size_y);
    dim3 grid_dim((unsigned int)((nx - 2 + block.x - 1) / block.x),
                  (unsigned int)((ny - 2 + block.y - 1) / block.y));

    // 1. Predictor: u*
    kernel_predictor<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, ctx->d_u_new,
                                                          ctx->d_v_new, nx, ny, dt, nu, inv_2dx,
                                                          inv_2dy, inv_dx2, inv_dy2);
    // Boundary using 2D grid
    kernel_apply_boundary_velocity<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u_new, ctx->d_v_new,
                                                                        nx, ny);

    // 2. Divergence of u*
    kernel_compute_divergence<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u_new, ctx->d_v_new, ctx->d_rhs, nx, ny, inv_2dx, inv_2dy);
    kernel_scale_rhs<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_rhs, nx, ny, rho / dt);

    // 3. Poisson: Solve for P
    double* p_src = ctx->d_p;
    double* p_dst = ctx->d_p_new;

    for (int i = 0; i < ctx->config.poisson_max_iter; i++) {
        kernel_poisson_jacobi<<<grid_dim, block, 0, ctx->stream>>>(p_src, p_dst, ctx->d_rhs, nx, ny,
                                                                   inv_dx2, inv_dy2, inv_factor);
        // Boundary using 2D grid
        kernel_apply_boundary_pressure<<<grid_dim, block, 0, ctx->stream>>>(p_dst, nx, ny);

        // Swap pointers
        double* temp = p_src;
        p_src = p_dst;
        p_dst = temp;
    }

    // Ensure final result is in d_p
    if (p_src != ctx->d_p) {
        cudaMemcpyAsync(ctx->d_p, p_src, ctx->size * sizeof(double), cudaMemcpyDeviceToDevice,
                        ctx->stream);
    }

    // 4. Corrector: u = u* - dt*grad(p)
    kernel_projection_correct<<<grid_dim, block, 0, ctx->stream>>>(
        ctx->d_u, ctx->d_v, ctx->d_u_new, ctx->d_v_new, ctx->d_p, nx, ny, dt_rho, inv_2dx, inv_2dy);
    // Boundary using 2D grid
    kernel_apply_boundary_velocity<<<grid_dim, block, 0, ctx->stream>>>(ctx->d_u, ctx->d_v, nx, ny);

    cudaStreamSynchronize(ctx->stream);

    if (stats) {
        stats->poisson_iterations = ctx->config.poisson_max_iter;
        stats->kernels_launched++;
    }

    return CFD_SUCCESS;
}

GPUSolverStats gpu_solver_get_stats(const GPUSolverContext* ctx) {
    if (!ctx) {
        GPUSolverStats s = {0};
        return s;
    }
    return ctx->stats;
}

void gpu_solver_reset_stats(GPUSolverContext* ctx) {
    if (ctx)
        memset(&ctx->stats, 0, sizeof(GPUSolverStats));
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
    for (int i = 0; i < params->max_iter; i++) {
        gpu_solver_step(ctx, grid, params, &stats);
    }

    gpu_solver_download(ctx, field);
    gpu_solver_destroy(ctx);
    return CFD_SUCCESS;
}

cfd_status_t solve_projection_method_gpu(FlowField* field, const Grid* grid,
                                         const SolverParams* params, const GPUConfig* config) {
    printf("DEBUG: Entering solve_projection_method_gpu (Fixed)\n");
    return solve_navier_stokes_gpu(field, grid, params, config);
}

}  // extern "C"
