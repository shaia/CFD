/**
 * GPU Solver Stub Implementation (No CUDA)
 *
 * This file provides stub implementations when CUDA is not available.
 * All functions report that GPU is not available and return appropriate errors.
 */

#include "solver_gpu.h"
#include "solver_interface.h"
#include <stdio.h>
#include <string.h>

// Forward declarations for CPU fallback solvers
// Forward declarations for CPU fallback solvers
extern cfd_status_t explicit_euler_impl(FlowField* field, const Grid* grid,
                                        const SolverParams* params);
extern cfd_status_t solve_projection_method(FlowField* field, const Grid* grid,
                                            const SolverParams* params);

GPUConfig gpu_config_default(void) {
    GPUConfig config;
    memset(&config, 0, sizeof(GPUConfig));
    config.enable_gpu = 0;  // Disabled by default when CUDA not available
    config.min_grid_size = 10000;
    config.min_steps = 10;
    config.block_size_x = 16;
    config.block_size_y = 16;
    config.poisson_max_iter = 1000;
    config.poisson_tolerance = 1e-6;
    config.persistent_memory = 1;
    config.async_transfers = 1;
    config.sync_after_kernel = 0;
    config.verbose = 0;
    return config;
}

int gpu_is_available(void) {
    return 0;  // No CUDA available
}

int gpu_get_device_info(GPUDeviceInfo* info, int max_devices) {
    (void)info;
    (void)max_devices;
    return 0;  // No devices
}

cfd_status_t gpu_select_device(int device_id) {
    (void)device_id;
    return CFD_ERROR;  // Cannot select device
}

int gpu_should_use(const GPUConfig* config, size_t nx, size_t ny, int num_steps) {
    (void)config;
    (void)nx;
    (void)ny;
    (void)num_steps;
    return 0;  // Never use GPU (not available)
}

GPUSolverContext* gpu_solver_create(size_t nx, size_t ny, const GPUConfig* config) {
    (void)nx;
    (void)ny;
    (void)config;
    fprintf(stderr, "GPU Solver: CUDA not available (compiled without CUDA support)\n");
    return NULL;
}

void gpu_solver_destroy(GPUSolverContext* ctx) {
    (void)ctx;
    // Nothing to do
}

cfd_status_t gpu_solver_upload(GPUSolverContext* ctx, const FlowField* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_download(GPUSolverContext* ctx, FlowField* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_step(GPUSolverContext* ctx, const Grid* grid, const SolverParams* params,
                             GPUSolverStats* stats) {
    (void)ctx;
    (void)grid;
    (void)params;
    (void)stats;
    return CFD_ERROR;
}

GPUSolverStats gpu_solver_get_stats(const GPUSolverContext* ctx) {
    (void)ctx;
    GPUSolverStats stats;
    memset(&stats, 0, sizeof(GPUSolverStats));
    return stats;
}

void gpu_solver_reset_stats(GPUSolverContext* ctx) {
    (void)ctx;
}

cfd_status_t solve_navier_stokes_gpu(FlowField* field, const Grid* grid, const SolverParams* params,
                                     const GPUConfig* config) {
    (void)config;

    // Fall back to CPU implementation
    if (config && config->verbose) {
        printf("GPU Solver: CUDA not available, using CPU solver\n");
    }

    return explicit_euler_impl(field, grid, params);
}

cfd_status_t solve_projection_method_gpu(FlowField* field, const Grid* grid,
                                         const SolverParams* params, const GPUConfig* config) {
    (void)config;

    // Fall back to CPU implementation
    if (config && config->verbose) {
        printf("GPU Projection: CUDA not available, using CPU solver\n");
    }

    return solve_projection_method(field, grid, params);
}
