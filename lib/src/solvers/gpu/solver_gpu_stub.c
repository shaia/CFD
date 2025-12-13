/**
 * GPU Solver Stub Implementation (No CUDA)
 *
 * This file provides stub implementations when CUDA is not available.
 * All functions report that GPU is not available and return appropriate errors.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/solver_gpu.h"
#include "cfd/solvers/solver_interface.h"
#include <stdio.h>
#include <string.h>

// Forward declarations for CPU fallback solvers

extern cfd_status_t explicit_euler_impl(flow_field* field, const grid* grid,
                                        const solver_params* params);
extern cfd_status_t solve_projection_method(flow_field* field, const grid* grid,
                                            const solver_params* params);

gpu_config gpu_config_default(void) {
    gpu_config config;
    memset(&config, 0, sizeof(gpu_config));
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

int gpu_get_device_info(gpu_device_info* info, int max_devices) {
    (void)info;
    (void)max_devices;
    return 0;  // No devices
}

cfd_status_t gpu_select_device(int device_id) {
    (void)device_id;
    return CFD_ERROR;  // Cannot select device
}

int gpu_should_use(const gpu_config* config, size_t nx, size_t ny, int num_steps) {
    (void)config;
    (void)nx;
    (void)ny;
    (void)num_steps;
    return 0;  // Never use GPU (not available)
}

gpu_solver_context* gpu_solver_create(size_t nx, size_t ny, const gpu_config* config) {
    (void)nx;
    (void)ny;
    (void)config;
    fprintf(stderr, "GPU Solver: CUDA not available (compiled without CUDA support)\n");
    return NULL;
}

void gpu_solver_destroy(gpu_solver_context* ctx) {
    (void)ctx;
    // Nothing to do
}

cfd_status_t gpu_solver_upload(gpu_solver_context* ctx, const flow_field* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_download(gpu_solver_context* ctx, flow_field* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_step(gpu_solver_context* ctx, const grid* grid, const solver_params* params,
                             gpu_solver_stats* stats) {
    (void)ctx;
    (void)grid;
    (void)params;
    (void)stats;
    return CFD_ERROR;
}

gpu_solver_stats gpu_solver_get_stats(const gpu_solver_context* ctx) {
    (void)ctx;
    gpu_solver_stats stats;
    memset(&stats, 0, sizeof(gpu_solver_stats));
    return stats;
}

void gpu_solver_reset_stats(gpu_solver_context* ctx) {
    (void)ctx;
}

cfd_status_t solve_navier_stokes_gpu(flow_field* field, const grid* grid,
                                     const solver_params* params, const gpu_config* config) {
    (void)config;

    // Fall back to CPU implementation
    if (config && config->verbose) {
        printf("GPU Solver: CUDA not available, using CPU solver\n");
    }

    return explicit_euler_impl(field, grid, params);
}

cfd_status_t solve_projection_method_gpu(flow_field* field, const grid* grid,
                                         const solver_params* params, const gpu_config* config) {
    (void)config;

    // Fall back to CPU implementation
    if (config && config->verbose) {
        printf("GPU Projection: CUDA not available, using CPU solver\n");
    }

    return solve_projection_method(field, grid, params);
}
