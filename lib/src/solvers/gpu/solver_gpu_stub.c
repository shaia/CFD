/**
 * GPU NSSolver Stub Implementation (No CUDA)
 *
 * This file provides stub implementations when CUDA is not available.
 * All functions report that GPU is not available and return appropriate errors.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include <stdio.h>
#include <string.h>

gpu_config_t gpu_config_default(void) {
    gpu_config_t config;
    memset(&config, 0, sizeof(gpu_config_t));
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

int gpu_get_device_info(gpu_device_info_t* info, int max_devices) {
    (void)info;
    (void)max_devices;
    return 0;  // No devices
}

cfd_status_t gpu_select_device(int device_id) {
    (void)device_id;
    return CFD_ERROR;  // Cannot select device
}

int gpu_should_use(const gpu_config_t* config, size_t nx, size_t ny, int num_steps) {
    (void)config;
    (void)nx;
    (void)ny;
    (void)num_steps;
    return 0;  // Never use GPU (not available)
}

gpu_solver_context_t* gpu_solver_create(size_t nx, size_t ny, const gpu_config_t* config) {
    (void)nx;
    (void)ny;
    (void)config;
    fprintf(stderr, "GPU NSSolver: CUDA not available (compiled without CUDA support)\n");
    return NULL;
}

void gpu_solver_destroy(gpu_solver_context_t* ctx) {
    (void)ctx;
    // Nothing to do
}

cfd_status_t gpu_solver_upload(gpu_solver_context_t* ctx, const flow_field* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_download(gpu_solver_context_t* ctx, flow_field* field) {
    (void)ctx;
    (void)field;
    return CFD_ERROR;
}

cfd_status_t gpu_solver_step(gpu_solver_context_t* ctx, const grid* grid, const ns_solver_params_t* params,
                             gpu_solver_stats_t* stats) {
    (void)ctx;
    (void)grid;
    (void)params;
    (void)stats;
    return CFD_ERROR;
}

gpu_solver_stats_t gpu_solver_get_stats(const gpu_solver_context_t* ctx) {
    (void)ctx;
    gpu_solver_stats_t stats;
    memset(&stats, 0, sizeof(gpu_solver_stats_t));
    return stats;
}

void gpu_solver_reset_stats(gpu_solver_context_t* ctx) {
    (void)ctx;
}

cfd_status_t solve_navier_stokes_gpu(flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params, const gpu_config_t* config) {
    (void)field;
    (void)grid;
    (void)params;
    (void)config;

    fprintf(stderr, "GPU NSSolver: CUDA not available (compiled without CUDA support)\n");
    return CFD_ERROR_UNSUPPORTED;
}

cfd_status_t solve_projection_method_gpu(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params, const gpu_config_t* config) {
    (void)field;
    (void)grid;
    (void)params;
    (void)config;

    fprintf(stderr, "GPU Projection: CUDA not available (compiled without CUDA support)\n");
    return CFD_ERROR_UNSUPPORTED;
}
