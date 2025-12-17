/**
 * @file gpu_device.h
 * @brief GPU device management and CUDA runtime abstraction
 *
 * This module provides GPU device management functionality:
 * - Device detection and selection
 * - Memory allocation and transfers
 * - Device information queries
 * - GPU solver context management
 *
 * This is infrastructure used by GPU-accelerated solvers and other
 * components that need GPU resources.
 */

#ifndef CFD_GPU_DEVICE_H
#define CFD_GPU_DEVICE_H

#include "cfd/cfd_export.h"

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU Configuration Settings
 */
typedef struct {
    int enable_gpu;            /**< Enable/disable GPU acceleration */

    /* Automatic selection thresholds */
    size_t min_grid_size;      /**< Minimum grid points for GPU (default: 10000) */
    int min_steps;             /**< Minimum steps to amortize transfer cost (default: 10) */

    /* Performance tuning */
    int block_size_x;          /**< CUDA block size X (default: 16) */
    int block_size_y;          /**< CUDA block size Y (default: 16) */
    int poisson_max_iter;      /**< Max Poisson iterations on GPU (default: 1000) */
    double poisson_tolerance;  /**< Poisson convergence tolerance (default: 1e-6) */

    /* Memory management */
    int persistent_memory;     /**< Keep GPU memory allocated between steps (default: 1) */
    int async_transfers;       /**< Use async memory transfers (default: 1) */

    /* Debug options */
    int sync_after_kernel;     /**< Synchronize after each kernel for debugging (default: 0) */
    int verbose;               /**< Print GPU info and timing (default: 0) */
} gpu_config_t;

/**
 * GPU Device Information
 */
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
    int is_available;
} gpu_device_info_t;

/**
 * GPU NSSolver Statistics
 */
typedef struct {
    double kernel_time_ms;     /**< Total kernel execution time */
    double transfer_time_ms;   /**< Total host-device transfer time */
    double poisson_time_ms;    /**< Time spent in Poisson solver */
    int poisson_iterations;    /**< Poisson solver iterations */
    double poisson_residual;   /**< Final Poisson residual */
    size_t memory_allocated;   /**< GPU memory allocated (bytes) */
    int kernels_launched;      /**< Number of kernels launched */
} gpu_solver_stats_t;

/**
 * GPU Solver Context (opaque)
 * Holds GPU memory pointers and configuration
 */
typedef struct gpu_solver_context_t gpu_solver_context_t;

//=============================================================================
// DEVICE MANAGEMENT
//=============================================================================

/**
 * Initialize default GPU configuration
 */
CFD_LIBRARY_EXPORT gpu_config_t gpu_config_default(void);

/**
 * Check if CUDA is available on this system
 * @return 1 if CUDA is available, 0 otherwise
 */
CFD_LIBRARY_EXPORT int gpu_is_available(void);

/**
 * Get information about available GPU devices
 * @param info Array to fill with device info
 * @param max_devices Maximum number of devices to query
 * @return Number of devices found
 */
CFD_LIBRARY_EXPORT int gpu_get_device_info(gpu_device_info_t* info, int max_devices);

/**
 * Select GPU device to use
 * @param device_id Device ID to select
 * @return CFD_SUCCESS on success, error code otherwise
 */
CFD_LIBRARY_EXPORT cfd_status_t gpu_select_device(int device_id);

/**
 * Determine if GPU should be used based on problem size and config
 * @param config GPU configuration
 * @param nx Grid points in x
 * @param ny Grid points in y
 * @param num_steps Number of steps to run
 * @return 1 if GPU should be used, 0 for CPU
 */
CFD_LIBRARY_EXPORT int gpu_should_use(const gpu_config_t* config, size_t nx, size_t ny,
                                      int num_steps);

//=============================================================================
// GPU SOLVER CONTEXT
//=============================================================================

/**
 * Create GPU solver context
 * Allocates GPU memory and initializes CUDA resources
 * @param nx Grid points in x
 * @param ny Grid points in y
 * @param config GPU configuration
 * @return New GPU solver context, or NULL on error
 */
CFD_LIBRARY_EXPORT gpu_solver_context_t* gpu_solver_create(size_t nx, size_t ny,
                                                           const gpu_config_t* config);

/**
 * Destroy GPU solver context
 * Frees GPU memory and releases CUDA resources
 * @param ctx Context to destroy
 */
CFD_LIBRARY_EXPORT void gpu_solver_destroy(gpu_solver_context_t* ctx);

/**
 * Transfer flow field data from host to GPU
 * @param ctx GPU solver context
 * @param field Flow field to upload
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t gpu_solver_upload(gpu_solver_context_t* ctx, const flow_field* field);

/**
 * Transfer flow field data from GPU to host
 * @param ctx GPU solver context
 * @param field Flow field to download to
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t gpu_solver_download(gpu_solver_context_t* ctx, flow_field* field);

/**
 * Run one solver step on GPU
 * @param ctx GPU solver context
 * @param grid Grid configuration
 * @param params NSSolver parameters
 * @param stats Output statistics
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t gpu_solver_step(gpu_solver_context_t* ctx, const grid* grid,
                                                const ns_solver_params_t* params,
                                                gpu_solver_stats_t* stats);

/**
 * Get GPU solver statistics
 * @param ctx GPU solver context
 * @return Current statistics
 */
CFD_LIBRARY_EXPORT gpu_solver_stats_t gpu_solver_get_stats(const gpu_solver_context_t* ctx);

/**
 * Reset GPU solver statistics
 * @param ctx GPU solver context
 */
CFD_LIBRARY_EXPORT void gpu_solver_reset_stats(gpu_solver_context_t* ctx);

//=============================================================================
// HIGH-LEVEL GPU SOLVERS
//=============================================================================

/**
 * High-level GPU-accelerated Navier-Stokes solver
 * Automatically handles data transfer and GPU selection
 * @param field Flow field to solve
 * @param grid Grid configuration
 * @param params NSSolver parameters
 * @param config GPU configuration
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t solve_navier_stokes_gpu(flow_field* field, const grid* grid,
                                                        const ns_solver_params_t* params,
                                                        const gpu_config_t* config);

/**
 * GPU-accelerated projection method solver
 * @param field Flow field to solve
 * @param grid Grid configuration
 * @param params NSSolver parameters
 * @param config GPU configuration
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t solve_projection_method_gpu(flow_field* field, const grid* grid,
                                                            const ns_solver_params_t* params,
                                                            const gpu_config_t* config);

#ifdef __cplusplus
}
#endif

#endif  /* CFD_GPU_DEVICE_H */
