/**
 * GPU-Accelerated Solver Interface
 *
 * Provides CUDA-accelerated pressure solving and velocity updates
 * for the Navier-Stokes equations. Uses Structure of Arrays (SoA)
 * layout for coalesced GPU memory access.
 *
 * GPU usage is configurable and can be automatically determined
 * based on grid size and simulation parameters.
 */

#ifndef CFD_SOLVER_GPU_H
#define CFD_SOLVER_GPU_H

#include "grid.h"
#include "solver_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU Configuration Settings
 */
typedef struct {
    // Enable/disable GPU acceleration
    int enable_gpu;

    // Automatic selection thresholds
    size_t min_grid_size;  // Minimum grid points for GPU (default: 10000)
    int min_steps;         // Minimum steps to amortize transfer cost (default: 10)

    // Performance tuning
    int block_size_x;          // CUDA block size X (default: 16)
    int block_size_y;          // CUDA block size Y (default: 16)
    int poisson_max_iter;      // Max Poisson iterations on GPU (default: 1000)
    double poisson_tolerance;  // Poisson convergence tolerance (default: 1e-6)

    // Memory management
    int persistent_memory;  // Keep GPU memory allocated between steps (default: 1)
    int async_transfers;    // Use async memory transfers (default: 1)

    // Debug options
    int sync_after_kernel;  // Synchronize after each kernel for debugging (default: 0)
    int verbose;            // Print GPU info and timing (default: 0)
} GPUConfig;

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
} GPUDeviceInfo;

/**
 * GPU Solver Statistics
 */
typedef struct {
    double kernel_time_ms;    // Total kernel execution time
    double transfer_time_ms;  // Total host-device transfer time
    double poisson_time_ms;   // Time spent in Poisson solver
    int poisson_iterations;   // Poisson solver iterations
    double poisson_residual;  // Final Poisson residual
    size_t memory_allocated;  // GPU memory allocated (bytes)
    int kernels_launched;     // Number of kernels launched
} GPUSolverStats;

/**
 * GPU Solver Context (opaque)
 * Holds GPU memory pointers and configuration
 */
typedef struct GPUSolverContext GPUSolverContext;

/**
 * Initialize default GPU configuration
 */
GPUConfig gpu_config_default(void);

/**
 * Check if CUDA is available on this system
 * Returns 1 if CUDA is available, 0 otherwise
 */
int gpu_is_available(void);

/**
 * Get information about available GPU devices
 * Returns number of devices found
 */
int gpu_get_device_info(GPUDeviceInfo* info, int max_devices);

/**
 * Select GPU device to use
 * Returns 0 on success, -1 on error
 */
int gpu_select_device(int device_id);

/**
 * Determine if GPU should be used based on problem size and config
 * Returns 1 if GPU should be used, 0 for CPU
 */
int gpu_should_use(const GPUConfig* config, size_t nx, size_t ny, int num_steps);

/**
 * Create GPU solver context
 * Allocates GPU memory and initializes CUDA resources
 */
GPUSolverContext* gpu_solver_create(size_t nx, size_t ny, const GPUConfig* config);

/**
 * Destroy GPU solver context
 * Frees GPU memory and releases CUDA resources
 */
void gpu_solver_destroy(GPUSolverContext* ctx);

/**
 * Transfer flow field data from host to GPU
 */
int gpu_solver_upload(GPUSolverContext* ctx, const FlowField* field);

/**
 * Transfer flow field data from GPU to host
 */
int gpu_solver_download(GPUSolverContext* ctx, FlowField* field);

/**
 * Run one solver step on GPU
 */
int gpu_solver_step(GPUSolverContext* ctx, const Grid* grid, const SolverParams* params,
                    GPUSolverStats* stats);

/**
 * Get GPU solver statistics
 */
GPUSolverStats gpu_solver_get_stats(const GPUSolverContext* ctx);

/**
 * Reset GPU solver statistics
 */
void gpu_solver_reset_stats(GPUSolverContext* ctx);

/**
 * High-level GPU-accelerated Navier-Stokes solver
 * Automatically handles data transfer and GPU selection
 */
void solve_navier_stokes_gpu(FlowField* field, const Grid* grid, const SolverParams* params,
                             const GPUConfig* config);

/**
 * GPU-accelerated projection method solver
 */
void solve_projection_method_gpu(FlowField* field, const Grid* grid, const SolverParams* params,
                                 const GPUConfig* config);

#ifdef __cplusplus
}
#endif

#endif  // CFD_SOLVER_GPU_H
