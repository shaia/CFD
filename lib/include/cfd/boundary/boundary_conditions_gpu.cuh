/**
 * GPU Boundary Conditions Interface (CUDA)
 *
 * CUDA device-side boundary condition implementations that mirror
 * the host-side API from boundary_conditions.h.
 *
 * This header provides:
 * - CUDA kernels for BC application on device memory
 * - Host-callable wrapper functions for kernel launch
 * - Uses the same bc_type_t enum for consistency
 */

#ifndef CFD_BOUNDARY_CONDITIONS_GPU_CUH
#define CFD_BOUNDARY_CONDITIONS_GPU_CUH

#include "cfd/boundary/boundary_conditions.h"

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Apply Neumann boundary conditions to a scalar field on GPU
 *
 * @param d_field  Device pointer to scalar field (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param stream   CUDA stream for async execution (0 for default)
 */
void bc_apply_neumann_gpu(double* d_field, size_t nx, size_t ny, cudaStream_t stream);

/**
 * Apply boundary conditions to a scalar field on GPU
 *
 * @param d_field  Device pointer to scalar field (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param type     Type of boundary condition to apply
 * @param stream   CUDA stream for async execution (0 for default)
 */
void bc_apply_scalar_gpu(double* d_field, size_t nx, size_t ny, bc_type_t type, cudaStream_t stream);

/**
 * Apply boundary conditions to velocity components on GPU
 *
 * @param d_u      Device pointer to x-velocity array (size nx*ny)
 * @param d_v      Device pointer to y-velocity array (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param type     Type of boundary condition to apply
 * @param stream   CUDA stream for async execution (0 for default)
 */
void bc_apply_velocity_gpu(double* d_u, double* d_v, size_t nx, size_t ny, bc_type_t type,
                           cudaStream_t stream);

/**
 * Apply Dirichlet boundary conditions to a scalar field on GPU
 *
 * Sets boundary values to specified fixed values:
 *   - Left:   field[0,j] = values->left
 *   - Right:  field[nx-1,j] = values->right
 *   - Bottom: field[i,0] = values->bottom
 *   - Top:    field[i,ny-1] = values->top
 *
 * @param d_field  Device pointer to scalar field (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param values   Pointer to struct containing boundary values (host memory)
 * @param stream   CUDA stream for async execution (0 for default)
 */
void bc_apply_dirichlet_scalar_gpu(double* d_field, size_t nx, size_t ny,
                                    const bc_dirichlet_values_t* values,
                                    cudaStream_t stream);

/**
 * Apply Dirichlet boundary conditions to velocity components on GPU
 *
 * @param d_u       Device pointer to x-velocity array (size nx*ny)
 * @param d_v       Device pointer to y-velocity array (size nx*ny)
 * @param nx        Number of grid points in x-direction
 * @param ny        Number of grid points in y-direction
 * @param u_values  Pointer to struct containing u-velocity boundary values (host memory)
 * @param v_values  Pointer to struct containing v-velocity boundary values (host memory)
 * @param stream    CUDA stream for async execution (0 for default)
 */
void bc_apply_dirichlet_velocity_gpu(double* d_u, double* d_v, size_t nx, size_t ny,
                                      const bc_dirichlet_values_t* u_values,
                                      const bc_dirichlet_values_t* v_values,
                                      cudaStream_t stream);

/**
 * Apply inlet velocity boundary conditions on GPU
 *
 * Supports uniform and parabolic velocity profiles.
 * Note: Custom profile callbacks are not supported on GPU.
 *
 * @param d_u      Device pointer to x-velocity array (size nx*ny)
 * @param d_v      Device pointer to y-velocity array (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param config   Pointer to inlet configuration struct (host memory)
 * @param stream   CUDA stream for async execution (0 for default)
 * @return         CFD_SUCCESS, CFD_ERROR_INVALID, or CFD_ERROR_UNSUPPORTED
 */
cfd_status_t bc_apply_inlet_gpu(double* d_u, double* d_v, size_t nx, size_t ny,
                                 const bc_inlet_config_t* config,
                                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // CFD_BOUNDARY_CONDITIONS_GPU_CUH
