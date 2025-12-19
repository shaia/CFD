#ifndef CFD_BOUNDARY_CONDITIONS_H
#define CFD_BOUNDARY_CONDITIONS_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Boundary Condition Types
 *
 * Defines the types of boundary conditions available for CFD simulations.
 */
typedef enum {
    BC_TYPE_PERIODIC,   // Wrap-around: left boundary = right interior, etc.
    BC_TYPE_NEUMANN,    // Zero gradient: boundary = adjacent interior value
    BC_TYPE_DIRICHLET,  // Fixed value: boundary = specified constant value
    BC_TYPE_NOSLIP,     // No-slip wall: velocity = 0 at all boundaries
    BC_TYPE_INLET,      // Inlet velocity specification (placeholder for future)
    BC_TYPE_OUTLET      // Outlet/convective (placeholder for future)
} bc_type_t;

/**
 * Boundary Condition Backend Types
 *
 * Specifies which implementation backend to use for boundary conditions.
 * This allows runtime selection of the most appropriate implementation
 * based on the solver type being used.
 */
typedef enum {
    BC_BACKEND_AUTO,      // Auto-select best available (SIMD_OMP > OMP > Scalar)
    BC_BACKEND_SCALAR,    // Force scalar implementation (single-threaded)
    BC_BACKEND_OMP,       // Force OpenMP implementation (multi-threaded, scalar loops)
    BC_BACKEND_SIMD_OMP,  // Force SIMD + OpenMP (runtime: AVX2 on x86, NEON on ARM)
    BC_BACKEND_CUDA       // Force CUDA GPU implementation
} bc_backend_t;

/**
 * Dirichlet Boundary Condition Values
 *
 * Specifies fixed values for each boundary in Dirichlet (fixed value) BCs.
 * Used with bc_apply_dirichlet_* functions.
 */
typedef struct {
    double left;    // Value at x=0 boundary (column 0)
    double right;   // Value at x=Lx boundary (column nx-1)
    double top;     // Value at y=Ly boundary (row ny-1)
    double bottom;  // Value at y=0 boundary (row 0)
} bc_dirichlet_values_t;

/**
 * Apply boundary conditions to a scalar field (raw array)
 *
 * @param field Pointer to the scalar field array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions to velocity components (u, v arrays)
 *
 * @param u     Pointer to x-velocity array (size nx*ny)
 * @param v     Pointer to y-velocity array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/* ============================================================================
 * Convenience Macros
 *
 * Shorthand macros for applying common boundary condition types.
 * These use the global backend setting (see bc_set_backend()).
 * ============================================================================ */

/**
 * Apply Neumann (zero-gradient) boundary conditions to a scalar field.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left:   field[0,j] = field[1,j]
 *   - Right:  field[nx-1,j] = field[nx-2,j]
 *   - Bottom: field[i,0] = field[i,1]
 *   - Top:    field[i,ny-1] = field[i,ny-2]
 *
 * @param field Pointer to scalar field array (size nx*ny, row-major)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 */
#define bc_apply_neumann(field, nx, ny)  bc_apply_scalar((field), (nx), (ny), BC_TYPE_NEUMANN)

/**
 * Apply periodic boundary conditions to a scalar field.
 *
 * Wraps values from opposite boundaries:
 *   - Left:   field[0,j] = field[nx-2,j]
 *   - Right:  field[nx-1,j] = field[1,j]
 *   - Bottom: field[i,0] = field[i,ny-2]
 *   - Top:    field[i,ny-1] = field[i,1]
 *
 * @param field Pointer to scalar field array (size nx*ny, row-major)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 */
#define bc_apply_periodic(field, nx, ny) bc_apply_scalar((field), (nx), (ny), BC_TYPE_PERIODIC)

/* ============================================================================
 * Backend Selection API
 * ============================================================================ */

/**
 * Get the currently active BC backend.
 *
 * @return The current backend type
 */
CFD_LIBRARY_EXPORT bc_backend_t bc_get_backend(void);

/**
 * Get the name of the currently active BC backend as a string.
 *
 * @return Human-readable backend name (e.g., "scalar", "omp", "simd_omp", "cuda")
 *         For simd_omp, may include architecture detail like "simd_omp (avx2)" or "simd_omp (neon)"
 */
CFD_LIBRARY_EXPORT const char* bc_get_backend_name(void);

/**
 * Set the BC backend to use for subsequent operations.
 *
 * @param backend The backend to use
 * @return true if the backend was set successfully, false if unavailable
 *
 * Note: BC_BACKEND_AUTO always succeeds and selects the best available.
 *       Other backends may fail if not compiled in or not supported.
 */
CFD_LIBRARY_EXPORT bool bc_set_backend(bc_backend_t backend);

/**
 * Check if a specific backend is available.
 *
 * @param backend The backend to check
 * @return true if the backend is available, false otherwise
 */
CFD_LIBRARY_EXPORT bool bc_backend_available(bc_backend_t backend);

/* ============================================================================
 * Explicit Backend API
 *
 * These functions allow direct selection of a specific implementation,
 * bypassing the global backend setting. Useful when different solvers
 * need different BC implementations.
 * ============================================================================ */

/**
 * Apply boundary conditions using scalar implementation.
 * Always available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_cpu(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_simd_omp(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_scalar_omp(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using scalar implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_cpu(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_simd_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply velocity boundary conditions using OpenMP implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_velocity_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/* ============================================================================
 * Dirichlet Boundary Conditions API
 *
 * Dirichlet BCs set boundary values to fixed specified values.
 * Unlike Neumann/Periodic, these require explicit values to be provided.
 * ============================================================================ */

/**
 * Apply Dirichlet (fixed value) boundary conditions to a scalar field.
 *
 * Sets each boundary to the corresponding value in the values struct:
 *   - Left:   field[0,j] = values->left
 *   - Right:  field[nx-1,j] = values->right
 *   - Bottom: field[i,0] = values->bottom
 *   - Top:    field[i,ny-1] = values->top
 *
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param field  Pointer to scalar field array (size nx*ny, row-major)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 * @param values Pointer to struct containing boundary values
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar(double* field, size_t nx, size_t ny,
                                                           const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions to velocity components (u, v).
 *
 * @param u        Pointer to x-velocity array (size nx*ny)
 * @param v        Pointer to y-velocity array (size nx*ny)
 * @param nx       Number of grid points in x-direction
 * @param ny       Number of grid points in y-direction
 * @param u_values Pointer to struct containing u-velocity boundary values
 * @param v_values Pointer to struct containing v-velocity boundary values
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                                             const bc_dirichlet_values_t* u_values,
                                                             const bc_dirichlet_values_t* v_values);

/* Backend-specific Dirichlet implementations */

/**
 * Apply Dirichlet boundary conditions using scalar implementation.
 * Always available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                                               const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_simd_omp(double* field, size_t nx, size_t ny,
                                                                    const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_scalar_omp(double* field, size_t nx, size_t ny,
                                                               const bc_dirichlet_values_t* values);

/**
 * Apply Dirichlet velocity boundary conditions using scalar implementation.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                                                 const bc_dirichlet_values_t* u_values,
                                                                 const bc_dirichlet_values_t* v_values);

/**
 * Apply Dirichlet velocity boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_simd_omp(double* u, double* v, size_t nx, size_t ny,
                                                                      const bc_dirichlet_values_t* u_values,
                                                                      const bc_dirichlet_values_t* v_values);

/**
 * Apply Dirichlet velocity boundary conditions using OpenMP implementation.
 * @return CFD_ERROR_UNSUPPORTED if OpenMP not available.
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_dirichlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                                                 const bc_dirichlet_values_t* u_values,
                                                                 const bc_dirichlet_values_t* v_values);

/**
 * Convenience macro for applying Dirichlet BCs to a scalar field.
 */
#define bc_apply_dirichlet(field, nx, ny, values) \
    bc_apply_dirichlet_scalar((field), (nx), (ny), (values))

/* ============================================================================
 * No-Slip Wall Boundary Conditions API
 *
 * No-slip BCs enforce zero velocity at solid walls.
 * This is the standard wall boundary condition for viscous flows.
 * Equivalent to Dirichlet BCs with all values set to 0.
 * ============================================================================ */

/**
 * Apply no-slip wall boundary conditions to velocity components.
 *
 * Sets both u and v velocity components to zero at all boundaries:
 *   - u = 0, v = 0 at left, right, top, and bottom walls
 *
 * This is the standard boundary condition for solid walls in viscous flow.
 * Uses the currently selected backend (see bc_set_backend()).
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using scalar implementation.
 * Always available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_cpu(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using SIMD + OpenMP implementation.
 * Automatically selects AVX2 (x86-64) or NEON (ARM64) at runtime.
 * Returns CFD_ERROR_UNSUPPORTED if SIMD or OpenMP not available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_simd_omp(double* u, double* v, size_t nx, size_t ny);

/**
 * Apply no-slip wall boundary conditions using OpenMP implementation.
 * Returns CFD_ERROR_UNSUPPORTED if OpenMP not available.
 *
 * @param u  Pointer to x-velocity array (size nx*ny)
 * @param v  Pointer to y-velocity array (size nx*ny)
 * @param nx Number of grid points in x-direction
 * @param ny Number of grid points in y-direction
 * @return CFD_SUCCESS on success, error code on failure
 */
CFD_LIBRARY_EXPORT cfd_status_t bc_apply_noslip_omp(double* u, double* v, size_t nx, size_t ny);

/**
 * Convenience macro for applying no-slip BCs to velocity fields.
 */
#define bc_apply_noslip_velocity(u, v, nx, ny) bc_apply_noslip((u), (v), (nx), (ny))

#ifdef __cplusplus
}
#endif

#endif  // CFD_BOUNDARY_CONDITIONS_H
