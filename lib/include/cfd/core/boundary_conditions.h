#ifndef CFD_BOUNDARY_CONDITIONS_H
#define CFD_BOUNDARY_CONDITIONS_H

#include "cfd/cfd_export.h"

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
    BC_TYPE_DIRICHLET,  // Fixed value (placeholder for future implementation)
    BC_TYPE_NOSLIP,     // No-slip wall: velocity = 0 (placeholder for future)
    BC_TYPE_INLET,      // Inlet velocity specification (placeholder for future)
    BC_TYPE_OUTLET      // Outlet/convective (placeholder for future)
} bc_type_t;

/**
 * Apply boundary conditions to a scalar field (raw array)
 *
 * @param field Pointer to the scalar field array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 */
CFD_LIBRARY_EXPORT void bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type);

/**
 * Apply boundary conditions to velocity components (u, v arrays)
 *
 * @param u     Pointer to x-velocity array (size nx*ny)
 * @param v     Pointer to y-velocity array (size nx*ny)
 * @param nx    Number of grid points in x-direction
 * @param ny    Number of grid points in y-direction
 * @param type  Type of boundary condition to apply
 */
CFD_LIBRARY_EXPORT void bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

/**
 * Convenience macros for common boundary condition types
 */
#define bc_apply_neumann(field, nx, ny)  bc_apply_scalar((field), (nx), (ny), BC_TYPE_NEUMANN)
#define bc_apply_periodic(field, nx, ny) bc_apply_scalar((field), (nx), (ny), BC_TYPE_PERIODIC)

#ifdef __cplusplus
}
#endif

#endif  // CFD_BOUNDARY_CONDITIONS_H
