/**
 * Boundary Conditions Implementation
 *
 * Unified boundary condition functions for CFD simulations.
 * Eliminates code duplication across CPU, SIMD, and OMP solvers.
 */

#include "cfd/core/boundary_conditions.h"

/**
 * Apply Neumann boundary conditions (zero gradient) to a scalar field.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left boundary (i=0): field[0] = field[1]
 *   - Right boundary (i=nx-1): field[nx-1] = field[nx-2]
 *   - Bottom boundary (j=0): field[j=0] = field[j=1]
 *   - Top boundary (j=ny-1): field[j=ny-1] = field[j=ny-2]
 */
static void apply_neumann_scalar(double* field, size_t nx, size_t ny) {
    // Left and right boundaries
    for (size_t j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }
    // Top and bottom boundaries
    for (size_t i = 0; i < nx; i++) {
        field[i] = field[nx + i];
        field[((ny - 1) * nx) + i] = field[((ny - 2) * nx) + i];
    }
}

/**
 * Apply periodic boundary conditions to a scalar field.
 *
 * Wraps values from opposite boundaries:
 *   - Left boundary (i=0): copies from right interior (i=nx-2)
 *   - Right boundary (i=nx-1): copies from left interior (i=1)
 *   - Bottom boundary (j=0): copies from top interior (j=ny-2)
 *   - Top boundary (j=ny-1): copies from bottom interior (j=1)
 */
static void apply_periodic_scalar(double* field, size_t nx, size_t ny) {
    // Left and right boundaries (periodic in x)
    for (size_t j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }
    // Top and bottom boundaries (periodic in y)
    for (size_t i = 0; i < nx; i++) {
        field[i] = field[((ny - 2) * nx) + i];
        field[((ny - 1) * nx) + i] = field[nx + i];
    }
}

void bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type) {
    if (!field || nx < 3 || ny < 3) {
        return;
    }

    switch (type) {
        case BC_TYPE_NEUMANN:
            apply_neumann_scalar(field, nx, ny);
            break;

        case BC_TYPE_PERIODIC:
            apply_periodic_scalar(field, nx, ny);
            break;

        case BC_TYPE_DIRICHLET:
        case BC_TYPE_NOSLIP:
        case BC_TYPE_INLET:
        case BC_TYPE_OUTLET:
            // Not yet implemented - fall through to default
        default:
            // Default to Neumann for safety
            apply_neumann_scalar(field, nx, ny);
            break;
    }
}

void bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return;
    }

    bc_apply_scalar(u, nx, ny, type);
    bc_apply_scalar(v, nx, ny, type);
}
