/**
 * Boundary Conditions Implementation
 *
 * Unified boundary condition functions for CFD simulations.
 * Eliminates code duplication across CPU, SIMD, and OMP solvers.
 *
 * OpenMP parallelization is enabled when CFD_ENABLE_OPENMP is defined,
 * ensuring no performance regression for OMP-based solvers.
 */

#include "cfd/core/boundary_conditions.h"

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>
#endif

/**
 * Apply Neumann boundary conditions (zero gradient) to a scalar field.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left boundary (i=0): field[0] = field[1]
 *   - Right boundary (i=nx-1): field[nx-1] = field[nx-2]
 *   - Bottom boundary (j=0): field[j=0] = field[j=1]
 *   - Top boundary (j=ny-1): field[j=ny-1] = field[j=ny-2]
 *
 * OpenMP parallelization is applied when CFD_ENABLE_OPENMP is defined.
 */
static void apply_neumann_scalar(double* field, size_t nx, size_t ny) {
    int j, i;

    // Left and right boundaries - parallelize over rows
#ifdef CFD_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (j = 0; j < (int)ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    // Top and bottom boundaries - parallelize over columns
#ifdef CFD_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < (int)nx; i++) {
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
 *
 * OpenMP parallelization is applied when CFD_ENABLE_OPENMP is defined.
 */
static void apply_periodic_scalar(double* field, size_t nx, size_t ny) {
    int j, i;

    // Left and right boundaries (periodic in x) - parallelize over rows
#ifdef CFD_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (j = 0; j < (int)ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    // Top and bottom boundaries (periodic in y) - parallelize over columns
#ifdef CFD_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < (int)nx; i++) {
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

    // For OMP builds, we can parallelize both calls together using sections
    // However, the simpler approach of calling sequentially is often just as efficient
    // because each call is already parallelized internally
    bc_apply_scalar(u, nx, ny, type);
    bc_apply_scalar(v, nx, ny, type);
}
