/**
 * Boundary Conditions - OpenMP Implementation
 *
 * OpenMP parallelized boundary condition implementations.
 * Parallelizes over rows for left/right boundaries and
 * over columns for top/bottom boundaries.
 */

#include "boundary_conditions_internal.h"

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>

/**
 * Apply Neumann boundary conditions (zero gradient) with OpenMP parallelization.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left boundary (i=0): field[0] = field[1]
 *   - Right boundary (i=nx-1): field[nx-1] = field[nx-2]
 *   - Bottom boundary (j=0): field[j=0] = field[j=1]
 *   - Top boundary (j=ny-1): field[j=ny-1] = field[j=ny-2]
 */
void bc_apply_neumann_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries - parallelize over rows */
#pragma omp parallel for schedule(static)
    for (j = 0; j < (int)ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    /* Top and bottom boundaries - parallelize over columns */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

#pragma omp parallel for schedule(static)
    for (i = 0; i < (int)nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with OpenMP parallelization.
 *
 * Wraps values from opposite boundaries:
 *   - Left boundary (i=0): copies from right interior (i=nx-2)
 *   - Right boundary (i=nx-1): copies from left interior (i=1)
 *   - Bottom boundary (j=0): copies from top interior (j=ny-2)
 *   - Top boundary (j=ny-1): copies from bottom interior (j=1)
 */
void bc_apply_periodic_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries (periodic in x) - parallelize over rows */
#pragma omp parallel for schedule(static)
    for (j = 0; j < (int)ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) - parallelize over columns */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

#pragma omp parallel for schedule(static)
    for (i = 0; i < (int)nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

#endif /* CFD_ENABLE_OPENMP */
