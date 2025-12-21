/**
 * Boundary Conditions - OpenMP Implementation
 *
 * OpenMP parallelized boundary condition implementations.
 * Parallelizes over rows for left/right boundaries and
 * over columns for top/bottom boundaries.
 *
 * Inlet boundary conditions are implemented in boundary_conditions_inlet_omp.c
 */

#include "../boundary_conditions_internal.h"

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>

/**
 * Apply Neumann boundary conditions (zero gradient) with OpenMP parallelization.
 */
static void bc_apply_neumann_omp_impl(double* field, size_t nx, size_t ny) {
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
 */
static void bc_apply_periodic_omp_impl(double* field, size_t nx, size_t ny) {
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

/**
 * Apply Dirichlet (fixed value) boundary conditions with OpenMP parallelization.
 */
static void bc_apply_dirichlet_omp_impl(double* field, size_t nx, size_t ny,
                                         const bc_dirichlet_values_t* values) {
    int j, i;

    /* Left and right boundaries - parallelize over rows */
#pragma omp parallel for schedule(static)
    for (j = 0; j < (int)ny; j++) {
        field[j * nx] = values->left;
        field[j * nx + (nx - 1)] = values->right;
    }

    /* Top and bottom boundaries - parallelize over columns */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);
    double val_bottom = values->bottom;
    double val_top = values->top;

#pragma omp parallel for schedule(static)
    for (i = 0; i < (int)nx; i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
    }
}

/* OpenMP backend implementation table
 * Note: bc_apply_inlet_omp_impl is defined in boundary_conditions_inlet_omp.c */
const bc_backend_impl_t bc_impl_omp = {
    .apply_neumann = bc_apply_neumann_omp_impl,
    .apply_periodic = bc_apply_periodic_omp_impl,
    .apply_dirichlet = bc_apply_dirichlet_omp_impl,
    .apply_inlet = bc_apply_inlet_omp_impl
};

#else /* !CFD_ENABLE_OPENMP */

/* OpenMP not available - provide empty table */
const bc_backend_impl_t bc_impl_omp = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL
};

#endif /* CFD_ENABLE_OPENMP */
