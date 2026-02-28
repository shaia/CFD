/**
 * Boundary Conditions - Parameterized Core Implementation
 *
 * Shared implementation for Neumann, Periodic, and Dirichlet BCs.
 * Before including, define:
 *   BC_CORE_FUNC_PREFIX  - function name prefix (scalar or omp)
 *   BC_CORE_USE_OMP      - 1 for OpenMP pragmas, 0 for plain loops
 *
 * Generates:
 *   bc_apply_neumann_<prefix>_impl()
 *   bc_apply_periodic_<prefix>_impl()
 *   bc_apply_dirichlet_<prefix>_impl()
 *
 * 3D support: all functions accept (nz, stride_z). When nz <= 1,
 * z-face loops are skipped and x/y-face loops run for a single plane.
 */

#include "boundary_conditions_internal.h"
#include "cfd/core/indexing.h"
#include <limits.h>

/* Token pasting helpers */
#define BC_CORE_PASTE2(a, b) a##b
#define BC_CORE_PASTE(a, b) BC_CORE_PASTE2(a, b)
#define BC_CORE_FN(name) BC_CORE_PASTE(BC_CORE_PASTE(bc_apply_, name), BC_CORE_PASTE(_, BC_CORE_PASTE(BC_CORE_FUNC_PREFIX, _impl)))

/* Loop variable type and cast for OMP compatibility */
#if BC_CORE_USE_OMP
#define BC_LOOP_VAR int
#define BC_LOOP_LIMIT(n) ((n) > (size_t)INT_MAX ? INT_MAX : (int)(n))
#define BC_OMP_FOR _Pragma("omp parallel for schedule(static)")
#else
#define BC_LOOP_VAR size_t
#define BC_LOOP_LIMIT(n) (n)
#define BC_OMP_FOR
#endif

/* --------------------------------------------------------------------
 * Neumann (zero-gradient)
 * -------------------------------------------------------------------- */
void BC_CORE_FN(neumann)(double* field, size_t nx, size_t ny,
                          size_t nz, size_t stride_z) {
    BC_LOOP_VAR j, i;
    size_t k;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        BC_OMP_FOR
        for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
            field[base + IDX_2D(0, j, nx)] = field[base + IDX_2D(1, j, nx)];
            field[base + IDX_2D(nx - 1, j, nx)] = field[base + IDX_2D(nx - 2, j, nx)];
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_dst = field + base;
        double* bottom_src = field + base + nx;
        double* top_dst = field + base + ((ny - 1) * nx);
        double* top_src = field + base + ((ny - 2) * nx);

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
            bottom_dst[i] = bottom_src[i];
            top_dst[i] = top_src[i];
        }
    }

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double* back_dst = field;                            /* k=0 plane */
        double* back_src = field + stride_z;                 /* k=1 plane */
        double* front_dst = field + ((nz - 1) * stride_z);  /* k=nz-1 plane */
        double* front_src = field + ((nz - 2) * stride_z);  /* k=nz-2 plane */
        size_t plane_size = nx * ny;

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(plane_size); i++) {
            back_dst[i] = back_src[i];
            front_dst[i] = front_src[i];
        }
    }
}

/* --------------------------------------------------------------------
 * Periodic
 * -------------------------------------------------------------------- */
void BC_CORE_FN(periodic)(double* field, size_t nx, size_t ny,
                           size_t nz, size_t stride_z) {
    BC_LOOP_VAR j, i;
    size_t k;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        BC_OMP_FOR
        for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
            field[base + IDX_2D(0, j, nx)] = field[base + IDX_2D(nx - 2, j, nx)];
            field[base + IDX_2D(nx - 1, j, nx)] = field[base + IDX_2D(1, j, nx)];
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_dst = field + base;
        double* bottom_src = field + base + ((ny - 2) * nx);
        double* top_dst = field + base + ((ny - 1) * nx);
        double* top_src = field + base + nx;

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
            bottom_dst[i] = bottom_src[i];
            top_dst[i] = top_src[i];
        }
    }

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double* back_dst = field;                            /* k=0 plane */
        double* back_src = field + ((nz - 2) * stride_z);   /* k=nz-2 plane */
        double* front_dst = field + ((nz - 1) * stride_z);  /* k=nz-1 plane */
        double* front_src = field + stride_z;                /* k=1 plane */
        size_t plane_size = nx * ny;

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(plane_size); i++) {
            back_dst[i] = back_src[i];
            front_dst[i] = front_src[i];
        }
    }
}

/* --------------------------------------------------------------------
 * Dirichlet (fixed value)
 * -------------------------------------------------------------------- */
void BC_CORE_FN(dirichlet)(double* field, size_t nx, size_t ny,
                            size_t nz, size_t stride_z,
                            const bc_dirichlet_values_t* values) {
    BC_LOOP_VAR j, i;
    size_t k;
    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        BC_OMP_FOR
        for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
            field[base + IDX_2D(0, j, nx)] = val_left;
            field[base + IDX_2D(nx - 1, j, nx)] = val_right;
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_row = field + base;
        double* top_row = field + base + ((ny - 1) * nx);

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
            bottom_row[i] = val_bottom;
            top_row[i] = val_top;
        }
    }

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double val_back = values->back;
        double val_front = values->front;
        double* back_plane = field;                         /* k=0 plane */
        double* front_plane = field + ((nz - 1) * stride_z); /* k=nz-1 plane */
        size_t plane_size = nx * ny;

        BC_OMP_FOR
        for (i = 0; i < BC_LOOP_LIMIT(plane_size); i++) {
            back_plane[i] = val_back;
            front_plane[i] = val_front;
        }
    }
}

/* Clean up macros to allow re-inclusion with different parameters */
#undef BC_LOOP_VAR
#undef BC_LOOP_LIMIT
#undef BC_OMP_FOR
#undef BC_CORE_FN
#undef BC_CORE_PASTE
#undef BC_CORE_PASTE2
#undef BC_CORE_FUNC_PREFIX
#undef BC_CORE_USE_OMP
