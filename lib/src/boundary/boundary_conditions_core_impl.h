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
 */

#include "boundary_conditions_internal.h"
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
void BC_CORE_FN(neumann)(double* field, size_t nx, size_t ny) {
    BC_LOOP_VAR j, i;

    BC_OMP_FOR
    for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    BC_OMP_FOR
    for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/* --------------------------------------------------------------------
 * Periodic
 * -------------------------------------------------------------------- */
void BC_CORE_FN(periodic)(double* field, size_t nx, size_t ny) {
    BC_LOOP_VAR j, i;

    BC_OMP_FOR
    for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;

    BC_OMP_FOR
    for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/* --------------------------------------------------------------------
 * Dirichlet (fixed value)
 * -------------------------------------------------------------------- */
void BC_CORE_FN(dirichlet)(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values) {
    BC_LOOP_VAR j, i;
    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    BC_OMP_FOR
    for (j = 0; j < BC_LOOP_LIMIT(ny); j++) {
        field[j * nx] = val_left;
        field[j * nx + (nx - 1)] = val_right;
    }

    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    BC_OMP_FOR
    for (i = 0; i < BC_LOOP_LIMIT(nx); i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
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
