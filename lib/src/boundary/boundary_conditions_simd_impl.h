/**
 * Boundary Conditions - Parameterized SIMD Implementation
 *
 * This header is included by both AVX2 and NEON main BC implementations.
 * Before including, define:
 *   BC_SIMD_STORE(dst, val)  - store SIMD vector to memory
 *   BC_SIMD_LOAD(src)        - load SIMD vector from memory
 *   BC_SIMD_BROADCAST(val)   - broadcast scalar to SIMD vector
 *   BC_SIMD_VEC_TYPE          - SIMD vector type (e.g., __m256d, float64x2_t)
 *   BC_SIMD_WIDTH             - number of doubles per SIMD vector (4 or 2)
 *   BC_SIMD_MASK              - alignment mask (3 for AVX2, 1 for NEON)
 *   BC_SIMD_THRESHOLD         - min width for OMP on SIMD loops
 *   BC_SIMD_FUNC_PREFIX       - function name prefix (e.g., avx2, neon)
 */

#include "boundary_conditions_internal.h"
#include <limits.h>

#define BC_SIMD_PASTE2(a, b) a##b
#define BC_SIMD_PASTE(a, b) BC_SIMD_PASTE2(a, b)
#define BC_SIMD_FN(name) BC_SIMD_PASTE(bc_apply_##name##_, BC_SIMD_PASTE(BC_SIMD_FUNC_PREFIX, _impl))

static inline int bc_simd_size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/**
 * Apply Neumann boundary conditions (zero gradient) with SIMD + OpenMP.
 */
static void BC_SIMD_FN(neumann)(double* field, size_t nx, size_t ny) {
    int j, i;

    #pragma omp parallel for schedule(static)
    for (j = 0; j < bc_simd_size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + 1];
        field[row + nx - 1] = field[row + nx - 2];
    }

    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

    if (nx >= BC_SIMD_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < bc_simd_size_to_int(simd_end); i += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_dst + i, BC_SIMD_LOAD(bottom_src + i));
            BC_SIMD_STORE(top_dst + i, BC_SIMD_LOAD(top_src + i));
        }
    } else {
        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_dst + ii, BC_SIMD_LOAD(bottom_src + ii));
            BC_SIMD_STORE(top_dst + ii, BC_SIMD_LOAD(top_src + ii));
        }
    }

    for (size_t ii = simd_end; ii < nx; ii++) {
        bottom_dst[ii] = bottom_src[ii];
        top_dst[ii] = top_src[ii];
    }
}

/**
 * Apply periodic boundary conditions with SIMD + OpenMP.
 */
static void BC_SIMD_FN(periodic)(double* field, size_t nx, size_t ny) {
    int j, i;

    #pragma omp parallel for schedule(static)
    for (j = 0; j < bc_simd_size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + nx - 2];
        field[row + nx - 1] = field[row + 1];
    }

    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;

    size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

    if (nx >= BC_SIMD_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < bc_simd_size_to_int(simd_end); i += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_dst + i, BC_SIMD_LOAD(bottom_src + i));
            BC_SIMD_STORE(top_dst + i, BC_SIMD_LOAD(top_src + i));
        }
    } else {
        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_dst + ii, BC_SIMD_LOAD(bottom_src + ii));
            BC_SIMD_STORE(top_dst + ii, BC_SIMD_LOAD(top_src + ii));
        }
    }

    for (size_t ii = simd_end; ii < nx; ii++) {
        bottom_dst[ii] = bottom_src[ii];
        top_dst[ii] = top_src[ii];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with SIMD + OpenMP.
 */
static void BC_SIMD_FN(dirichlet)(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values) {
    int j, i;

    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    #pragma omp parallel for schedule(static)
    for (j = 0; j < bc_simd_size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = val_left;
        field[row + nx - 1] = val_right;
    }

    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    BC_SIMD_VEC_TYPE bottom_broadcast = BC_SIMD_BROADCAST(val_bottom);
    BC_SIMD_VEC_TYPE top_broadcast = BC_SIMD_BROADCAST(val_top);
    size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

    if (nx >= BC_SIMD_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < bc_simd_size_to_int(simd_end); i += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_row + i, bottom_broadcast);
            BC_SIMD_STORE(top_row + i, top_broadcast);
        }
    } else {
        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
            BC_SIMD_STORE(bottom_row + ii, bottom_broadcast);
            BC_SIMD_STORE(top_row + ii, top_broadcast);
        }
    }

    for (size_t ii = simd_end; ii < nx; ii++) {
        bottom_row[ii] = val_bottom;
        top_row[ii] = val_top;
    }
}

#undef BC_SIMD_PASTE2
#undef BC_SIMD_PASTE
#undef BC_SIMD_FN
