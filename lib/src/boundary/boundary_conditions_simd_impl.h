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
 *   BC_SIMD_MASK              - low-bit mask for rounding down (3 for AVX2, 1 for NEON)
 *   BC_SIMD_THRESHOLD         - min width for OMP on SIMD loops
 *   BC_SIMD_FUNC_PREFIX       - function name prefix (e.g., avx2, neon)
 *
 * 3D support: all functions accept (nz, stride_z). When nz <= 1,
 * z-face loops are skipped and x/y-face loops run for a single plane.
 */

#include "boundary_conditions_internal.h"
#include "cfd/core/indexing.h"
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
static void BC_SIMD_FN(neumann)(double* field, size_t nx, size_t ny,
                                 size_t nz, size_t stride_z) {
    int j, i;
    size_t k;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        #pragma omp parallel for schedule(static)
        for (j = 0; j < bc_simd_size_to_int(ny); j++) {
            size_t row = base + (size_t)j * nx;
            field[row] = field[row + 1];
            field[row + nx - 1] = field[row + nx - 2];
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_dst = field + base;
        double* bottom_src = field + base + nx;
        double* top_dst = field + base + ((ny - 1) * nx);
        double* top_src = field + base + ((ny - 2) * nx);

        size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

        if (nx >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
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

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double* back_dst = field;
        double* back_src = field + stride_z;
        double* front_dst = field + ((nz - 1) * stride_z);
        double* front_src = field + ((nz - 2) * stride_z);
        size_t plane_size = nx * ny;
        size_t simd_end = plane_size & ~(size_t)BC_SIMD_MASK;

        if (plane_size >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_dst + i, BC_SIMD_LOAD(back_src + i));
                BC_SIMD_STORE(front_dst + i, BC_SIMD_LOAD(front_src + i));
            }
        } else {
            for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_dst + ii, BC_SIMD_LOAD(back_src + ii));
                BC_SIMD_STORE(front_dst + ii, BC_SIMD_LOAD(front_src + ii));
            }
        }

        for (size_t ii = simd_end; ii < plane_size; ii++) {
            back_dst[ii] = back_src[ii];
            front_dst[ii] = front_src[ii];
        }
    }
}

/**
 * Apply periodic boundary conditions with SIMD + OpenMP.
 */
static void BC_SIMD_FN(periodic)(double* field, size_t nx, size_t ny,
                                  size_t nz, size_t stride_z) {
    int j, i;
    size_t k;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        #pragma omp parallel for schedule(static)
        for (j = 0; j < bc_simd_size_to_int(ny); j++) {
            size_t row = base + (size_t)j * nx;
            field[row] = field[row + nx - 2];
            field[row + nx - 1] = field[row + 1];
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_dst = field + base;
        double* bottom_src = field + base + ((ny - 2) * nx);
        double* top_dst = field + base + ((ny - 1) * nx);
        double* top_src = field + base + nx;

        size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

        if (nx >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
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

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double* back_dst = field;
        double* back_src = field + ((nz - 2) * stride_z);
        double* front_dst = field + ((nz - 1) * stride_z);
        double* front_src = field + stride_z;
        size_t plane_size = nx * ny;
        size_t simd_end = plane_size & ~(size_t)BC_SIMD_MASK;

        if (plane_size >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_dst + i, BC_SIMD_LOAD(back_src + i));
                BC_SIMD_STORE(front_dst + i, BC_SIMD_LOAD(front_src + i));
            }
        } else {
            for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_dst + ii, BC_SIMD_LOAD(back_src + ii));
                BC_SIMD_STORE(front_dst + ii, BC_SIMD_LOAD(front_src + ii));
            }
        }

        for (size_t ii = simd_end; ii < plane_size; ii++) {
            back_dst[ii] = back_src[ii];
            front_dst[ii] = front_src[ii];
        }
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with SIMD + OpenMP.
 */
static void BC_SIMD_FN(dirichlet)(double* field, size_t nx, size_t ny,
                                   size_t nz, size_t stride_z,
                                   const bc_dirichlet_values_t* values) {
    int j, i;
    size_t k;

    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    /* x-faces (left/right) for each z-plane */
    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        #pragma omp parallel for schedule(static)
        for (j = 0; j < bc_simd_size_to_int(ny); j++) {
            size_t row = base + (size_t)j * nx;
            field[row] = val_left;
            field[row + nx - 1] = val_right;
        }
    }

    /* y-faces (bottom/top) for each z-plane */
    BC_SIMD_VEC_TYPE bottom_broadcast = BC_SIMD_BROADCAST(val_bottom);
    BC_SIMD_VEC_TYPE top_broadcast = BC_SIMD_BROADCAST(val_top);

    for (k = 0; k < nz; k++) {
        size_t base = k * stride_z;
        double* bottom_row = field + base;
        double* top_row = field + base + ((ny - 1) * nx);
        size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

        if (nx >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
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

    /* z-faces (back/front) - only when nz > 1 */
    if (nz > 1) {
        double val_back = values->back;
        double val_front = values->front;
        BC_SIMD_VEC_TYPE back_broadcast = BC_SIMD_BROADCAST(val_back);
        BC_SIMD_VEC_TYPE front_broadcast = BC_SIMD_BROADCAST(val_front);
        double* back_plane = field;
        double* front_plane = field + ((nz - 1) * stride_z);
        size_t plane_size = nx * ny;
        size_t simd_end = plane_size & ~(size_t)BC_SIMD_MASK;

        if (plane_size >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
            #pragma omp parallel for schedule(static)
            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_plane + i, back_broadcast);
                BC_SIMD_STORE(front_plane + i, front_broadcast);
            }
        } else {
            for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                BC_SIMD_STORE(back_plane + ii, back_broadcast);
                BC_SIMD_STORE(front_plane + ii, front_broadcast);
            }
        }

        for (size_t ii = simd_end; ii < plane_size; ii++) {
            back_plane[ii] = val_back;
            front_plane[ii] = val_front;
        }
    }
}

#undef BC_SIMD_PASTE2
#undef BC_SIMD_PASTE
#undef BC_SIMD_FN
#undef BC_SIMD_STORE
#undef BC_SIMD_LOAD
#undef BC_SIMD_BROADCAST
#undef BC_SIMD_VEC_TYPE
#undef BC_SIMD_WIDTH
#undef BC_SIMD_MASK
#undef BC_SIMD_THRESHOLD
#undef BC_SIMD_FUNC_PREFIX
