/**
 * Boundary Conditions - SIMD Implementation
 *
 * AVX2 optimized boundary condition implementations.
 * Uses SIMD for contiguous memory operations (top/bottom boundaries).
 * Falls back to scalar for strided access (left/right boundaries).
 *
 * Requires AVX2 support (available on all x86-64 CPUs from 2013+).
 * Falls back to scalar implementation when AVX2 is not available.
 */

#include "../boundary_conditions_internal.h"

/* AVX2 detection */
#if defined(__AVX2__)
#define BC_HAS_AVX2 1
#include <immintrin.h>
#endif

#if defined(BC_HAS_AVX2)

/**
 * Apply Neumann boundary conditions (zero gradient) with SIMD optimization.
 */
static void bc_apply_neumann_simd_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    /* AVX2: Process 4 doubles per iteration */
    size_t simd_end = nx & ~(size_t)3;  /* Round down to multiple of 4 */

    for (i = 0; i < simd_end; i += 4) {
        __m256d bottom_vals = _mm256_loadu_pd(bottom_src + i);
        __m256d top_vals = _mm256_loadu_pd(top_src + i);
        _mm256_storeu_pd(bottom_dst + i, bottom_vals);
        _mm256_storeu_pd(top_dst + i, top_vals);
    }

    /* Handle remaining elements */
    for (i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with SIMD optimization.
 */
static void bc_apply_periodic_simd_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries (periodic in x) - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

    /* AVX2: Process 4 doubles per iteration */
    size_t simd_end = nx & ~(size_t)3;

    for (i = 0; i < simd_end; i += 4) {
        __m256d bottom_vals = _mm256_loadu_pd(bottom_src + i);
        __m256d top_vals = _mm256_loadu_pd(top_src + i);
        _mm256_storeu_pd(bottom_dst + i, bottom_vals);
        _mm256_storeu_pd(top_dst + i, top_vals);
    }

    /* Handle remaining elements */
    for (i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with SIMD optimization.
 */
static void bc_apply_dirichlet_simd_impl(double* field, size_t nx, size_t ny,
                                          const bc_dirichlet_values_t* values) {
    size_t j, i;

    /* Left and right boundaries - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[j * nx] = values->left;
        field[j * nx + (nx - 1)] = values->right;
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    /* AVX2: Broadcast value to 4 doubles and store */
    __m256d bottom_broadcast = _mm256_set1_pd(values->bottom);
    __m256d top_broadcast = _mm256_set1_pd(values->top);
    size_t simd_end = nx & ~(size_t)3;  /* Round down to multiple of 4 */

    for (i = 0; i < simd_end; i += 4) {
        _mm256_storeu_pd(bottom_row + i, bottom_broadcast);
        _mm256_storeu_pd(top_row + i, top_broadcast);
    }

    /* Handle remaining elements */
    for (i = simd_end; i < nx; i++) {
        bottom_row[i] = values->bottom;
        top_row[i] = values->top;
    }
}

/* SIMD backend implementation table */
const bc_backend_impl_t bc_impl_simd = {
    .apply_neumann = bc_apply_neumann_simd_impl,
    .apply_periodic = bc_apply_periodic_simd_impl,
    .apply_dirichlet = bc_apply_dirichlet_simd_impl
};

#else /* !BC_HAS_AVX2 */

/* AVX2 not available - provide empty table (falls back to scalar) */
const bc_backend_impl_t bc_impl_simd = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL
};

#endif /* BC_HAS_AVX2 */
