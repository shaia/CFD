/**
 * Boundary Conditions - SIMD Implementation
 *
 * AVX2 and SSE2 optimized boundary condition implementations.
 * Uses SIMD for contiguous memory operations (top/bottom boundaries).
 * Falls back to scalar for strided access (left/right boundaries).
 */

#include "../boundary_conditions_internal.h"

/* SIMD headers based on detected support */
#if defined(BC_HAS_AVX2)
#include <immintrin.h>
#elif defined(BC_HAS_SSE2)
#include <emmintrin.h>
#endif

#if defined(BC_HAS_SIMD)

/**
 * Apply Neumann boundary conditions (zero gradient) with SIMD optimization.
 *
 * Sets boundary values equal to adjacent interior values:
 *   - Left boundary (i=0): field[0] = field[1]
 *   - Right boundary (i=nx-1): field[nx-1] = field[nx-2]
 *   - Bottom boundary (j=0): field[j=0] = field[j=1]
 *   - Top boundary (j=ny-1): field[j=ny-1] = field[j=ny-2]
 *
 * SIMD is used for contiguous top/bottom boundaries.
 */
void bc_apply_neumann_simd_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use SIMD */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

#if defined(BC_HAS_AVX2)
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
#elif defined(BC_HAS_SSE2)
    /* SSE2: Process 2 doubles per iteration */
    size_t simd_end = nx & ~(size_t)1;  /* Round down to multiple of 2 */

    for (i = 0; i < simd_end; i += 2) {
        __m128d bottom_vals = _mm_loadu_pd(bottom_src + i);
        __m128d top_vals = _mm_loadu_pd(top_src + i);
        _mm_storeu_pd(bottom_dst + i, bottom_vals);
        _mm_storeu_pd(top_dst + i, top_vals);
    }

    /* Handle remaining element */
    for (i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
#endif
}

/**
 * Apply periodic boundary conditions with SIMD optimization.
 *
 * Wraps values from opposite boundaries:
 *   - Left boundary (i=0): copies from right interior (i=nx-2)
 *   - Right boundary (i=nx-1): copies from left interior (i=1)
 *   - Bottom boundary (j=0): copies from top interior (j=ny-2)
 *   - Top boundary (j=ny-1): copies from bottom interior (j=1)
 *
 * SIMD is used for contiguous top/bottom boundaries.
 */
void bc_apply_periodic_simd_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries (periodic in x) - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use SIMD */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

#if defined(BC_HAS_AVX2)
    /* AVX2: Process 4 doubles per iteration */
    size_t simd_end = nx & ~(size_t)3;

    for (i = 0; i < simd_end; i += 4) {
        __m256d bottom_vals = _mm256_loadu_pd(bottom_src + i);
        __m256d top_vals = _mm256_loadu_pd(top_src + i);
        _mm256_storeu_pd(bottom_dst + i, bottom_vals);
        _mm256_storeu_pd(top_dst + i, top_vals);
    }

    for (i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
#elif defined(BC_HAS_SSE2)
    /* SSE2: Process 2 doubles per iteration */
    size_t simd_end = nx & ~(size_t)1;

    for (i = 0; i < simd_end; i += 2) {
        __m128d bottom_vals = _mm_loadu_pd(bottom_src + i);
        __m128d top_vals = _mm_loadu_pd(top_src + i);
        _mm_storeu_pd(bottom_dst + i, bottom_vals);
        _mm_storeu_pd(top_dst + i, top_vals);
    }

    for (i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
#endif
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with SIMD optimization.
 *
 * Sets boundary values to specified fixed values:
 *   - Left boundary (i=0): field[0,j] = values->left
 *   - Right boundary (i=nx-1): field[nx-1,j] = values->right
 *   - Bottom boundary (j=0): field[i,0] = values->bottom
 *   - Top boundary (j=ny-1): field[i,ny-1] = values->top
 *
 * SIMD is used for contiguous top/bottom boundaries using broadcast.
 */
void bc_apply_dirichlet_simd_impl(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values) {
    size_t j, i;

    /* Left and right boundaries - strided access, use scalar */
    for (j = 0; j < ny; j++) {
        field[j * nx] = values->left;
        field[j * nx + (nx - 1)] = values->right;
    }

    /* Top and bottom boundaries - contiguous memory, use SIMD broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

#if defined(BC_HAS_AVX2)
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
#elif defined(BC_HAS_SSE2)
    /* SSE2: Broadcast value to 2 doubles and store */
    __m128d bottom_broadcast = _mm_set1_pd(values->bottom);
    __m128d top_broadcast = _mm_set1_pd(values->top);
    size_t simd_end = nx & ~(size_t)1;  /* Round down to multiple of 2 */

    for (i = 0; i < simd_end; i += 2) {
        _mm_storeu_pd(bottom_row + i, bottom_broadcast);
        _mm_storeu_pd(top_row + i, top_broadcast);
    }

    /* Handle remaining element */
    for (i = simd_end; i < nx; i++) {
        bottom_row[i] = values->bottom;
        top_row[i] = values->top;
    }
#endif
}

#endif /* BC_HAS_SIMD */
