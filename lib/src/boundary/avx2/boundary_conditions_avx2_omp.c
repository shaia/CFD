/**
 * Boundary Conditions - AVX2 + OpenMP Implementation
 *
 * Combined AVX2 SIMD and OpenMP parallelized boundary condition implementations.
 * Uses OpenMP for thread-level parallelism across rows.
 * Uses AVX2 SIMD for instruction-level parallelism on contiguous memory.
 *
 * Requires both AVX2 support and OpenMP:
 * - AVX2 available on all x86-64 CPUs from 2013+ (Intel Haswell, AMD Excavator)
 * - OpenMP for multi-threading
 *
 * Falls back to NULL pointers when either AVX2 or OpenMP is not available.
 */

#include "../boundary_conditions_internal.h"

/* AVX2 + OpenMP detection */
#if defined(__AVX2__) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_AVX2_OMP 1
#include <immintrin.h>
#include <omp.h>
#endif

#if defined(BC_HAS_AVX2_OMP)

/**
 * Apply Neumann boundary conditions (zero gradient) with AVX2 + OpenMP.
 */
static void bc_apply_neumann_avx2_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;
    int inx = (int)nx;
    int iny = (int)ny;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < iny; j++) {
        field[(j * inx) + 0] = field[(j * inx) + 1];
        field[(j * inx) + inx - 1] = field[(j * inx) + inx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    /* AVX2: Process 4 doubles per iteration */
    int simd_end = inx & ~3;  /* Round down to multiple of 4 */

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 4) {
        __m256d bottom_vals = _mm256_loadu_pd(bottom_src + i);
        __m256d top_vals = _mm256_loadu_pd(top_src + i);
        _mm256_storeu_pd(bottom_dst + i, bottom_vals);
        _mm256_storeu_pd(top_dst + i, top_vals);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with AVX2 + OpenMP.
 */
static void bc_apply_periodic_avx2_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;
    int inx = (int)nx;
    int iny = (int)ny;

    /* Left and right boundaries (periodic in x) - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < iny; j++) {
        field[(j * inx) + 0] = field[(j * inx) + inx - 2];
        field[(j * inx) + inx - 1] = field[(j * inx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

    /* AVX2: Process 4 doubles per iteration */
    int simd_end = inx & ~3;

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 4) {
        __m256d bottom_vals = _mm256_loadu_pd(bottom_src + i);
        __m256d top_vals = _mm256_loadu_pd(top_src + i);
        _mm256_storeu_pd(bottom_dst + i, bottom_vals);
        _mm256_storeu_pd(top_dst + i, top_vals);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with AVX2 + OpenMP.
 */
static void bc_apply_dirichlet_avx2_omp_impl(double* field, size_t nx, size_t ny,
                                              const bc_dirichlet_values_t* values) {
    int j, i;
    int inx = (int)nx;
    int iny = (int)ny;

    /* Store values in locals for OpenMP */
    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < iny; j++) {
        field[j * inx] = val_left;
        field[j * inx + (inx - 1)] = val_right;
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    /* AVX2: Broadcast value to 4 doubles and store */
    __m256d bottom_broadcast = _mm256_set1_pd(val_bottom);
    __m256d top_broadcast = _mm256_set1_pd(val_top);
    int simd_end = inx & ~3;  /* Round down to multiple of 4 */

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 4) {
        _mm256_storeu_pd(bottom_row + i, bottom_broadcast);
        _mm256_storeu_pd(top_row + i, top_broadcast);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
    }
}

/* AVX2 + OpenMP backend implementation table */
const bc_backend_impl_t bc_impl_avx2_omp = {
    .apply_neumann = bc_apply_neumann_avx2_omp_impl,
    .apply_periodic = bc_apply_periodic_avx2_omp_impl,
    .apply_dirichlet = bc_apply_dirichlet_avx2_omp_impl
};

#else /* !BC_HAS_AVX2_OMP */

/* AVX2 + OpenMP not available - provide empty table */
const bc_backend_impl_t bc_impl_avx2_omp = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL
};

#endif /* BC_HAS_AVX2_OMP */
