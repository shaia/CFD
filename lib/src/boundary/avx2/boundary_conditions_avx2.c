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

/* AVX2 + OpenMP detection
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#include <assert.h>
#endif

#if defined(BC_HAS_AVX2)

/**
 * Minimum row width to use OpenMP for horizontal boundary loops.
 * For smaller grids, the thread synchronization overhead exceeds the benefit.
 * With AVX2 (4 doubles/iteration), 256 elements = 64 iterations.
 * On a typical 8-thread system, this gives 8 iterations per thread.
 */
#define BC_AVX2_THRESHOLD 256

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 *
 * OpenMP requires signed integer loop counters. This function safely converts
 * size_t to int, clamping to INT_MAX if the value would overflow.
 * In practice, grids exceeding INT_MAX (~2 billion) would require ~16GB
 * per row of doubles, making this limit unlikely to be hit.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/**
 * Apply Neumann boundary conditions (zero gradient) with AVX2 + OpenMP.
 */
static void bc_apply_neumann_avx2_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + 1];
        field[row + nx - 1] = field[row + nx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    /* AVX2: Process 4 doubles per iteration */
    size_t simd_end = nx & ~(size_t)3;  /* Round down to multiple of 4 */

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_AVX2_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 4) {
            _mm256_storeu_pd(bottom_dst + i, _mm256_loadu_pd(bottom_src + i));
            _mm256_storeu_pd(top_dst + i, _mm256_loadu_pd(top_src + i));
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 4) {
            _mm256_storeu_pd(bottom_dst + i, _mm256_loadu_pd(bottom_src + i));
            _mm256_storeu_pd(top_dst + i, _mm256_loadu_pd(top_src + i));
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with AVX2 + OpenMP.
 */
static void bc_apply_periodic_avx2_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries (periodic in x) - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + nx - 2];
        field[row + nx - 1] = field[row + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use AVX2 */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

    /* AVX2: Process 4 doubles per iteration */
    size_t simd_end = nx & ~(size_t)3;

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_AVX2_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 4) {
            _mm256_storeu_pd(bottom_dst + i, _mm256_loadu_pd(bottom_src + i));
            _mm256_storeu_pd(top_dst + i, _mm256_loadu_pd(top_src + i));
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 4) {
            _mm256_storeu_pd(bottom_dst + i, _mm256_loadu_pd(bottom_src + i));
            _mm256_storeu_pd(top_dst + i, _mm256_loadu_pd(top_src + i));
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with AVX2 + OpenMP.
 */
static void bc_apply_dirichlet_avx2_impl(double* field, size_t nx, size_t ny,
                                          const bc_dirichlet_values_t* values) {
    int j, i;

    /* Store values in locals for OpenMP */
    double val_left = values->left;
    double val_right = values->right;
    double val_bottom = values->bottom;
    double val_top = values->top;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = val_left;
        field[row + nx - 1] = val_right;
    }

    /* Top and bottom boundaries - contiguous memory, use AVX2 broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    /* AVX2: Broadcast value to 4 doubles and store */
    __m256d bottom_broadcast = _mm256_set1_pd(val_bottom);
    __m256d top_broadcast = _mm256_set1_pd(val_top);
    size_t simd_end = nx & ~(size_t)3;  /* Round down to multiple of 4 */

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_AVX2_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 4) {
            _mm256_storeu_pd(bottom_row + i, bottom_broadcast);
            _mm256_storeu_pd(top_row + i, top_broadcast);
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 4) {
            _mm256_storeu_pd(bottom_row + i, bottom_broadcast);
            _mm256_storeu_pd(top_row + i, top_broadcast);
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
    }
}

/* AVX2 + OpenMP backend implementation table
 * Note: bc_apply_inlet_avx2_impl is defined in boundary_conditions_inlet_avx2.c
 * Note: bc_apply_outlet_avx2_impl is defined in boundary_conditions_outlet_avx2.c */
const bc_backend_impl_t bc_impl_avx2 = {
    .apply_neumann = bc_apply_neumann_avx2_impl,
    .apply_periodic = bc_apply_periodic_avx2_impl,
    .apply_dirichlet = bc_apply_dirichlet_avx2_impl,
    .apply_inlet = bc_apply_inlet_avx2_impl,
    .apply_outlet = bc_apply_outlet_avx2_impl,
    .apply_symmetry = NULL  /* Falls back to scalar */
};

#else /* !BC_HAS_AVX2 */

/* AVX2 + OpenMP not available - provide empty table */
const bc_backend_impl_t bc_impl_avx2 = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL,
    .apply_outlet = NULL,
    .apply_symmetry = NULL
};

#endif /* BC_HAS_AVX2 */
