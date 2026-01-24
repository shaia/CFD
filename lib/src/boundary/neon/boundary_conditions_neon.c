/**
 * Boundary Conditions - ARM NEON Implementation
 *
 * ARM NEON SIMD boundary condition implementations with OpenMP parallelization.
 * Uses OpenMP for thread-level parallelism across rows.
 * Uses NEON SIMD for instruction-level parallelism on contiguous memory.
 *
 * Requires both NEON support and OpenMP:
 * - NEON available on all ARMv7+ and ARM64 processors
 * - OpenMP for multi-threading
 *
 * Falls back to NULL pointers when either NEON or OpenMP is not available.
 */

#include "../boundary_conditions_internal.h"

/* ARM NEON + OpenMP detection
 * NEON is mandatory in ARMv8-A (AArch64/ARM64), so we enable NEON code
 * whenever building for ARM64, even if __ARM_NEON isn't explicitly defined.
 * For ARMv7 (32-bit ARM), we require the __ARM_NEON macro.
 */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#include <limits.h>
#include <assert.h>
#endif

#if defined(BC_HAS_NEON)

/**
 * Minimum row width to use OpenMP for horizontal boundary loops.
 * For smaller grids, the thread synchronization overhead exceeds the benefit.
 * With NEON (2 doubles/iteration), 256 elements = 128 iterations.
 * On a typical 8-thread system, this gives 16 iterations per thread.
 */
#define BC_NEON_THRESHOLD 256

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
 * Apply Neumann boundary conditions (zero gradient) with NEON + OpenMP.
 */
static void bc_apply_neumann_neon_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + 1];
        field[row + nx - 1] = field[row + nx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use NEON */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    /* NEON: Process 2 doubles per iteration (float64x2_t) */
    size_t simd_end = nx & ~(size_t)1;  /* Round down to multiple of 2 */

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_NEON_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 2) {
            vst1q_f64(bottom_dst + i, vld1q_f64(bottom_src + i));
            vst1q_f64(top_dst + i, vld1q_f64(top_src + i));
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 2) {
            vst1q_f64(bottom_dst + i, vld1q_f64(bottom_src + i));
            vst1q_f64(top_dst + i, vld1q_f64(top_src + i));
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with NEON + OpenMP.
 */
static void bc_apply_periodic_neon_impl(double* field, size_t nx, size_t ny) {
    int j, i;

    /* Left and right boundaries (periodic in x) - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < size_to_int(ny); j++) {
        size_t row = (size_t)j * nx;
        field[row] = field[row + nx - 2];
        field[row + nx - 1] = field[row + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use NEON */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

    /* NEON: Process 2 doubles per iteration */
    size_t simd_end = nx & ~(size_t)1;

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_NEON_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 2) {
            vst1q_f64(bottom_dst + i, vld1q_f64(bottom_src + i));
            vst1q_f64(top_dst + i, vld1q_f64(top_src + i));
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 2) {
            vst1q_f64(bottom_dst + i, vld1q_f64(bottom_src + i));
            vst1q_f64(top_dst + i, vld1q_f64(top_src + i));
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with NEON + OpenMP.
 */
static void bc_apply_dirichlet_neon_impl(double* field, size_t nx, size_t ny,
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

    /* Top and bottom boundaries - contiguous memory, use NEON broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    /* NEON: Duplicate value to 2 doubles and store */
    float64x2_t bottom_broadcast = vdupq_n_f64(val_bottom);
    float64x2_t top_broadcast = vdupq_n_f64(val_top);
    size_t simd_end = nx & ~(size_t)1;  /* Round down to multiple of 2 */

    /* Only parallelize if row is wide enough to justify overhead */
    if (nx >= BC_NEON_THRESHOLD) {
        #pragma omp parallel for schedule(static)
        for (i = 0; i < size_to_int(simd_end); i += 2) {
            vst1q_f64(bottom_row + i, bottom_broadcast);
            vst1q_f64(top_row + i, top_broadcast);
        }
    } else {
        /* Sequential SIMD for small grids */
        for (size_t i = 0; i < simd_end; i += 2) {
            vst1q_f64(bottom_row + i, bottom_broadcast);
            vst1q_f64(top_row + i, top_broadcast);
        }
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (size_t i = simd_end; i < nx; i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
    }
}

/* NEON backend implementation table
 * Note: bc_apply_inlet_neon_impl is defined in boundary_conditions_inlet_neon.c
 * Note: bc_apply_outlet_neon_impl is defined in boundary_conditions_outlet_neon.c */
const bc_backend_impl_t bc_impl_neon = {
    .apply_neumann = bc_apply_neumann_neon_impl,
    .apply_periodic = bc_apply_periodic_neon_impl,
    .apply_dirichlet = bc_apply_dirichlet_neon_impl,
    .apply_inlet = bc_apply_inlet_neon_impl,
    .apply_outlet = bc_apply_outlet_neon_impl,
    .apply_symmetry = NULL  /* Falls back to scalar */
};

#else /* !BC_HAS_NEON */

/* NEON not available - provide empty table */
const bc_backend_impl_t bc_impl_neon = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL,
    .apply_outlet = NULL,
    .apply_symmetry = NULL
};

#endif /* BC_HAS_NEON */
