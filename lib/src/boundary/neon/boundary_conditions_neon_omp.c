/**
 * Boundary Conditions - ARM NEON + OpenMP Implementation
 *
 * Combined ARM NEON SIMD and OpenMP parallelized boundary condition implementations.
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

/* ARM NEON + OpenMP detection */
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_NEON_OMP 1
#include <arm_neon.h>
#include <omp.h>
#endif

#if defined(BC_HAS_NEON_OMP)

/**
 * Apply Neumann boundary conditions (zero gradient) with NEON + OpenMP.
 */
static void bc_apply_neumann_neon_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;
    int inx = (int)nx;
    int iny = (int)ny;

    /* Left and right boundaries - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < iny; j++) {
        field[(j * inx) + 0] = field[(j * inx) + 1];
        field[(j * inx) + inx - 1] = field[(j * inx) + inx - 2];
    }

    /* Top and bottom boundaries - contiguous memory, use NEON */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    /* NEON: Process 2 doubles per iteration (float64x2_t) */
    int simd_end = inx & ~1;  /* Round down to multiple of 2 */

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 2) {
        float64x2_t bottom_vals = vld1q_f64(bottom_src + i);
        float64x2_t top_vals = vld1q_f64(top_src + i);
        vst1q_f64(bottom_dst + i, bottom_vals);
        vst1q_f64(top_dst + i, top_vals);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply periodic boundary conditions with NEON + OpenMP.
 */
static void bc_apply_periodic_neon_omp_impl(double* field, size_t nx, size_t ny) {
    int j, i;
    int inx = (int)nx;
    int iny = (int)ny;

    /* Left and right boundaries (periodic in x) - parallelize over rows */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < iny; j++) {
        field[(j * inx) + 0] = field[(j * inx) + inx - 2];
        field[(j * inx) + inx - 1] = field[(j * inx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) - contiguous memory, use NEON */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);  /* Copy from top interior */
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;  /* Copy from bottom interior */

    /* NEON: Process 2 doubles per iteration */
    int simd_end = inx & ~1;

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 2) {
        float64x2_t bottom_vals = vld1q_f64(bottom_src + i);
        float64x2_t top_vals = vld1q_f64(top_src + i);
        vst1q_f64(bottom_dst + i, bottom_vals);
        vst1q_f64(top_dst + i, top_vals);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

/**
 * Apply Dirichlet (fixed value) boundary conditions with NEON + OpenMP.
 */
static void bc_apply_dirichlet_neon_omp_impl(double* field, size_t nx, size_t ny,
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

    /* Top and bottom boundaries - contiguous memory, use NEON broadcast */
    double* bottom_row = field;
    double* top_row = field + ((ny - 1) * nx);

    /* NEON: Duplicate value to 2 doubles and store */
    float64x2_t bottom_broadcast = vdupq_n_f64(val_bottom);
    float64x2_t top_broadcast = vdupq_n_f64(val_top);
    int simd_end = inx & ~1;  /* Round down to multiple of 2 */

    /* Parallelize the SIMD loop across columns */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < simd_end; i += 2) {
        vst1q_f64(bottom_row + i, bottom_broadcast);
        vst1q_f64(top_row + i, top_broadcast);
    }

    /* Handle remaining elements (sequential, minimal work) */
    for (i = simd_end; i < inx; i++) {
        bottom_row[i] = val_bottom;
        top_row[i] = val_top;
    }
}

/* NEON + OpenMP backend implementation table */
const bc_backend_impl_t bc_impl_neon_omp = {
    .apply_neumann = bc_apply_neumann_neon_omp_impl,
    .apply_periodic = bc_apply_periodic_neon_omp_impl,
    .apply_dirichlet = bc_apply_dirichlet_neon_omp_impl
};

#else /* !BC_HAS_NEON_OMP */

/* NEON + OpenMP not available - provide empty table */
const bc_backend_impl_t bc_impl_neon_omp = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL
};

#endif /* BC_HAS_NEON_OMP */
