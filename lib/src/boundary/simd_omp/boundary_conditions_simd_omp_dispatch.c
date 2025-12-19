/**
 * Boundary Conditions - SIMD + OpenMP Dispatcher with Runtime Detection
 *
 * This file provides the unified bc_impl_simd_omp interface by selecting
 * the correct architecture-specific implementation at RUNTIME:
 * - AVX2 + OpenMP on x86-64 (detected via CPUID)
 * - NEON + OpenMP on ARM64 (always available on ARM64)
 *
 * The actual implementations remain in separate files:
 * - avx2/boundary_conditions_avx2_omp.c
 * - neon/boundary_conditions_neon_omp.c
 *
 * Runtime detection is provided by the cpu_features module.
 *
 * IMPORTANT: These dispatcher functions will abort if called when no SIMD
 * backend is available. Callers MUST check bc_simd_omp_backend_available()
 * before using this backend. This is a programming error, not a runtime error.
 */

#include "../boundary_conditions_internal.h"
#include "cfd/core/cpu_features.h"
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

/* ============================================================================
 * Runtime Dispatching Functions
 *
 * These functions check the detected architecture and delegate to the
 * appropriate backend (AVX2 or NEON) at runtime.
 * ============================================================================ */

static void bc_simd_omp_neumann(double* field, size_t nx, size_t ny) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2 && bc_impl_avx2_omp.apply_neumann != NULL) {
        bc_impl_avx2_omp.apply_neumann(field, nx, ny);
        return;
    }
    if (arch == CFD_SIMD_NEON && bc_impl_neon_omp.apply_neumann != NULL) {
        bc_impl_neon_omp.apply_neumann(field, nx, ny);
        return;
    }

    /* No SIMD backend available - this is a programming error.
     * Caller should have checked bc_simd_omp_backend_available() first. */
    fprintf(stderr, "FATAL: SIMD+OMP neumann called but no SIMD backend available "
                    "(detected arch: %s). Check bc_simd_omp_backend_available() "
                    "before using BC_BACKEND_SIMD_OMP.\n", cfd_get_simd_name());
    assert(0 && "SIMD+OMP backend called without available implementation");
    abort();
}

static void bc_simd_omp_periodic(double* field, size_t nx, size_t ny) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2 && bc_impl_avx2_omp.apply_periodic != NULL) {
        bc_impl_avx2_omp.apply_periodic(field, nx, ny);
        return;
    }
    if (arch == CFD_SIMD_NEON && bc_impl_neon_omp.apply_periodic != NULL) {
        bc_impl_neon_omp.apply_periodic(field, nx, ny);
        return;
    }

    /* No SIMD backend available - this is a programming error. */
    fprintf(stderr, "FATAL: SIMD+OMP periodic called but no SIMD backend available "
                    "(detected arch: %s). Check bc_simd_omp_backend_available() "
                    "before using BC_BACKEND_SIMD_OMP.\n", cfd_get_simd_name());
    assert(0 && "SIMD+OMP backend called without available implementation");
    abort();
}

static void bc_simd_omp_dirichlet(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2 && bc_impl_avx2_omp.apply_dirichlet != NULL) {
        bc_impl_avx2_omp.apply_dirichlet(field, nx, ny, values);
        return;
    }
    if (arch == CFD_SIMD_NEON && bc_impl_neon_omp.apply_dirichlet != NULL) {
        bc_impl_neon_omp.apply_dirichlet(field, nx, ny, values);
        return;
    }

    /* No SIMD backend available - this is a programming error. */
    fprintf(stderr, "FATAL: SIMD+OMP dirichlet called but no SIMD backend available "
                    "(detected arch: %s). Check bc_simd_omp_backend_available() "
                    "before using BC_BACKEND_SIMD_OMP.\n", cfd_get_simd_name());
    assert(0 && "SIMD+OMP backend called without available implementation");
    abort();
}

/* ============================================================================
 * Check if SIMD+OMP backend is available at runtime
 * ============================================================================ */

/**
 * Check if any SIMD implementation is available.
 * Called during initialization to determine if bc_impl_simd_omp should be used.
 */
static bool simd_omp_available(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2) {
        return bc_impl_avx2_omp.apply_neumann != NULL;
    } else if (arch == CFD_SIMD_NEON) {
        return bc_impl_neon_omp.apply_neumann != NULL;
    }
    return false;
}

/* ============================================================================
 * Unified SIMD + OMP Interface
 *
 * bc_impl_simd_omp provides runtime-dispatching functions.
 * The functions check availability internally.
 * ============================================================================ */

const bc_backend_impl_t bc_impl_simd_omp = {
    .apply_neumann = bc_simd_omp_neumann,
    .apply_periodic = bc_simd_omp_periodic,
    .apply_dirichlet = bc_simd_omp_dirichlet
};

/**
 * Query function for external code to check if SIMD+OMP is actually available.
 * This is used by the backend availability check since bc_impl_simd_omp
 * always has non-NULL function pointers (they do runtime dispatch).
 */
bool bc_simd_omp_backend_available(void) {
    return simd_omp_available();
}

/**
 * Get the name of the detected SIMD architecture.
 * Returns "avx2", "neon", or "none" based on runtime detection.
 */
const char* bc_simd_omp_get_arch_name(void) {
    return cfd_get_simd_name();
}
