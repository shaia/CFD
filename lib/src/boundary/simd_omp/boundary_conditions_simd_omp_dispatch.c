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
 * Error Handling:
 * If called when no SIMD backend is available (programming error), these
 * dispatcher functions will:
 * 1. Call the user-configurable error handler (or print to stderr if none set)
 * 2. Assert in debug builds
 * 3. Fall back to scalar implementation to avoid leaving fields in invalid state
 *
 * Callers SHOULD check bc_simd_omp_backend_available() before using this backend.
 */

#include "../boundary_conditions_internal.h"
#include "cfd/core/cpu_features.h"
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

/* ============================================================================
 * Helper: Report error and fall back to scalar
 * ============================================================================ */

static void report_no_simd_error(const char* function) {
    char message[128];
    snprintf(message, sizeof(message),
             "SIMD+OMP backend called but no SIMD available (detected: %s). "
             "Falling back to scalar.",
             cfd_get_simd_name());
    bc_report_error(BC_ERROR_NO_SIMD_BACKEND, function, message);
    assert(0 && "SIMD+OMP backend called without available implementation");
}

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

    /* No SIMD backend available - report error and fall back to scalar */
    report_no_simd_error("bc_simd_omp_neumann");
    bc_apply_neumann_scalar_impl(field, nx, ny);
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

    /* No SIMD backend available - report error and fall back to scalar */
    report_no_simd_error("bc_simd_omp_periodic");
    bc_apply_periodic_scalar_impl(field, nx, ny);
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

    /* No SIMD backend available - report error and fall back to scalar */
    report_no_simd_error("bc_simd_omp_dirichlet");
    bc_apply_dirichlet_scalar_impl(field, nx, ny, values);
}

/* ============================================================================
 * Check if SIMD+OMP backend is available at runtime
 * ============================================================================ */

/**
 * Check if a backend implementation table is fully populated.
 * All three function pointers must be non-NULL for the backend to be usable.
 */
static bool backend_impl_complete(const bc_backend_impl_t* impl) {
    return impl->apply_neumann != NULL &&
           impl->apply_periodic != NULL &&
           impl->apply_dirichlet != NULL;
}

/**
 * Check if any SIMD implementation is available.
 * Called during initialization to determine if bc_impl_simd_omp should be used.
 *
 * Verifies all three function pointers (neumann, periodic, dirichlet) are present.
 * This ensures the backend is fully functional, not just partially implemented.
 */
static bool simd_omp_available(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2) {
        return backend_impl_complete(&bc_impl_avx2_omp);
    } else if (arch == CFD_SIMD_NEON) {
        return backend_impl_complete(&bc_impl_neon_omp);
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
