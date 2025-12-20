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
 * Compile-Time vs Runtime Detection:
 * ----------------------------------
 * The availability check (simd_omp_available) uses BOTH:
 * 1. Runtime CPU detection: cfd_detect_simd_arch() checks if CPU supports AVX2/NEON
 * 2. Compile-time availability: Checks if function pointers are non-NULL
 *
 * This two-phase check handles the case where:
 * - CPU supports AVX2, but code was compiled without -mavx2 flag
 * - In this case, bc_impl_avx2_omp has NULL pointers, so simd_omp_available()
 *   returns false even though runtime detection reports AVX2 support.
 *
 * This design ensures safe operation: SIMD backend is only used when BOTH
 * the CPU supports it AND the code was compiled with SIMD instructions.
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
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

/* ============================================================================
 * Helper: Get SIMD backend based on runtime detection
 *
 * Returns the appropriate backend implementation table (AVX2 or NEON) based
 * on runtime CPU detection. Returns NULL if no SIMD backend is available.
 *
 * The result is cached after first call since SIMD architecture doesn't change
 * at runtime. This avoids redundant calls to cfd_detect_simd_arch() on every
 * boundary condition operation.
 * ============================================================================ */

/* Cache for the SIMD backend pointer. NULL means not yet initialized,
 * a valid pointer means the backend is available, and a special sentinel
 * (void*)-1 means no SIMD backend is available. */
static const bc_backend_impl_t* g_simd_backend_cache = NULL;
static int g_simd_backend_initialized = 0;

static const bc_backend_impl_t* get_simd_backend(void) {
    /* Fast path: return cached result */
    if (g_simd_backend_initialized) {
        /* Check for "no backend" sentinel */
        if (g_simd_backend_cache == (const bc_backend_impl_t*)(intptr_t)-1) {
            return NULL;
        }
        return g_simd_backend_cache;
    }

    /* Slow path: detect and cache */
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    const bc_backend_impl_t* result = NULL;

    if (arch == CFD_SIMD_AVX2 && bc_impl_avx2_omp.apply_neumann != NULL) {
        result = &bc_impl_avx2_omp;
    } else if (arch == CFD_SIMD_NEON && bc_impl_neon_omp.apply_neumann != NULL) {
        result = &bc_impl_neon_omp;
    }

    /* Cache result (use sentinel for NULL to distinguish from uninitialized) */
    g_simd_backend_cache = result ? result : (const bc_backend_impl_t*)(intptr_t)-1;
    g_simd_backend_initialized = 1;

    return result;
}

/**
 * Report error when SIMD backend is unavailable.
 * Called as a programming error fallback - callers should check availability first.
 */
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
 * These functions use get_simd_backend() for unified dispatch logic.
 * ============================================================================ */

static void bc_simd_omp_neumann(double* field, size_t nx, size_t ny) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_neumann(field, nx, ny);
        return;
    }
    report_no_simd_error("bc_simd_omp_neumann");
    bc_apply_neumann_scalar_impl(field, nx, ny);
}

static void bc_simd_omp_periodic(double* field, size_t nx, size_t ny) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_periodic(field, nx, ny);
        return;
    }
    report_no_simd_error("bc_simd_omp_periodic");
    bc_apply_periodic_scalar_impl(field, nx, ny);
}

static void bc_simd_omp_dirichlet(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_dirichlet(field, nx, ny, values);
        return;
    }
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
