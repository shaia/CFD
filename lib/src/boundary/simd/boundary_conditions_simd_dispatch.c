/**
 * Boundary Conditions - SIMD Dispatcher with Runtime Detection
 *
 * This file provides the unified bc_impl_simd interface by selecting
 * the correct architecture-specific implementation at RUNTIME:
 * - AVX2 on x86-64 (detected via CPUID)
 * - NEON on ARM64 (always available on ARM64)
 *
 * The actual implementations remain in separate files:
 * - avx2/boundary_conditions_avx2.c
 * - neon/boundary_conditions_neon.c
 *
 * Compile-Time vs Runtime Detection:
 * ----------------------------------
 * The availability check (simd_available) uses BOTH:
 * 1. Runtime CPU detection: cfd_detect_simd_arch() checks if CPU supports AVX2/NEON
 * 2. Compile-time availability: Checks if function pointers are non-NULL
 *
 * This two-phase check handles the case where:
 * - CPU supports AVX2, but code was compiled without -mavx2 flag
 * - In this case, bc_impl_avx2 has NULL pointers, so simd_available()
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
 * Callers SHOULD check bc_simd_backend_available() before using this backend.
 */

#include "../boundary_conditions_internal.h"
#include "cfd/core/cpu_features.h"
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>

/* Platform-specific atomic operations for thread-safe caching */
#ifdef _MSC_VER
#include <intrin.h>
#define ATOMIC_LOAD(ptr) _InterlockedCompareExchange64((volatile long long*)(ptr), 0, 0)
#define ATOMIC_CAS(ptr, expected, desired) \
    (_InterlockedCompareExchange64((volatile long long*)(ptr), (long long)(desired), (long long)(*(expected))) == (long long)(*(expected)))
#define COMPILER_BARRIER() _ReadWriteBarrier()
#else
/* GCC/Clang built-in atomics */
#define ATOMIC_LOAD(ptr) __atomic_load_n((ptr), __ATOMIC_ACQUIRE)
#define ATOMIC_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n((ptr), (expected), (desired), 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)
#define COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#endif

/* ============================================================================
 * Helper: Get SIMD backend based on runtime detection
 *
 * Returns the appropriate backend implementation table (AVX2 or NEON) based
 * on runtime CPU detection. Returns NULL if no SIMD backend is available.
 *
 * The result is cached after first call since SIMD architecture doesn't change
 * at runtime. This avoids redundant calls to cfd_detect_simd_arch() on every
 * boundary condition operation.
 *
 * Thread Safety:
 * Uses atomic compare-and-swap to ensure proper synchronization. Only one
 * thread will successfully initialize the cache; others will use the
 * already-cached result. Memory barriers ensure visibility across threads.
 * ============================================================================ */

/* Cache for the SIMD backend pointer.
 * Values: 0 = not initialized, 1 = no backend, 2+ = valid backend pointer + 1
 * Using intptr_t allows atomic operations and encodes state in a single variable. */
static volatile intptr_t g_simd_backend_cache = 0;

/* Sentinel values for cache state */
#define CACHE_UNINITIALIZED 0
#define CACHE_NO_BACKEND    1

static const bc_backend_impl_t* get_simd_backend(void) {
    /* Fast path: check if already initialized */
    intptr_t cached = (intptr_t)ATOMIC_LOAD(&g_simd_backend_cache);

    if (cached != CACHE_UNINITIALIZED) {
        if (cached == CACHE_NO_BACKEND) {
            return NULL;
        }
        /* Decode pointer: subtract 1 and cast back */
        return (const bc_backend_impl_t*)(cached - 1);
    }

    /* Slow path: detect SIMD backend */
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    const bc_backend_impl_t* result = NULL;

    if (arch == CFD_SIMD_AVX2 && bc_impl_avx2.apply_neumann != NULL) {
        result = &bc_impl_avx2;
    } else if (arch == CFD_SIMD_NEON && bc_impl_neon.apply_neumann != NULL) {
        result = &bc_impl_neon;
    }

    /* Encode result: NULL becomes CACHE_NO_BACKEND, valid pointer becomes ptr+1 */
    intptr_t new_value = result ? ((intptr_t)result + 1) : CACHE_NO_BACKEND;

    /* Try to set cache atomically. If another thread beat us, use their result. */
    intptr_t expected = CACHE_UNINITIALIZED;
    if (!ATOMIC_CAS(&g_simd_backend_cache, &expected, new_value)) {
        /* Another thread initialized first - use their cached value */
        if (expected == CACHE_NO_BACKEND) {
            return NULL;
        }
        return (const bc_backend_impl_t*)(expected - 1);
    }

    return result;
}

/**
 * Report error when SIMD backend is unavailable.
 * Called as a programming error fallback - callers should check availability first.
 */
static void report_no_simd_error(const char* function) {
    char message[128];
    snprintf(message, sizeof(message),
             "SIMD backend called but no SIMD available (detected: %s). "
             "Falling back to scalar.",
             cfd_get_simd_name());
    bc_report_error(BC_ERROR_NO_SIMD_BACKEND, function, message);
    assert(0 && "SIMD backend called without available implementation");
}

/* ============================================================================
 * Runtime Dispatching Functions
 *
 * These functions use get_simd_backend() for unified dispatch logic.
 * ============================================================================ */

static void bc_simd_neumann(double* field, size_t nx, size_t ny) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_neumann(field, nx, ny);
        return;
    }
    report_no_simd_error("bc_simd_neumann");
    bc_apply_neumann_scalar_impl(field, nx, ny);
}

static void bc_simd_periodic(double* field, size_t nx, size_t ny) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_periodic(field, nx, ny);
        return;
    }
    report_no_simd_error("bc_simd_periodic");
    bc_apply_periodic_scalar_impl(field, nx, ny);
}

static void bc_simd_dirichlet(double* field, size_t nx, size_t ny,
                               const bc_dirichlet_values_t* values) {
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL) {
        impl->apply_dirichlet(field, nx, ny, values);
        return;
    }
    report_no_simd_error("bc_simd_dirichlet");
    bc_apply_dirichlet_scalar_impl(field, nx, ny, values);
}

static cfd_status_t bc_simd_inlet(double* u, double* v, size_t nx, size_t ny,
                                   const bc_inlet_config_t* config) {
    /* Inlet BCs operate on 1D boundaries - SIMD provides limited benefit.
     * Delegate to the architecture-specific backend if available, otherwise
     * fall back to scalar implementation. */
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL && impl->apply_inlet != NULL) {
        return impl->apply_inlet(u, v, nx, ny, config);
    }
    /* Fall back to scalar implementation for inlet */
    return bc_apply_inlet_scalar_impl(u, v, nx, ny, config);
}

static cfd_status_t bc_simd_outlet(double* field, size_t nx, size_t ny,
                                    const bc_outlet_config_t* config) {
    /* Outlet BCs operate on 1D boundaries - SIMD provides limited benefit
     * except for top/bottom edges where memory is contiguous.
     * Delegate to the architecture-specific backend if available, otherwise
     * fall back to scalar implementation. */
    const bc_backend_impl_t* impl = get_simd_backend();
    if (impl != NULL && impl->apply_outlet != NULL) {
        return impl->apply_outlet(field, nx, ny, config);
    }
    /* Fall back to scalar implementation for outlet */
    return bc_apply_outlet_scalar_impl(field, nx, ny, config);
}

/* ============================================================================
 * Check if SIMD backend is available at runtime
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
static bool simd_available(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2) {
        return backend_impl_complete(&bc_impl_avx2);
    } else if (arch == CFD_SIMD_NEON) {
        return backend_impl_complete(&bc_impl_neon);
    }
    return false;
}

/* ============================================================================
 * Unified SIMD Interface
 *
 * bc_impl_simd provides runtime-dispatching functions.
 * The functions check availability internally.
 * ============================================================================ */

const bc_backend_impl_t bc_impl_simd = {
    .apply_neumann = bc_simd_neumann,
    .apply_periodic = bc_simd_periodic,
    .apply_dirichlet = bc_simd_dirichlet,
    .apply_inlet = bc_simd_inlet,
    .apply_outlet = bc_simd_outlet
};

/**
 * Query function for external code to check if SIMD is actually available.
 * This is used by the backend availability check since bc_impl_simd
 * always has non-NULL function pointers (they do runtime dispatch).
 */
bool bc_simd_backend_available(void) {
    return simd_available();
}

/**
 * Get the name of the detected SIMD architecture.
 * Returns "avx2", "neon", or "none" based on runtime detection.
 */
const char* bc_simd_get_arch_name(void) {
    return cfd_get_simd_name();
}
