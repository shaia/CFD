/**
 * @file ns_simd_backend.c
 * @brief SIMD Navier-Stokes backend availability (compile-time + runtime).
 *
 * This is the single place that decides whether a SIMD NS solver can run. The
 * registry's backend-availability query and every SIMD solver's init guard go
 * through it, so they cannot disagree.
 *
 * The compile-time guard lives here, in the backend file, rather than in the
 * registry dispatcher.
 */

#include "../ns_simd_backend_internal.h"

#include "cfd/core/cpu_features.h"

/* CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON, consistently across
 * GCC, Clang and MSVC. There are no NEON NS kernels, so AVX2 is the only
 * compiled SIMD implementation. */
#if defined(CFD_HAS_AVX2)
#define NS_HAS_SIMD_IMPL 1
#else
#define NS_HAS_SIMD_IMPL 0
#endif

bool ns_simd_backend_available(void) {
#if NS_HAS_SIMD_IMPL
    return cfd_detect_simd_arch() == CFD_SIMD_AVX2;
#else
    return false;
#endif
}

cfd_status_t ns_check_simd_backend(void) {
    if (ns_simd_backend_available()) {
        return CFD_SUCCESS;
    }
#if NS_HAS_SIMD_IMPL
    cfd_set_error(CFD_ERROR_UNSUPPORTED,
                  "SIMD Navier-Stokes backend unavailable: CPU does not support AVX2");
#else
    cfd_set_error(CFD_ERROR_UNSUPPORTED,
                  "SIMD Navier-Stokes backend unavailable: built without AVX2 "
                  "(configure with -DCFD_ENABLE_AVX2=ON)");
#endif
    return CFD_ERROR_UNSUPPORTED;
}
