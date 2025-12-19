/**
 * CPU Feature Detection
 *
 * Runtime detection of CPU SIMD capabilities.
 * Provides a portable interface to detect AVX2, NEON, and other features
 * across different platforms and compilers.
 */

#ifndef CFD_CPU_FEATURES_H
#define CFD_CPU_FEATURES_H

#include "cfd/cfd_export.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * SIMD architecture types detected at runtime.
 */
typedef enum {
    CFD_SIMD_NONE = 0,   /**< No SIMD or unknown architecture */
    CFD_SIMD_AVX2 = 1,   /**< x86-64 AVX2 (256-bit, 4 doubles) */
    CFD_SIMD_NEON = 2    /**< ARM NEON (128-bit, 2 doubles) */
} cfd_simd_arch_t;

/**
 * Detect the best available SIMD architecture at runtime.
 *
 * On x86/x64: Uses CPUID to check for AVX2 support.
 * On ARM64: NEON is always available.
 * On other platforms: Returns CFD_SIMD_NONE.
 *
 * The result is cached after the first call for performance.
 *
 * @return The detected SIMD architecture type
 */
CFD_LIBRARY_EXPORT cfd_simd_arch_t cfd_detect_simd_arch(void);

/**
 * Check if AVX2 SIMD is available on the current CPU.
 *
 * @return true if AVX2 is supported, false otherwise
 */
CFD_LIBRARY_EXPORT bool cfd_has_avx2(void);

/**
 * Check if ARM NEON SIMD is available on the current CPU.
 *
 * @return true if NEON is supported, false otherwise
 */
CFD_LIBRARY_EXPORT bool cfd_has_neon(void);

/**
 * Check if any SIMD architecture is available.
 *
 * @return true if AVX2 or NEON is available, false otherwise
 */
CFD_LIBRARY_EXPORT bool cfd_has_simd(void);

/**
 * Get the name of the detected SIMD architecture.
 *
 * @return "avx2", "neon", or "none"
 */
CFD_LIBRARY_EXPORT const char* cfd_get_simd_name(void);

#ifdef __cplusplus
}
#endif

#endif /* CFD_CPU_FEATURES_H */
