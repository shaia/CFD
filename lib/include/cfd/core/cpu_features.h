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
    CFD_SIMD_AVX2 = 1,   /**< x86-64 AVX2 + FMA3 (256-bit, 4 doubles) */
    CFD_SIMD_NEON = 2    /**< ARM NEON (128-bit, 2 doubles) */
} cfd_simd_arch_t;

/**
 * Detect the best available SIMD architecture at runtime.
 *
 * On x86/x64: Verifies both CPU and OS support for AVX2 with FMA3:
 *   1. CPUID leaf 7 EBX bit 5 for CPU AVX2 support
 *   2. CPUID leaf 1 ECX bit 12 for FMA3, a separate capability that the
 *      AVX2 kernels and the AVX2 build's compiler flags both rely on
 *   3. OSXSAVE enabled (CPUID leaf 1 ECX bit 27)
 *   4. XCR0 bits 1-2 set (OS saves AVX state on context switch)
 *   AVX2 is only reported if all four conditions are met. This prevents
 *   illegal instruction exceptions on CPUs or VMs that expose AVX2 without
 *   FMA, and on systems where the OS hasn't enabled AVX state saving.
 *
 * On ARM64: NEON is always available (mandatory in ARMv8-A).
 *
 * On other platforms: Returns CFD_SIMD_NONE.
 *
 * The result is cached after the first call. Thread-safe.
 *
 * @return The detected SIMD architecture type
 */
CFD_LIBRARY_EXPORT cfd_simd_arch_t cfd_detect_simd_arch(void);

/**
 * Check if AVX2 SIMD, with the FMA3 it is paired with, is available on the
 * current CPU.
 *
 * @return true if AVX2 and FMA3 are both supported, false otherwise
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
