/**
 * CPU Feature Detection Implementation
 *
 * Runtime detection of CPU SIMD capabilities using platform-specific methods:
 * - x86/x64 (MSVC): __cpuid/__cpuidex intrinsics
 * - x86/x64 (GCC/Clang): __get_cpuid_count from cpuid.h
 * - ARM64: NEON is always available
 */

#include "cfd/core/cpu_features.h"

/* Include platform-specific headers at file scope */
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define CFD_RUNTIME_X86_MSVC 1
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define CFD_RUNTIME_X86_GCC 1
#endif

/* ============================================================================
 * Cached Detection Result
 * ============================================================================ */

/**
 * Cached architecture detection result.
 * -1 = not yet detected
 *  0 = no SIMD available (CFD_SIMD_NONE)
 *  1 = AVX2 available (CFD_SIMD_AVX2)
 *  2 = NEON available (CFD_SIMD_NEON)
 */
static int g_simd_arch_cache = -1;

/* ============================================================================
 * Runtime Detection Implementation
 * ============================================================================ */

cfd_simd_arch_t cfd_detect_simd_arch(void) {
    /* Return cached result if available */
    if (g_simd_arch_cache >= 0) {
        return (cfd_simd_arch_t)g_simd_arch_cache;
    }

#if defined(CFD_RUNTIME_X86_MSVC)
    /* MSVC on x86/x64: Use __cpuid intrinsic */
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 7) {
        __cpuidex(cpuInfo, 7, 0);
        /* Check AVX2 bit (EBX bit 5) */
        if (cpuInfo[1] & (1 << 5)) {
            g_simd_arch_cache = CFD_SIMD_AVX2;
            return CFD_SIMD_AVX2;
        }
    }
    g_simd_arch_cache = CFD_SIMD_NONE;

#elif defined(CFD_RUNTIME_X86_GCC)
    /* GCC/Clang on x86/x64: Use __get_cpuid_count */
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        /* Check AVX2 bit (EBX bit 5) */
        if (ebx & (1 << 5)) {
            g_simd_arch_cache = CFD_SIMD_AVX2;
            return CFD_SIMD_AVX2;
        }
    }
    g_simd_arch_cache = CFD_SIMD_NONE;

#elif defined(__aarch64__) || defined(_M_ARM64)
    /* ARM64: NEON is always available on AArch64 */
    g_simd_arch_cache = CFD_SIMD_NEON;

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* ARMv7 with NEON: Assume available if compiled with NEON support */
    g_simd_arch_cache = CFD_SIMD_NEON;

#else
    /* Unknown architecture */
    g_simd_arch_cache = CFD_SIMD_NONE;

#endif

    return (cfd_simd_arch_t)g_simd_arch_cache;
}

/* ============================================================================
 * Convenience Functions
 * ============================================================================ */

bool cfd_has_avx2(void) {
    return cfd_detect_simd_arch() == CFD_SIMD_AVX2;
}

bool cfd_has_neon(void) {
    return cfd_detect_simd_arch() == CFD_SIMD_NEON;
}

bool cfd_has_simd(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    return arch == CFD_SIMD_AVX2 || arch == CFD_SIMD_NEON;
}

const char* cfd_get_simd_name(void) {
    switch (cfd_detect_simd_arch()) {
        case CFD_SIMD_AVX2:
            return "avx2";
        case CFD_SIMD_NEON:
            return "neon";
        default:
            return "none";
    }
}
