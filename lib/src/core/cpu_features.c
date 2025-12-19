/**
 * CPU Feature Detection Implementation
 *
 * Runtime detection of CPU SIMD capabilities using platform-specific methods:
 * - x86/x64 (MSVC): __cpuid/__cpuidex intrinsics + XGETBV for OS support
 * - x86/x64 (GCC/Clang): __get_cpuid_count + xgetbv for OS support
 * - ARM64: NEON is always available
 *
 * IMPORTANT: For AVX2, we must verify both:
 * 1. CPU support (CPUID leaf 7, EBX bit 5)
 * 2. OS support (OSXSAVE enabled + XCR0 bits 1,2 set for AVX state saving)
 *
 * Without OS support verification, AVX instructions will cause illegal
 * instruction exceptions on systems where the OS hasn't enabled AVX.
 *
 * Thread Safety:
 * The cache uses atomic operations with proper memory ordering to ensure
 * thread-safe initialization. The detection logic is idempotent (always
 * produces the same result), so multiple threads racing to initialize is
 * safe - they will all compute and store the same value.
 *
 * We use:
 * - C11 stdatomic.h when available (GCC 4.9+, Clang 3.1+, MSVC 2022+)
 * - GCC __atomic builtins as fallback (GCC 4.7+)
 * - MSVC Interlocked functions as final fallback
 */

#include "cfd/core/cpu_features.h"

/* ============================================================================
 * Atomic Operations Abstraction
 *
 * Provides atomic load/store with proper memory ordering across platforms.
 * Priority: C11 atomics > GCC atomics > MSVC Interlocked > volatile fallback
 * ============================================================================ */

/* Check for C11 atomics support */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    /* C11 atomics available */
    #include <stdatomic.h>
    #define CFD_HAS_C11_ATOMICS 1
#elif defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7))
    /* GCC 4.7+ atomic builtins */
    #define CFD_HAS_GCC_ATOMICS 1
#elif defined(_MSC_VER)
    /* MSVC Interlocked functions */
    #include <intrin.h>
    #define CFD_HAS_MSVC_ATOMICS 1
#endif

/* Atomic cache type and operations */
#if defined(CFD_HAS_C11_ATOMICS)
    static _Atomic int g_simd_arch_cache = -1;

    static inline int atomic_cache_load(void) {
        return atomic_load_explicit(&g_simd_arch_cache, memory_order_acquire);
    }

    static inline void atomic_cache_store(int value) {
        atomic_store_explicit(&g_simd_arch_cache, value, memory_order_release);
    }

#elif defined(CFD_HAS_GCC_ATOMICS)
    static int g_simd_arch_cache = -1;

    static inline int atomic_cache_load(void) {
        return __atomic_load_n(&g_simd_arch_cache, __ATOMIC_ACQUIRE);
    }

    static inline void atomic_cache_store(int value) {
        __atomic_store_n(&g_simd_arch_cache, value, __ATOMIC_RELEASE);
    }

#elif defined(CFD_HAS_MSVC_ATOMICS)
    static volatile long g_simd_arch_cache = -1;

    static inline int atomic_cache_load(void) {
        /* _InterlockedCompareExchange provides full barrier on x86/x64
         * This is the intrinsic form (note the underscore prefix) */
        return (int)_InterlockedCompareExchange(&g_simd_arch_cache, -1, -1);
    }

    static inline void atomic_cache_store(int value) {
        _InterlockedExchange(&g_simd_arch_cache, (long)value);
    }

#else
    /* Fallback: volatile with no guarantees beyond single-threaded correctness.
     * This should rarely be hit on modern compilers. */
    #warning "No atomic primitives available - cache may not be thread-safe"
    static volatile int g_simd_arch_cache = -1;

    static inline int atomic_cache_load(void) {
        return g_simd_arch_cache;
    }

    static inline void atomic_cache_store(int value) {
        g_simd_arch_cache = value;
    }
#endif

/* ============================================================================
 * Platform-Specific CPU Detection Headers
 * ============================================================================ */

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define CFD_RUNTIME_X86_MSVC 1
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define CFD_RUNTIME_X86_GCC 1

/**
 * Inline xgetbv for GCC/Clang.
 *
 * We use inline assembly instead of _xgetbv() intrinsic because the intrinsic
 * requires -mxsave compiler flag, but we want this file to compile without
 * requiring any special CPU feature flags (since it's doing runtime detection).
 *
 * IMPORTANT: This function must ONLY be called after verifying OSXSAVE support
 * via CPUID leaf 1, ECX bit 27. The OSXSAVE bit indicates that:
 * 1. The CPU supports the XGETBV instruction
 * 2. The OS has enabled XSAVE/XRSTOR for context switching
 *
 * Calling xgetbv without OSXSAVE support will cause an undefined opcode (#UD)
 * exception. The detect_simd_arch_impl() function enforces this requirement
 * by only calling cfd_xgetbv() inside an if(osxsave) block.
 *
 * Register usage:
 * - Input: ECX = XCR number (0 for XCR0)
 * - Output: EDX:EAX = XCR value (64-bit)
 * - No XMM/YMM registers are modified by xgetbv itself
 * - No additional clobbers needed beyond the output registers
 */
static inline unsigned long long cfd_xgetbv(unsigned int xcr) {
    unsigned int eax, edx;
    __asm__ __volatile__(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(xcr)
    );
    return ((unsigned long long)edx << 32) | eax;
}
#endif

/* ============================================================================
 * Runtime Detection Implementation
 * ============================================================================ */

/**
 * Perform the actual SIMD detection.
 * This is a pure function - always returns the same result for a given CPU.
 */
static cfd_simd_arch_t detect_simd_arch_impl(void) {
#if defined(CFD_RUNTIME_X86_MSVC)
    /* MSVC on x86/x64: Use __cpuid intrinsic */
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    int nIds = cpuInfo[0];

    if (nIds >= 7) {
        /* First check CPUID leaf 1 for OSXSAVE support (ECX bit 27) */
        __cpuid(cpuInfo, 1);
        int osxsave = (cpuInfo[2] & (1 << 27)) != 0;

        if (osxsave) {
            /* OSXSAVE is enabled, now check XCR0 for AVX state support */
            /* XCR0 bits: bit 1 = SSE state, bit 2 = AVX state */
            /* Both must be set for AVX to work */
            unsigned long long xcr0 = _xgetbv(0);
            int avx_os_support = ((xcr0 & 0x6) == 0x6);

            if (avx_os_support) {
                /* Now check CPU support for AVX2 */
                __cpuidex(cpuInfo, 7, 0);
                /* Check AVX2 bit (EBX bit 5) */
                if (cpuInfo[1] & (1 << 5)) {
                    return CFD_SIMD_AVX2;
                }
            }
        }
    }
    return CFD_SIMD_NONE;

#elif defined(CFD_RUNTIME_X86_GCC)
    /* GCC/Clang on x86/x64: Use __get_cpuid_count */
    unsigned int eax, ebx, ecx, edx;

    /* First check CPUID leaf 1 for OSXSAVE support (ECX bit 27) */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        int osxsave = (ecx & (1 << 27)) != 0;

        if (osxsave) {
            /* OSXSAVE is enabled, now check XCR0 for AVX state support */
            /* XCR0 bits: bit 1 = SSE state, bit 2 = AVX state */
            /* Both must be set for AVX to work */
            unsigned long long xcr0 = cfd_xgetbv(0);
            int avx_os_support = ((xcr0 & 0x6) == 0x6);

            if (avx_os_support) {
                /* Now check CPU support for AVX2 */
                if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                    /* Check AVX2 bit (EBX bit 5) */
                    if (ebx & (1 << 5)) {
                        return CFD_SIMD_AVX2;
                    }
                }
            }
        }
    }
    return CFD_SIMD_NONE;

#elif defined(__aarch64__) || defined(_M_ARM64)
    /* ARM64: NEON is always available on AArch64 */
    return CFD_SIMD_NEON;

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    /* ARMv7 with NEON: Assume available if compiled with NEON support */
    return CFD_SIMD_NEON;

#else
    return CFD_SIMD_NONE;
#endif
}

cfd_simd_arch_t cfd_detect_simd_arch(void) {
    /* Read cached result with acquire semantics */
    int cached = atomic_cache_load();

    if (cached >= 0) {
        return (cfd_simd_arch_t)cached;
    }

    /* Perform detection - result will be the same regardless of which thread
     * computes it, so racing here is safe (just wastes some CPU cycles). */
    cfd_simd_arch_t detected = detect_simd_arch_impl();

    /* Store result with release semantics to ensure visibility to other threads */
    atomic_cache_store((int)detected);

    return detected;
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
