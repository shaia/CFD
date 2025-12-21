/**
 * @file linear_solver_simd_dispatch.c
 * @brief SIMD Poisson solver dispatcher with runtime CPU detection
 *
 * This file provides unified SIMD solvers that select the appropriate
 * architecture-specific implementation at RUNTIME:
 * - AVX2 on x86-64 (detected via CPUID)
 * - NEON on ARM64 (always available on ARM64)
 *
 * The actual implementations remain in separate files:
 * - avx2/linear_solver_jacobi_avx2.c
 * - avx2/linear_solver_redblack_avx2.c
 * - avx2/linear_solver_cg_avx2.c
 * - neon/linear_solver_jacobi_neon.c
 * - neon/linear_solver_redblack_neon.c
 * - neon/linear_solver_cg_neon.c
 *
 * This design is identical to the boundary conditions SIMD dispatcher.
 */

#include "../linear_solver_internal.h"
#include "cfd/core/cpu_features.h"

#include <stdio.h>

/* ============================================================================
 * LOGGING
 * ============================================================================ */

/**
 * Log when SIMD backend is unavailable.
 * Callers should handle the NULL return and fall back to scalar if needed.
 */
static void log_no_simd_available(const char* solver_type) {
    /* Silent in release builds - caller handles fallback */
#ifdef CFD_DEBUG
    fprintf(stderr,
            "DEBUG: SIMD %s solver not available "
            "(detected arch: %s). Returning NULL for fallback.\n",
            solver_type, cfd_get_simd_name());
#else
    (void)solver_type;  /* Suppress unused parameter warning */
#endif
}

/* ============================================================================
 * FORWARD DECLARATIONS FOR ARCHITECTURE-SPECIFIC IMPLEMENTATIONS
 *
 * These are defined in avx2/ and neon/ subdirectories.
 * If a backend is not compiled (e.g., NEON on x86), these functions return NULL.
 * ============================================================================ */

/* AVX2 implementations (x86-64) */
extern poisson_solver_t* create_jacobi_avx2_solver(void);
extern poisson_solver_t* create_redblack_avx2_solver(void);
extern poisson_solver_t* create_cg_avx2_solver(void);

/* NEON implementations (ARM64) */
extern poisson_solver_t* create_jacobi_neon_solver(void);
extern poisson_solver_t* create_redblack_neon_solver(void);
extern poisson_solver_t* create_cg_neon_solver(void);

/* ============================================================================
 * SIMD BACKEND AVAILABILITY (Runtime + Compile-time detection)
 *
 * This checks BOTH:
 * 1. Compile-time: Was code compiled with SIMD support (-mavx2 or ARM64)?
 * 2. Compile-time: Was OpenMP enabled at build time?
 * 3. Runtime: Does CPU support the SIMD instructions?
 *
 * All must be true for SIMD to be available.
 * ============================================================================ */

/* Check if AVX2 implementation was compiled
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define HAS_AVX2_IMPL 1
#else
#define HAS_AVX2_IMPL 0
#endif

/* Check if NEON implementation was compiled */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define HAS_NEON_IMPL 1
#else
#define HAS_NEON_IMPL 0
#endif

bool poisson_solver_simd_backend_available(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    /* Check AVX2: compiled with AVX2 AND runtime CPU supports AVX2 */
    if (HAS_AVX2_IMPL && arch == CFD_SIMD_AVX2) {
        return true;
    }

    /* Check NEON: compiled with NEON AND runtime CPU supports NEON */
    if (HAS_NEON_IMPL && arch == CFD_SIMD_NEON) {
        return true;
    }

    return false;
}

const char* poisson_solver_simd_get_arch_name(void) {
    return cfd_get_simd_name();
}

/* Public API wrappers */
bool poisson_solver_simd_available(void) {
    return poisson_solver_simd_backend_available();
}

const char* poisson_solver_get_simd_arch_name(void) {
    return poisson_solver_simd_get_arch_name();
}

/* ============================================================================
 * JACOBI SIMD DISPATCHER
 * ============================================================================ */

poisson_solver_t* create_jacobi_simd_solver(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    /* Check compile-time AND runtime availability before calling factory */
    if (HAS_AVX2_IMPL && arch == CFD_SIMD_AVX2) {
        return create_jacobi_avx2_solver();
    }

    if (HAS_NEON_IMPL && arch == CFD_SIMD_NEON) {
        return create_jacobi_neon_solver();
    }

    /* No SIMD backend available - report error and return NULL (no fallback) */
    log_no_simd_available("Jacobi");
    return NULL;
}

/* ============================================================================
 * RED-BLACK SOR SIMD DISPATCHER
 * ============================================================================ */

poisson_solver_t* create_redblack_simd_solver(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    /* Check compile-time AND runtime availability before calling factory */
    if (HAS_AVX2_IMPL && arch == CFD_SIMD_AVX2) {
        return create_redblack_avx2_solver();
    }

    if (HAS_NEON_IMPL && arch == CFD_SIMD_NEON) {
        return create_redblack_neon_solver();
    }

    /* No SIMD backend available - report error and return NULL (no fallback) */
    log_no_simd_available("Red-Black");
    return NULL;
}

/* ============================================================================
 * CONJUGATE GRADIENT SIMD DISPATCHER
 * ============================================================================ */

poisson_solver_t* create_cg_simd_solver(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    /* Check compile-time AND runtime availability before calling factory */
    if (HAS_AVX2_IMPL && arch == CFD_SIMD_AVX2) {
        return create_cg_avx2_solver();
    }

    if (HAS_NEON_IMPL && arch == CFD_SIMD_NEON) {
        return create_cg_neon_solver();
    }

    /* No SIMD backend available - report error and return NULL (no fallback) */
    log_no_simd_available("CG");
    return NULL;
}
