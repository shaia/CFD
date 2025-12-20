/**
 * @file linear_solver_simd_omp_dispatch.c
 * @brief SIMD+OMP Poisson solver dispatcher with runtime CPU detection
 *
 * This file provides unified SIMD+OMP solvers that select the appropriate
 * architecture-specific implementation at RUNTIME:
 * - AVX2 + OpenMP on x86-64 (detected via CPUID)
 * - NEON + OpenMP on ARM64 (always available on ARM64)
 *
 * The actual implementations remain in separate files:
 * - avx2/linear_solver_jacobi_avx2_omp.c
 * - avx2/linear_solver_redblack_avx2_omp.c
 * - neon/linear_solver_jacobi_neon_omp.c
 * - neon/linear_solver_redblack_neon_omp.c
 *
 * This design is identical to the boundary conditions SIMD_OMP dispatcher.
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
            "DEBUG: SIMD+OMP %s solver not available "
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
 * If a backend is not compiled (e.g., NEON on x86), the pointers will be NULL.
 * ============================================================================ */

/* AVX2 + OMP implementations (x86-64) */
extern poisson_solver_t* create_jacobi_avx2_omp_solver(void);
extern poisson_solver_t* create_redblack_avx2_omp_solver(void);

/* NEON + OMP implementations (ARM64) */
extern poisson_solver_t* create_jacobi_neon_omp_solver(void);
extern poisson_solver_t* create_redblack_neon_omp_solver(void);

/* ============================================================================
 * SIMD+OMP BACKEND AVAILABILITY (Runtime + Compile-time detection)
 *
 * This checks BOTH:
 * 1. Compile-time: Was code compiled with SIMD support (-mavx2 or ARM64)?
 * 2. Compile-time: Was OpenMP enabled at build time?
 * 3. Runtime: Does CPU support the SIMD instructions?
 *
 * All must be true for SIMD+OMP to be available.
 * ============================================================================ */

/* Check if AVX2+OMP implementation was compiled */
#if defined(__AVX2__) && defined(CFD_ENABLE_OPENMP)
#define HAS_AVX2_OMP_IMPL 1
#else
#define HAS_AVX2_OMP_IMPL 0
#endif

/* Check if NEON+OMP implementation was compiled */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define HAS_NEON_OMP_IMPL 1
#else
#define HAS_NEON_OMP_IMPL 0
#endif

bool poisson_solver_simd_omp_backend_available(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    /* Check AVX2: compiled with AVX2 + OMP AND runtime CPU supports AVX2 */
    if (HAS_AVX2_OMP_IMPL && arch == CFD_SIMD_AVX2) {
        return true;
    }

    /* Check NEON: compiled with NEON + OMP AND runtime CPU supports NEON */
    if (HAS_NEON_OMP_IMPL && arch == CFD_SIMD_NEON) {
        return true;
    }

    return false;
}

const char* poisson_solver_simd_omp_get_arch_name(void) {
    return cfd_get_simd_name();
}

/* Public API wrappers */
bool poisson_solver_simd_omp_available(void) {
    return poisson_solver_simd_omp_backend_available();
}

const char* poisson_solver_get_simd_arch_name(void) {
    return poisson_solver_simd_omp_get_arch_name();
}

/* ============================================================================
 * JACOBI SIMD+OMP DISPATCHER
 * ============================================================================ */

poisson_solver_t* create_jacobi_simd_omp_solver(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2) {
        poisson_solver_t* solver = create_jacobi_avx2_omp_solver();
        if (solver) {
            return solver;
        }
    }

    if (arch == CFD_SIMD_NEON) {
        poisson_solver_t* solver = create_jacobi_neon_omp_solver();
        if (solver) {
            return solver;
        }
    }

    /* No SIMD backend available - report error and return NULL (no fallback) */
    log_no_simd_available("Jacobi");
    return NULL;
}

/* ============================================================================
 * RED-BLACK SOR SIMD+OMP DISPATCHER
 * ============================================================================ */

poisson_solver_t* create_redblack_simd_omp_solver(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();

    if (arch == CFD_SIMD_AVX2) {
        poisson_solver_t* solver = create_redblack_avx2_omp_solver();
        if (solver) {
            return solver;
        }
    }

    if (arch == CFD_SIMD_NEON) {
        poisson_solver_t* solver = create_redblack_neon_omp_solver();
        if (solver) {
            return solver;
        }
    }

    /* No SIMD backend available - report error and return NULL (no fallback) */
    log_no_simd_available("Red-Black");
    return NULL;
}
