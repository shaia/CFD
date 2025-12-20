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

#include <assert.h>
#include <stdio.h>

/* ============================================================================
 * ERROR REPORTING
 * ============================================================================ */

/**
 * Report error when SIMD backend is unavailable.
 * This is a programming error - callers should check availability first.
 */
static void report_no_simd_error(const char* solver_type) {
    fprintf(stderr,
            "ERROR: SIMD+OMP %s solver requested but no SIMD backend available "
            "(detected arch: %s). Check poisson_solver_simd_omp_available() first.\n",
            solver_type, cfd_get_simd_name());
    assert(0 && "SIMD+OMP solver requested without available implementation");
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
 * 1. Runtime CPU detection: Does CPU support AVX2/NEON?
 * 2. Compile-time availability: Was OpenMP enabled at build time?
 *
 * Both must be true for SIMD+OMP to be available.
 * ============================================================================ */

bool poisson_solver_simd_omp_backend_available(void) {
#ifndef CFD_ENABLE_OPENMP
    /* OpenMP not enabled at compile time - SIMD+OMP not available */
    return false;
#else
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    return (arch == CFD_SIMD_AVX2 || arch == CFD_SIMD_NEON);
#endif
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
    report_no_simd_error("Jacobi");
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
    report_no_simd_error("Red-Black");
    return NULL;
}
