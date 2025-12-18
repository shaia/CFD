/**
 * @file linear_solver_internal.h
 * @brief Internal declarations for linear solver implementations
 *
 * This header is not part of the public API.
 */

#ifndef CFD_LINEAR_SOLVER_INTERNAL_H
#define CFD_LINEAR_SOLVER_INTERNAL_H

#include "cfd/solvers/poisson_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FACTORY FUNCTIONS
 * ============================================================================ */

/* Jacobi solvers */
poisson_solver_t* create_jacobi_scalar_solver(void);
poisson_solver_t* create_jacobi_simd_solver(void);

/* SOR solvers */
poisson_solver_t* create_sor_scalar_solver(void);

/* Red-Black SOR solvers */
poisson_solver_t* create_redblack_scalar_solver(void);
poisson_solver_t* create_redblack_simd_solver(void);

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_redblack_omp_solver(void);
#endif

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Common solve loop used by all iterative solvers
 *
 * Implements the iteration control, convergence checking, and statistics.
 *
 * @param solver Initialized solver
 * @param x Solution vector
 * @param x_temp Temporary buffer
 * @param rhs Right-hand side
 * @param stats Output statistics
 * @return CFD_SUCCESS if converged
 */
cfd_status_t poisson_solver_solve_common(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats);

/**
 * Get current time in milliseconds (platform-independent)
 */
double poisson_solver_get_time_ms(void);

/* ============================================================================
 * SIMD DETECTION (Compile-time)
 * ============================================================================ */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
        #define POISSON_HAS_AVX2 1
    #else
        #define POISSON_HAS_AVX2 0
    #endif
    #ifdef __SSE2__
        #define POISSON_HAS_SSE2 1
    #elif defined(_M_X64) || defined(_M_AMD64)
        /* MSVC on x64 always has SSE2 */
        #define POISSON_HAS_SSE2 1
    #else
        #define POISSON_HAS_SSE2 0
    #endif
#else
    #define POISSON_HAS_AVX2 0
    #define POISSON_HAS_SSE2 0
#endif

#ifdef __ARM_NEON
    #define POISSON_HAS_NEON 1
#else
    #define POISSON_HAS_NEON 0
#endif

/* SIMD implementations currently only support AVX2 */
#define POISSON_HAS_SIMD POISSON_HAS_AVX2

#ifdef __cplusplus
}
#endif

#endif /* CFD_LINEAR_SOLVER_INTERNAL_H */
