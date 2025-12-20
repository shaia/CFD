/**
 * @file linear_solver_internal.h
 * @brief Internal declarations for linear solver implementations
 *
 * This header is not part of the public API.
 */

#ifndef CFD_LINEAR_SOLVER_INTERNAL_H
#define CFD_LINEAR_SOLVER_INTERNAL_H

#include "cfd/solvers/poisson_solver.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FACTORY FUNCTIONS
 *
 * All SIMD backends use runtime CPU detection (AVX2/NEON) via the SIMD+OMP
 * dispatcher. See simd_omp/linear_solver_simd_omp_dispatch.c for details.
 * ============================================================================ */

/* Jacobi solvers */
poisson_solver_t* create_jacobi_scalar_solver(void);
poisson_solver_t* create_jacobi_simd_omp_solver(void);

/* SOR solvers */
poisson_solver_t* create_sor_scalar_solver(void);

/* Red-Black SOR solvers */
poisson_solver_t* create_redblack_scalar_solver(void);
poisson_solver_t* create_redblack_simd_omp_solver(void);

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_redblack_omp_solver(void);
#endif

/* ============================================================================
 * SIMD+OMP BACKEND AVAILABILITY (Runtime detection)
 * ============================================================================ */

/**
 * Check if SIMD+OMP backend is available at runtime.
 * Uses cfd_detect_simd_arch() from cpu_features.h.
 */
bool poisson_solver_simd_omp_backend_available(void);

/**
 * Get the name of the detected SIMD architecture.
 * Returns "avx2", "neon", or "none".
 */
const char* poisson_solver_simd_omp_get_arch_name(void);

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

#ifdef __cplusplus
}
#endif

#endif /* CFD_LINEAR_SOLVER_INTERNAL_H */
