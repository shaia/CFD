/**
 * SIMD-Optimized Poisson Solver Interface
 *
 * Common interface for different Poisson solver implementations.
 * Each implementation provides vectorized computation where possible.
 */

#ifndef POISSON_SOLVER_SIMD_H
#define POISSON_SOLVER_SIMD_H

#include <stddef.h>

// Poisson solver parameters
#define POISSON_MAX_ITER        1000
#define POISSON_MAX_ITER_JACOBI 2000  // Jacobi needs ~2x more iterations than SOR
#define POISSON_TOLERANCE       1e-6
#define POISSON_OMEGA           1.5   // SOR relaxation parameter (1 < omega < 2)

/**
 * Poisson solver type enumeration
 */
typedef enum {
    POISSON_SOLVER_SOR_SCALAR,     // Original scalar SOR (Gauss-Seidel)
    POISSON_SOLVER_JACOBI_SIMD,    // Jacobi iteration with SIMD
    POISSON_SOLVER_REDBLACK_SIMD   // Red-Black SOR with partial SIMD
} poisson_solver_type;

// Default solver: Red-Black SIMD (best balance of convergence and parallelism)
#define DEFAULT_POISSON_SOLVER POISSON_SOLVER_REDBLACK_SIMD

/**
 * Apply Neumann boundary conditions (zero gradient) to pressure field
 */
void poisson_apply_bc(double* p, size_t nx, size_t ny);

/**
 * Compute maximum residual for convergence check
 */
double poisson_compute_residual(const double* p, const double* rhs, size_t nx, size_t ny,
                                 double dx2, double dy2);

/**
 * Solve the Poisson equation using Jacobi iteration with AVX2 SIMD
 *
 * Jacobi iteration is fully parallelizable because all updates read from
 * the OLD array and write to a NEW array. This allows SIMD vectorization
 * of the entire inner loop.
 *
 * Trade-off: Converges ~2x slower than Gauss-Seidel/SOR, but each iteration
 * is much faster due to SIMD parallelism.
 *
 * @param p       Pressure field (in/out)
 * @param p_temp  Temporary buffer for double-buffering
 * @param rhs     Right-hand side of Poisson equation
 * @param nx      Grid size in x direction
 * @param ny      Grid size in y direction
 * @param dx      Grid spacing in x direction
 * @param dy      Grid spacing in y direction
 * @return        Number of iterations if converged, -1 if not
 */
int poisson_solve_jacobi_simd(double* p, double* p_temp, const double* rhs,
                               size_t nx, size_t ny, double dx, double dy);

/**
 * Solve the Poisson equation using Red-Black SOR with AVX2 SIMD
 *
 * Red-Black ordering allows partial vectorization: within each "color" sweep,
 * all updates are independent (they only read from the other color).
 * This combines SOR's fast convergence with SIMD parallelism.
 *
 * Same-color cells (stride 2 apart) are gathered into contiguous memory
 * for SIMD processing.
 *
 * @param p       Pressure field (in/out)
 * @param p_temp  Temporary buffer (optional, can be NULL for this solver)
 * @param rhs     Right-hand side of Poisson equation
 * @param nx      Grid size in x direction
 * @param ny      Grid size in y direction
 * @param dx      Grid spacing in x direction
 * @param dy      Grid spacing in y direction
 * @return        Number of iterations if converged, -1 if not
 */
int poisson_solve_redblack_simd(double* p, double* p_temp, const double* rhs,
                                 size_t nx, size_t ny, double dx, double dy);

/**
 * Solve the Poisson equation using scalar SOR (Gauss-Seidel)
 *
 * Original scalar implementation - fast convergence but not parallelizable.
 * Used as fallback when SIMD is not available.
 *
 * @param p       Pressure field (in/out)
 * @param rhs     Right-hand side of Poisson equation
 * @param nx      Grid size in x direction
 * @param ny      Grid size in y direction
 * @param dx      Grid spacing in x direction
 * @param dy      Grid spacing in y direction
 * @return        Number of iterations if converged, -1 if not
 */
int poisson_solve_sor_scalar(double* p, const double* rhs, size_t nx, size_t ny,
                              double dx, double dy);

/**
 * Unified Poisson solver interface - selects the best available method
 *
 * @param p            Pressure field (in/out)
 * @param p_temp       Temporary buffer (required for Jacobi, optional for Red-Black)
 * @param rhs          Right-hand side of Poisson equation
 * @param nx           Grid size in x direction
 * @param ny           Grid size in y direction
 * @param dx           Grid spacing in x direction
 * @param dy           Grid spacing in y direction
 * @param solver_type  Which solver to use
 * @return             Number of iterations if converged, -1 if not
 */
int poisson_solve(double* p, double* p_temp, const double* rhs, size_t nx, size_t ny,
                  double dx, double dy, poisson_solver_type solver_type);

#endif // POISSON_SOLVER_SIMD_H
