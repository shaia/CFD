/**
 * Common Poisson Solver Utilities
 *
 * Shared functions for all Poisson solver implementations:
 * - Boundary condition application
 * - Residual computation
 * - Scalar SOR solver (fallback)
 * - Unified solver interface
 */

#include "poisson_solver_simd.h"
#include <math.h>

/**
 * Apply Neumann boundary conditions (zero gradient) to pressure field
 */
void poisson_apply_bc(double* p, size_t nx, size_t ny) {
    // Left and right boundaries
    for (size_t j = 0; j < ny; j++) {
        p[(j * nx) + 0] = p[(j * nx) + 1];
        p[(j * nx) + nx - 1] = p[(j * nx) + nx - 2];
    }
    // Top and bottom boundaries
    for (size_t i = 0; i < nx; i++) {
        p[i] = p[nx + i];
        p[((ny - 1) * nx) + i] = p[((ny - 2) * nx) + i];
    }
}

/**
 * Compute maximum residual for convergence check
 */
double poisson_compute_residual(const double* p, const double* rhs, size_t nx, size_t ny,
                                 double dx2, double dy2) {
    double max_residual = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = (j * nx) + i;
            double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
            double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;
            double residual = fabs(p_xx + p_yy - rhs[idx]);
            if (residual > max_residual) {
                max_residual = residual;
            }
        }
    }
    return max_residual;
}

/**
 * Solve the Poisson equation using scalar SOR (Gauss-Seidel)
 * Original scalar implementation - fast convergence but not parallelizable.
 */
int poisson_solve_sor_scalar(double* p, const double* rhs, size_t nx, size_t ny,
                              double dx, double dy) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    if (factor < 1e-10) {
        return -1;
    }

    double inv_factor = 1.0 / factor;
    int converged = 0;
    int iter;

    for (iter = 0; iter < POISSON_MAX_ITER; iter++) {
        double max_residual = 0.0;

        // Red-black Gauss-Seidel with SOR
        for (int color = 0; color < 2; color++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    if ((int)((i + j) % 2) != color) {
                        continue;
                    }

                    size_t idx = (j * nx) + i;
                    double p_c = p[idx];
                    double p_xp = p[idx + 1];
                    double p_xm = p[idx - 1];
                    double p_yp = p[idx + nx];
                    double p_ym = p[idx - nx];

                    // Compute SOR update first
                    double p_new = (rhs[idx] - (p_xp + p_xm) / dx2 -
                                    (p_yp + p_ym) / dy2) * (-inv_factor);
                    double p_updated = p_c + (POISSON_OMEGA * (p_new - p_c));
                    p[idx] = p_updated;

                    // Track residual using UPDATED value for accurate convergence check
                    double p_xx = (p_xp - 2.0 * p_updated + p_xm) / dx2;
                    double p_yy = (p_yp - 2.0 * p_updated + p_ym) / dy2;
                    double residual = fabs(p_xx + p_yy - rhs[idx]);

                    if (residual > max_residual) {
                        max_residual = residual;
                    }
                }
            }
        }

        poisson_apply_bc(p, nx, ny);

        if (max_residual < POISSON_TOLERANCE) {
            converged = 1;
            break;
        }
    }

    return converged ? iter : -1;
}

/**
 * Unified Poisson solver interface - selects the best available method
 */
int poisson_solve(double* p, double* p_temp, const double* rhs, size_t nx, size_t ny,
                  double dx, double dy, poisson_solver_type solver_type) {
    switch (solver_type) {
        case POISSON_SOLVER_JACOBI_SIMD:
            return poisson_solve_jacobi_simd(p, p_temp, rhs, nx, ny, dx, dy);
        case POISSON_SOLVER_REDBLACK_SIMD:
            return poisson_solve_redblack_simd(p, p_temp, rhs, nx, ny, dx, dy);
        case POISSON_SOLVER_SOR_SCALAR:
        default:
            return poisson_solve_sor_scalar(p, rhs, nx, ny, dx, dy);
    }
}
