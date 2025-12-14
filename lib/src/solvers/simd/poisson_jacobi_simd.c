/**
 * Jacobi Iteration Poisson Solver with AVX2 SIMD
 *
 * This implementation uses Jacobi iteration which is fully parallelizable
 * because all updates read from the OLD array and write to a NEW array.
 * This allows SIMD vectorization of the entire inner loop.
 *
 * Trade-off: Converges ~2x slower than Gauss-Seidel/SOR, but each iteration
 * is much faster due to SIMD parallelism.
 *
 * Mathematical formulation:
 *   p_new[i,j] = -(rhs[i,j] - (p_old[i+1,j] + p_old[i-1,j])/dx^2
 *                           - (p_old[i,j+1] + p_old[i,j-1])/dy^2) / factor
 *
 * where factor = 2 * (1/dx^2 + 1/dy^2)
 */

#include "poisson_solver_simd.h"
#include <math.h>
#include <string.h>
#include <immintrin.h>

int poisson_solve_jacobi_simd(double* p, double* p_temp, const double* rhs,
                               size_t nx, size_t ny, double dx, double dy) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    if (factor < 1e-10) {
        return -1;
    }

    double inv_factor = 1.0 / factor;
    int converged = 0;
    int iter;

    // SIMD constants
    __m256d dx2_inv_vec = _mm256_set1_pd(1.0 / dx2);
    __m256d dy2_inv_vec = _mm256_set1_pd(1.0 / dy2);
    __m256d inv_factor_vec = _mm256_set1_pd(-inv_factor);

    // Jacobi needs ~2x more iterations than SOR for same convergence
    int jacobi_max_iter = POISSON_MAX_ITER * 2;

    for (iter = 0; iter < jacobi_max_iter; iter++) {
        // Jacobi update: p_new[i] depends only on p_old neighbors
        for (size_t j = 1; j < ny - 1; j++) {
            size_t i;

            // SIMD loop - process 4 cells at once
            for (i = 1; i + 4 <= nx - 1; i += 4) {
                size_t idx = (j * nx) + i;

                // Load neighbor values from OLD array (p_c not needed for Jacobi)
                __m256d p_xp = _mm256_loadu_pd(&p[idx + 1]);
                __m256d p_xm = _mm256_loadu_pd(&p[idx - 1]);
                __m256d p_yp = _mm256_loadu_pd(&p[idx + nx]);
                __m256d p_ym = _mm256_loadu_pd(&p[idx - nx]);
                __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

                // Compute: p_new = -(rhs - (p_xp + p_xm)/dx2 - (p_yp + p_ym)/dy2) / factor
                __m256d sum_x = _mm256_add_pd(p_xp, p_xm);
                __m256d sum_y = _mm256_add_pd(p_yp, p_ym);

                __m256d term_x = _mm256_mul_pd(sum_x, dx2_inv_vec);
                __m256d term_y = _mm256_mul_pd(sum_y, dy2_inv_vec);

                __m256d p_new = _mm256_mul_pd(
                    _mm256_sub_pd(rhs_vec, _mm256_add_pd(term_x, term_y)),
                    inv_factor_vec);

                // Store to NEW array (p_temp)
                _mm256_storeu_pd(&p_temp[idx], p_new);
            }

            // Scalar remainder
            for (; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;
                double p_new = (rhs[idx] - (p[idx + 1] + p[idx - 1]) / dx2
                                         - (p[idx + nx] + p[idx - nx]) / dy2) * (-inv_factor);
                p_temp[idx] = p_new;
            }
        }

        // Swap: copy p_temp back to p
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;
                p[idx] = p_temp[idx];
            }
        }

        poisson_apply_bc(p, nx, ny);

        // Check convergence every 10 iterations (residual computation is expensive)
        if (iter % 10 == 0) {
            double max_residual = poisson_compute_residual(p, rhs, nx, ny, dx2, dy2);
            if (max_residual < POISSON_TOLERANCE) {
                converged = 1;
                break;
            }
        }
    }

    return converged ? iter : -1;
}
