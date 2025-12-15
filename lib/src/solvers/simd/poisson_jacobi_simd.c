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

// Check for x86/x64 architecture before including AVX headers
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif
#else
#define USE_AVX2 0
#endif

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

#if USE_AVX2
    // SIMD constants
    __m256d dx2_inv_vec = _mm256_set1_pd(1.0 / dx2);
    __m256d dy2_inv_vec = _mm256_set1_pd(1.0 / dy2);
    __m256d inv_factor_vec = _mm256_set1_pd(-inv_factor);
#endif

    // Use local pointers for double-buffering with pointer swapping
    double* p_old = p;
    double* p_new = p_temp;

    for (iter = 0; iter < POISSON_MAX_ITER_JACOBI; iter++) {
        // Jacobi update: p_new[i] depends only on p_old neighbors
        for (size_t j = 1; j < ny - 1; j++) {
            size_t i = 1;

#if USE_AVX2
            // SIMD loop - process 4 cells at once
            for (; i + 4 <= nx - 1; i += 4) {
                size_t idx = (j * nx) + i;

                // Load neighbor values from OLD array
                __m256d p_xp = _mm256_loadu_pd(&p_old[idx + 1]);
                __m256d p_xm = _mm256_loadu_pd(&p_old[idx - 1]);
                __m256d p_yp = _mm256_loadu_pd(&p_old[idx + nx]);
                __m256d p_ym = _mm256_loadu_pd(&p_old[idx - nx]);
                __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

                // Compute: p_new = -(rhs - (p_xp + p_xm)/dx2 - (p_yp + p_ym)/dy2) / factor
                __m256d sum_x = _mm256_add_pd(p_xp, p_xm);
                __m256d sum_y = _mm256_add_pd(p_yp, p_ym);

                __m256d term_x = _mm256_mul_pd(sum_x, dx2_inv_vec);
                __m256d term_y = _mm256_mul_pd(sum_y, dy2_inv_vec);

                __m256d p_result = _mm256_mul_pd(
                    _mm256_sub_pd(rhs_vec, _mm256_add_pd(term_x, term_y)),
                    inv_factor_vec);

                // Store to NEW array
                _mm256_storeu_pd(&p_new[idx], p_result);
            }
#endif

            // Scalar remainder (or full scalar loop on non-x86)
            for (; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;
                double p_result = (rhs[idx] - (p_old[idx + 1] + p_old[idx - 1]) / dx2
                                            - (p_old[idx + nx] + p_old[idx - nx]) / dy2) * (-inv_factor);
                p_new[idx] = p_result;
            }
        }

        // Swap pointers instead of copying data
        double* tmp = p_old;
        p_old = p_new;
        p_new = tmp;

        poisson_apply_bc(p_old, nx, ny);

        // Check convergence every 10 iterations (residual computation is expensive)
        if (iter % 10 == 0) {
            double max_residual = poisson_compute_residual(p_old, rhs, nx, ny, dx2, dy2);
            if (max_residual < POISSON_TOLERANCE) {
                converged = 1;
                break;
            }
        }
    }

    // Ensure final result is in p (the caller's buffer)
    // After the loop, p_old points to the latest result
    if (p_old != p) {
        // Copy final result back to p
        memcpy(p, p_old, nx * ny * sizeof(double));
    }

    return converged ? iter : -1;
}
