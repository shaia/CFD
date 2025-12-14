/**
 * Red-Black SOR Poisson Solver with AVX2 SIMD
 *
 * Red-Black ordering allows partial vectorization: within each "color" sweep,
 * all updates are independent (they only read from the other color).
 * This combines SOR's fast convergence with SIMD parallelism.
 *
 * The red-black pattern:
 *   Red cells:   (i+j) % 2 == 0  -> Updated in first sweep
 *   Black cells: (i+j) % 2 == 1  -> Updated in second sweep
 *
 * SIMD Strategy:
 *   Same-color cells (stride 2 apart) are gathered into contiguous memory
 *   for SIMD processing. After computation, results are scattered back.
 *
 * Advantages:
 *   - Maintains SOR's fast convergence (~2x faster than Jacobi)
 *   - Allows partial SIMD parallelism within each color sweep
 *
 * Trade-offs:
 *   - Gather/scatter overhead reduces SIMD efficiency compared to Jacobi
 *   - Still faster than scalar SOR due to SIMD arithmetic
 */

#include "poisson_solver_simd.h"
#include <math.h>
#include <string.h>
#include <immintrin.h>

int poisson_solve_redblack_simd(double* p, double* p_temp, const double* rhs,
                                 size_t nx, size_t ny, double dx, double dy) {
    (void)p_temp;  // Red-Black SOR doesn't need temporary buffer (in-place update)

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
    __m256d omega_vec = _mm256_set1_pd(POISSON_OMEGA);

    for (iter = 0; iter < POISSON_MAX_ITER; iter++) {
        double max_residual = 0.0;

        // Two sweeps: red cells (color=0) then black cells (color=1)
        for (int color = 0; color < 2; color++) {
            // Process rows where we can get contiguous same-color cells
            // On even rows (j%2==0): red at even i, black at odd i
            // On odd rows (j%2==1): red at odd i, black at even i

            for (size_t j = 1; j < ny - 1; j++) {
                size_t i_start = 1 + ((j + color) % 2);

                // For rows with enough same-color cells, use SIMD
                // Gather 4 same-color values (stride 2 apart)
                size_t i;
                for (i = i_start; i + 6 < nx - 1; i += 8) {
                    // Process 4 same-color cells at positions i, i+2, i+4, i+6
                    // These are all the same color since stride is 2

                    // Manually gather values (AVX2 gather is slower for this pattern)
                    double vals[4], p_xp[4], p_xm[4], p_yp[4], p_ym[4], rhs_v[4];

                    for (int k = 0; k < 4; k++) {
                        size_t idx = (j * nx) + i + (k * 2);
                        vals[k] = p[idx];
                        p_xp[k] = p[idx + 1];
                        p_xm[k] = p[idx - 1];
                        p_yp[k] = p[idx + nx];
                        p_ym[k] = p[idx - nx];
                        rhs_v[k] = rhs[idx];
                    }

                    // Load into SIMD registers
                    __m256d p_c = _mm256_loadu_pd(vals);
                    __m256d p_xp_v = _mm256_loadu_pd(p_xp);
                    __m256d p_xm_v = _mm256_loadu_pd(p_xm);
                    __m256d p_yp_v = _mm256_loadu_pd(p_yp);
                    __m256d p_ym_v = _mm256_loadu_pd(p_ym);
                    __m256d rhs_vec = _mm256_loadu_pd(rhs_v);

                    // Compute SOR update
                    __m256d sum_x = _mm256_add_pd(p_xp_v, p_xm_v);
                    __m256d sum_y = _mm256_add_pd(p_yp_v, p_ym_v);

                    __m256d term_x = _mm256_mul_pd(sum_x, dx2_inv_vec);
                    __m256d term_y = _mm256_mul_pd(sum_y, dy2_inv_vec);

                    __m256d p_new = _mm256_mul_pd(
                        _mm256_sub_pd(rhs_vec, _mm256_add_pd(term_x, term_y)),
                        inv_factor_vec);

                    // SOR relaxation: p = p + omega * (p_new - p)
                    __m256d delta = _mm256_sub_pd(p_new, p_c);
                    __m256d p_updated = _mm256_add_pd(p_c, _mm256_mul_pd(omega_vec, delta));

                    // Scatter back (manual)
                    double results[4];
                    _mm256_storeu_pd(results, p_updated);

                    for (int k = 0; k < 4; k++) {
                        size_t idx = (j * nx) + i + (k * 2);
                        p[idx] = results[k];

                        // Track residual
                        double p_xx = (p_xp[k] - 2.0 * vals[k] + p_xm[k]) / dx2;
                        double p_yy = (p_yp[k] - 2.0 * vals[k] + p_ym[k]) / dy2;
                        double res = fabs(p_xx + p_yy - rhs_v[k]);
                        if (res > max_residual) max_residual = res;
                    }
                }

                // Scalar remainder
                for (; i < nx - 1; i += 2) {
                    size_t idx = (j * nx) + i;

                    double p_c = p[idx];
                    double p_xp_s = p[idx + 1];
                    double p_xm_s = p[idx - 1];
                    double p_yp_s = p[idx + nx];
                    double p_ym_s = p[idx - nx];

                    double p_xx = (p_xp_s - 2.0 * p_c + p_xm_s) / dx2;
                    double p_yy = (p_yp_s - 2.0 * p_c + p_ym_s) / dy2;
                    double res = fabs(p_xx + p_yy - rhs[idx]);
                    if (res > max_residual) max_residual = res;

                    double p_new = (rhs[idx] - (p_xp_s + p_xm_s) / dx2
                                             - (p_yp_s + p_ym_s) / dy2) * (-inv_factor);
                    p[idx] = p_c + POISSON_OMEGA * (p_new - p_c);
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
