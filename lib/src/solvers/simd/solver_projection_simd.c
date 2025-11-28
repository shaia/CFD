/**
 * Optimized Projection Method Solver (Chorin's Method) with SIMD
 *
 * This is the SIMD-optimized version of the projection method solver
 * using AVX2/FMA intrinsics for improved performance on modern CPUs.
 *
 * The projection method solves the incompressible Navier-Stokes equations
 * by splitting the solution into two steps:
 *
 * 1. Predictor Step: Compute intermediate velocity u* ignoring pressure
 *    u* = u^n + dt * (-u·∇u + ν∇²u + f)
 *
 * 2. Pressure Projection: Solve Poisson equation for pressure
 *    ∇²p = (ρ/dt) * ∇·u*
 *
 * 3. Corrector Step: Project velocity to be divergence-free
 *    u^(n+1) = u* - (dt/ρ) * ∇p
 */

#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Try to use SIMD intrinsics
#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
    #define USE_AVX 1
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #include <immintrin.h>
    #define USE_AVX 1
#else
    #define USE_AVX 0
#endif

// Poisson solver parameters
#define POISSON_MAX_ITER 1000
#define POISSON_TOLERANCE 1e-6
#define POISSON_OMEGA 1.5  // SOR relaxation parameter

// Physical limits
#define MAX_VELOCITY 100.0
#define MAX_PRESSURE 1000.0

#if USE_AVX
/**
 * AVX-optimized horizontal sum of a __m256d vector
 */
static inline double hsum_avx(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}

/**
 * AVX-optimized maximum of a __m256d vector
 */
static inline double hmax_avx(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_max_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_max_sd(vlow, high64));
}

/**
 * AVX-optimized absolute value of a __m256d vector
 */
static inline __m256d abs_avx(__m256d v) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    return _mm256_andnot_pd(sign_mask, v);
}

/**
 * SIMD-optimized Poisson solver using SOR with AVX2
 */
static int solve_poisson_sor_simd(double* p, const double* rhs,
                                   size_t nx, size_t ny,
                                   double dx, double dy,
                                   int max_iter, double tolerance) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0/dx2 + 1.0/dy2);

    if (factor < 1e-10) return -1;

    double inv_factor = 1.0 / factor;
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;

    // Vectorized constants
    __m256d omega_vec = _mm256_set1_pd(POISSON_OMEGA);
    __m256d one_minus_omega = _mm256_set1_pd(1.0 - POISSON_OMEGA);
    __m256d neg_inv_factor = _mm256_set1_pd(-inv_factor);
    __m256d inv_dx2_vec = _mm256_set1_pd(inv_dx2);
    __m256d inv_dy2_vec = _mm256_set1_pd(inv_dy2);

    int converged = 0;
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        __m256d max_residual_vec = _mm256_setzero_pd();
        double max_residual_scalar = 0.0;

        // Red-black Gauss-Seidel with SOR
        for (int color = 0; color < 2; color++) {
            for (size_t j = 1; j < ny - 1; j++) {
                // Determine starting point based on color
                size_t i_start = 1 + ((j + color) % 2);

                // Process 4 elements at a time where possible
                size_t i;
                for (i = i_start; i + 7 < nx - 1; i += 8) {
                    // Only process every other element for red-black
                    // Process elements i, i+2, i+4, i+6
                    size_t idx0 = j * nx + i;
                    size_t idx1 = j * nx + i + 2;
                    size_t idx2 = j * nx + i + 4;
                    size_t idx3 = j * nx + i + 6;

                    // Load current values
                    __m256d p_curr = _mm256_set_pd(p[idx3], p[idx2], p[idx1], p[idx0]);

                    // Load neighbors for Laplacian
                    __m256d p_xp = _mm256_set_pd(p[idx3 + 1], p[idx2 + 1], p[idx1 + 1], p[idx0 + 1]);
                    __m256d p_xm = _mm256_set_pd(p[idx3 - 1], p[idx2 - 1], p[idx1 - 1], p[idx0 - 1]);
                    __m256d p_yp = _mm256_set_pd(p[idx3 + nx], p[idx2 + nx], p[idx1 + nx], p[idx0 + nx]);
                    __m256d p_ym = _mm256_set_pd(p[idx3 - nx], p[idx2 - nx], p[idx1 - nx], p[idx0 - nx]);

                    // Load RHS
                    __m256d rhs_vec = _mm256_set_pd(rhs[idx3], rhs[idx2], rhs[idx1], rhs[idx0]);

                    // Compute Laplacian: (p_xp - 2*p + p_xm)/dx² + (p_yp - 2*p + p_ym)/dy²
                    __m256d two = _mm256_set1_pd(2.0);
                    __m256d p_xx = _mm256_mul_pd(_mm256_sub_pd(_mm256_add_pd(p_xp, p_xm),
                                                               _mm256_mul_pd(two, p_curr)),
                                                 inv_dx2_vec);
                    __m256d p_yy = _mm256_mul_pd(_mm256_sub_pd(_mm256_add_pd(p_yp, p_ym),
                                                               _mm256_mul_pd(two, p_curr)),
                                                 inv_dy2_vec);

                    // Residual
                    __m256d residual = _mm256_sub_pd(_mm256_add_pd(p_xx, p_yy), rhs_vec);
                    max_residual_vec = _mm256_max_pd(max_residual_vec, abs_avx(residual));

                    // SOR update
                    __m256d sum_neighbors = _mm256_add_pd(
                        _mm256_mul_pd(_mm256_add_pd(p_xp, p_xm), inv_dx2_vec),
                        _mm256_mul_pd(_mm256_add_pd(p_yp, p_ym), inv_dy2_vec));
                    __m256d p_new = _mm256_mul_pd(_mm256_sub_pd(rhs_vec, sum_neighbors), neg_inv_factor);

                    // Apply relaxation: p = (1-omega)*p + omega*p_new
                    __m256d p_updated = _mm256_add_pd(
                        _mm256_mul_pd(one_minus_omega, p_curr),
                        _mm256_mul_pd(omega_vec, p_new));

                    // Store results
                    p[idx0] = ((double*)&p_updated)[0];
                    p[idx1] = ((double*)&p_updated)[1];
                    p[idx2] = ((double*)&p_updated)[2];
                    p[idx3] = ((double*)&p_updated)[3];
                }

                // Handle remaining elements
                for (; i < nx - 1; i += 2) {
                    size_t idx = j * nx + i;

                    double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
                    double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;

                    double residual = p_xx + p_yy - rhs[idx];
                    if (fabs(residual) > max_residual_scalar) {
                        max_residual_scalar = fabs(residual);
                    }

                    double p_new = (rhs[idx] - (p[idx + 1] + p[idx - 1]) / dx2
                                             - (p[idx + nx] + p[idx - nx]) / dy2) * (-inv_factor);
                    p[idx] = p[idx] + POISSON_OMEGA * (p_new - p[idx]);
                }
            }
        }

        // Apply Neumann boundary conditions
        for (size_t j = 0; j < ny; j++) {
            p[j * nx + 0] = p[j * nx + 1];
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        }
        for (size_t i = 0; i < nx; i++) {
            p[i] = p[nx + i];
            p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
        }

        // Check convergence
        double max_residual = fmax(hmax_avx(max_residual_vec), max_residual_scalar);
        if (max_residual < tolerance) {
            converged = 1;
            break;
        }
    }

    return converged ? iter : -1;
}

#else
// Fallback scalar Poisson solver
static int solve_poisson_sor_simd(double* p, const double* rhs,
                                   size_t nx, size_t ny,
                                   double dx, double dy,
                                   int max_iter, double tolerance) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0/dx2 + 1.0/dy2);

    if (factor < 1e-10) return -1;

    double inv_factor = 1.0 / factor;

    int converged = 0;
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        double max_residual = 0.0;

        for (int color = 0; color < 2; color++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    if ((i + j) % 2 != color) continue;

                    size_t idx = j * nx + i;

                    double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
                    double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;

                    double residual = p_xx + p_yy - rhs[idx];
                    if (fabs(residual) > max_residual) {
                        max_residual = fabs(residual);
                    }

                    double p_new = (rhs[idx] - (p[idx + 1] + p[idx - 1]) / dx2
                                             - (p[idx + nx] + p[idx - nx]) / dy2) * (-inv_factor);
                    p[idx] = p[idx] + POISSON_OMEGA * (p_new - p[idx]);
                }
            }
        }

        for (size_t j = 0; j < ny; j++) {
            p[j * nx + 0] = p[j * nx + 1];
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        }
        for (size_t i = 0; i < nx; i++) {
            p[i] = p[nx + i];
            p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
        }

        if (max_residual < tolerance) {
            converged = 1;
            break;
        }
    }

    return converged ? iter : -1;
}
#endif

/**
 * SIMD-optimized Projection Method Solver
 */
void solve_projection_method_optimized(FlowField* field, const Grid* grid, const SolverParams* params) {
    if (!field || !grid || !params) return;
    if (field->nx < 3 || field->ny < 3) return;

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;

    // Allocate temporary arrays
    double* u_star = (double*)cfd_calloc(size, sizeof(double));
    double* v_star = (double*)cfd_calloc(size, sizeof(double));
    double* p_new = (double*)cfd_calloc(size, sizeof(double));
    double* rhs = (double*)cfd_calloc(size, sizeof(double));

    if (!u_star || !v_star || !p_new || !rhs) {
        cfd_free(u_star);
        cfd_free(v_star);
        cfd_free(p_new);
        cfd_free(rhs);
        return;
    }

    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(p_new, field->p, size * sizeof(double));

#if USE_AVX
    // Vectorized constants
    __m256d dt_vec = _mm256_set1_pd(dt);
    __m256d nu_vec = _mm256_set1_pd(nu);
    __m256d two_dx_vec = _mm256_set1_pd(2.0 * dx);
    __m256d two_dy_vec = _mm256_set1_pd(2.0 * dy);
    __m256d dx2_vec = _mm256_set1_pd(dx * dx);
    __m256d dy2_vec = _mm256_set1_pd(dy * dy);
    __m256d max_vel_vec = _mm256_set1_pd(MAX_VELOCITY);
    __m256d neg_max_vel_vec = _mm256_set1_pd(-MAX_VELOCITY);
#endif

    for (int iter = 0; iter < params->max_iter; iter++) {

        // ============================================================
        // STEP 1: Predictor - Compute intermediate velocity u*
        // ============================================================
        for (size_t j = 1; j < ny - 1; j++) {
#if USE_AVX
            size_t i;
            for (i = 1; i + 3 < nx - 1; i += 4) {
                size_t idx = j * nx + i;

                // Load velocities
                __m256d u = _mm256_loadu_pd(&field->u[idx]);
                __m256d v = _mm256_loadu_pd(&field->v[idx]);

                // Load neighbors for gradients
                __m256d u_xp = _mm256_loadu_pd(&field->u[idx + 1]);
                __m256d u_xm = _mm256_loadu_pd(&field->u[idx - 1]);
                __m256d u_yp = _mm256_loadu_pd(&field->u[idx + nx]);
                __m256d u_ym = _mm256_loadu_pd(&field->u[idx - nx]);

                __m256d v_xp = _mm256_loadu_pd(&field->v[idx + 1]);
                __m256d v_xm = _mm256_loadu_pd(&field->v[idx - 1]);
                __m256d v_yp = _mm256_loadu_pd(&field->v[idx + nx]);
                __m256d v_ym = _mm256_loadu_pd(&field->v[idx - nx]);

                // Convective terms: -u·∇u
                __m256d du_dx = _mm256_div_pd(_mm256_sub_pd(u_xp, u_xm), two_dx_vec);
                __m256d du_dy = _mm256_div_pd(_mm256_sub_pd(u_yp, u_ym), two_dy_vec);
                __m256d dv_dx = _mm256_div_pd(_mm256_sub_pd(v_xp, v_xm), two_dx_vec);
                __m256d dv_dy = _mm256_div_pd(_mm256_sub_pd(v_yp, v_ym), two_dy_vec);

                __m256d conv_u = _mm256_add_pd(_mm256_mul_pd(u, du_dx), _mm256_mul_pd(v, du_dy));
                __m256d conv_v = _mm256_add_pd(_mm256_mul_pd(u, dv_dx), _mm256_mul_pd(v, dv_dy));

                // Viscous terms: ν∇²u
                __m256d two = _mm256_set1_pd(2.0);
                __m256d d2u_dx2 = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_xp, u_xm), _mm256_mul_pd(two, u)), dx2_vec);
                __m256d d2u_dy2 = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_yp, u_ym), _mm256_mul_pd(two, u)), dy2_vec);
                __m256d d2v_dx2 = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(v_xp, v_xm), _mm256_mul_pd(two, v)), dx2_vec);
                __m256d d2v_dy2 = _mm256_div_pd(
                    _mm256_sub_pd(_mm256_add_pd(v_yp, v_ym), _mm256_mul_pd(two, v)), dy2_vec);

                __m256d visc_u = _mm256_mul_pd(nu_vec, _mm256_add_pd(d2u_dx2, d2u_dy2));
                __m256d visc_v = _mm256_mul_pd(nu_vec, _mm256_add_pd(d2v_dx2, d2v_dy2));

                // Intermediate velocity
                __m256d u_star_new = _mm256_add_pd(u,
                    _mm256_mul_pd(dt_vec, _mm256_sub_pd(visc_u, conv_u)));
                __m256d v_star_new = _mm256_add_pd(v,
                    _mm256_mul_pd(dt_vec, _mm256_sub_pd(visc_v, conv_v)));

                // Clamp velocities
                u_star_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, u_star_new));
                v_star_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, v_star_new));

                _mm256_storeu_pd(&u_star[idx], u_star_new);
                _mm256_storeu_pd(&v_star[idx], v_star_new);
            }

            // Handle remaining elements
            for (; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double u_val = field->u[idx];
                double v_val = field->v[idx];

                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

                double conv_u = u_val * du_dx + v_val * du_dy;
                double conv_v = u_val * dv_dx + v_val * dv_dy;

                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u_val + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u_val + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v_val + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v_val + field->v[idx - nx]) / (dy * dy);

                double visc_u = nu * (d2u_dx2 + d2u_dy2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2);

                u_star[idx] = u_val + dt * (-conv_u + visc_u);
                v_star[idx] = v_val + dt * (-conv_v + visc_v);

                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
            }
#else
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double u_val = field->u[idx];
                double v_val = field->v[idx];

                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

                double conv_u = u_val * du_dx + v_val * du_dy;
                double conv_v = u_val * dv_dx + v_val * dv_dy;

                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u_val + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u_val + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v_val + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v_val + field->v[idx - nx]) / (dy * dy);

                double visc_u = nu * (d2u_dx2 + d2u_dy2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2);

                u_star[idx] = u_val + dt * (-conv_u + visc_u);
                v_star[idx] = v_val + dt * (-conv_v + visc_v);

                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
            }
#endif
        }

        // Apply boundary conditions to intermediate velocity
        for (size_t j = 0; j < ny; j++) {
            u_star[j * nx + 0] = u_star[j * nx + 1];
            u_star[j * nx + nx - 1] = u_star[j * nx + nx - 2];
            v_star[j * nx + 0] = v_star[j * nx + 1];
            v_star[j * nx + nx - 1] = v_star[j * nx + nx - 2];
        }
        for (size_t i = 0; i < nx; i++) {
            u_star[i] = u_star[nx + i];
            u_star[(ny - 1) * nx + i] = u_star[(ny - 2) * nx + i];
            v_star[i] = v_star[nx + i];
            v_star[(ny - 1) * nx + i] = v_star[(ny - 2) * nx + i];
        }

        // ============================================================
        // STEP 2: Solve Poisson equation for pressure
        // ============================================================
        double rho = field->rho[0];
        if (rho < 1e-10) rho = 1.0;

        // Compute RHS: divergence of intermediate velocity
        for (size_t j = 1; j < ny - 1; j++) {
#if USE_AVX
            size_t i;
            __m256d rho_dt_vec = _mm256_set1_pd(rho / dt);

            for (i = 1; i + 3 < nx - 1; i += 4) {
                size_t idx = j * nx + i;

                __m256d u_xp = _mm256_loadu_pd(&u_star[idx + 1]);
                __m256d u_xm = _mm256_loadu_pd(&u_star[idx - 1]);
                __m256d v_yp = _mm256_loadu_pd(&v_star[idx + nx]);
                __m256d v_ym = _mm256_loadu_pd(&v_star[idx - nx]);

                __m256d du_dx = _mm256_div_pd(_mm256_sub_pd(u_xp, u_xm), two_dx_vec);
                __m256d dv_dy = _mm256_div_pd(_mm256_sub_pd(v_yp, v_ym), two_dy_vec);

                __m256d divergence = _mm256_add_pd(du_dx, dv_dy);
                __m256d rhs_val = _mm256_mul_pd(rho_dt_vec, divergence);

                _mm256_storeu_pd(&rhs[idx], rhs_val);
            }

            for (; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double du_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                rhs[idx] = (rho / dt) * (du_dx + dv_dy);
            }
#else
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                double du_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                rhs[idx] = (rho / dt) * (du_dx + dv_dy);
            }
#endif
        }

        // Solve Poisson equation using optimized SOR
        int poisson_iters = solve_poisson_sor_simd(p_new, rhs, nx, ny, dx, dy,
                                                    POISSON_MAX_ITER, POISSON_TOLERANCE);

        if (poisson_iters < 0) {
            for (size_t idx = 0; idx < size; idx++) {
                p_new[idx] = field->p[idx] - 0.1 * dt * rhs[idx];
            }
        }

        // ============================================================
        // STEP 3: Corrector - Project velocity to be divergence-free
        // ============================================================
        double dt_rho = dt / rho;

        for (size_t j = 1; j < ny - 1; j++) {
#if USE_AVX
            size_t i;
            __m256d dt_rho_vec = _mm256_set1_pd(dt_rho);

            for (i = 1; i + 3 < nx - 1; i += 4) {
                size_t idx = j * nx + i;

                __m256d u_s = _mm256_loadu_pd(&u_star[idx]);
                __m256d v_s = _mm256_loadu_pd(&v_star[idx]);

                __m256d p_xp = _mm256_loadu_pd(&p_new[idx + 1]);
                __m256d p_xm = _mm256_loadu_pd(&p_new[idx - 1]);
                __m256d p_yp = _mm256_loadu_pd(&p_new[idx + nx]);
                __m256d p_ym = _mm256_loadu_pd(&p_new[idx - nx]);

                __m256d dp_dx = _mm256_div_pd(_mm256_sub_pd(p_xp, p_xm), two_dx_vec);
                __m256d dp_dy = _mm256_div_pd(_mm256_sub_pd(p_yp, p_ym), two_dy_vec);

                __m256d u_new = _mm256_sub_pd(u_s, _mm256_mul_pd(dt_rho_vec, dp_dx));
                __m256d v_new = _mm256_sub_pd(v_s, _mm256_mul_pd(dt_rho_vec, dp_dy));

                // Clamp
                u_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, u_new));
                v_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, v_new));

                _mm256_storeu_pd(&field->u[idx], u_new);
                _mm256_storeu_pd(&field->v[idx], v_new);
            }

            for (; i < nx - 1; i++) {
                size_t idx = j * nx + i;

                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);

                field->u[idx] = u_star[idx] - dt_rho * dp_dx;
                field->v[idx] = v_star[idx] - dt_rho * dp_dy;

                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
            }
#else
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;

                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);

                field->u[idx] = u_star[idx] - dt_rho * dp_dx;
                field->v[idx] = v_star[idx] - dt_rho * dp_dy;

                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
            }
#endif
        }

        // Update pressure
        memcpy(field->p, p_new, size * sizeof(double));

        // Apply boundary conditions
        apply_boundary_conditions(field, grid);

        // Check for NaN
        int has_nan = 0;
        for (size_t k = 0; k < size; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                has_nan = 1;
                break;
            }
        }

        if (has_nan) {
            printf("Warning: NaN detected in optimized projection solver at iteration %d\n", iter);
            break;
        }
    }

    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(p_new);
    cfd_free(rhs);
}
