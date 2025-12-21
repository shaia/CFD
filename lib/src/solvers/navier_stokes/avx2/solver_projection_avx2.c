/**
 * Optimized Projection Method NSSolver (Chorin's Method) with SIMD + OpenMP
 *
 * This implementation combines SIMD vectorization (AVX2) with OpenMP
 * parallelization for maximum performance on multi-core CPUs.
 *
 * - Predictor step: OpenMP parallelized (scalar inner loops)
 * - Corrector step: OpenMP parallelized with AVX2 SIMD inner loops
 * - Poisson solver: Uses SIMD Poisson solver for pressure computation
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* AVX2 detection
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

// Physical limits
#define MAX_VELOCITY 100.0

typedef struct {
    double* u_star;
    double* v_star;
    double* p_new;
    double* rhs;
    double* u_new;
    double* v_new;
    size_t nx;
    size_t ny;
    int initialized;
} projection_simd_context;

// Public API
cfd_status_t projection_simd_init(struct NSSolver* solver, const grid* grid,
                                  const ns_solver_params_t* params);
void projection_simd_destroy(struct NSSolver* solver);
cfd_status_t projection_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params, ns_solver_stats_t* stats);

cfd_status_t projection_simd_init(struct NSSolver* solver, const grid* grid,
                                  const ns_solver_params_t* params) {
    (void)params;
    if (!solver || !grid) {
        return CFD_ERROR_INVALID;
    }

    projection_simd_context* ctx =
        (projection_simd_context*)cfd_calloc(1, sizeof(projection_simd_context));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->nx = grid->nx;
    ctx->ny = grid->ny;
    size_t size = ctx->nx * ctx->ny * sizeof(double);

    ctx->u_star = (double*)cfd_aligned_malloc(size);
    ctx->v_star = (double*)cfd_aligned_malloc(size);
    ctx->p_new = (double*)cfd_aligned_malloc(size);
    ctx->rhs = (double*)cfd_aligned_malloc(size);
    ctx->u_new = (double*)cfd_aligned_malloc(size);
    ctx->v_new = (double*)cfd_aligned_malloc(size);

    if (!ctx->u_star || !ctx->v_star || !ctx->p_new || !ctx->rhs || !ctx->u_new || !ctx->v_new) {
        if (ctx->u_star) {
            cfd_aligned_free(ctx->u_star);
        }
        if (ctx->v_star) {
            cfd_aligned_free(ctx->v_star);
        }
        if (ctx->p_new) {
            cfd_aligned_free(ctx->p_new);
        }
        if (ctx->rhs) {
            cfd_aligned_free(ctx->rhs);
        }
        if (ctx->u_new) {
            cfd_aligned_free(ctx->u_new);
        }
        if (ctx->v_new) {
            cfd_aligned_free(ctx->v_new);
        }
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

void projection_simd_destroy(struct NSSolver* solver) {
    if (solver && solver->context) {
        projection_simd_context* ctx = (projection_simd_context*)solver->context;
        if (ctx->initialized) {
            cfd_aligned_free(ctx->u_star);
            cfd_aligned_free(ctx->v_star);
            cfd_aligned_free(ctx->p_new);
            cfd_aligned_free(ctx->rhs);
            cfd_aligned_free(ctx->u_new);
            cfd_aligned_free(ctx->v_new);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

cfd_status_t projection_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !solver->context || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    projection_simd_context* ctx = (projection_simd_context*)solver->context;

    // Verify context matches current grid
    if (ctx->nx != field->nx || ctx->ny != field->ny) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;  // Viscosity (treated as kinematic for ρ=1)

    double* u_star = ctx->u_star;
    double* v_star = ctx->v_star;
    double* p_new = ctx->p_new;
    double* rhs = ctx->rhs;

    // Copy current field values to work buffers (includes boundaries)
    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(p_new, field->p, size * sizeof(double));
    memset(rhs, 0, size * sizeof(double));

    // ============================================================
    // STEP 1: Predictor - Compute intermediate velocity u*
    // (OpenMP parallelized outer loop, scalar inner loop)
    // ============================================================
    int ny_int = (int)ny;
    int nx_int = (int)nx;
    int jj;
    (void)nx_int;  /* suppress unused variable warning */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = (j * nx) + i;

            double u = field->u[idx];
            double v = field->v[idx];

            // Convective terms: -u·∇u (central differences)
            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
            double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
            double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
            double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

            double conv_u = (u * du_dx) + (v * du_dy);
            double conv_v = (u * dv_dx) + (v * dv_dy);

            // Viscous terms: ν∇²u
            double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
            double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
            double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
            double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);

            double visc_u = nu * (d2u_dx2 + d2u_dy2);
            double visc_v = nu * (d2v_dx2 + d2v_dy2);

            // Source terms (for maintaining flow)
            double source_u = 0.0;
            double source_v = 0.0;
            if (params->source_amplitude_u > 0) {
                double x = grid->x[i];
                double y = grid->y[j];
                source_u = params->source_amplitude_u * sin(M_PI * y);
                source_v = params->source_amplitude_v * sin(2.0 * M_PI * x);
            }

            // Intermediate velocity (without pressure gradient)
            u_star[idx] = u + (dt * (-conv_u + visc_u + source_u));
            v_star[idx] = v + (dt * (-conv_v + visc_v + source_v));

            // Limit velocities
            u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
            v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
        }
    }

    // Apply boundary conditions to intermediate velocity (using SIMD backend)
    bc_apply_velocity_simd(u_star, v_star, nx, ny, BC_TYPE_NEUMANN);

    // ============================================================
    // STEP 2: Solve Poisson equation for pressure
    // ∇²p = (ρ/dt) * ∇·u*
    // ============================================================

    double rho = field->rho[0];
    if (rho < 1e-10) {
        rho = 1.0;
    }

    // Compute RHS: divergence of intermediate velocity
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = (j * nx) + i;

            double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
            double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);

            double divergence = du_star_dx + dv_star_dy;
            rhs[idx] = (rho / dt) * divergence;
        }
    }

    // Use SIMD Poisson solver (Red-Black SOR with SIMD)
    // ctx->u_new is used as temp buffer for the Poisson solver
    int poisson_iters = poisson_solve(p_new, ctx->u_new, rhs, nx, ny, dx, dy,
                                       DEFAULT_POISSON_SOLVER);

    if (poisson_iters < 0) {
        // Poisson solver didn't converge - use simple pressure update as fallback
        for (size_t idx = 0; idx < size; idx++) {
            p_new[idx] = field->p[idx] - (0.1 * dt * rhs[idx]);
        }
    }

    // ============================================================
    // STEP 3: Corrector - Project velocity to be divergence-free
    // u^(n+1) = u* - (dt/ρ) * ∇p
    // (OpenMP parallelized with SIMD inner loop)
    // ============================================================

    double dt_over_rho = dt / rho;
    double inv_2dx = 1.0 / (2.0 * dx);
    double inv_2dy = 1.0 / (2.0 * dy);

#if USE_AVX
    __m256d dt_rho_vec = _mm256_set1_pd(dt_over_rho);
    __m256d inv_2dx_vec = _mm256_set1_pd(inv_2dx);
    __m256d inv_2dy_vec = _mm256_set1_pd(inv_2dy);
    __m256d max_vel_vec = _mm256_set1_pd(MAX_VELOCITY);
    __m256d neg_max_vel_vec = _mm256_set1_pd(-MAX_VELOCITY);
#endif

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

#if USE_AVX
        // SIMD loop - process 4 cells at once
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = (j * nx) + i;

            // Load pressure neighbors for gradient computation
            __m256d p_xp = _mm256_loadu_pd(&p_new[idx + 1]);
            __m256d p_xm = _mm256_loadu_pd(&p_new[idx - 1]);
            __m256d p_yp = _mm256_loadu_pd(&p_new[idx + nx]);
            __m256d p_ym = _mm256_loadu_pd(&p_new[idx - nx]);

            // Compute pressure gradients: dp/dx = (p[i+1] - p[i-1]) / (2*dx)
            __m256d dp_dx = _mm256_mul_pd(_mm256_sub_pd(p_xp, p_xm), inv_2dx_vec);
            __m256d dp_dy = _mm256_mul_pd(_mm256_sub_pd(p_yp, p_ym), inv_2dy_vec);

            // Load intermediate velocities
            __m256d u_s = _mm256_loadu_pd(&u_star[idx]);
            __m256d v_s = _mm256_loadu_pd(&v_star[idx]);

            // Corrector: u = u* - (dt/rho) * dp/dx
            __m256d u_new = _mm256_sub_pd(u_s, _mm256_mul_pd(dt_rho_vec, dp_dx));
            __m256d v_new = _mm256_sub_pd(v_s, _mm256_mul_pd(dt_rho_vec, dp_dy));

            // Clamp velocities to [-MAX_VELOCITY, MAX_VELOCITY]
            u_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, u_new));
            v_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, v_new));

            // Store results
            _mm256_storeu_pd(&field->u[idx], u_new);
            _mm256_storeu_pd(&field->v[idx], v_new);
        }
#endif

        // Scalar remainder
        for (; i < nx - 1; i++) {
            size_t idx = (j * nx) + i;

            double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) * inv_2dx;
            double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) * inv_2dy;

            field->u[idx] = u_star[idx] - (dt_over_rho * dp_dx);
            field->v[idx] = v_star[idx] - (dt_over_rho * dp_dy);

            // Limit velocities
            field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
            field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
        }
    }

    // Update pressure field
    memcpy(field->p, p_new, size * sizeof(double));

    // Apply boundary conditions to final velocity
    apply_boundary_conditions(field, grid);

    // Check for NaN
    for (size_t k = 0; k < size; k++) {
        if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
            return CFD_ERROR_DIVERGED;
        }
    }

    if (stats) {
        stats->iterations = 1;
    }

    return CFD_SUCCESS;
}
