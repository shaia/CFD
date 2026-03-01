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
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include "../boundary_copy_utils.h"

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
    double* w_star;
    double* p_new;
    double* rhs;
    double* u_new;  /* used as p_temp for Poisson solver */
    size_t nx;
    size_t ny;
    size_t nz;
    size_t stride_z;
    size_t k_start;
    size_t k_end;
    double inv_2dz;
    double inv_dz2;
    int initialized;
    int iter_count;
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

    /* Verify SIMD CG Poisson solver is available before allocating resources */
    poisson_solver_t* test_solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);
    if (!test_solver) {
        fprintf(stderr, "projection_simd_init: SIMD CG Poisson solver not available\n");
        return CFD_ERROR_UNSUPPORTED;
    }
    poisson_solver_destroy(test_solver);

    projection_simd_context* ctx =
        (projection_simd_context*)cfd_calloc(1, sizeof(projection_simd_context));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->nx = grid->nx;
    ctx->ny = grid->ny;
    ctx->nz = grid->nz;
    size_t size = ctx->nx * ctx->ny * grid->nz * sizeof(double);

    /* Reject non-uniform z-spacing (solver uses constant dz) */
    if (grid->nz > 1 && grid->dz) {
        for (size_t kk = 1; kk < grid->nz - 1; kk++) {
            if (fabs(grid->dz[kk] - grid->dz[0]) > 1e-14) {
                cfd_free(ctx);
                return CFD_ERROR_INVALID;
            }
        }
    }

    size_t plane = ctx->nx * ctx->ny;
    ctx->stride_z = (grid->nz > 1) ? plane : 0;
    ctx->k_start  = (grid->nz > 1) ? 1 : 0;
    ctx->k_end    = (grid->nz > 1) ? (grid->nz - 1) : 1;
    double dz = (grid->nz > 1 && grid->dz) ? grid->dz[0] : 0.0;
    ctx->inv_2dz  = (grid->nz > 1 && grid->dz) ? 1.0 / (2.0 * dz) : 0.0;
    ctx->inv_dz2  = (grid->nz > 1 && grid->dz) ? 1.0 / (dz * dz) : 0.0;

    ctx->u_star = (double*)cfd_aligned_malloc(size);
    ctx->v_star = (double*)cfd_aligned_malloc(size);
    ctx->w_star = (double*)cfd_aligned_malloc(size);
    ctx->p_new  = (double*)cfd_aligned_malloc(size);
    ctx->rhs    = (double*)cfd_aligned_malloc(size);
    ctx->u_new  = (double*)cfd_aligned_malloc(size);

    if (!ctx->u_star || !ctx->v_star || !ctx->w_star || !ctx->p_new ||
        !ctx->rhs || !ctx->u_new) {
        if (ctx->u_star) {
            cfd_aligned_free(ctx->u_star);
        }
        if (ctx->v_star) {
            cfd_aligned_free(ctx->v_star);
        }
        if (ctx->w_star) {
            cfd_aligned_free(ctx->w_star);
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
            cfd_aligned_free(ctx->w_star);
            cfd_aligned_free(ctx->p_new);
            cfd_aligned_free(ctx->rhs);
            cfd_aligned_free(ctx->u_new);
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
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    projection_simd_context* ctx = (projection_simd_context*)solver->context;

    // Verify context matches current grid
    if (ctx->nx != field->nx || ctx->ny != field->ny || ctx->nz != field->nz) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny * ctx->nz;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dz = (ctx->nz > 1 && grid->dz) ? grid->dz[0] : 0.0;
    double dt = params->dt;
    double nu = params->mu;  // Viscosity (treated as kinematic for ρ=1)

    double* u_star = ctx->u_star;
    double* v_star = ctx->v_star;
    double* w_star = ctx->w_star;
    double* p_new = ctx->p_new;
    double* rhs = ctx->rhs;

    // Copy current field values to work buffers (includes boundaries)
    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(w_star, field->w, size * sizeof(double));
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

    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        size_t k_off = k * ctx->stride_z;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k_off + IDX_2D(i, j, nx);

                double u = field->u[idx];
                double v = field->v[idx];
                double w = field->w[idx];

                // Convective terms: -u·∇u (central differences)
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double du_dz = (field->u[idx + ctx->stride_z] - field->u[idx - ctx->stride_z]) *
                               ctx->inv_2dz;

                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
                double dv_dz = (field->v[idx + ctx->stride_z] - field->v[idx - ctx->stride_z]) *
                               ctx->inv_2dz;

                double dw_dx = (field->w[idx + 1] - field->w[idx - 1]) / (2.0 * dx);
                double dw_dy = (field->w[idx + nx] - field->w[idx - nx]) / (2.0 * dy);
                double dw_dz = (field->w[idx + ctx->stride_z] - field->w[idx - ctx->stride_z]) *
                               ctx->inv_2dz;

                double conv_u = (u * du_dx) + (v * du_dy) + (w * du_dz);
                double conv_v = (u * dv_dx) + (v * dv_dy) + (w * dv_dz);
                double conv_w = (u * dw_dx) + (v * dw_dy) + (w * dw_dz);

                // Viscous terms: ν∇²u
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                double d2u_dz2 = (field->u[idx + ctx->stride_z] - 2.0 * u +
                                  field->u[idx - ctx->stride_z]) * ctx->inv_dz2;

                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);
                double d2v_dz2 = (field->v[idx + ctx->stride_z] - 2.0 * v +
                                  field->v[idx - ctx->stride_z]) * ctx->inv_dz2;

                double d2w_dx2 = (field->w[idx + 1] - 2.0 * w + field->w[idx - 1]) / (dx * dx);
                double d2w_dy2 = (field->w[idx + nx] - 2.0 * w + field->w[idx - nx]) / (dy * dy);
                double d2w_dz2 = (field->w[idx + ctx->stride_z] - 2.0 * w +
                                  field->w[idx - ctx->stride_z]) * ctx->inv_dz2;

                double visc_u = nu * (d2u_dx2 + d2u_dy2 + d2u_dz2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2 + d2v_dz2);
                double visc_w = nu * (d2w_dx2 + d2w_dy2 + d2w_dz2);

                // Source terms
                double source_u = 0.0;
                double source_v = 0.0;
                double source_w = 0.0;
                double x_coord = grid->x[i];
                double y_coord = grid->y[j];
                double z_coord = (ctx->nz > 1 && grid->z) ? grid->z[k] : 0.0;
                compute_source_terms(x_coord, y_coord, z_coord, ctx->iter_count, dt, params,
                                     &source_u, &source_v, &source_w);

                // Intermediate velocity (without pressure gradient)
                u_star[idx] = u + (dt * (-conv_u + visc_u + source_u));
                v_star[idx] = v + (dt * (-conv_v + visc_v + source_v));
                w_star[idx] = w + (dt * (-conv_w + visc_w + source_w));

                // Limit velocities
                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
                w_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, w_star[idx]));
            }
        }
    }

    // Copy boundary values from field to star arrays
    copy_boundary_velocities_3d(u_star, v_star, w_star, field->u, field->v, field->w,
                                nx, ny, ctx->nz);

    // ============================================================
    // STEP 2: Solve Poisson equation for pressure
    // ∇²p = (ρ/dt) * ∇·u*
    // ============================================================

    double rho = field->rho[0];
    if (rho < 1e-10) {
        rho = 1.0;
    }

    // Compute RHS: divergence of intermediate velocity
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        size_t k_off = k * ctx->stride_z;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k_off + IDX_2D(i, j, nx);

                double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                double dw_star_dz = (w_star[idx + ctx->stride_z] -
                                     w_star[idx - ctx->stride_z]) * ctx->inv_2dz;

                double divergence = du_star_dx + dv_star_dy + dw_star_dz;
                rhs[idx] = (rho / dt) * divergence;
            }
        }
    }

    // Use SIMD Poisson solver (Conjugate Gradient with SIMD)
    // ctx->u_new is used as temp buffer for the Poisson solver
    int poisson_iters = poisson_solve_3d(p_new, ctx->u_new, rhs, nx, ny, ctx->nz,
                                         dx, dy, dz, POISSON_SOLVER_CG_SIMD);

    if (poisson_iters < 0) {
        return CFD_ERROR_MAX_ITER;
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
    __m256d dt_rho_vec      = _mm256_set1_pd(dt_over_rho);
    __m256d inv_2dx_vec     = _mm256_set1_pd(inv_2dx);
    __m256d inv_2dy_vec     = _mm256_set1_pd(inv_2dy);
    __m256d inv_2dz_vec     = _mm256_set1_pd(ctx->inv_2dz);
    __m256d max_vel_vec     = _mm256_set1_pd(MAX_VELOCITY);
    __m256d neg_max_vel_vec = _mm256_set1_pd(-MAX_VELOCITY);
#endif

    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        size_t k_off = k * ctx->stride_z;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;

#if USE_AVX
            // SIMD loop - process 4 cells at once
            for (; i + 4 <= nx - 1; i += 4) {
                size_t idx = k_off + IDX_2D(i, j, nx);

                // Load pressure neighbors for gradient computation
                __m256d p_xp = _mm256_loadu_pd(&p_new[idx + 1]);
                __m256d p_xm = _mm256_loadu_pd(&p_new[idx - 1]);
                __m256d p_yp = _mm256_loadu_pd(&p_new[idx + nx]);
                __m256d p_ym = _mm256_loadu_pd(&p_new[idx - nx]);
                __m256d p_zp = _mm256_loadu_pd(&p_new[idx + ctx->stride_z]);
                __m256d p_zm = _mm256_loadu_pd(&p_new[idx - ctx->stride_z]);

                // Compute pressure gradients
                __m256d dp_dx = _mm256_mul_pd(_mm256_sub_pd(p_xp, p_xm), inv_2dx_vec);
                __m256d dp_dy = _mm256_mul_pd(_mm256_sub_pd(p_yp, p_ym), inv_2dy_vec);
                __m256d dp_dz = _mm256_mul_pd(_mm256_sub_pd(p_zp, p_zm), inv_2dz_vec);

                // Load intermediate velocities
                __m256d u_s = _mm256_loadu_pd(&u_star[idx]);
                __m256d v_s = _mm256_loadu_pd(&v_star[idx]);
                __m256d w_s = _mm256_loadu_pd(&w_star[idx]);

                // Corrector: u = u* - (dt/rho) * dp/dx
                __m256d u_new = _mm256_sub_pd(u_s, _mm256_mul_pd(dt_rho_vec, dp_dx));
                __m256d v_new = _mm256_sub_pd(v_s, _mm256_mul_pd(dt_rho_vec, dp_dy));
                __m256d w_new = _mm256_sub_pd(w_s, _mm256_mul_pd(dt_rho_vec, dp_dz));

                // Clamp velocities to [-MAX_VELOCITY, MAX_VELOCITY]
                u_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, u_new));
                v_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, v_new));
                w_new = _mm256_max_pd(neg_max_vel_vec, _mm256_min_pd(max_vel_vec, w_new));

                // Store results
                _mm256_storeu_pd(&field->u[idx], u_new);
                _mm256_storeu_pd(&field->v[idx], v_new);
                _mm256_storeu_pd(&field->w[idx], w_new);
            }
#endif

            // Scalar remainder
            for (; i < nx - 1; i++) {
                size_t idx = k_off + IDX_2D(i, j, nx);

                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) * inv_2dx;
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) * inv_2dy;
                double dp_dz = (p_new[idx + ctx->stride_z] - p_new[idx - ctx->stride_z]) *
                               ctx->inv_2dz;

                field->u[idx] = u_star[idx] - (dt_over_rho * dp_dx);
                field->v[idx] = v_star[idx] - (dt_over_rho * dp_dy);
                field->w[idx] = w_star[idx] - (dt_over_rho * dp_dz);

                // Limit velocities
                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
                field->w[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->w[idx]));
            }
        }
    }

    // Update pressure field
    memcpy(field->p, p_new, size * sizeof(double));

    // Copy boundary velocity values from star arrays (which have caller's BCs)
    copy_boundary_velocities_3d(field->u, field->v, field->w, u_star, v_star, w_star,
                                nx, ny, ctx->nz);

    // Check for NaN
    for (size_t n = 0; n < size; n++) {
        if (!isfinite(field->u[n]) || !isfinite(field->v[n]) ||
            !isfinite(field->w[n]) || !isfinite(field->p[n])) {
            return CFD_ERROR_DIVERGED;
        }
    }

    ctx->iter_count++;

    if (stats) {
        stats->iterations = 1;
    }

    return CFD_SUCCESS;
}
