// Enable C11 features for aligned_alloc
#define _POSIX_C_SOURCE 200112L
#define _ISOC11_SOURCE

#include "cfd_status.h"
#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

// Physical stability limits
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DIVERGENCE_LIMIT        10.0
#define DT_CONSERVATIVE_LIMIT       0.0001
#define UPDATE_LIMIT                1.0
#define PRESSURE_UPDATE_FACTOR      0.1

typedef struct {
    double* u_new;
    double* v_new;
    double* p_new;
    double* dx_inv;
    double* dy_inv;
    size_t nx;
    size_t ny;
    int initialized;
} ExplicitEulerSIMDContext;

// Public API functions
cfd_status_t explicit_euler_simd_init(Solver* solver, const Grid* grid, const SolverParams* params);
void explicit_euler_simd_destroy(Solver* solver);
cfd_status_t explicit_euler_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                      const SolverParams* params, SolverStats* stats);

cfd_status_t explicit_euler_simd_init(Solver* solver, const Grid* grid,
                                      const SolverParams* params) {
    (void)params;  // Unused
    if (!solver || !grid)
        return CFD_ERROR_INVALID;

    ExplicitEulerSIMDContext* ctx =
        (ExplicitEulerSIMDContext*)cfd_calloc(1, sizeof(ExplicitEulerSIMDContext));
    if (!ctx)
        return CFD_ERROR_NOMEM;

    ctx->nx = grid->nx;
    ctx->ny = grid->ny;
    size_t field_size = ctx->nx * ctx->ny * sizeof(double);

    ctx->u_new = (double*)cfd_aligned_malloc(field_size);
    ctx->v_new = (double*)cfd_aligned_malloc(field_size);
    ctx->p_new = (double*)cfd_aligned_malloc(field_size);
    ctx->dx_inv = (double*)cfd_aligned_malloc(ctx->nx * sizeof(double));
    ctx->dy_inv = (double*)cfd_aligned_malloc(ctx->ny * sizeof(double));

    if (!ctx->u_new || !ctx->v_new || !ctx->p_new || !ctx->dx_inv || !ctx->dy_inv) {
        if (ctx->u_new)
            cfd_aligned_free(ctx->u_new);
        if (ctx->v_new)
            cfd_aligned_free(ctx->v_new);
        if (ctx->p_new)
            cfd_aligned_free(ctx->p_new);
        if (ctx->dx_inv)
            cfd_aligned_free(ctx->dx_inv);
        if (ctx->dy_inv)
            cfd_aligned_free(ctx->dy_inv);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    // Pre-compute inverses
    for (size_t i = 0; i < ctx->nx; i++) {
        ctx->dx_inv[i] = (i < ctx->nx - 1) ? 1.0 / (2.0 * grid->dx[i]) : 0.0;
    }
    for (size_t j = 0; j < ctx->ny; j++) {
        ctx->dy_inv[j] = (j < ctx->ny - 1) ? 1.0 / (2.0 * grid->dy[j]) : 0.0;
    }

    ctx->initialized = 1;
    solver->context = ctx;

#if USE_AVX
    printf("Explicit Euler SIMD: AVX2 optimizations ENABLED (Profiling Mode)\n");
#else
    printf("Explicit Euler SIMD: AVX2 optimizations DISABLED (Profiling Mode)\n");
#endif

    return CFD_SUCCESS;
}

void explicit_euler_simd_destroy(Solver* solver) {
    if (solver && solver->context) {
        ExplicitEulerSIMDContext* ctx = (ExplicitEulerSIMDContext*)solver->context;
        if (ctx->initialized) {
            cfd_aligned_free(ctx->u_new);
            cfd_aligned_free(ctx->v_new);
            cfd_aligned_free(ctx->p_new);
            cfd_aligned_free(ctx->dx_inv);
            cfd_aligned_free(ctx->dy_inv);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

#if USE_AVX
static inline __m256d vector_fmax(__m256d a, __m256d b) {
    return _mm256_max_pd(a, b);
}
static inline __m256d vector_fmin(__m256d a, __m256d b) {
    return _mm256_min_pd(a, b);
}
#endif

cfd_status_t explicit_euler_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                      const SolverParams* params, SolverStats* stats) {
    if (!solver || !solver->context || !field || !grid || !params)
        return CFD_ERROR_INVALID;

    ExplicitEulerSIMDContext* ctx = (ExplicitEulerSIMDContext*)solver->context;

    if (field->nx != ctx->nx || field->ny != ctx->ny) {
        return CFD_ERROR_INVALID;
    }

    // Use conservative time step to match basic solver stability
    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

#if USE_AVX
    __m256d dt_vec = _mm256_set1_pd(conservative_dt);
    __m256d max_deriv = _mm256_set1_pd(MAX_DERIVATIVE_LIMIT);
    __m256d min_deriv = _mm256_set1_pd(-MAX_DERIVATIVE_LIMIT);
    __m256d max_diverg = _mm256_set1_pd(MAX_DIVERGENCE_LIMIT);
    __m256d min_diverg = _mm256_set1_pd(-MAX_DIVERGENCE_LIMIT);
    __m256d max_vel_limit = _mm256_set1_pd(MAX_VELOCITY_LIMIT);
    __m256d min_vel_limit = _mm256_set1_pd(-MAX_VELOCITY_LIMIT);
    __m256d one_vec = _mm256_set1_pd(UPDATE_LIMIT);
    __m256d neg_one_vec = _mm256_set1_pd(-UPDATE_LIMIT);
    __m256d pressure_factor = _mm256_set1_pd(-PRESSURE_UPDATE_FACTOR);
    __m256d two = _mm256_set1_pd(2.0);
    __m256d four = _mm256_set1_pd(4.0);
    __m256d epsilon = _mm256_set1_pd(1e-10);
    __m256d mu_vec = _mm256_set1_pd(params->mu);
    __m256d zero = _mm256_setzero_pd();
#endif


    // Copy current state to temp buffers (SIMD copy)
    size_t size = ctx->nx * ctx->ny;
    memcpy(ctx->u_new, field->u, size * sizeof(double));
    memcpy(ctx->v_new, field->v, size * sizeof(double));
    memcpy(ctx->p_new, field->p, size * sizeof(double));

    // No internal loop for step function!
    // for (int iter = 0; iter < params->max_iter; iter++) { <--- REMOVED

    for (size_t j = 1; j < ctx->ny - 1; j++) {
#if USE_AVX
        double dy2 = grid->dy[j] * grid->dy[j];
        __m256d dy_inv_val = _mm256_set1_pd(ctx->dy_inv[j]);
        __m256d dy2_val = _mm256_set1_pd(dy2);
        __m256d dy2_recip = _mm256_div_pd(one_vec, dy2_val);  // Hoisted division

        size_t i;
        for (i = 1; i + 3 < ctx->nx - 1; i += 4) {
            size_t idx = j * ctx->nx + i;

            // Load data
            __m256d u = _mm256_loadu_pd(&field->u[idx]);
            __m256d v = _mm256_loadu_pd(&field->v[idx]);
            // p and rho loaded on demand or here?
            // derivatives need neighbors.

            __m256d rho = _mm256_loadu_pd(&field->rho[idx]);
            __m256d rho_inv =
                _mm256_div_pd(one_vec, _mm256_max_pd(rho, epsilon));  // One division per vector

            __m256d dx_inv_val = _mm256_loadu_pd(&ctx->dx_inv[i]);

            // U derivatives
            __m256d u_xp = _mm256_loadu_pd(&field->u[idx + 1]);
            __m256d u_xm = _mm256_loadu_pd(&field->u[idx - 1]);
            __m256d u_yp = _mm256_loadu_pd(&field->u[idx + ctx->nx]);
            __m256d u_ym = _mm256_loadu_pd(&field->u[idx - ctx->nx]);

            __m256d du_dx = _mm256_mul_pd(_mm256_sub_pd(u_xp, u_xm), dx_inv_val);
            __m256d du_dy = _mm256_mul_pd(_mm256_sub_pd(u_yp, u_ym), dy_inv_val);

            du_dx = vector_fmax(min_deriv, vector_fmin(max_deriv, du_dx));
            du_dy = vector_fmax(min_deriv, vector_fmin(max_deriv, du_dy));

            // V derivatives
            __m256d v_xp = _mm256_loadu_pd(&field->v[idx + 1]);
            __m256d v_xm = _mm256_loadu_pd(&field->v[idx - 1]);
            __m256d v_yp = _mm256_loadu_pd(&field->v[idx + ctx->nx]);
            __m256d v_ym = _mm256_loadu_pd(&field->v[idx - ctx->nx]);

            __m256d dv_dx = _mm256_mul_pd(_mm256_sub_pd(v_xp, v_xm), dx_inv_val);
            __m256d dv_dy = _mm256_mul_pd(_mm256_sub_pd(v_yp, v_ym), dy_inv_val);

            dv_dx = vector_fmax(min_deriv, vector_fmin(max_deriv, dv_dx));
            dv_dy = vector_fmax(min_deriv, vector_fmin(max_deriv, dv_dy));

            // Pressure derivatives
            __m256d p_xp = _mm256_loadu_pd(&field->p[idx + 1]);
            __m256d p_xm = _mm256_loadu_pd(&field->p[idx - 1]);
            __m256d p_yp = _mm256_loadu_pd(&field->p[idx + ctx->nx]);
            __m256d p_ym = _mm256_loadu_pd(&field->p[idx - ctx->nx]);

            __m256d dp_dx = _mm256_mul_pd(_mm256_sub_pd(p_xp, p_xm), dx_inv_val);
            __m256d dp_dy = _mm256_mul_pd(_mm256_sub_pd(p_yp, p_ym), dy_inv_val);

            dp_dx = vector_fmax(min_deriv, vector_fmin(max_deriv, dp_dx));
            dp_dy = vector_fmax(min_deriv, vector_fmin(max_deriv, dp_dy));

            // Second derivatives
            // dx_inv stores 1/(2*dx). We need 1/(dx^2) for second derivative.
            // (1/(2*dx))^2 * 4 = 1/(4*dx^2) * 4 = 1/(dx^2)
            __m256d inv_dx_sq = _mm256_mul_pd(four, _mm256_mul_pd(dx_inv_val, dx_inv_val));

            __m256d d2u_dx2 = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_add_pd(u_xp, u_xm), _mm256_mul_pd(two, u)), inv_dx_sq);
            __m256d d2u_dy2 = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_add_pd(u_yp, u_ym), _mm256_mul_pd(two, u)), dy2_recip);
            __m256d d2v_dx2 = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_add_pd(v_xp, v_xm), _mm256_mul_pd(two, v)), inv_dx_sq);
            __m256d d2v_dy2 = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_add_pd(v_yp, v_ym), _mm256_mul_pd(two, v)), dy2_recip);

            // Nu
            __m256d nu = _mm256_min_pd(
                one_vec, _mm256_mul_pd(mu_vec, rho_inv));  // Multiplication instead of division

            // Update U
            __m256d term_pres_x = _mm256_mul_pd(dp_dx, rho_inv);  // Multiplication
            __m256d term_visc_u = _mm256_mul_pd(nu, _mm256_add_pd(d2u_dx2, d2u_dy2));
            __m256d conv_u = _mm256_add_pd(_mm256_mul_pd(u, du_dx), _mm256_mul_pd(v, du_dy));
            __m256d du =
                _mm256_mul_pd(dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_u, term_pres_x),
                                                    _mm256_sub_pd(zero, conv_u)));

            // Update V
            __m256d term_pres_y = _mm256_mul_pd(dp_dy, rho_inv);  // Multiplication
            __m256d term_visc_v = _mm256_mul_pd(nu, _mm256_add_pd(d2v_dx2, d2v_dy2));
            __m256d conv_v = _mm256_add_pd(_mm256_mul_pd(u, dv_dx), _mm256_mul_pd(v, dv_dy));
            __m256d dv =
                _mm256_mul_pd(dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_v, term_pres_y),
                                                    _mm256_sub_pd(zero, conv_v)));

            du = vector_fmin(one_vec, vector_fmax(neg_one_vec, du));
            dv = vector_fmin(one_vec, vector_fmax(neg_one_vec, dv));

            __m256d u_next = _mm256_add_pd(u, du);
            __m256d v_next = _mm256_add_pd(v, dv);

            u_next = vector_fmin(max_vel_limit, vector_fmax(min_vel_limit, u_next));
            v_next = vector_fmin(max_vel_limit, vector_fmax(min_vel_limit, v_next));

            // Pressure update
            __m256d divergence = _mm256_add_pd(du_dx, dv_dy);
            divergence = vector_fmin(max_diverg, vector_fmax(min_diverg, divergence));
            __m256d p = _mm256_loadu_pd(&field->p[idx]);
            __m256d dp = _mm256_mul_pd(
                dt_vec, _mm256_mul_pd(pressure_factor, _mm256_mul_pd(rho, divergence)));
            dp = vector_fmin(one_vec, vector_fmax(neg_one_vec, dp));
            __m256d p_next = _mm256_add_pd(p, dp);

            _mm256_storeu_pd(&ctx->u_new[idx], u_next);
            _mm256_storeu_pd(&ctx->v_new[idx], v_next);
            _mm256_storeu_pd(&ctx->p_new[idx], p_next);
        }
#else
        // Scalar fallback implementation
        for (size_t i = 1; i < ctx->nx - 1; i++) {
            size_t idx = j * ctx->nx + i;

            // Scalar derivatives
            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
            double du_dy =
                (field->u[idx + ctx->nx] - field->u[idx - ctx->nx]) / (2.0 * grid->dy[j]);
            double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
            double dv_dy =
                (field->v[idx + ctx->nx] - field->v[idx - ctx->nx]) / (2.0 * grid->dy[j]);

            double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
            double dp_dy =
                (field->p[idx + ctx->nx] - field->p[idx - ctx->nx]) / (2.0 * grid->dy[j]);

            double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) /
                             (grid->dx[i] * grid->dx[i]);
            double d2u_dy2 =
                (field->u[idx + ctx->nx] - 2.0 * field->u[idx] + field->u[idx - ctx->nx]) /
                (grid->dy[j] * grid->dy[j]);
            double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) /
                             (grid->dx[i] * grid->dx[i]);
            double d2v_dy2 =
                (field->v[idx + ctx->nx] - 2.0 * field->v[idx] + field->v[idx - ctx->nx]) /
                (grid->dy[j] * grid->dy[j]);

            double rho = fmax(field->rho[idx], 1e-10);
            double nu = fmin(params->mu / rho, 1.0);

            double source_u = 0.0;
            double source_v = 0.0;
            if (params->source_amplitude_u > 0) {
                source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]);
                source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]);
            }

            // Updates
            double du = conservative_dt * (-field->u[idx] * du_dx - field->v[idx] * du_dy -
                                           dp_dx / rho + nu * (d2u_dx2 + d2u_dy2) + source_u);
            double dv = conservative_dt * (-field->u[idx] * dv_dx - field->v[idx] * dv_dy -
                                           dp_dy / rho + nu * (d2v_dx2 + d2v_dy2) + source_v);

            // Limit changes
            du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
            dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));

            ctx->u_new[idx] = field->u[idx] + du;
            ctx->v_new[idx] = field->v[idx] + dv;

            // Limit velocity
            ctx->u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, ctx->u_new[idx]));
            ctx->v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, ctx->v_new[idx]));

            // Pressure update (simple)
            double divergence = du_dx + dv_dy;
            divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
            double dp = -PRESSURE_UPDATE_FACTOR * conservative_dt * rho * divergence;
            dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));
            ctx->p_new[idx] = field->p[idx] + dp;
        }
#endif
    }

    // Apply boundary, check NaNs, etc.
    memcpy(field->u, ctx->u_new, size * sizeof(double));
    memcpy(field->v, ctx->v_new, size * sizeof(double));
    memcpy(field->p, ctx->p_new, size * sizeof(double));
    apply_boundary_conditions(field, grid);

    if (stats) {
        stats->iterations = 1;
    }

    return CFD_SUCCESS;
}