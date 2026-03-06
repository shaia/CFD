/**
 * Optimized Explicit Euler NSSolver with SIMD + OpenMP
 *
 * This implementation combines SIMD vectorization (AVX2) with OpenMP
 * parallelization for maximum performance on multi-core CPUs.
 *
 * - Outer loops are parallelized with OpenMP
 * - Inner loops use AVX2 SIMD intrinsics for vectorization
 *
 * Note: When AVX2 is not enabled at compile time (CFD_ENABLE_AVX2=OFF),
 * this solver uses scalar code paths. Use the base explicit_euler solver
 * for guaranteed scalar-only execution.
 */

// Enable C11 features for aligned_alloc
#define _POSIX_C_SOURCE 200809L
#define _ISOC11_SOURCE
#define _USE_MATH_DEFINES

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include "../boundary_copy_utils.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    double* w_new;
    double* p_new;
    double* dx_inv;
    double* dy_inv;
    size_t nx;
    size_t ny;
    size_t nz;
    size_t stride_z;
    size_t k_start;
    size_t k_end;
    double inv_2dz;
    double inv_dz2;
    int initialized;
} explicit_euler_simd_context;

// Public API functions
cfd_status_t explicit_euler_simd_init(struct NSSolver* solver, const grid* grid,
                                      const ns_solver_params_t* params);
void explicit_euler_simd_destroy(struct NSSolver* solver);
cfd_status_t explicit_euler_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params, ns_solver_stats_t* stats);

cfd_status_t explicit_euler_simd_init(struct NSSolver* solver, const grid* grid,
                                      const ns_solver_params_t* params) {
    (void)params;  // Unused
    if (!solver || !grid) {
        return CFD_ERROR_INVALID;
    }
    if (grid->nx < 3 || grid->ny < 3 || (grid->nz > 1 && grid->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    explicit_euler_simd_context* ctx =
        (explicit_euler_simd_context*)cfd_calloc(1, sizeof(explicit_euler_simd_context));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->nx = grid->nx;
    ctx->ny = grid->ny;
    ctx->nz = grid->nz;
    size_t field_size = ctx->nx * ctx->ny * ctx->nz * sizeof(double);

    /* Reject non-uniform z-spacing (solver uses constant inv_2dz/inv_dz2) */
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
    ctx->inv_2dz  = (grid->nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    ctx->inv_dz2  = (grid->nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    ctx->u_new = (double*)cfd_aligned_malloc(field_size);
    ctx->v_new = (double*)cfd_aligned_malloc(field_size);
    ctx->w_new = (double*)cfd_aligned_malloc(field_size);
    ctx->p_new = (double*)cfd_aligned_malloc(field_size);
    ctx->dx_inv = (double*)cfd_aligned_malloc(ctx->nx * sizeof(double));
    ctx->dy_inv = (double*)cfd_aligned_malloc(ctx->ny * sizeof(double));

    if (!ctx->u_new || !ctx->v_new || !ctx->w_new || !ctx->p_new || !ctx->dx_inv || !ctx->dy_inv) {
        if (ctx->u_new) {
            cfd_aligned_free(ctx->u_new);
        }
        if (ctx->v_new) {
            cfd_aligned_free(ctx->v_new);
        }
        if (ctx->w_new) {
            cfd_aligned_free(ctx->w_new);
        }
        if (ctx->p_new) {
            cfd_aligned_free(ctx->p_new);
        }
        if (ctx->dx_inv) {
            cfd_aligned_free(ctx->dx_inv);
        }
        if (ctx->dy_inv) {
            cfd_aligned_free(ctx->dy_inv);
        }
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
    #ifdef _OPENMP
    CFD_LOG_INFO("solver", "Explicit Euler SIMD: AVX2 + OpenMP enabled (%d threads)", omp_get_max_threads());
    #else
    CFD_LOG_INFO("solver", "Explicit Euler SIMD: AVX2 enabled (OpenMP disabled)");
    #endif
#else
    #ifdef _OPENMP
    CFD_LOG_INFO("solver", "Explicit Euler OMP: Scalar + OpenMP enabled (%d threads)", omp_get_max_threads());
    #else
    CFD_LOG_INFO("solver", "Explicit Euler: Scalar fallback (no SIMD or OpenMP)");
    #endif
#endif

    return CFD_SUCCESS;
}

void explicit_euler_simd_destroy(struct NSSolver* solver) {
    if (solver && solver->context) {
        explicit_euler_simd_context* ctx = (explicit_euler_simd_context*)solver->context;
        if (ctx->initialized) {
            cfd_aligned_free(ctx->u_new);
            cfd_aligned_free(ctx->v_new);
            cfd_aligned_free(ctx->w_new);
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

typedef struct {
    __m256d dt_vec;
    __m256d max_deriv;
    __m256d min_deriv;
    __m256d max_diverg;
    __m256d min_diverg;
    __m256d max_vel_limit;
    __m256d min_vel_limit;
    __m256d one_vec;
    __m256d neg_one_vec;
    __m256d pressure_factor;
    __m256d two;
    __m256d four;
    __m256d epsilon;
    __m256d mu_vec;
    __m256d zero;
    __m256d inv_2dz_vec;
    __m256d inv_dz2_vec;
} simd_constants;

static void init_simd_constants(simd_constants* c, const ns_solver_params_t* params,
                                double conservative_dt, double inv_2dz, double inv_dz2) {
    c->dt_vec = _mm256_set1_pd(conservative_dt);
    c->max_deriv = _mm256_set1_pd(MAX_DERIVATIVE_LIMIT);
    c->min_deriv = _mm256_set1_pd(-MAX_DERIVATIVE_LIMIT);
    c->max_diverg = _mm256_set1_pd(MAX_DIVERGENCE_LIMIT);
    c->min_diverg = _mm256_set1_pd(-MAX_DIVERGENCE_LIMIT);
    c->max_vel_limit = _mm256_set1_pd(MAX_VELOCITY_LIMIT);
    c->min_vel_limit = _mm256_set1_pd(-MAX_VELOCITY_LIMIT);
    c->one_vec = _mm256_set1_pd(UPDATE_LIMIT);
    c->neg_one_vec = _mm256_set1_pd(-UPDATE_LIMIT);
    c->pressure_factor = _mm256_set1_pd(-PRESSURE_UPDATE_FACTOR);
    c->two = _mm256_set1_pd(2.0);
    c->four = _mm256_set1_pd(4.0);
    c->epsilon = _mm256_set1_pd(1e-10);
    c->mu_vec = _mm256_set1_pd(params->mu);
    c->zero = _mm256_setzero_pd();
    c->inv_2dz_vec = _mm256_set1_pd(inv_2dz);
    c->inv_dz2_vec = _mm256_set1_pd(inv_dz2);
}

static void process_simd_row(explicit_euler_simd_context* ctx, flow_field* field, const grid* grid,
                             size_t j, const simd_constants* sc,
                             size_t stride_z, size_t k_offset) {
    double dy2 = grid->dy[j] * grid->dy[j];
    __m256d dy_inv_val = _mm256_set1_pd(ctx->dy_inv[j]);
    __m256d dy2_val = _mm256_set1_pd(dy2);
    __m256d dy2_recip = _mm256_div_pd(sc->one_vec, dy2_val);

    for (size_t i = 1; i + 3 < ctx->nx - 1; i += 4) {
        size_t idx = k_offset + IDX_2D(i, j, ctx->nx);

        __m256d u = _mm256_loadu_pd(&field->u[idx]);
        __m256d v = _mm256_loadu_pd(&field->v[idx]);
        __m256d rho = _mm256_loadu_pd(&field->rho[idx]);
        __m256d rho_inv = _mm256_div_pd(sc->one_vec, _mm256_max_pd(rho, sc->epsilon));
        __m256d dx_inv_val = _mm256_loadu_pd(&ctx->dx_inv[i]);

        __m256d u_xp = _mm256_loadu_pd(&field->u[idx + 1]);
        __m256d u_xm = _mm256_loadu_pd(&field->u[idx - 1]);
        __m256d u_yp = _mm256_loadu_pd(&field->u[idx + ctx->nx]);
        __m256d u_ym = _mm256_loadu_pd(&field->u[idx - ctx->nx]);
        __m256d u_zp = _mm256_loadu_pd(&field->u[idx + stride_z]);
        __m256d u_zm = _mm256_loadu_pd(&field->u[idx - stride_z]);

        __m256d du_dx = _mm256_mul_pd(_mm256_sub_pd(u_xp, u_xm), dx_inv_val);
        __m256d du_dy = _mm256_mul_pd(_mm256_sub_pd(u_yp, u_ym), dy_inv_val);
        __m256d du_dz = _mm256_mul_pd(_mm256_sub_pd(u_zp, u_zm), sc->inv_2dz_vec);

        du_dx = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, du_dx));
        du_dy = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, du_dy));
        du_dz = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, du_dz));

        __m256d v_xp = _mm256_loadu_pd(&field->v[idx + 1]);
        __m256d v_xm = _mm256_loadu_pd(&field->v[idx - 1]);
        __m256d v_yp = _mm256_loadu_pd(&field->v[idx + ctx->nx]);
        __m256d v_ym = _mm256_loadu_pd(&field->v[idx - ctx->nx]);
        __m256d v_zp = _mm256_loadu_pd(&field->v[idx + stride_z]);
        __m256d v_zm = _mm256_loadu_pd(&field->v[idx - stride_z]);

        __m256d dv_dx = _mm256_mul_pd(_mm256_sub_pd(v_xp, v_xm), dx_inv_val);
        __m256d dv_dy = _mm256_mul_pd(_mm256_sub_pd(v_yp, v_ym), dy_inv_val);
        __m256d dv_dz = _mm256_mul_pd(_mm256_sub_pd(v_zp, v_zm), sc->inv_2dz_vec);

        dv_dx = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dv_dx));
        dv_dy = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dv_dy));
        dv_dz = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dv_dz));

        __m256d p_xp = _mm256_loadu_pd(&field->p[idx + 1]);
        __m256d p_xm = _mm256_loadu_pd(&field->p[idx - 1]);
        __m256d p_yp = _mm256_loadu_pd(&field->p[idx + ctx->nx]);
        __m256d p_ym = _mm256_loadu_pd(&field->p[idx - ctx->nx]);
        __m256d p_zp = _mm256_loadu_pd(&field->p[idx + stride_z]);
        __m256d p_zm = _mm256_loadu_pd(&field->p[idx - stride_z]);

        __m256d dp_dx = _mm256_mul_pd(_mm256_sub_pd(p_xp, p_xm), dx_inv_val);
        __m256d dp_dy = _mm256_mul_pd(_mm256_sub_pd(p_yp, p_ym), dy_inv_val);
        __m256d dp_dz = _mm256_mul_pd(_mm256_sub_pd(p_zp, p_zm), sc->inv_2dz_vec);

        dp_dx = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dp_dx));
        dp_dy = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dp_dy));
        dp_dz = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dp_dz));

        __m256d w = _mm256_loadu_pd(&field->w[idx]);
        __m256d w_xp = _mm256_loadu_pd(&field->w[idx + 1]);
        __m256d w_xm = _mm256_loadu_pd(&field->w[idx - 1]);
        __m256d w_yp = _mm256_loadu_pd(&field->w[idx + ctx->nx]);
        __m256d w_ym = _mm256_loadu_pd(&field->w[idx - ctx->nx]);
        __m256d w_zp = _mm256_loadu_pd(&field->w[idx + stride_z]);
        __m256d w_zm = _mm256_loadu_pd(&field->w[idx - stride_z]);

        __m256d dw_dx = _mm256_mul_pd(_mm256_sub_pd(w_xp, w_xm), dx_inv_val);
        __m256d dw_dy = _mm256_mul_pd(_mm256_sub_pd(w_yp, w_ym), dy_inv_val);
        __m256d dw_dz = _mm256_mul_pd(_mm256_sub_pd(w_zp, w_zm), sc->inv_2dz_vec);

        dw_dx = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dw_dx));
        dw_dy = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dw_dy));
        dw_dz = vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, dw_dz));

        __m256d inv_dx_sq = _mm256_mul_pd(sc->four, _mm256_mul_pd(dx_inv_val, dx_inv_val));

        __m256d d2u_dx2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(u_xp, u_xm), _mm256_mul_pd(sc->two, u)), inv_dx_sq);
        __m256d d2u_dy2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(u_yp, u_ym), _mm256_mul_pd(sc->two, u)), dy2_recip);
        __m256d d2u_dz2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(u_zp, u_zm), _mm256_mul_pd(sc->two, u)), sc->inv_dz2_vec);

        __m256d d2v_dx2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(v_xp, v_xm), _mm256_mul_pd(sc->two, v)), inv_dx_sq);
        __m256d d2v_dy2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(v_yp, v_ym), _mm256_mul_pd(sc->two, v)), dy2_recip);
        __m256d d2v_dz2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(v_zp, v_zm), _mm256_mul_pd(sc->two, v)), sc->inv_dz2_vec);

        __m256d d2w_dx2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(w_xp, w_xm), _mm256_mul_pd(sc->two, w)), inv_dx_sq);
        __m256d d2w_dy2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(w_yp, w_ym), _mm256_mul_pd(sc->two, w)), dy2_recip);
        __m256d d2w_dz2 = _mm256_mul_pd(
            _mm256_sub_pd(_mm256_add_pd(w_zp, w_zm), _mm256_mul_pd(sc->two, w)), sc->inv_dz2_vec);

        __m256d nu = _mm256_min_pd(sc->one_vec, _mm256_mul_pd(sc->mu_vec, rho_inv));

        __m256d term_pres_x = _mm256_mul_pd(dp_dx, rho_inv);
        __m256d term_visc_u = _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2u_dx2, d2u_dy2), d2u_dz2));
        __m256d conv_u = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(u, du_dx), _mm256_mul_pd(v, du_dy)),
            _mm256_mul_pd(w, du_dz));
        __m256d du =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_u, term_pres_x),
                                                    _mm256_sub_pd(sc->zero, conv_u)));

        __m256d term_pres_y = _mm256_mul_pd(dp_dy, rho_inv);
        __m256d term_visc_v = _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2v_dx2, d2v_dy2), d2v_dz2));
        __m256d conv_v = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(u, dv_dx), _mm256_mul_pd(v, dv_dy)),
            _mm256_mul_pd(w, dv_dz));
        __m256d dv =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_v, term_pres_y),
                                                    _mm256_sub_pd(sc->zero, conv_v)));

        __m256d term_pres_z = _mm256_mul_pd(dp_dz, rho_inv);
        __m256d term_visc_w = _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2w_dx2, d2w_dy2), d2w_dz2));
        __m256d conv_w = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(u, dw_dx), _mm256_mul_pd(v, dw_dy)),
            _mm256_mul_pd(w, dw_dz));
        __m256d dw =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_w, term_pres_z),
                                                    _mm256_sub_pd(sc->zero, conv_w)));

        du = vector_fmin(sc->one_vec, vector_fmax(sc->neg_one_vec, du));
        dv = vector_fmin(sc->one_vec, vector_fmax(sc->neg_one_vec, dv));
        dw = vector_fmin(sc->one_vec, vector_fmax(sc->neg_one_vec, dw));

        __m256d u_next = _mm256_add_pd(u, du);
        __m256d v_next = _mm256_add_pd(v, dv);
        __m256d w_next = _mm256_add_pd(w, dw);

        u_next = vector_fmin(sc->max_vel_limit, vector_fmax(sc->min_vel_limit, u_next));
        v_next = vector_fmin(sc->max_vel_limit, vector_fmax(sc->min_vel_limit, v_next));
        w_next = vector_fmin(sc->max_vel_limit, vector_fmax(sc->min_vel_limit, w_next));

        __m256d divergence = _mm256_add_pd(_mm256_add_pd(du_dx, dv_dy), dw_dz);
        divergence = vector_fmin(sc->max_diverg, vector_fmax(sc->min_diverg, divergence));
        __m256d p = _mm256_loadu_pd(&field->p[idx]);
        __m256d dp = _mm256_mul_pd(
            sc->dt_vec, _mm256_mul_pd(sc->pressure_factor, _mm256_mul_pd(rho, divergence)));
        dp = vector_fmin(sc->one_vec, vector_fmax(sc->neg_one_vec, dp));
        __m256d p_next = _mm256_add_pd(p, dp);

        _mm256_storeu_pd(&ctx->u_new[idx], u_next);
        _mm256_storeu_pd(&ctx->v_new[idx], v_next);
        _mm256_storeu_pd(&ctx->w_new[idx], w_next);
        _mm256_storeu_pd(&ctx->p_new[idx], p_next);
    }
}
#endif

#if !USE_AVX
static void process_scalar_row(explicit_euler_simd_context* ctx, flow_field* field,
                               const grid* grid, const ns_solver_params_t* params, size_t j,
                               double conservative_dt, double t, size_t stride_z, size_t k_offset) {
    for (size_t i = 1; i < ctx->nx - 1; i++) {
        size_t idx = k_offset + IDX_2D(i, j, ctx->nx);

        double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
        double du_dy = (field->u[idx + ctx->nx] - field->u[idx - ctx->nx]) / (2.0 * grid->dy[j]);
        double du_dz = (field->u[idx + stride_z] - field->u[idx - stride_z]) * ctx->inv_2dz;
        double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
        double dv_dy = (field->v[idx + ctx->nx] - field->v[idx - ctx->nx]) / (2.0 * grid->dy[j]);
        double dv_dz = (field->v[idx + stride_z] - field->v[idx - stride_z]) * ctx->inv_2dz;
        double dw_dx = (field->w[idx + 1] - field->w[idx - 1]) / (2.0 * grid->dx[i]);
        double dw_dy = (field->w[idx + ctx->nx] - field->w[idx - ctx->nx]) / (2.0 * grid->dy[j]);
        double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) * ctx->inv_2dz;

        double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
        double dp_dy = (field->p[idx + ctx->nx] - field->p[idx - ctx->nx]) / (2.0 * grid->dy[j]);
        double dp_dz = (field->p[idx + stride_z] - field->p[idx - stride_z]) * ctx->inv_2dz;

        double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) /
                         (grid->dx[i] * grid->dx[i]);
        double d2u_dy2 = (field->u[idx + ctx->nx] - 2.0 * field->u[idx] + field->u[idx - ctx->nx]) /
                         (grid->dy[j] * grid->dy[j]);
        double d2u_dz2 = (field->u[idx + stride_z] - 2.0 * field->u[idx] + field->u[idx - stride_z]) *
                         ctx->inv_dz2;
        double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) /
                         (grid->dx[i] * grid->dx[i]);
        double d2v_dy2 = (field->v[idx + ctx->nx] - 2.0 * field->v[idx] + field->v[idx - ctx->nx]) /
                         (grid->dy[j] * grid->dy[j]);
        double d2v_dz2 = (field->v[idx + stride_z] - 2.0 * field->v[idx] + field->v[idx - stride_z]) *
                         ctx->inv_dz2;
        double d2w_dx2 = (field->w[idx + 1] - 2.0 * field->w[idx] + field->w[idx - 1]) /
                         (grid->dx[i] * grid->dx[i]);
        double d2w_dy2 = (field->w[idx + ctx->nx] - 2.0 * field->w[idx] + field->w[idx - ctx->nx]) /
                         (grid->dy[j] * grid->dy[j]);
        double d2w_dz2 = (field->w[idx + stride_z] - 2.0 * field->w[idx] + field->w[idx - stride_z]) *
                         ctx->inv_dz2;

        double rho = fmax(field->rho[idx], 1e-10);
        // Using manual define for M_PI just in case it is missed in fallback
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
        double nu = fmin(params->mu / rho, 1.0);

        du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
        du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
        du_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dz));
        dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
        dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
        dv_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dz));
        dw_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dx));
        dw_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dy));
        dw_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dz));
        dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
        dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
        dp_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dz));

        double source_u = 0.0;
        double source_v = 0.0;
        double source_w = 0.0;
        if (params->source_func) {
            params->source_func(grid->x[i], grid->y[j], 0.0, t,
                                params->source_context,
                                &source_u, &source_v, &source_w);
        } else if (params->source_amplitude_u > 0) {
            source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]);
            source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]);
        }

        double u_c = field->u[idx];
        double v_c = field->v[idx];
        double w_c = field->w[idx];

        double du = conservative_dt * (-u_c * du_dx - v_c * du_dy - w_c * du_dz -
                                       dp_dx / rho + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2) + source_u);
        double dv = conservative_dt * (-u_c * dv_dx - v_c * dv_dy - w_c * dv_dz -
                                       dp_dy / rho + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2) + source_v);
        double dw = conservative_dt * (-u_c * dw_dx - v_c * dw_dy - w_c * dw_dz -
                                       dp_dz / rho + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2) + source_w);

        du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
        dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));
        dw = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dw));

        ctx->u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_c + du));
        ctx->v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_c + dv));
        ctx->w_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, w_c + dw));

        double divergence = du_dx + dv_dy + dw_dz;
        divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
        double dp = -PRESSURE_UPDATE_FACTOR * conservative_dt * rho * divergence;
        dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));
        ctx->p_new[idx] = field->p[idx] + dp;
    }
}
#endif

cfd_status_t explicit_euler_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !solver->context || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }

    explicit_euler_simd_context* ctx = (explicit_euler_simd_context*)solver->context;

    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    if (field->nx != ctx->nx || field->ny != ctx->ny || field->nz != ctx->nz) {
        return CFD_ERROR_INVALID;
    }

    // Use conservative time step to match basic solver stability
    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

    // Copy current state to temp buffers
    size_t size = ctx->nx * ctx->ny * ctx->nz;
    memcpy(ctx->u_new, field->u, size * sizeof(double));
    memcpy(ctx->v_new, field->v, size * sizeof(double));
    memcpy(ctx->w_new, field->w, size * sizeof(double));
    memcpy(ctx->p_new, field->p, size * sizeof(double));

    size_t nx = ctx->nx;
    size_t ny = ctx->ny;

#if USE_AVX
    simd_constants sc;
    init_simd_constants(&sc, params, conservative_dt, ctx->inv_2dz, ctx->inv_dz2);

    int ny_int = (int)(ctx->ny);
    int j;
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        size_t k_offset = k * ctx->stride_z;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (j = 1; j < ny_int - 1; j++) {
            process_simd_row(ctx, field, grid, (size_t)j, &sc, ctx->stride_z, k_offset);
        }
    }
#else
    int ny_int = (int)(ctx->ny);
    int j;
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        size_t k_offset = k * ctx->stride_z;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (j = 1; j < ny_int - 1; j++) {
            process_scalar_row(ctx, field, grid, params, (size_t)j, conservative_dt, 0.0,
                               ctx->stride_z, k_offset);
        }
    }
#endif

    // Apply boundary, check NaNs, etc.
    memcpy(field->u, ctx->u_new, size * sizeof(double));
    memcpy(field->v, ctx->v_new, size * sizeof(double));
    memcpy(field->w, ctx->w_new, size * sizeof(double));
    memcpy(field->p, ctx->p_new, size * sizeof(double));

    // Store caller-set boundary values before apply_boundary_conditions overwrites them
    copy_boundary_velocities_3d(ctx->u_new, ctx->v_new, ctx->w_new,
                                field->u, field->v, field->w, nx, ny, ctx->nz);
    apply_boundary_conditions(field, grid);
    copy_boundary_velocities_3d(field->u, field->v, field->w,
                                ctx->u_new, ctx->v_new, ctx->w_new, nx, ny, ctx->nz);

    if (stats) {
        stats->iterations = 1;
    }

    // NaN/Inf check
    int has_nan = 0;
    for (size_t n = 0; n < size; n++) {
        if (!isfinite(field->u[n]) || !isfinite(field->v[n]) ||
            !isfinite(field->w[n]) || !isfinite(field->p[n])) {
            has_nan = 1;
            break;
        }
    }
    if (has_nan) {
        cfd_set_error(CFD_ERROR_DIVERGED,
                      "NaN/Inf detected in explicit_euler_simd step");
        return CFD_ERROR_DIVERGED;
    }

    return CFD_SUCCESS;
}
