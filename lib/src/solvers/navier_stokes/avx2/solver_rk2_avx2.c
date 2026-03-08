/**
 * @file solver_rk2_avx2.c
 * @brief RK2 (Heun's method) - AVX2 + OpenMP implementation
 *
 * AVX2-only backend. Returns CFD_ERROR_UNSUPPORTED if AVX2 is unavailable.
 * Uses persistent aligned buffers in context to avoid per-step allocation.
 *
 * SIMD strategy per row (fixed j, k):
 *   i = 1       : scalar (periodic il wrap)
 *   i = 2..nx-4 : AVX2 vectorized (interior, no wrapping)
 *   i = nx-2    : scalar (periodic ir wrap)
 *   Remainder between AVX2 end and nx-2: scalar (no wrapping)
 *
 * Algorithm: identical to scalar/OMP RK2.
 *   k1 = RHS(Q^n)
 *   Q_pred = Q^n + dt*k1
 *   k2 = RHS(Q_pred)
 *   Q^{n+1} = Q^n + (dt/2)*(k1 + k2)
 *   BCs applied to final state only.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing bit-identical results to the 2D code path.
 */

#define _POSIX_C_SOURCE 200809L
#define _ISOC11_SOURCE
#define _USE_MATH_DEFINES

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(CFD_HAS_AVX2)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

/* Physical stability limits (identical to scalar/OMP RK2) */
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DIVERGENCE_LIMIT        10.0
#define PRESSURE_UPDATE_FACTOR      0.1

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    double* k1_u; double* k1_v; double* k1_w; double* k1_p;
    double* k2_u; double* k2_v; double* k2_w; double* k2_p;
    double* u0;   double* v0;   double* w0;   double* p0;
    double* dx_inv;  /* 1/(2*dx[i]) for i = 0..nx-1 */
    double* dy_inv;  /* 1/(2*dy[j]) for j = 0..ny-1 */
    size_t nx, ny, nz;
    size_t stride_z;
    size_t k_start, k_end;
    double inv_2dz, inv_dz2;
    int initialized;
} rk2_avx2_context_t;

/* ============================================================================
 * PUBLIC API DECLARATIONS
 * ============================================================================ */

cfd_status_t rk2_avx2_init(ns_solver_t* solver, const grid* g,
                             const ns_solver_params_t* params);
void         rk2_avx2_destroy(ns_solver_t* solver);
cfd_status_t rk2_avx2_step(ns_solver_t* solver, flow_field* field, const grid* g,
                             const ns_solver_params_t* params, ns_solver_stats_t* stats);

/* ============================================================================
 * SCALAR POINT HELPER
 * Handles a single interior grid point given explicit stencil indices.
 * Used for periodic-edge columns (i=1 with il wrap, i=nx-2 with ir wrap)
 * and for scalar remainder after the AVX2 group.
 * ============================================================================ */

static void rk2_rhs_point(
    const double* u, const double* v, const double* w,
    const double* p, const double* rho,
    double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
    const grid* g, const ns_solver_params_t* params,
    size_t idx, size_t il, size_t ir, size_t jd, size_t ju,
    size_t kd, size_t ku,
    double inv_2dz, double inv_dz2,
    size_t i, size_t j, size_t k, size_t nz,
    int iter, double dt)
{
    if (rho[idx] <= 1e-10 || fabs(g->dx[i]) < 1e-10 || fabs(g->dy[j]) < 1e-10) {
        rhs_u[idx] = 0.0;
        rhs_v[idx] = 0.0;
        rhs_w[idx] = 0.0;
        rhs_p[idx] = 0.0;
        return;
    }

    double dx = g->dx[i], dy = g->dy[j];
    double dx2 = dx * dx, dy2 = dy * dy;

    double du_dx = (u[ir] - u[il]) / (2.0 * dx);
    double du_dy = (u[ju] - u[jd]) / (2.0 * dy);
    double du_dz = (u[ku] - u[kd]) * inv_2dz;
    double dv_dx = (v[ir] - v[il]) / (2.0 * dx);
    double dv_dy = (v[ju] - v[jd]) / (2.0 * dy);
    double dv_dz = (v[ku] - v[kd]) * inv_2dz;
    double dw_dx = (w[ir] - w[il]) / (2.0 * dx);
    double dw_dy = (w[ju] - w[jd]) / (2.0 * dy);
    double dw_dz = (w[ku] - w[kd]) * inv_2dz;
    double dp_dx = (p[ir] - p[il]) / (2.0 * dx);
    double dp_dy = (p[ju] - p[jd]) / (2.0 * dy);
    double dp_dz = (p[ku] - p[kd]) * inv_2dz;

    double d2u_dx2 = (u[ir] - 2.0 * u[idx] + u[il]) / dx2;
    double d2u_dy2 = (u[ju] - 2.0 * u[idx] + u[jd]) / dy2;
    double d2u_dz2 = (u[ku] - 2.0 * u[idx] + u[kd]) * inv_dz2;
    double d2v_dx2 = (v[ir] - 2.0 * v[idx] + v[il]) / dx2;
    double d2v_dy2 = (v[ju] - 2.0 * v[idx] + v[jd]) / dy2;
    double d2v_dz2 = (v[ku] - 2.0 * v[idx] + v[kd]) * inv_dz2;
    double d2w_dx2 = (w[ir] - 2.0 * w[idx] + w[il]) / dx2;
    double d2w_dy2 = (w[ju] - 2.0 * w[idx] + w[jd]) / dy2;
    double d2w_dz2 = (w[ku] - 2.0 * w[idx] + w[kd]) * inv_dz2;

    double nu = params->mu / fmax(rho[idx], 1e-10);
    nu = fmin(nu, 1.0);

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
    d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
    d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
    d2u_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dz2));
    d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
    d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));
    d2v_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dz2));
    d2w_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dx2));
    d2w_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dy2));
    d2w_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dz2));

    double source_u = 0.0, source_v = 0.0, source_w = 0.0;
    double z_coord = (nz > 1 && g->z) ? g->z[k] : 0.0;
    compute_source_terms(g->x[i], g->y[j], z_coord, iter, dt, params,
                         &source_u, &source_v, &source_w);

    rhs_u[idx] = -(u[idx] * du_dx) - (v[idx] * du_dy) - (w[idx] * du_dz)
                 - dp_dx / rho[idx]
                 + (nu * (d2u_dx2 + d2u_dy2 + d2u_dz2))
                 + source_u;

    rhs_v[idx] = -(u[idx] * dv_dx) - (v[idx] * dv_dy) - (w[idx] * dv_dz)
                 - dp_dy / rho[idx]
                 + (nu * (d2v_dx2 + d2v_dy2 + d2v_dz2))
                 + source_v;

    rhs_w[idx] = -(u[idx] * dw_dx) - (v[idx] * dw_dy) - (w[idx] * dw_dz)
                 - dp_dz / rho[idx]
                 + (nu * (d2w_dx2 + d2w_dy2 + d2w_dz2))
                 + source_w;

    double divergence = du_dx + dv_dy + dw_dz;
    divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
    rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
}

/* ============================================================================
 * AVX2 ROW-LEVEL RHS  (compiled only when AVX2 is available)
 * ============================================================================ */

#if USE_AVX2

static inline __m256d avx2_clamp(__m256d x, __m256d lo, __m256d hi) {
    return _mm256_max_pd(lo, _mm256_min_pd(hi, x));
}

/*
 * Compute RHS for one row j at z-plane k, dispatching:
 *   i=1        : scalar (periodic il wrap)
 *   i=2..nx-4  : AVX2 (4-wide, no wrapping)
 *   i=nx-2     : scalar (periodic ir wrap), plus any scalar remainder
 */
static void compute_rhs_row(
    const double* u, const double* v, const double* w,
    const double* p, const double* rho,
    double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
    const rk2_avx2_context_t* ctx, const grid* g,
    const ns_solver_params_t* params, size_t j, size_t k, int iter, double dt)
{
    size_t nx       = ctx->nx;
    size_t ny       = ctx->ny;
    size_t nz       = ctx->nz;
    size_t stride_z = ctx->stride_z;
    double inv_2dz  = ctx->inv_2dz;
    double inv_dz2  = ctx->inv_dz2;

    size_t k_off  = k * stride_z;
    size_t j_off  = k_off + j * nx;

    /* Degenerate dy: zero RHS for entire row (matches scalar path) */
    if (fabs(g->dy[j]) < 1e-10) {
        return;  /* rhs arrays already memset to 0 before this call */
    }

    /* j-direction stencil row offsets: uniform across entire row */
    size_t jd_row = (j > 1)      ? k_off + (j - 1) * nx : k_off + (ny - 2) * nx;
    size_t ju_row = (j < ny - 2) ? k_off + (j + 1) * nx : k_off + nx;

    /* z-direction stencil plane offsets: uniform across entire row.
     * When nz==1: stride_z=0, so kd_off=ku_off=0 and z-terms vanish. */
    size_t kd_off = (k > 1)      ? (k - 1) * stride_z : (nz - 2) * stride_z;
    size_t ku_off = (k < nz - 2) ? (k + 1) * stride_z : 1 * stride_z;

    /* --- Scalar: i = 1 (il wraps periodically) --- */
    {
        size_t si  = 1;
        size_t idx = j_off + si;
        size_t il  = j_off + (nx - 2);  /* periodic wrap */
        size_t ir  = idx + 1;
        size_t kd  = kd_off + j * nx + si;
        size_t ku  = ku_off + j * nx + si;
        rk2_rhs_point(u, v, w, p, rho, rhs_u, rhs_v, rhs_w, rhs_p, g, params,
                       idx, il, ir, jd_row + si, ju_row + si, kd, ku,
                       inv_2dz, inv_dz2, si, j, k, nz, iter, dt);
    }

    /* --- AVX2: i = 2 while i+3 <= nx-3 (interior, no wrapping needed) --- */
    {
        __m256d dy_inv_v  = _mm256_set1_pd(ctx->dy_inv[j]);
        /* 1/dy^2 = 4 * (1/(2*dy))^2, derived from pre-guarded dy_inv */
        __m256d dy2_inv_v = _mm256_set1_pd(4.0 * ctx->dy_inv[j] * ctx->dy_inv[j]);
        __m256d dz_inv_v  = _mm256_set1_pd(inv_2dz);
        __m256d dz2_inv_v = _mm256_set1_pd(inv_dz2);
        __m256d two       = _mm256_set1_pd(2.0);
        __m256d four      = _mm256_set1_pd(4.0);
        __m256d max_d1    = _mm256_set1_pd( MAX_DERIVATIVE_LIMIT);
        __m256d min_d1    = _mm256_set1_pd(-MAX_DERIVATIVE_LIMIT);
        __m256d max_d2    = _mm256_set1_pd( MAX_SECOND_DERIVATIVE_LIMIT);
        __m256d min_d2    = _mm256_set1_pd(-MAX_SECOND_DERIVATIVE_LIMIT);
        __m256d max_div   = _mm256_set1_pd( MAX_DIVERGENCE_LIMIT);
        __m256d min_div   = _mm256_set1_pd(-MAX_DIVERGENCE_LIMIT);
        __m256d mu_v      = _mm256_set1_pd(params->mu);
        __m256d eps       = _mm256_set1_pd(1e-10);
        __m256d one       = _mm256_set1_pd(1.0);
        __m256d neg_puf   = _mm256_set1_pd(-PRESSURE_UPDATE_FACTOR);

        ptrdiff_t i;
        for (i = 2; i + 3 <= (ptrdiff_t)(nx - 3); i += 4) {
            size_t idx = j_off + (size_t)i;

            /* Load u stencil */
            __m256d u_c  = _mm256_loadu_pd(&u[idx]);
            __m256d u_il = _mm256_loadu_pd(&u[idx - 1]);
            __m256d u_ir = _mm256_loadu_pd(&u[idx + 1]);
            __m256d u_jd = _mm256_loadu_pd(&u[jd_row + (size_t)i]);
            __m256d u_ju = _mm256_loadu_pd(&u[ju_row + (size_t)i]);
            __m256d u_kd = _mm256_loadu_pd(&u[kd_off + j * nx + (size_t)i]);
            __m256d u_ku = _mm256_loadu_pd(&u[ku_off + j * nx + (size_t)i]);

            /* Load v stencil */
            __m256d v_c  = _mm256_loadu_pd(&v[idx]);
            __m256d v_il = _mm256_loadu_pd(&v[idx - 1]);
            __m256d v_ir = _mm256_loadu_pd(&v[idx + 1]);
            __m256d v_jd = _mm256_loadu_pd(&v[jd_row + (size_t)i]);
            __m256d v_ju = _mm256_loadu_pd(&v[ju_row + (size_t)i]);
            __m256d v_kd = _mm256_loadu_pd(&v[kd_off + j * nx + (size_t)i]);
            __m256d v_ku = _mm256_loadu_pd(&v[ku_off + j * nx + (size_t)i]);

            /* Load w stencil */
            __m256d w_c  = _mm256_loadu_pd(&w[idx]);
            __m256d w_il = _mm256_loadu_pd(&w[idx - 1]);
            __m256d w_ir = _mm256_loadu_pd(&w[idx + 1]);
            __m256d w_jd = _mm256_loadu_pd(&w[jd_row + (size_t)i]);
            __m256d w_ju = _mm256_loadu_pd(&w[ju_row + (size_t)i]);
            __m256d w_kd = _mm256_loadu_pd(&w[kd_off + j * nx + (size_t)i]);
            __m256d w_ku = _mm256_loadu_pd(&w[ku_off + j * nx + (size_t)i]);

            /* Load p stencil */
            __m256d p_il = _mm256_loadu_pd(&p[idx - 1]);
            __m256d p_ir = _mm256_loadu_pd(&p[idx + 1]);
            __m256d p_jd = _mm256_loadu_pd(&p[jd_row + (size_t)i]);
            __m256d p_ju = _mm256_loadu_pd(&p[ju_row + (size_t)i]);
            __m256d p_kd = _mm256_loadu_pd(&p[kd_off + j * nx + (size_t)i]);
            __m256d p_ku = _mm256_loadu_pd(&p[ku_off + j * nx + (size_t)i]);

            /* Density: clamp for safe division, but track validity */
            __m256d rho_raw   = _mm256_loadu_pd(&rho[idx]);
            __m256d rho_valid = _mm256_cmp_pd(rho_raw, eps, _CMP_GT_OQ);
            __m256d rho_c     = _mm256_max_pd(rho_raw, eps);
            __m256d rho_inv   = _mm256_div_pd(one, rho_c);

            /* dx stencil: dx_inv[i] = 1/(2*dx[i]), 0 when dx < 1e-10 */
            __m256d dx_inv_v  = _mm256_loadu_pd(&ctx->dx_inv[(size_t)i]);
            __m256d dx_valid  = _mm256_cmp_pd(dx_inv_v, _mm256_setzero_pd(), _CMP_GT_OQ);

            /* Combined validity: rho > eps AND dx > 0 (dy already checked at row level) */
            __m256d valid = _mm256_and_pd(rho_valid, dx_valid);
            /* 1/dx^2 = 4 * (1/(2*dx))^2 */
            __m256d dx2_inv_v = _mm256_mul_pd(four, _mm256_mul_pd(dx_inv_v, dx_inv_v));

            /* First derivatives — u */
            __m256d du_dx = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(u_ir, u_il), dx_inv_v), min_d1, max_d1);
            __m256d du_dy = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(u_ju, u_jd), dy_inv_v), min_d1, max_d1);
            __m256d du_dz = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(u_ku, u_kd), dz_inv_v), min_d1, max_d1);

            /* First derivatives — v */
            __m256d dv_dx = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(v_ir, v_il), dx_inv_v), min_d1, max_d1);
            __m256d dv_dy = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(v_ju, v_jd), dy_inv_v), min_d1, max_d1);
            __m256d dv_dz = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(v_ku, v_kd), dz_inv_v), min_d1, max_d1);

            /* First derivatives — w */
            __m256d dw_dx = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(w_ir, w_il), dx_inv_v), min_d1, max_d1);
            __m256d dw_dy = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(w_ju, w_jd), dy_inv_v), min_d1, max_d1);
            __m256d dw_dz = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(w_ku, w_kd), dz_inv_v), min_d1, max_d1);

            /* Pressure gradients */
            __m256d dp_dx = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(p_ir, p_il), dx_inv_v), min_d1, max_d1);
            __m256d dp_dy = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(p_ju, p_jd), dy_inv_v), min_d1, max_d1);
            __m256d dp_dz = avx2_clamp(
                _mm256_mul_pd(_mm256_sub_pd(p_ku, p_kd), dz_inv_v), min_d1, max_d1);

            /* Second derivatives — u */
            __m256d d2u_dx2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_ir, u_il), _mm256_mul_pd(two, u_c)),
                    dx2_inv_v),
                min_d2, max_d2);
            __m256d d2u_dy2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_ju, u_jd), _mm256_mul_pd(two, u_c)),
                    dy2_inv_v),
                min_d2, max_d2);
            __m256d d2u_dz2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(u_ku, u_kd), _mm256_mul_pd(two, u_c)),
                    dz2_inv_v),
                min_d2, max_d2);

            /* Second derivatives — v */
            __m256d d2v_dx2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(v_ir, v_il), _mm256_mul_pd(two, v_c)),
                    dx2_inv_v),
                min_d2, max_d2);
            __m256d d2v_dy2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(v_ju, v_jd), _mm256_mul_pd(two, v_c)),
                    dy2_inv_v),
                min_d2, max_d2);
            __m256d d2v_dz2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(v_ku, v_kd), _mm256_mul_pd(two, v_c)),
                    dz2_inv_v),
                min_d2, max_d2);

            /* Second derivatives — w */
            __m256d d2w_dx2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(w_ir, w_il), _mm256_mul_pd(two, w_c)),
                    dx2_inv_v),
                min_d2, max_d2);
            __m256d d2w_dy2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(w_ju, w_jd), _mm256_mul_pd(two, w_c)),
                    dy2_inv_v),
                min_d2, max_d2);
            __m256d d2w_dz2 = avx2_clamp(
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(w_ku, w_kd), _mm256_mul_pd(two, w_c)),
                    dz2_inv_v),
                min_d2, max_d2);

            /* Kinematic viscosity: nu = min(mu/rho, 1.0) */
            __m256d nu = _mm256_min_pd(one, _mm256_mul_pd(mu_v, rho_inv));

            /* Source terms: computed scalar per lane and packed */
            double src_u_arr[4] = {0.0, 0.0, 0.0, 0.0};
            double src_v_arr[4] = {0.0, 0.0, 0.0, 0.0};
            double src_w_arr[4] = {0.0, 0.0, 0.0, 0.0};
            for (int lane = 0; lane < 4; lane++) {
                double z_coord = (nz > 1 && g->z) ? g->z[k] : 0.0;
                compute_source_terms(g->x[(size_t)i + (size_t)lane], g->y[j],
                                     z_coord, iter, dt, params,
                                     &src_u_arr[lane], &src_v_arr[lane],
                                     &src_w_arr[lane]);
            }
            __m256d src_u_v = _mm256_loadu_pd(src_u_arr);
            __m256d src_v_v = _mm256_loadu_pd(src_v_arr);
            __m256d src_w_v = _mm256_loadu_pd(src_w_arr);

            /* RHS u: nu*(d2u_dx2+d2u_dy2+d2u_dz2) - dp_dx/rho - conv_u + src_u */
            __m256d conv_u = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(u_c, du_dx),
                    _mm256_mul_pd(v_c, du_dy)),
                _mm256_mul_pd(w_c, du_dz));
            __m256d rhs_u_v = _mm256_add_pd(
                _mm256_sub_pd(
                    _mm256_sub_pd(
                        _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2u_dx2, d2u_dy2), d2u_dz2)),
                        _mm256_mul_pd(dp_dx, rho_inv)),
                    conv_u),
                src_u_v);

            /* RHS v: nu*(d2v_dx2+d2v_dy2+d2v_dz2) - dp_dy/rho - conv_v + src_v */
            __m256d conv_v = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(u_c, dv_dx),
                    _mm256_mul_pd(v_c, dv_dy)),
                _mm256_mul_pd(w_c, dv_dz));
            __m256d rhs_v_v = _mm256_add_pd(
                _mm256_sub_pd(
                    _mm256_sub_pd(
                        _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2v_dx2, d2v_dy2), d2v_dz2)),
                        _mm256_mul_pd(dp_dy, rho_inv)),
                    conv_v),
                src_v_v);

            /* RHS w: nu*(d2w_dx2+d2w_dy2+d2w_dz2) - dp_dz/rho - conv_w + src_w */
            __m256d conv_w = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_mul_pd(u_c, dw_dx),
                    _mm256_mul_pd(v_c, dw_dy)),
                _mm256_mul_pd(w_c, dw_dz));
            __m256d rhs_w_v = _mm256_add_pd(
                _mm256_sub_pd(
                    _mm256_sub_pd(
                        _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2w_dx2, d2w_dy2), d2w_dz2)),
                        _mm256_mul_pd(dp_dz, rho_inv)),
                    conv_w),
                src_w_v);

            /* Pressure RHS: -0.1 * rho * divergence */
            __m256d divergence = avx2_clamp(
                _mm256_add_pd(_mm256_add_pd(du_dx, dv_dy), dw_dz), min_div, max_div);
            __m256d rhs_p_v = _mm256_mul_pd(
                neg_puf, _mm256_mul_pd(rho_c, divergence));

            /* Zero RHS for invalid lanes (rho <= eps or dx degenerate) */
            rhs_u_v = _mm256_and_pd(rhs_u_v, valid);
            rhs_v_v = _mm256_and_pd(rhs_v_v, valid);
            rhs_w_v = _mm256_and_pd(rhs_w_v, valid);
            rhs_p_v = _mm256_and_pd(rhs_p_v, valid);

            _mm256_storeu_pd(&rhs_u[idx], rhs_u_v);
            _mm256_storeu_pd(&rhs_v[idx], rhs_v_v);
            _mm256_storeu_pd(&rhs_w[idx], rhs_w_v);
            _mm256_storeu_pd(&rhs_p[idx], rhs_p_v);
        }

        /* --- Scalar: remainder from end of AVX2 to nx-2 (inclusive) ---
         * Handles 0-3 leftover interior points plus the periodic ir-wrap edge. */
        for (; i <= (ptrdiff_t)(nx - 2); i++) {
            size_t si  = (size_t)i;
            size_t idx = j_off + si;
            size_t il  = idx - 1;
            size_t ir  = (si == nx - 2) ? j_off + 1 : idx + 1;  /* periodic at last point */
            size_t kd  = kd_off + j * nx + si;
            size_t ku  = ku_off + j * nx + si;
            rk2_rhs_point(u, v, w, p, rho, rhs_u, rhs_v, rhs_w, rhs_p, g, params,
                           idx, il, ir, jd_row + si, ju_row + si, kd, ku,
                           inv_2dz, inv_dz2, si, j, k, nz, iter, dt);
        }
    }
}

/*
 * Compute RHS for all interior rows and z-planes.
 * k-loop is outside the OMP parallel for j-loop (MSVC OMP 2.0 compatibility).
 */
static void compute_rhs_avx2(
    const double* u, const double* v, const double* w,
    const double* p, const double* rho,
    double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
    const rk2_avx2_context_t* ctx, const grid* g,
    const ns_solver_params_t* params, int iter, double dt)
{
    ptrdiff_t ny_int = (ptrdiff_t)ctx->ny;
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        ptrdiff_t j;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (j = 1; j < ny_int - 1; j++) {
            compute_rhs_row(u, v, w, p, rho, rhs_u, rhs_v, rhs_w, rhs_p,
                            ctx, g, params, (size_t)j, k, iter, dt);
        }
    }
}

/* ============================================================================
 * RK2 AVX2 MAIN LOOP
 * ============================================================================ */

static cfd_status_t rk2_avx2_impl(flow_field* field, rk2_avx2_context_t* ctx,
                                    const grid* g, const ns_solver_params_t* params)
{
    size_t n     = ctx->nx * ctx->ny * ctx->nz;
    size_t bytes = n * sizeof(double);
    double dt    = params->dt;
    ptrdiff_t n_int = (ptrdiff_t)n;
    cfd_status_t status = CFD_SUCCESS;

    for (int iter = 0; iter < params->max_iter; iter++) {
        /* Save Q^n */
        memcpy(ctx->u0, field->u, bytes);
        memcpy(ctx->v0, field->v, bytes);
        memcpy(ctx->w0, field->w, bytes);
        memcpy(ctx->p0, field->p, bytes);

        /* ---- Stage 1: k1 = RHS(Q^n) ---- */
        memset(ctx->k1_u, 0, bytes);
        memset(ctx->k1_v, 0, bytes);
        memset(ctx->k1_w, 0, bytes);
        memset(ctx->k1_p, 0, bytes);
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho,
                         ctx->k1_u, ctx->k1_v, ctx->k1_w, ctx->k1_p,
                         ctx, g, params, iter, dt);

        /* ---- Intermediate: field = Q^n + dt*k1 ---- */
        {
            ptrdiff_t k;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (k = 0; k < n_int; k++) {
                field->u[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->u0[k] + dt * ctx->k1_u[k]));
                field->v[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->v0[k] + dt * ctx->k1_v[k]));
                field->w[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->w0[k] + dt * ctx->k1_w[k]));
                field->p[k] = ctx->p0[k] + dt * ctx->k1_p[k];
            }
        }

        /* NOTE: Do NOT apply BCs between RK stages.
         * The ghost cells carry zero-derivative evolution (k1[ghost]=0),
         * consistent with the semi-discrete ODE. Applying BCs here would
         * reduce RK2 to first-order temporal accuracy. */

        /* ---- Stage 2: k2 = RHS(Q_pred) ---- */
        memset(ctx->k2_u, 0, bytes);
        memset(ctx->k2_v, 0, bytes);
        memset(ctx->k2_w, 0, bytes);
        memset(ctx->k2_p, 0, bytes);
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho,
                         ctx->k2_u, ctx->k2_v, ctx->k2_w, ctx->k2_p,
                         ctx, g, params, iter, dt);

        /* ---- Final update: Q^{n+1} = Q^n + (dt/2)*(k1 + k2) ---- */
        {
            double half_dt = 0.5 * dt;
            ptrdiff_t k;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (k = 0; k < n_int; k++) {
                field->u[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->u0[k] + half_dt * (ctx->k1_u[k] + ctx->k2_u[k])));
                field->v[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->v0[k] + half_dt * (ctx->k1_v[k] + ctx->k2_v[k])));
                field->w[k] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->w0[k] + half_dt * (ctx->k1_w[k] + ctx->k2_w[k])));
                field->p[k] = ctx->p0[k] + half_dt * (ctx->k1_p[k] + ctx->k2_p[k]);
            }
        }

        /* Apply BCs to final state only */
        apply_boundary_conditions(field, g);

        /* NaN / Inf check (parallelized) */
        {
            int has_nan = 0;
            ptrdiff_t k;
#ifdef _OPENMP
            #pragma omp parallel for reduction(|:has_nan) schedule(static)
#endif
            for (k = 0; k < n_int; k++) {
                if (!isfinite(field->u[k]) || !isfinite(field->v[k]) ||
                    !isfinite(field->w[k]) || !isfinite(field->p[k])) {
                    has_nan = 1;
                }
            }
            if (has_nan) {
                status = CFD_ERROR_DIVERGED;
                goto cleanup;
            }
        }
    }

cleanup:
    return status;
}

#endif /* USE_AVX2 */

/* ============================================================================
 * PUBLIC API IMPLEMENTATIONS
 * ============================================================================ */

cfd_status_t rk2_avx2_init(ns_solver_t* solver, const grid* g,
                             const ns_solver_params_t* params)
{
    (void)params;

#if !USE_AVX2
    (void)solver;
    (void)g;
    cfd_set_error(CFD_ERROR_UNSUPPORTED, "AVX2 not available in this build");
    return CFD_ERROR_UNSUPPORTED;
#else
    if (!solver || !g) {
        return CFD_ERROR_INVALID;
    }
    if (g->nx < 3 || g->ny < 3 || (g->nz > 1 && g->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    rk2_avx2_context_t* ctx =
        (rk2_avx2_context_t*)cfd_calloc(1, sizeof(rk2_avx2_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->nx = g->nx;
    ctx->ny = g->ny;
    ctx->nz = g->nz;
    size_t bytes = ctx->nx * ctx->ny * g->nz * sizeof(double);

    /* Reject non-uniform z-spacing (solver uses constant inv_2dz/inv_dz2) */
    if (g->nz > 1 && g->dz) {
        for (size_t kk = 1; kk < g->nz - 1; kk++) {
            if (fabs(g->dz[kk] - g->dz[0]) > 1e-14) {
                cfd_free(ctx);
                return CFD_ERROR_INVALID;
            }
        }
    }

    /* Branch-free 3D constants */
    size_t plane   = ctx->nx * ctx->ny;
    ctx->stride_z  = (g->nz > 1) ? plane : 0;
    ctx->k_start   = (g->nz > 1) ? 1 : 0;
    ctx->k_end     = (g->nz > 1) ? (g->nz - 1) : 1;
    ctx->inv_2dz   = (g->nz > 1 && g->dz) ? 1.0 / (2.0 * g->dz[0]) : 0.0;
    ctx->inv_dz2   = (g->nz > 1 && g->dz) ? 1.0 / (g->dz[0] * g->dz[0]) : 0.0;

    ctx->k1_u   = (double*)cfd_aligned_malloc(bytes);
    ctx->k1_v   = (double*)cfd_aligned_malloc(bytes);
    ctx->k1_w   = (double*)cfd_aligned_malloc(bytes);
    ctx->k1_p   = (double*)cfd_aligned_malloc(bytes);
    ctx->k2_u   = (double*)cfd_aligned_malloc(bytes);
    ctx->k2_v   = (double*)cfd_aligned_malloc(bytes);
    ctx->k2_w   = (double*)cfd_aligned_malloc(bytes);
    ctx->k2_p   = (double*)cfd_aligned_malloc(bytes);
    ctx->u0     = (double*)cfd_aligned_malloc(bytes);
    ctx->v0     = (double*)cfd_aligned_malloc(bytes);
    ctx->w0     = (double*)cfd_aligned_malloc(bytes);
    ctx->p0     = (double*)cfd_aligned_malloc(bytes);
    ctx->dx_inv = (double*)cfd_aligned_malloc(ctx->nx * sizeof(double));
    ctx->dy_inv = (double*)cfd_aligned_malloc(ctx->ny * sizeof(double));

    if (!ctx->k1_u || !ctx->k1_v || !ctx->k1_w || !ctx->k1_p ||
        !ctx->k2_u || !ctx->k2_v || !ctx->k2_w || !ctx->k2_p ||
        !ctx->u0   || !ctx->v0   || !ctx->w0   || !ctx->p0   ||
        !ctx->dx_inv || !ctx->dy_inv) {
        cfd_aligned_free(ctx->k1_u); cfd_aligned_free(ctx->k1_v);
        cfd_aligned_free(ctx->k1_w); cfd_aligned_free(ctx->k1_p);
        cfd_aligned_free(ctx->k2_u); cfd_aligned_free(ctx->k2_v);
        cfd_aligned_free(ctx->k2_w); cfd_aligned_free(ctx->k2_p);
        cfd_aligned_free(ctx->u0);   cfd_aligned_free(ctx->v0);
        cfd_aligned_free(ctx->w0);   cfd_aligned_free(ctx->p0);
        cfd_aligned_free(ctx->dx_inv);
        cfd_aligned_free(ctx->dy_inv);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    for (size_t i = 0; i < ctx->nx; i++) {
        ctx->dx_inv[i] = (g->dx[i] > 1e-10) ? 1.0 / (2.0 * g->dx[i]) : 0.0;
    }
    for (size_t j = 0; j < ctx->ny; j++) {
        ctx->dy_inv[j] = (g->dy[j] > 1e-10) ? 1.0 / (2.0 * g->dy[j]) : 0.0;
    }

    ctx->initialized = 1;
    solver->context  = ctx;

#ifdef _OPENMP
    CFD_LOG_INFO("solver", "RK2 SIMD: AVX2 + OpenMP enabled (%d threads)", omp_get_max_threads());
#else
    CFD_LOG_INFO("solver", "RK2 SIMD: AVX2 enabled (OpenMP disabled)");
#endif

    return CFD_SUCCESS;
#endif
}

void rk2_avx2_destroy(ns_solver_t* solver)
{
    if (!solver || !solver->context) {
        return;
    }
    rk2_avx2_context_t* ctx = (rk2_avx2_context_t*)solver->context;
    if (ctx->initialized) {
        cfd_aligned_free(ctx->k1_u); cfd_aligned_free(ctx->k1_v);
        cfd_aligned_free(ctx->k1_w); cfd_aligned_free(ctx->k1_p);
        cfd_aligned_free(ctx->k2_u); cfd_aligned_free(ctx->k2_v);
        cfd_aligned_free(ctx->k2_w); cfd_aligned_free(ctx->k2_p);
        cfd_aligned_free(ctx->u0);   cfd_aligned_free(ctx->v0);
        cfd_aligned_free(ctx->w0);   cfd_aligned_free(ctx->p0);
        cfd_aligned_free(ctx->dx_inv);
        cfd_aligned_free(ctx->dy_inv);
    }
    cfd_free(ctx);
    solver->context = NULL;
}

cfd_status_t rk2_avx2_step(ns_solver_t* solver, flow_field* field, const grid* g,
                             const ns_solver_params_t* params, ns_solver_stats_t* stats)
{
#if !USE_AVX2
    (void)solver; (void)field; (void)g; (void)params; (void)stats;
    return CFD_ERROR_UNSUPPORTED;
#else
    if (!solver || !solver->context || !field || !g || !params) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    rk2_avx2_context_t* ctx = (rk2_avx2_context_t*)solver->context;
    if (field->nx != ctx->nx || field->ny != ctx->ny || field->nz != ctx->nz) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    cfd_status_t status = rk2_avx2_impl(field, ctx, g, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        ptrdiff_t n_s = (ptrdiff_t)(field->nx * field->ny * field->nz);
        ptrdiff_t ks;
#if defined(_OPENMP) && (_OPENMP >= 201107)
        #pragma omp parallel for reduction(max: max_vel, max_p) schedule(static)
#endif
        for (ks = 0; ks < n_s; ks++) {
            double vel = sqrt(field->u[ks] * field->u[ks] +
                              field->v[ks] * field->v[ks] +
                              field->w[ks] * field->w[ks]);
            if (vel > max_vel) max_vel = vel;
            double ap = fabs(field->p[ks]);
            if (ap > max_p) max_p = ap;
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
        double max_t = 0.0;
        if (field->T) {
            for (ks = 0; ks < n_s; ks++) {
                double at = fabs(field->T[ks]);
                if (at > max_t) max_t = at;
            }
        }
        stats->max_temperature = max_t;
    }

    return status;
#endif
}

cfd_status_t rk2_avx2_solve(ns_solver_t* solver, flow_field* field, const grid* g,
                              const ns_solver_params_t* params, ns_solver_stats_t* stats)
{
#if !USE_AVX2
    (void)solver; (void)field; (void)g; (void)params; (void)stats;
    return CFD_ERROR_UNSUPPORTED;
#else
    if (!solver || !solver->context || !field || !g || !params) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    rk2_avx2_context_t* ctx = (rk2_avx2_context_t*)solver->context;
    if (field->nx != ctx->nx || field->ny != ctx->ny || field->nz != ctx->nz) {
        return CFD_ERROR_INVALID;
    }

    /* Call impl directly so its internal loop runs iter=0..max_iter-1,
     * giving compute_source_terms the correct iteration index. */
    cfd_status_t status = rk2_avx2_impl(field, ctx, g, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        ptrdiff_t n_s = (ptrdiff_t)(field->nx * field->ny * field->nz);
        ptrdiff_t ks;
#if defined(_OPENMP) && (_OPENMP >= 201107)
        #pragma omp parallel for reduction(max: max_vel, max_p) schedule(static)
#endif
        for (ks = 0; ks < n_s; ks++) {
            double vel = sqrt(field->u[ks] * field->u[ks] +
                              field->v[ks] * field->v[ks] +
                              field->w[ks] * field->w[ks]);
            if (vel > max_vel) max_vel = vel;
            double ap = fabs(field->p[ks]);
            if (ap > max_p) max_p = ap;
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
        double max_t = 0.0;
        if (field->T) {
            for (ks = 0; ks < n_s; ks++) {
                double at = fabs(field->T[ks]);
                if (at > max_t) max_t = at;
            }
        }
        stats->max_temperature = max_t;
    }
    return status;
#endif
}
