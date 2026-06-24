/**
 * @file ns_momentum_rhs_avx2.h
 * @brief Shared AVX2 momentum RHS kernel for the explicit RK time integrators.
 *
 * AVX2 + OpenMP counterpart of ns_momentum_rhs_scalar.h, used identically by the
 * AVX2 RK2 (`solver_rk2_avx2.c`) and RK4 (`solver_rk4_avx2.c`) solvers. The
 * vectorized kernels read only the geometry/coefficient fields common to both
 * solvers' context structs (`nx, ny, nz, stride_z, k_start, k_end, inv_2dz,
 * inv_dz2, dx_inv, dy_inv`), so the includer selects its own context type via:
 *
 *     #define RHS_CTX_T rk4_avx2_context_t
 *     #include "../momentum_rhs/ns_momentum_rhs_avx2.h"
 *     #undef RHS_CTX_T
 *
 * (mirrors the SIMD template idiom in lib/src/solvers/linear/simd_template/).
 *
 * Include requirements (the includer must satisfy these BEFORE #include-ing):
 *   - <math.h>, <stddef.h>; <immintrin.h> when CFD_HAS_AVX2; <omp.h> when _OPENMP
 *   - USE_AVX2 #defined (1 when CFD_HAS_AVX2, else 0)
 *   - RHS_CTX_T #defined to the includer's AVX2 context struct type
 *   - "cfd/core/grid.h" (grid), "cfd/solvers/navier_stokes_solver.h"
 *     (ns_solver_params_t)
 *   - "../../energy/energy_solver_internal.h" (compute_source_terms,
 *     energy_compute_buoyancy)
 *
 * This header is #include-only and is never compiled as a standalone TU.
 */
#ifndef CFD_NS_MOMENTUM_RHS_AVX2_H
#define CFD_NS_MOMENTUM_RHS_AVX2_H

/* Physical stability limits (shared by the RK2/RK4 AVX2 RHS kernel) */
#ifndef MAX_DERIVATIVE_LIMIT
#define MAX_DERIVATIVE_LIMIT        100.0
#endif
#ifndef MAX_SECOND_DERIVATIVE_LIMIT
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#endif
#ifndef MAX_DIVERGENCE_LIMIT
#define MAX_DIVERGENCE_LIMIT        10.0
#endif
#ifndef PRESSURE_UPDATE_FACTOR
#define PRESSURE_UPDATE_FACTOR      0.1
#endif

/* ============================================================================
 * SCALAR POINT HELPER
 * Handles a single interior grid point given explicit stencil indices.
 * Used for periodic-edge columns (i=1 with il wrap, i=nx-2 with ir wrap)
 * and for scalar remainder after the AVX2 group.
 * ============================================================================ */

static void ns_rhs_point(
    const double* u, const double* v, const double* w,
    const double* p, const double* rho, const double* T,
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

    /* Boussinesq buoyancy source (no-op when beta == 0) */
    if (T) {
        energy_compute_buoyancy(T[idx], params, &source_u, &source_v, &source_w);
    }

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
    const double* p, const double* rho, const double* T,
    double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
    const RHS_CTX_T* ctx, const grid* g,
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
        ns_rhs_point(u, v, w, p, rho, T, rhs_u, rhs_v, rhs_w, rhs_p, g, params,
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
                /* Boussinesq buoyancy source (no-op when beta == 0) */
                if (T) {
                    energy_compute_buoyancy(T[j_off + (size_t)i + (size_t)lane], params,
                                            &src_u_arr[lane], &src_v_arr[lane],
                                            &src_w_arr[lane]);
                }
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
            ns_rhs_point(u, v, w, p, rho, T, rhs_u, rhs_v, rhs_w, rhs_p, g, params,
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
    const double* p, const double* rho, const double* T,
    double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
    const RHS_CTX_T* ctx, const grid* g,
    const ns_solver_params_t* params, int iter, double dt)
{
    ptrdiff_t ny_int = (ptrdiff_t)ctx->ny;
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        ptrdiff_t j;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (j = 1; j < ny_int - 1; j++) {
            compute_rhs_row(u, v, w, p, rho, T, rhs_u, rhs_v, rhs_w, rhs_p,
                            ctx, g, params, (size_t)j, k, iter, dt);
        }
    }
}

#endif /* USE_AVX2 */

#endif /* CFD_NS_MOMENTUM_RHS_AVX2_H */
