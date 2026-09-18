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

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/energy_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "../../energy/energy_solver_internal.h"
#include "../../turbulence/turbulence_solver_internal.h"

#include "../boundary_copy_utils.h"
#include "../ns_convection_internal.h"
#include "../ns_simd_backend_internal.h"
#include "upwind_avx2.h"

#include <math.h>
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
    double* T_ws;    /* Reusable scratch for the energy step (avoids per-step alloc) */
    double* turb_ws; /* Reusable scratch for the turbulence step; always allocated */
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
    int iter_count;  /* Step counter for advancing time-dependent thermal terms */
} explicit_euler_simd_context;

// Public API functions
cfd_status_t explicit_euler_simd_init(struct NSSolver* solver, const grid* grid,
                                      const ns_solver_params_t* params);
void explicit_euler_simd_destroy(struct NSSolver* solver);
cfd_status_t explicit_euler_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params, ns_solver_stats_t* stats);

cfd_status_t explicit_euler_simd_init(struct NSSolver* solver, const grid* grid,
                                      const ns_solver_params_t* params) {
    if (!solver || !grid) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t scheme_status = ns_check_convection_scheme(params, 1);
    if (scheme_status != CFD_SUCCESS) {
        return scheme_status;
    }

    /* No scalar fallback: a SIMD solver that quietly runs scalar kernels turns a
     * configuration error into a performance mystery. Fail so the caller can pick
     * the scalar "explicit_euler" deliberately. */
    cfd_status_t simd_status = ns_check_simd_backend();
    if (simd_status != CFD_SUCCESS) {
        return simd_status;
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

    size_t n_total = ctx->nx * ctx->ny * ctx->nz;
    ctx->u_new   = (double*)cfd_aligned_malloc(field_size);
    ctx->v_new   = (double*)cfd_aligned_malloc(field_size);
    ctx->w_new   = (double*)cfd_aligned_malloc(field_size);
    ctx->p_new   = (double*)cfd_aligned_malloc(field_size);
    ctx->T_ws    = (double*)cfd_aligned_malloc(field_size);
    ctx->turb_ws = (double*)cfd_calloc(TURB_WORKSPACE_SIZE(n_total), sizeof(double));
    ctx->dx_inv  = (double*)cfd_aligned_malloc(ctx->nx * sizeof(double));
    ctx->dy_inv  = (double*)cfd_aligned_malloc(ctx->ny * sizeof(double));

    if (!ctx->u_new || !ctx->v_new || !ctx->w_new || !ctx->p_new || !ctx->T_ws ||
        !ctx->turb_ws || !ctx->dx_inv || !ctx->dy_inv) {
        if (ctx->u_new)   { cfd_aligned_free(ctx->u_new); }
        if (ctx->v_new)   { cfd_aligned_free(ctx->v_new); }
        if (ctx->w_new)   { cfd_aligned_free(ctx->w_new); }
        if (ctx->p_new)   { cfd_aligned_free(ctx->p_new); }
        if (ctx->T_ws)    { cfd_aligned_free(ctx->T_ws); }
        if (ctx->turb_ws) { cfd_free(ctx->turb_ws); }
        if (ctx->dx_inv)  { cfd_aligned_free(ctx->dx_inv); }
        if (ctx->dy_inv)  { cfd_aligned_free(ctx->dy_inv); }
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

#ifdef _OPENMP
    CFD_LOG_INFO("solver", "Explicit Euler SIMD: AVX2 + OpenMP enabled (%d threads)", omp_get_max_threads());
#else
    CFD_LOG_INFO("solver", "Explicit Euler SIMD: AVX2 enabled (OpenMP disabled)");
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
            cfd_aligned_free(ctx->T_ws);
            cfd_free(ctx->turb_ws);
            cfd_aligned_free(ctx->dx_inv);
            cfd_aligned_free(ctx->dy_inv);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

/* Momentum source terms at column i of row j, shared by the vector lanes and
 * the scalar row. Iteration 0 matches the scalar explicit_euler step, which
 * runs every step as a one-iteration solve. */
static void cell_source_terms(const grid* grid, const ns_solver_params_t* params, size_t i,
                              size_t j, double z_coord, double dt, double* source_u,
                              double* source_v, double* source_w) {
    compute_source_terms(grid->x[i], grid->y[j], z_coord, 0, dt, params,
                         source_u, source_v, source_w);
}

#if USE_AVX
static inline __m256d vector_fmax(__m256d a, __m256d b) {
    return _mm256_max_pd(a, b);
}
static inline __m256d vector_fmin(__m256d a, __m256d b) {
    return _mm256_min_pd(a, b);
}

typedef struct {
    double dt;
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
    __m256d inv_dz_vec;
    __m256d inv_dz2_vec;
    /* Boussinesq buoyancy: accel = -beta*(T - T_ref)*g. Zero beta -> no-op. */
    __m256d neg_beta_vec;
    __m256d t_ref_vec;
    __m256d gx_vec;
    __m256d gy_vec;
    __m256d gz_vec;
    int upwind;      /* First-order upwind convective derivatives */
    int has_source;  /* Momentum sources can be nonzero; skips per-lane evaluation otherwise */
} simd_constants;

static void init_simd_constants(simd_constants* c, const ns_solver_params_t* params,
                                double conservative_dt, double inv_2dz, double inv_dz2) {
    c->dt = conservative_dt;
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
    c->inv_dz_vec = _mm256_set1_pd(2.0 * inv_2dz);
    c->inv_dz2_vec = _mm256_set1_pd(inv_dz2);
    c->neg_beta_vec = _mm256_set1_pd(-params->beta);
    c->t_ref_vec = _mm256_set1_pd(params->T_ref);
    c->gx_vec = _mm256_set1_pd(params->gravity[0]);
    c->gy_vec = _mm256_set1_pd(params->gravity[1]);
    c->gz_vec = _mm256_set1_pd(params->gravity[2]);
    c->upwind = (params->convection_scheme == NS_CONVECTION_SCHEME_UPWIND);
    c->has_source = (params->source_func != NULL || params->source_amplitude_u != 0.0 ||
                     params->source_amplitude_v != 0.0);
}

static inline __m256d clamp_deriv(const simd_constants* sc, __m256d x) {
    return vector_fmax(sc->min_deriv, vector_fmin(sc->max_deriv, x));
}

/* Vectorized update of row j in 4-wide groups. Returns the first interior
 * column not processed (1-3 columns remain when (nx-2) % 4 != 0); the caller
 * finishes the row with the scalar row path. */
static size_t process_simd_row(explicit_euler_simd_context* ctx, flow_field* field, const grid* grid,
                               const ns_solver_params_t* params, size_t j,
                               const simd_constants* sc, double z_coord,
                               size_t stride_z, size_t k_offset) {
    double dy2 = grid->dy[j] * grid->dy[j];
    __m256d dy_inv_val = _mm256_set1_pd(ctx->dy_inv[j]);
    __m256d dy2_val = _mm256_set1_pd(dy2);
    __m256d dy2_recip = _mm256_div_pd(sc->one_vec, dy2_val);

    size_t i = 1;
    for (; i + 3 < ctx->nx - 1; i += 4) {
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

        /* Convective derivatives: the clamped central ones, or first-order
         * upwind. The divergence below stays central. */
        __m256d cdu_dx = du_dx, cdu_dy = du_dy, cdu_dz = du_dz;
        __m256d cdv_dx = dv_dx, cdv_dy = dv_dy, cdv_dz = dv_dz;
        __m256d cdw_dx = dw_dx, cdw_dy = dw_dy, cdw_dz = dw_dz;
        if (sc->upwind) {
            /* dx_inv and dy_inv hold 1/(2h); upwind differences need 1/h */
            __m256d inv_dx = _mm256_add_pd(dx_inv_val, dx_inv_val);
            __m256d inv_dy = _mm256_add_pd(dy_inv_val, dy_inv_val);
            cdu_dx = clamp_deriv(sc, upwind_deriv_avx2(u, u_xm, u, u_xp, inv_dx));
            cdu_dy = clamp_deriv(sc, upwind_deriv_avx2(v, u_ym, u, u_yp, inv_dy));
            cdu_dz = clamp_deriv(sc, upwind_deriv_avx2(w, u_zm, u, u_zp, sc->inv_dz_vec));
            cdv_dx = clamp_deriv(sc, upwind_deriv_avx2(u, v_xm, v, v_xp, inv_dx));
            cdv_dy = clamp_deriv(sc, upwind_deriv_avx2(v, v_ym, v, v_yp, inv_dy));
            cdv_dz = clamp_deriv(sc, upwind_deriv_avx2(w, v_zm, v, v_zp, sc->inv_dz_vec));
            cdw_dx = clamp_deriv(sc, upwind_deriv_avx2(u, w_xm, w, w_xp, inv_dx));
            cdw_dy = clamp_deriv(sc, upwind_deriv_avx2(v, w_ym, w, w_yp, inv_dy));
            cdw_dz = clamp_deriv(sc, upwind_deriv_avx2(w, w_zm, w, w_zp, sc->inv_dz_vec));
        }

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
            _mm256_add_pd(_mm256_mul_pd(u, cdu_dx), _mm256_mul_pd(v, cdu_dy)),
            _mm256_mul_pd(w, cdu_dz));
        __m256d du =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_u, term_pres_x),
                                                    _mm256_sub_pd(sc->zero, conv_u)));

        __m256d term_pres_y = _mm256_mul_pd(dp_dy, rho_inv);
        __m256d term_visc_v = _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2v_dx2, d2v_dy2), d2v_dz2));
        __m256d conv_v = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(u, cdv_dx), _mm256_mul_pd(v, cdv_dy)),
            _mm256_mul_pd(w, cdv_dz));
        __m256d dv =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_v, term_pres_y),
                                                    _mm256_sub_pd(sc->zero, conv_v)));

        __m256d term_pres_z = _mm256_mul_pd(dp_dz, rho_inv);
        __m256d term_visc_w = _mm256_mul_pd(nu, _mm256_add_pd(_mm256_add_pd(d2w_dx2, d2w_dy2), d2w_dz2));
        __m256d conv_w = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(u, cdw_dx), _mm256_mul_pd(v, cdw_dy)),
            _mm256_mul_pd(w, cdw_dz));
        __m256d dw =
            _mm256_mul_pd(sc->dt_vec, _mm256_add_pd(_mm256_sub_pd(term_visc_w, term_pres_z),
                                                    _mm256_sub_pd(sc->zero, conv_w)));

        /* Momentum sources: dvel += dt * source, evaluated per lane */
        if (sc->has_source) {
            double src_u[4] = {0.0, 0.0, 0.0, 0.0};
            double src_v[4] = {0.0, 0.0, 0.0, 0.0};
            double src_w[4] = {0.0, 0.0, 0.0, 0.0};
            for (size_t lane = 0; lane < 4; lane++) {
                cell_source_terms(grid, params, i + lane, j, z_coord, sc->dt,
                                  &src_u[lane], &src_v[lane], &src_w[lane]);
            }
            du = _mm256_add_pd(du, _mm256_mul_pd(sc->dt_vec, _mm256_loadu_pd(src_u)));
            dv = _mm256_add_pd(dv, _mm256_mul_pd(sc->dt_vec, _mm256_loadu_pd(src_v)));
            dw = _mm256_add_pd(dw, _mm256_mul_pd(sc->dt_vec, _mm256_loadu_pd(src_w)));
        }

        /* Boussinesq buoyancy: dvel += dt * (-beta*(T - T_ref)) * g.
         * When beta == 0 this adds exactly 0.0, matching the scalar path. */
        __m256d Tcell = _mm256_loadu_pd(&field->T[idx]);
        __m256d buoy = _mm256_mul_pd(sc->neg_beta_vec, _mm256_sub_pd(Tcell, sc->t_ref_vec));
        __m256d dt_buoy = _mm256_mul_pd(sc->dt_vec, buoy);
        du = _mm256_add_pd(du, _mm256_mul_pd(dt_buoy, sc->gx_vec));
        dv = _mm256_add_pd(dv, _mm256_mul_pd(dt_buoy, sc->gy_vec));
        dw = _mm256_add_pd(dw, _mm256_mul_pd(dt_buoy, sc->gz_vec));

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
    return i;
}
#endif

/* Scalar update of row j for interior columns i_start..nx-2 */
static void process_scalar_row_turb(explicit_euler_simd_context* ctx, flow_field* field,
                                    const grid* grid, const ns_solver_params_t* params, size_t j,
                                    double conservative_dt, double z_coord, size_t stride_z,
                                    size_t k_offset, int turb_on_flag, size_t i_start) {
    for (size_t i = i_start; i < ctx->nx - 1; i++) {
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

        /* Convective derivatives: the clamped central ones, or first-order
         * upwind. The divergence below stays central. */
        ns_conv_derivs_t cd = {du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz};
        if (params->convection_scheme == NS_CONVECTION_SCHEME_UPWIND) {
            ns_upwind_conv_derivs(field->u, field->v, field->w, idx, idx - 1, idx + 1,
                                  idx - ctx->nx, idx + ctx->nx, idx - stride_z, idx + stride_z,
                                  grid->dx[i], grid->dy[j], 2.0 * ctx->inv_2dz, &cd);
            ns_clamp_conv_derivs(&cd, MAX_DERIVATIVE_LIMIT);
        }

        double source_u = 0.0;
        double source_v = 0.0;
        double source_w = 0.0;
        cell_source_terms(grid, params, i, j, z_coord, conservative_dt,
                          &source_u, &source_v, &source_w);

        /* Boussinesq buoyancy source (no-op when beta == 0) */
        energy_compute_buoyancy(field->T[idx], params, &source_u, &source_v, &source_w);

        double u_c = field->u[idx];
        double v_c = field->v[idx];
        double w_c = field->w[idx];

        double visc_u, visc_v, visc_w;
        if (!turb_on_flag) {
            visc_u = nu * (d2u_dx2 + d2u_dy2 + d2u_dz2);
            visc_v = nu * (d2v_dx2 + d2v_dy2 + d2v_dz2);
            visc_w = nu * (d2w_dx2 + d2w_dy2 + d2w_dz2);
        } else {
            const double* nu_t = field->nu_t;
            double dx2 = grid->dx[i] * grid->dx[i];
            double dy2 = grid->dy[j] * grid->dy[j];
            /* Face-averaged nu_eff = nu + nu_t (nu_t is bounded by the
             * realizability clamp) */
            double nu_xp = nu + 0.5 * (nu_t[idx] + nu_t[idx + 1]);
            double nu_xm = nu + 0.5 * (nu_t[idx] + nu_t[idx - 1]);
            double nu_yp = nu + 0.5 * (nu_t[idx] + nu_t[idx + ctx->nx]);
            double nu_ym = nu + 0.5 * (nu_t[idx] + nu_t[idx - ctx->nx]);
            visc_u = (nu_xp * (field->u[idx + 1]      - u_c) -
                      nu_xm * (u_c - field->u[idx - 1])) / dx2 +
                     (nu_yp * (field->u[idx + ctx->nx] - u_c) -
                      nu_ym * (u_c - field->u[idx - ctx->nx])) / dy2 +
                     nu * d2u_dz2;
            visc_v = (nu_xp * (field->v[idx + 1]      - v_c) -
                      nu_xm * (v_c - field->v[idx - 1])) / dx2 +
                     (nu_yp * (field->v[idx + ctx->nx] - v_c) -
                      nu_ym * (v_c - field->v[idx - ctx->nx])) / dy2 +
                     nu * d2v_dz2;
            visc_w = (nu_xp * (field->w[idx + 1]      - w_c) -
                      nu_xm * (w_c - field->w[idx - 1])) / dx2 +
                     (nu_yp * (field->w[idx + ctx->nx] - w_c) -
                      nu_ym * (w_c - field->w[idx - ctx->nx])) / dy2 +
                     nu * d2w_dz2;
            visc_u = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, visc_u));
            visc_v = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, visc_v));
            visc_w = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, visc_w));
        }

        double du = conservative_dt * (-u_c * cd.du_dx - v_c * cd.du_dy - w_c * cd.du_dz -
                                       dp_dx / rho + visc_u + source_u);
        double dv = conservative_dt * (-u_c * cd.dv_dx - v_c * cd.dv_dy - w_c * cd.dv_dz -
                                       dp_dy / rho + visc_v + source_v);
        double dw = conservative_dt * (-u_c * cd.dw_dx - v_c * cd.dw_dy - w_c * cd.dw_dz -
                                       dp_dz / rho + visc_w + source_w);

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

cfd_status_t explicit_euler_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !solver->context || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }

    explicit_euler_simd_context* ctx = (explicit_euler_simd_context*)solver->context;
    const int turb_on = (params->turb_model != TURB_MODEL_NONE);

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
    int ny_int = (int)(ctx->ny);
    int j;
    if (!turb_on) {
        simd_constants sc;
        init_simd_constants(&sc, params, conservative_dt, ctx->inv_2dz, ctx->inv_dz2);
        for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
            size_t k_offset = k * ctx->stride_z;
            double z_coord = (ctx->nz > 1 && grid->z) ? grid->z[k] : 0.0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (j = 1; j < ny_int - 1; j++) {
                size_t i_tail = process_simd_row(ctx, field, grid, params, (size_t)j, &sc,
                                                 z_coord, ctx->stride_z, k_offset);
                process_scalar_row_turb(ctx, field, grid, params, (size_t)j, conservative_dt,
                                        z_coord, ctx->stride_z, k_offset, 0, i_tail);
            }
        }
    } else {
        for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
            size_t k_offset = k * ctx->stride_z;
            double z_coord = (ctx->nz > 1 && grid->z) ? grid->z[k] : 0.0;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (j = 1; j < ny_int - 1; j++) {
                process_scalar_row_turb(ctx, field, grid, params, (size_t)j, conservative_dt,
                                        z_coord, ctx->stride_z, k_offset, 1, 1);
            }
        }
    }
#endif

    // Apply boundary, check NaNs, etc.
    memcpy(field->u, ctx->u_new, size * sizeof(double));
    memcpy(field->v, ctx->v_new, size * sizeof(double));
    memcpy(field->w, ctx->w_new, size * sizeof(double));
    memcpy(field->p, ctx->p_new, size * sizeof(double));

    // Energy equation: advance temperature using updated velocity
    {
        cfd_status_t energy_status = energy_step_explicit_avx2_with_workspace(
            field, grid, params, conservative_dt,
            ctx->iter_count * conservative_dt, ctx->T_ws, size);
        if (energy_status != CFD_SUCCESS) {
            return energy_status;
        }
    }

    // Store caller-set boundary values before apply_boundary_conditions overwrites them,
    // then restore them. Then apply configured thermal BCs.
    copy_boundary_velocities_3d(ctx->u_new, ctx->v_new, ctx->w_new,
                                field->u, field->v, field->w, nx, ny, ctx->nz);
    apply_boundary_conditions(field, grid);
    copy_boundary_velocities_3d(field->u, field->v, field->w,
                                ctx->u_new, ctx->v_new, ctx->w_new, nx, ny, ctx->nz);
    {
        cfd_status_t bc_status = energy_apply_thermal_bcs(field, params);
        if (bc_status != CFD_SUCCESS) {
            return bc_status;
        }
    }

    /* Turbulence transport: advance k-eps/SA with the updated velocity,
     * then apply turbulence BCs. nu_t is updated once per step. */
    {
        cfd_status_t turb_status = turbulence_step_explicit_avx2_with_workspace(
            field, grid, params, conservative_dt,
            ctx->iter_count * conservative_dt, ctx->turb_ws,
            turb_on ? TURB_WORKSPACE_SIZE(size) : 0);
        if (turb_status == CFD_SUCCESS) {
            turb_status = turbulence_apply_bcs(field, grid, params);
        }
        if (turb_status != CFD_SUCCESS) {
            return turb_status;
        }
    }

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

    ctx->iter_count++;
    return CFD_SUCCESS;
}
