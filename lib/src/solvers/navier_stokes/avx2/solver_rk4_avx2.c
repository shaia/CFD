/**
 * @file solver_rk4_avx2.c
 * @brief RK4 (classical Runge-Kutta) - AVX2 + OpenMP implementation
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
 * Algorithm: identical to scalar/OMP RK4.
 *   k1 = RHS(Q^n)
 *   k2 = RHS(Q^n + (dt/2)*k1)
 *   k3 = RHS(Q^n + (dt/2)*k2)
 *   k4 = RHS(Q^n + dt*k3)
 *   Q^{n+1} = Q^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
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
#include "cfd/solvers/energy_solver.h"
#include "../../energy/energy_solver_internal.h"

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

/* Velocity clamp used by the time-step update (not the RHS kernel).
 * The RHS clamp limits live in ../momentum_rhs/ns_momentum_rhs_avx2.h. */
#define MAX_VELOCITY_LIMIT          100.0

/* ============================================================================
 * CONTEXT
 * ============================================================================ */

typedef struct {
    double* k1_u; double* k1_v; double* k1_w; double* k1_p;
    double* k2_u; double* k2_v; double* k2_w; double* k2_p;
    double* k3_u; double* k3_v; double* k3_w; double* k3_p;
    double* k4_u; double* k4_v; double* k4_w; double* k4_p;
    double* u0;   double* v0;   double* w0;   double* p0;
    double* T_ws;    /* Reusable scratch for the energy step (avoids per-step alloc) */
    double* dx_inv;  /* 1/(2*dx[i]) for i = 0..nx-1 */
    double* dy_inv;  /* 1/(2*dy[j]) for j = 0..ny-1 */
    size_t nx, ny, nz;
    size_t stride_z;
    size_t k_start, k_end;
    double inv_2dz, inv_dz2;
    int initialized;
} rk4_avx2_context_t;

/* Shared AVX2 momentum RHS kernel (also #include-d by solver_rk2_avx2.c).
 * RHS_CTX_T selects this solver's context type for the vectorized kernels;
 * USE_AVX2 and <immintrin.h> are already set above. Provides ns_rhs_point,
 * avx2_clamp, compute_rhs_row, compute_rhs_avx2 and the shared RHS clamp
 * constants. */
#define RHS_CTX_T rk4_avx2_context_t
#include "../momentum_rhs/ns_momentum_rhs_avx2.h"
#undef RHS_CTX_T

/* ============================================================================
 * PUBLIC API DECLARATIONS
 * ============================================================================ */

cfd_status_t rk4_avx2_init(ns_solver_t* solver, const grid* g,
                             const ns_solver_params_t* params);
void         rk4_avx2_destroy(ns_solver_t* solver);
cfd_status_t rk4_avx2_step(ns_solver_t* solver, flow_field* field, const grid* g,
                             const ns_solver_params_t* params, ns_solver_stats_t* stats);


/* ============================================================================
 * AVX2 STAGE UPDATE + MAIN LOOP  (compiled only when AVX2 is available;
 * the vectorized RHS kernels live in ../momentum_rhs/ns_momentum_rhs_avx2.h)
 * ============================================================================ */

#if USE_AVX2

/* Intermediate stage update (parallelized): field = Q^n + factor * k_stage,
 * with velocity clamping. */
static void apply_stage_update_avx2(
    flow_field* field, const rk4_avx2_context_t* ctx,
    const double* ku, const double* kv, const double* kw, const double* kp,
    double factor, ptrdiff_t n_int)
{
    ptrdiff_t kk;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (kk = 0; kk < n_int; kk++) {
        field->u[kk] = fmax(-MAX_VELOCITY_LIMIT,
                      fmin( MAX_VELOCITY_LIMIT, ctx->u0[kk] + factor * ku[kk]));
        field->v[kk] = fmax(-MAX_VELOCITY_LIMIT,
                      fmin( MAX_VELOCITY_LIMIT, ctx->v0[kk] + factor * kv[kk]));
        field->w[kk] = fmax(-MAX_VELOCITY_LIMIT,
                      fmin( MAX_VELOCITY_LIMIT, ctx->w0[kk] + factor * kw[kk]));
        field->p[kk] = ctx->p0[kk] + factor * kp[kk];
    }
}

/* ============================================================================
 * RK4 AVX2 MAIN LOOP
 * ============================================================================ */

static cfd_status_t rk4_avx2_impl(flow_field* field, rk4_avx2_context_t* ctx,
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
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho, field->T,
                         ctx->k1_u, ctx->k1_v, ctx->k1_w, ctx->k1_p,
                         ctx, g, params, iter, dt);

        /* ---- Stage 2: k2 = RHS(Q^n + (dt/2)*k1) ---- */
        apply_stage_update_avx2(field, ctx, ctx->k1_u, ctx->k1_v, ctx->k1_w, ctx->k1_p,
                                0.5 * dt, n_int);
        memset(ctx->k2_u, 0, bytes);
        memset(ctx->k2_v, 0, bytes);
        memset(ctx->k2_w, 0, bytes);
        memset(ctx->k2_p, 0, bytes);
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho, field->T,
                         ctx->k2_u, ctx->k2_v, ctx->k2_w, ctx->k2_p,
                         ctx, g, params, iter, dt);

        /* ---- Stage 3: k3 = RHS(Q^n + (dt/2)*k2) ---- */
        apply_stage_update_avx2(field, ctx, ctx->k2_u, ctx->k2_v, ctx->k2_w, ctx->k2_p,
                                0.5 * dt, n_int);
        memset(ctx->k3_u, 0, bytes);
        memset(ctx->k3_v, 0, bytes);
        memset(ctx->k3_w, 0, bytes);
        memset(ctx->k3_p, 0, bytes);
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho, field->T,
                         ctx->k3_u, ctx->k3_v, ctx->k3_w, ctx->k3_p,
                         ctx, g, params, iter, dt);

        /* ---- Stage 4: k4 = RHS(Q^n + dt*k3) ---- */
        apply_stage_update_avx2(field, ctx, ctx->k3_u, ctx->k3_v, ctx->k3_w, ctx->k3_p,
                                dt, n_int);
        memset(ctx->k4_u, 0, bytes);
        memset(ctx->k4_v, 0, bytes);
        memset(ctx->k4_w, 0, bytes);
        memset(ctx->k4_p, 0, bytes);
        compute_rhs_avx2(field->u, field->v, field->w, field->p, field->rho, field->T,
                         ctx->k4_u, ctx->k4_v, ctx->k4_w, ctx->k4_p,
                         ctx, g, params, iter, dt);

        /* NOTE: Do NOT apply BCs between RK stages.
         * The ghost cells carry zero-derivative evolution (k[ghost]=0),
         * consistent with the semi-discrete ODE. Applying BCs here would
         * reduce RK4 to first-order temporal accuracy. */

        /* ---- Final update: Q^{n+1} = Q^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) ---- */
        {
            double sixth_dt = dt / 6.0;
            ptrdiff_t kk;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (kk = 0; kk < n_int; kk++) {
                field->u[kk] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->u0[kk] + sixth_dt * (ctx->k1_u[kk] + 2.0 * ctx->k2_u[kk] + 2.0 * ctx->k3_u[kk] + ctx->k4_u[kk])));
                field->v[kk] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->v0[kk] + sixth_dt * (ctx->k1_v[kk] + 2.0 * ctx->k2_v[kk] + 2.0 * ctx->k3_v[kk] + ctx->k4_v[kk])));
                field->w[kk] = fmax(-MAX_VELOCITY_LIMIT,
                              fmin( MAX_VELOCITY_LIMIT,
                                    ctx->w0[kk] + sixth_dt * (ctx->k1_w[kk] + 2.0 * ctx->k2_w[kk] + 2.0 * ctx->k3_w[kk] + ctx->k4_w[kk])));
                field->p[kk] = ctx->p0[kk] + sixth_dt * (ctx->k1_p[kk] + 2.0 * ctx->k2_p[kk] + 2.0 * ctx->k3_p[kk] + ctx->k4_p[kk]);
            }
        }

        /* Energy equation: advance temperature after the full RK4 step */
        {
            cfd_status_t energy_status = energy_step_explicit_avx2_with_workspace(
                field, g, params, dt, iter * dt, ctx->T_ws, n);
            if (energy_status != CFD_SUCCESS) {
                status = energy_status;
                goto cleanup;
            }
        }

        /* Apply BCs to final state only, then configured thermal BCs */
        apply_boundary_conditions(field, g);
        status = energy_apply_thermal_bcs(field, params);
        if (status != CFD_SUCCESS) {
            goto cleanup;
        }

        /* NaN / Inf check (parallelized) */
        {
            int has_nan = 0;
            ptrdiff_t kk;
#ifdef _OPENMP
            #pragma omp parallel for reduction(|:has_nan) schedule(static)
#endif
            for (kk = 0; kk < n_int; kk++) {
                if (!isfinite(field->u[kk]) || !isfinite(field->v[kk]) ||
                    !isfinite(field->w[kk]) || !isfinite(field->p[kk])) {
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

cfd_status_t rk4_avx2_init(ns_solver_t* solver, const grid* g,
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

    rk4_avx2_context_t* ctx =
        (rk4_avx2_context_t*)cfd_calloc(1, sizeof(rk4_avx2_context_t));
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
    ctx->k3_u   = (double*)cfd_aligned_malloc(bytes);
    ctx->k3_v   = (double*)cfd_aligned_malloc(bytes);
    ctx->k3_w   = (double*)cfd_aligned_malloc(bytes);
    ctx->k3_p   = (double*)cfd_aligned_malloc(bytes);
    ctx->k4_u   = (double*)cfd_aligned_malloc(bytes);
    ctx->k4_v   = (double*)cfd_aligned_malloc(bytes);
    ctx->k4_w   = (double*)cfd_aligned_malloc(bytes);
    ctx->k4_p   = (double*)cfd_aligned_malloc(bytes);
    ctx->u0     = (double*)cfd_aligned_malloc(bytes);
    ctx->v0     = (double*)cfd_aligned_malloc(bytes);
    ctx->w0     = (double*)cfd_aligned_malloc(bytes);
    ctx->p0     = (double*)cfd_aligned_malloc(bytes);
    ctx->T_ws   = (double*)cfd_aligned_malloc(bytes);
    ctx->dx_inv = (double*)cfd_aligned_malloc(ctx->nx * sizeof(double));
    ctx->dy_inv = (double*)cfd_aligned_malloc(ctx->ny * sizeof(double));

    if (!ctx->k1_u || !ctx->k1_v || !ctx->k1_w || !ctx->k1_p ||
        !ctx->k2_u || !ctx->k2_v || !ctx->k2_w || !ctx->k2_p ||
        !ctx->k3_u || !ctx->k3_v || !ctx->k3_w || !ctx->k3_p ||
        !ctx->k4_u || !ctx->k4_v || !ctx->k4_w || !ctx->k4_p ||
        !ctx->u0   || !ctx->v0   || !ctx->w0   || !ctx->p0   ||
        !ctx->T_ws || !ctx->dx_inv || !ctx->dy_inv) {
        cfd_aligned_free(ctx->k1_u); cfd_aligned_free(ctx->k1_v);
        cfd_aligned_free(ctx->k1_w); cfd_aligned_free(ctx->k1_p);
        cfd_aligned_free(ctx->k2_u); cfd_aligned_free(ctx->k2_v);
        cfd_aligned_free(ctx->k2_w); cfd_aligned_free(ctx->k2_p);
        cfd_aligned_free(ctx->k3_u); cfd_aligned_free(ctx->k3_v);
        cfd_aligned_free(ctx->k3_w); cfd_aligned_free(ctx->k3_p);
        cfd_aligned_free(ctx->k4_u); cfd_aligned_free(ctx->k4_v);
        cfd_aligned_free(ctx->k4_w); cfd_aligned_free(ctx->k4_p);
        cfd_aligned_free(ctx->u0);   cfd_aligned_free(ctx->v0);
        cfd_aligned_free(ctx->w0);   cfd_aligned_free(ctx->p0);
        cfd_aligned_free(ctx->T_ws);
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
    CFD_LOG_INFO("solver", "RK4 SIMD: AVX2 + OpenMP enabled (%d threads)", omp_get_max_threads());
#else
    CFD_LOG_INFO("solver", "RK4 SIMD: AVX2 enabled (OpenMP disabled)");
#endif

    return CFD_SUCCESS;
#endif
}

void rk4_avx2_destroy(ns_solver_t* solver)
{
    if (!solver || !solver->context) {
        return;
    }
    rk4_avx2_context_t* ctx = (rk4_avx2_context_t*)solver->context;
    if (ctx->initialized) {
        cfd_aligned_free(ctx->k1_u); cfd_aligned_free(ctx->k1_v);
        cfd_aligned_free(ctx->k1_w); cfd_aligned_free(ctx->k1_p);
        cfd_aligned_free(ctx->k2_u); cfd_aligned_free(ctx->k2_v);
        cfd_aligned_free(ctx->k2_w); cfd_aligned_free(ctx->k2_p);
        cfd_aligned_free(ctx->k3_u); cfd_aligned_free(ctx->k3_v);
        cfd_aligned_free(ctx->k3_w); cfd_aligned_free(ctx->k3_p);
        cfd_aligned_free(ctx->k4_u); cfd_aligned_free(ctx->k4_v);
        cfd_aligned_free(ctx->k4_w); cfd_aligned_free(ctx->k4_p);
        cfd_aligned_free(ctx->u0);   cfd_aligned_free(ctx->v0);
        cfd_aligned_free(ctx->w0);   cfd_aligned_free(ctx->p0);
        cfd_aligned_free(ctx->T_ws);
        cfd_aligned_free(ctx->dx_inv);
        cfd_aligned_free(ctx->dy_inv);
    }
    cfd_free(ctx);
    solver->context = NULL;
}

cfd_status_t rk4_avx2_step(ns_solver_t* solver, flow_field* field, const grid* g,
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

    rk4_avx2_context_t* ctx = (rk4_avx2_context_t*)solver->context;
    if (field->nx != ctx->nx || field->ny != ctx->ny || field->nz != ctx->nz) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    cfd_status_t status = rk4_avx2_impl(field, ctx, g, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        ptrdiff_t n_s = (ptrdiff_t)(field->nx * field->ny * field->nz);
        double max_t = (field->T && n_s > 0) ? field->T[0] : 0.0;
        ptrdiff_t ks;
#if defined(_OPENMP) && (_OPENMP >= 201107)
        #pragma omp parallel for reduction(max: max_vel, max_p, max_t) schedule(static)
#endif
        for (ks = 0; ks < n_s; ks++) {
            double vel = sqrt(field->u[ks] * field->u[ks] +
                              field->v[ks] * field->v[ks] +
                              field->w[ks] * field->w[ks]);
            if (vel > max_vel) max_vel = vel;
            double ap = fabs(field->p[ks]);
            if (ap > max_p) max_p = ap;
            if (field->T && field->T[ks] > max_t) max_t = field->T[ks];
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
        stats->max_temperature = max_t;
    }

    return status;
#endif
}

cfd_status_t rk4_avx2_solve(ns_solver_t* solver, flow_field* field, const grid* g,
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

    rk4_avx2_context_t* ctx = (rk4_avx2_context_t*)solver->context;
    if (field->nx != ctx->nx || field->ny != ctx->ny || field->nz != ctx->nz) {
        return CFD_ERROR_INVALID;
    }

    /* Call impl directly so its internal loop runs iter=0..max_iter-1,
     * giving compute_source_terms the correct iteration index. */
    cfd_status_t status = rk4_avx2_impl(field, ctx, g, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        ptrdiff_t n_s = (ptrdiff_t)(field->nx * field->ny * field->nz);
        double max_t = (field->T && n_s > 0) ? field->T[0] : 0.0;
        ptrdiff_t ks;
#if defined(_OPENMP) && (_OPENMP >= 201107)
        #pragma omp parallel for reduction(max: max_vel, max_p, max_t) schedule(static)
#endif
        for (ks = 0; ks < n_s; ks++) {
            double vel = sqrt(field->u[ks] * field->u[ks] +
                              field->v[ks] * field->v[ks] +
                              field->w[ks] * field->w[ks]);
            if (vel > max_vel) max_vel = vel;
            double ap = fabs(field->p[ks]);
            if (ap > max_p) max_p = ap;
            if (field->T && field->T[ks] > max_t) max_t = field->T[ks];
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
        stats->max_temperature = max_t;
    }
    return status;
#endif
}
