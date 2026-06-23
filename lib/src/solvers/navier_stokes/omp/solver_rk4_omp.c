/**
 * @file solver_rk4_omp.c
 * @brief RK4 (classical Runge-Kutta) time integration - OpenMP parallelized
 *
 * Same algorithm as scalar RK4 but with OpenMP-parallelized loops.
 * Achieves O(dt^4) temporal accuracy with multi-threaded execution.
 *
 *   k1 = RHS(Q^n)
 *   k2 = RHS(Q^n + (dt/2)*k1)
 *   k3 = RHS(Q^n + (dt/2)*k2)
 *   k4 = RHS(Q^n + dt*k3)
 *   Q^{n+1} = Q^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing bit-identical results to the 2D code path.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/energy_solver.h"
#include "../../energy/energy_solver_internal.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

/* OpenMP version-gated pragmas */
#if _OPENMP >= 201307  /* OMP 4.0 */
#  define OMP_FOR_SIMD _Pragma("omp parallel for simd schedule(static)")
#else
#  define OMP_FOR_SIMD _Pragma("omp parallel for schedule(static)")
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Velocity clamp used by the time-step update (not the RHS kernel) */
#define MAX_VELOCITY_LIMIT          100.0

/* Shared OpenMP momentum RHS kernel (also #include-d by solver_rk2_omp.c).
 * Included inside the CFD_ENABLE_OPENMP block, after <omp.h>/<math.h>/grid/
 * energy. Defines the shared clamp constants MAX_DERIVATIVE_LIMIT /
 * MAX_SECOND_DERIVATIVE_LIMIT / MAX_DIVERGENCE_LIMIT / PRESSURE_UPDATE_FACTOR. */
#include "../momentum_rhs/ns_momentum_rhs_omp.h"


/* Intermediate stage update (parallelized): field = Q^n + factor * k_stage,
 * with velocity clamping. */
static void apply_stage_update_omp(flow_field* field,
                                   const double* u0, const double* v0,
                                   const double* w0, const double* p0,
                                   const double* ku, const double* kv,
                                   const double* kw, const double* kp,
                                   double factor, ptrdiff_t total_int) {
    ptrdiff_t kk;
    OMP_FOR_SIMD
    for (kk = 0; kk < total_int; kk++) {
        field->u[kk] = u0[kk] + factor * ku[kk];
        field->v[kk] = v0[kk] + factor * kv[kk];
        field->w[kk] = w0[kk] + factor * kw[kk];
        field->p[kk] = p0[kk] + factor * kp[kk];

        field->u[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[kk]));
        field->v[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[kk]));
        field->w[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[kk]));
    }
}

/* ============================================================================
 * RK4 OMP SOLVER
 * ============================================================================ */

cfd_status_t rk4_omp_impl(flow_field* field, const grid* grid,
                            const ns_solver_params_t* params) {
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;

    /* Reject non-uniform z-spacing (solver uses constant inv_2dz/inv_dz2) */
    if (nz > 1 && grid->dz) {
        for (size_t k = 1; k < nz - 1; k++) {
            if (fabs(grid->dz[k] - grid->dz[0]) > 1e-14) {
                return CFD_ERROR_INVALID;
            }
        }
    }

    size_t plane = nx * ny;
    size_t total = plane * nz;
    size_t bytes = total * sizeof(double);

    /* Branch-free 3D constants */
    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start  = (nz > 1) ? 1 : 0;
    size_t k_end    = (nz > 1) ? (nz - 1) : 1;
    double inv_2dz  = (nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    double inv_dz2  = (nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    /* Allocate working arrays */
    double* k1_u = (double*)cfd_calloc(total, sizeof(double));
    double* k1_v = (double*)cfd_calloc(total, sizeof(double));
    double* k1_w = (double*)cfd_calloc(total, sizeof(double));
    double* k1_p = (double*)cfd_calloc(total, sizeof(double));
    double* k2_u = (double*)cfd_calloc(total, sizeof(double));
    double* k2_v = (double*)cfd_calloc(total, sizeof(double));
    double* k2_w = (double*)cfd_calloc(total, sizeof(double));
    double* k2_p = (double*)cfd_calloc(total, sizeof(double));
    double* k3_u = (double*)cfd_calloc(total, sizeof(double));
    double* k3_v = (double*)cfd_calloc(total, sizeof(double));
    double* k3_w = (double*)cfd_calloc(total, sizeof(double));
    double* k3_p = (double*)cfd_calloc(total, sizeof(double));
    double* k4_u = (double*)cfd_calloc(total, sizeof(double));
    double* k4_v = (double*)cfd_calloc(total, sizeof(double));
    double* k4_w = (double*)cfd_calloc(total, sizeof(double));
    double* k4_p = (double*)cfd_calloc(total, sizeof(double));
    double* u0 = (double*)cfd_calloc(total, sizeof(double));
    double* v0 = (double*)cfd_calloc(total, sizeof(double));
    double* w0 = (double*)cfd_calloc(total, sizeof(double));
    double* p0 = (double*)cfd_calloc(total, sizeof(double));
    int needs_T_ws = (params->alpha > 0.0 || params->beta != 0.0);
    double* T_energy_ws = needs_T_ws
        ? (double*)cfd_calloc(total, sizeof(double)) : NULL;

    if (!k1_u || !k1_v || !k1_w || !k1_p ||
        !k2_u || !k2_v || !k2_w || !k2_p ||
        !k3_u || !k3_v || !k3_w || !k3_p ||
        !k4_u || !k4_v || !k4_w || !k4_p ||
        !u0 || !v0 || !w0 || !p0 ||
        (needs_T_ws && !T_energy_ws)) {
        cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_w); cfd_free(k1_p);
        cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_w); cfd_free(k2_p);
        cfd_free(k3_u); cfd_free(k3_v); cfd_free(k3_w); cfd_free(k3_p);
        cfd_free(k4_u); cfd_free(k4_v); cfd_free(k4_w); cfd_free(k4_p);
        cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);
        cfd_free(T_energy_ws);
        return CFD_ERROR_NOMEM;
    }

    double dt = params->dt;
    ptrdiff_t total_int = (ptrdiff_t)total;
    cfd_status_t status = CFD_SUCCESS;

    for (int iter = 0; iter < params->max_iter; iter++) {
        /* Save Q^n */
        memcpy(u0, field->u, bytes);
        memcpy(v0, field->v, bytes);
        memcpy(w0, field->w, bytes);
        memcpy(p0, field->p, bytes);

        /* ---- Stage 1: k1 = RHS(Q^n) ---- */
        memset(k1_u, 0, bytes);
        memset(k1_v, 0, bytes);
        memset(k1_w, 0, bytes);
        memset(k1_p, 0, bytes);
        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho, field->T,
                         k1_u, k1_v, k1_w, k1_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* ---- Stage 2: k2 = RHS(Q^n + (dt/2)*k1) ---- */
        apply_stage_update_omp(field, u0, v0, w0, p0, k1_u, k1_v, k1_w, k1_p,
                               0.5 * dt, total_int);
        memset(k2_u, 0, bytes);
        memset(k2_v, 0, bytes);
        memset(k2_w, 0, bytes);
        memset(k2_p, 0, bytes);
        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho, field->T,
                         k2_u, k2_v, k2_w, k2_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* ---- Stage 3: k3 = RHS(Q^n + (dt/2)*k2) ---- */
        apply_stage_update_omp(field, u0, v0, w0, p0, k2_u, k2_v, k2_w, k2_p,
                               0.5 * dt, total_int);
        memset(k3_u, 0, bytes);
        memset(k3_v, 0, bytes);
        memset(k3_w, 0, bytes);
        memset(k3_p, 0, bytes);
        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho, field->T,
                         k3_u, k3_v, k3_w, k3_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* ---- Stage 4: k4 = RHS(Q^n + dt*k3) ---- */
        apply_stage_update_omp(field, u0, v0, w0, p0, k3_u, k3_v, k3_w, k3_p,
                               dt, total_int);
        memset(k4_u, 0, bytes);
        memset(k4_v, 0, bytes);
        memset(k4_w, 0, bytes);
        memset(k4_p, 0, bytes);
        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho, field->T,
                         k4_u, k4_v, k4_w, k4_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* NOTE: Do NOT apply BCs between RK stages. The ghost cells carry
         * zero-derivative evolution (k[ghost]=0), consistent with the
         * semi-discrete ODE. Applying BCs here would reduce RK4 to
         * first-order temporal accuracy. */

        /* ---- Final update: Q^{n+1} = Q^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) ---- */
        {
            double sixth_dt = dt / 6.0;
            ptrdiff_t kk;
            OMP_FOR_SIMD
            for (kk = 0; kk < total_int; kk++) {
                field->u[kk] = u0[kk] + sixth_dt * (k1_u[kk] + 2.0 * k2_u[kk] + 2.0 * k3_u[kk] + k4_u[kk]);
                field->v[kk] = v0[kk] + sixth_dt * (k1_v[kk] + 2.0 * k2_v[kk] + 2.0 * k3_v[kk] + k4_v[kk]);
                field->w[kk] = w0[kk] + sixth_dt * (k1_w[kk] + 2.0 * k2_w[kk] + 2.0 * k3_w[kk] + k4_w[kk]);
                field->p[kk] = p0[kk] + sixth_dt * (k1_p[kk] + 2.0 * k2_p[kk] + 2.0 * k3_p[kk] + k4_p[kk]);

                field->u[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[kk]));
                field->v[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[kk]));
                field->w[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[kk]));
            }
        }

        /* Energy equation: advance temperature after the full RK4 velocity update */
        {
            cfd_status_t energy_status = energy_step_explicit_omp_with_workspace(
                field, grid, params, dt, iter * dt, T_energy_ws, total);
            if (energy_status != CFD_SUCCESS) {
                status = energy_status;
                goto cleanup;
            }
        }

        /* Apply BCs to final state only (after the full RK4 step).
         * Then apply configured thermal BCs (overwrites periodic T values). */
        apply_boundary_conditions(field, grid);
        status = energy_apply_thermal_bcs(field, params);
        if (status != CFD_SUCCESS) {
            goto cleanup;
        }

        /* NaN / Inf check (parallelized) */
        {
            int has_nan = 0;
            ptrdiff_t kk;
#pragma omp parallel for reduction(| : has_nan) schedule(static)
            for (kk = 0; kk < total_int; kk++) {
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
    cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_w); cfd_free(k1_p);
    cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_w); cfd_free(k2_p);
    cfd_free(k3_u); cfd_free(k3_v); cfd_free(k3_w); cfd_free(k3_p);
    cfd_free(k4_u); cfd_free(k4_v); cfd_free(k4_w); cfd_free(k4_p);
    cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);
    cfd_free(T_energy_ws);

    return status;
}

#endif /* CFD_ENABLE_OPENMP */
