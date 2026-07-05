/**
 * @file solver_rk2.c
 * @brief RK2 (Heun's method) time integration for Navier-Stokes
 *
 * Second-order Runge-Kutta time stepping:
 *   k1 = RHS(Q^n)
 *   Q_pred = Q^n + dt * k1
 *   k2 = RHS(Q_pred)
 *   Q^{n+1} = Q^n + (dt/2) * (k1 + k2)
 *
 * Uses the same spatial discretisation (central differences) and physics
 * as the explicit Euler solver, but achieves O(dt^2) temporal accuracy
 * instead of O(dt).
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing bit-identical results to the 2D code path.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/energy_solver.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"

#include "../../energy/energy_solver_internal.h"
#include "../../turbulence/turbulence_solver_internal.h"

#include <math.h>
#include <string.h>

/* Shared momentum RHS kernel (also #include-d by solver_rk4.c). Must come after
 * the grid/indexing/energy/<math.h> includes above, which it depends on. Defines
 * the shared clamp constants MAX_DERIVATIVE_LIMIT / MAX_SECOND_DERIVATIVE_LIMIT /
 * MAX_DIVERGENCE_LIMIT / PRESSURE_UPDATE_FACTOR. */
#include "../momentum_rhs/ns_momentum_rhs_scalar.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Velocity clamp used by the time-step update (not the RHS kernel) */
#define MAX_VELOCITY_LIMIT          100.0

/* ============================================================================
 * RK2 SOLVER
 * ============================================================================ */

cfd_status_t rk2_impl(flow_field* field, const grid* grid,
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

    /* Allocate working arrays:
     *   k1_u/v/w/p : Stage 1 derivatives
     *   k2_u/v/w/p : Stage 2 derivatives
     *   u0/v0/w0/p0 : Saved state Q^n
     */
    double* k1_u = (double*)cfd_calloc(total, sizeof(double));
    double* k1_v = (double*)cfd_calloc(total, sizeof(double));
    double* k1_w = (double*)cfd_calloc(total, sizeof(double));
    double* k1_p = (double*)cfd_calloc(total, sizeof(double));
    double* k2_u = (double*)cfd_calloc(total, sizeof(double));
    double* k2_v = (double*)cfd_calloc(total, sizeof(double));
    double* k2_w = (double*)cfd_calloc(total, sizeof(double));
    double* k2_p = (double*)cfd_calloc(total, sizeof(double));
    double* u0 = (double*)cfd_calloc(total, sizeof(double));
    double* v0 = (double*)cfd_calloc(total, sizeof(double));
    double* w0 = (double*)cfd_calloc(total, sizeof(double));
    double* p0 = (double*)cfd_calloc(total, sizeof(double));
    int needs_T_ws = (params->alpha > 0.0 || params->beta != 0.0);
    double* T_energy_ws = needs_T_ws
        ? (double*)cfd_calloc(total, sizeof(double)) : NULL;
    const int turb_on = (params->turb_model != TURB_MODEL_NONE);
    double* turb_ws = turb_on
        ? (double*)cfd_calloc(TURB_WORKSPACE_SIZE(total), sizeof(double)) : NULL;

    if (!k1_u || !k1_v || !k1_w || !k1_p ||
        !k2_u || !k2_v || !k2_w || !k2_p ||
        !u0 || !v0 || !w0 || !p0 ||
        (needs_T_ws && !T_energy_ws) || (turb_on && !turb_ws)) {
        cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_w); cfd_free(k1_p);
        cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_w); cfd_free(k2_p);
        cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);
        cfd_free(T_energy_ws); cfd_free(turb_ws);
        return CFD_ERROR_NOMEM;
    }

    double dt = params->dt;
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

        compute_rhs(field->u, field->v, field->w, field->p, field->rho, field->T,
                     field->nu_t,
                     k1_u, k1_v, k1_w, k1_p,
                     grid, params, nx, ny, nz,
                     stride_z, k_start, k_end, inv_2dz, inv_dz2,
                     iter, dt);

        /* ---- Intermediate: field = Q^n + dt * k1 ---- */
        for (size_t n = 0; n < total; n++) {
            field->u[n] = u0[n] + dt * k1_u[n];
            field->v[n] = v0[n] + dt * k1_v[n];
            field->w[n] = w0[n] + dt * k1_w[n];
            field->p[n] = p0[n] + dt * k1_p[n];

            field->u[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[n]));
            field->v[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[n]));
            field->w[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[n]));
        }

        /* NOTE: Do NOT apply BCs between RK stages. The ghost cells carry
         * zero-derivative evolution (k1[ghost]=0), which is consistent with
         * the semi-discrete ODE system. Applying BCs here would modify the
         * intermediate state outside the ODE trajectory and reduce RK2 to
         * first-order temporal accuracy. */

        /* ---- Stage 2: k2 = RHS(Q_pred) ---- */
        memset(k2_u, 0, bytes);
        memset(k2_v, 0, bytes);
        memset(k2_w, 0, bytes);
        memset(k2_p, 0, bytes);

        compute_rhs(field->u, field->v, field->w, field->p, field->rho, field->T,
                     field->nu_t,
                     k2_u, k2_v, k2_w, k2_p,
                     grid, params, nx, ny, nz,
                     stride_z, k_start, k_end, inv_2dz, inv_dz2,
                     iter, dt);

        /* ---- Final update: Q^{n+1} = Q^n + (dt/2)*(k1 + k2) ---- */
        double half_dt = 0.5 * dt;
        for (size_t n = 0; n < total; n++) {
            field->u[n] = u0[n] + half_dt * (k1_u[n] + k2_u[n]);
            field->v[n] = v0[n] + half_dt * (k1_v[n] + k2_v[n]);
            field->w[n] = w0[n] + half_dt * (k1_w[n] + k2_w[n]);
            field->p[n] = p0[n] + half_dt * (k1_p[n] + k2_p[n]);

            field->u[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[n]));
            field->v[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[n]));
            field->w[n] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[n]));
        }

        /* Energy equation: advance temperature after RK2 velocity update */
        {
            cfd_status_t energy_status = energy_step_explicit_with_workspace(
                field, grid, params, dt, iter * dt, T_energy_ws, total);
            if (energy_status != CFD_SUCCESS) {
                status = energy_status;
                goto cleanup;
            }
        }

        /* Apply BCs to final state only (after the full RK2 step).
         * This updates ghost cells for the next step's k1 evaluation.
         * Then apply configured thermal BCs (overwrites periodic T values). */
        apply_boundary_conditions(field, grid);
        status = energy_apply_thermal_bcs(field, params);
        if (status != CFD_SUCCESS) {
            goto cleanup;
        }

        /* Turbulence transport: advance k-eps/SA with the updated velocity,
         * then apply turbulence BCs (including wall functions). nu_t is
         * frozen across RK stages and updated once per full step. */
        status = turbulence_step_explicit_with_workspace(
            field, grid, params, dt, iter * dt, turb_ws,
            turb_on ? TURB_WORKSPACE_SIZE(total) : 0);
        if (status == CFD_SUCCESS) {
            status = turbulence_apply_bcs(field, grid, params);
        }
        if (status != CFD_SUCCESS) {
            goto cleanup;
        }

        /* NaN / Inf check */
        for (size_t n = 0; n < total; n++) {
            if (!isfinite(field->u[n]) || !isfinite(field->v[n]) ||
                !isfinite(field->w[n]) || !isfinite(field->p[n])) {
                status = CFD_ERROR_DIVERGED;
                goto cleanup;
            }
        }
    }

cleanup:
    cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_w); cfd_free(k1_p);
    cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_w); cfd_free(k2_p);
    cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);
    cfd_free(T_energy_ws); cfd_free(turb_ws);

    return status;
}
