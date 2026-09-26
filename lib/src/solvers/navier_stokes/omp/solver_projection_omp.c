#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/energy_solver.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "../../energy/energy_solver_internal.h"
#include "../../linear/multigrid_internal.h"
#include "../../turbulence/turbulence_solver_internal.h"

#include "../boundary_copy_utils.h"
#include "../ns_convection_internal.h"
#include "../ns_pressure_internal.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical limits
#define MAX_VELOCITY 100.0

/**
 * Turn the explicit increment in `star` into the implicit viscous one:
 *
 *     (I - theta*nu*dt*lap) delta = star - cur,   star = cur + delta
 *
 * on the OpenMP CG solver the projection owns. Same numerics as the scalar
 * implicit_viscous_component(); see ns_viscous_internal.h.
 */
static cfd_status_t implicit_viscous_component_omp(poisson_solver_t* viscous,
                                                   const double* cur, double* star,
                                                   double* rhs, double* delta,
                                                   size_t nx, size_t ny, size_t total,
                                                   size_t k_start, size_t k_end,
                                                   size_t stride_z) {
    double sigma = viscous->params.helmholtz_shift;

    memset(delta, 0, total * sizeof(double));
    for (size_t kk = k_start; kk < k_end; kk++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (int i = 1; i < (int)nx - 1; i++) {
                size_t idx = kk * stride_z + IDX_2D(i, j, nx);
                rhs[idx] = -sigma * (star[idx] - cur[idx]);
            }
        }
    }

    poisson_solver_stats_t vstats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(viscous, delta, NULL, rhs, &vstats);
    if (status != CFD_SUCCESS) {
        return status;
    }

    for (size_t kk = k_start; kk < k_end; kk++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (int i = 1; i < (int)nx - 1; i++) {
                size_t idx = kk * stride_z + IDX_2D(i, j, nx);
                star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, cur[idx] + delta[idx]));
            }
        }
    }
    return CFD_SUCCESS;
}

cfd_status_t solve_projection_method_omp(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params,
                                         poisson_solver_t* pressure,
                                         poisson_solver_t* viscous) {
    if (!field || !grid || !params || !pressure) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }
    /* The buffers below are sized from the field; the solver from the grid. */
    cfd_status_t shape_status =
        ns_pressure_check_shape(pressure, field->nx, field->ny, field->nz);
    if (shape_status == CFD_SUCCESS && viscous) {
        shape_status = ns_pressure_check_shape(viscous, field->nx, field->ny, field->nz);
    }
    if (shape_status != CFD_SUCCESS) {
        return shape_status;
    }
    /* The implicit operator has one constant viscosity; nu + nu_t does not. */
    if (viscous && params->turb_model != TURB_MODEL_NONE) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "The implicit viscous solve does not support turbulence models");
        return CFD_ERROR_UNSUPPORTED;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;

    /* Reject non-uniform z-spacing */
    if (nz > 1 && grid->dz) {
        for (size_t kk = 1; kk < nz - 1; kk++) {
            if (fabs(grid->dz[kk] - grid->dz[0]) > 1e-14) {
                return CFD_ERROR_INVALID;
            }
        }
    }

    size_t plane = nx * ny;
    size_t total = plane * nz;

    /* Branch-free 3D constants */
    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start  = (nz > 1) ? 1 : 0;
    size_t k_end    = (nz > 1) ? (nz - 1) : 1;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dz = (nz > 1 && grid->dz) ? grid->dz[0] : 0.0;
    double dt = params->dt;
    double nu = params->mu;
    double inv_2dz = (nz > 1 && grid->dz) ? 1.0 / (2.0 * dz) : 0.0;
    double inv_dz2 = (nz > 1 && grid->dz) ? 1.0 / (dz * dz) : 0.0;
    double inv_dz  = 2.0 * inv_2dz;
    const int upwind = (params->convection_scheme == NS_CONVECTION_SCHEME_UPWIND);


    double* u_star = (double*)cfd_calloc(total, sizeof(double));
    double* v_star = (double*)cfd_calloc(total, sizeof(double));
    double* w_star = (double*)cfd_calloc(total, sizeof(double));
    double* p_new = (double*)cfd_calloc(total, sizeof(double));
    double* p_temp = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    int needs_T_ws = (params->alpha > 0.0 || params->beta != 0.0);
    double* T_energy_ws = needs_T_ws
        ? (double*)cfd_calloc(total, sizeof(double)) : NULL;
    const int turb_on = (params->turb_model != TURB_MODEL_NONE);
    double* turb_ws = turb_on
        ? (double*)cfd_calloc(TURB_WORKSPACE_SIZE(total), sizeof(double)) : NULL;
    const int implicit_visc = (viscous != NULL);
    double* delta = implicit_visc ? (double*)cfd_calloc(total, sizeof(double)) : NULL;

    if (!u_star || !v_star || !w_star || !p_new || !p_temp || !rhs ||
        (needs_T_ws && !T_energy_ws) || (turb_on && !turb_ws) ||
        (implicit_visc && !delta)) {
        cfd_free(u_star);
        cfd_free(v_star);
        cfd_free(w_star);
        cfd_free(p_new);
        cfd_free(p_temp);
        cfd_free(rhs);
        cfd_free(T_energy_ws);
        cfd_free(turb_ws);
        cfd_free(delta);
        return CFD_ERROR_NOMEM;
    }

    memcpy(u_star, field->u, total * sizeof(double));
    memcpy(v_star, field->v, total * sizeof(double));
    memcpy(w_star, field->w, total * sizeof(double));
    memcpy(p_new, field->p, total * sizeof(double));

    for (int iter = 0; iter < params->max_iter; iter++) {
        /* STEP 1: Predictor — compute u_star, v_star, w_star without pressure */
        for (size_t kk = k_start; kk < k_end; kk++) {
            int j;
#pragma omp parallel for schedule(static)
            for (j = 1; j < (int)ny - 1; j++) {
                for (int i = 1; i < (int)nx - 1; i++) {
                    size_t idx = kk * stride_z + IDX_2D(i, j, nx);

                    double u = field->u[idx];
                    double v = field->v[idx];
                    double w = field->w[idx];

                    double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                    double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                    double du_dz = (field->u[idx + stride_z] - field->u[idx - stride_z]) * inv_2dz;
                    double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                    double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
                    double dv_dz = (field->v[idx + stride_z] - field->v[idx - stride_z]) * inv_2dz;
                    double dw_dx = (field->w[idx + 1] - field->w[idx - 1]) / (2.0 * dx);
                    double dw_dy = (field->w[idx + nx] - field->w[idx - nx]) / (2.0 * dy);
                    double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) * inv_2dz;

                    /* Convective derivatives: central, or first-order upwind */
                    ns_conv_derivs_t cd = {du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz,
                                           dw_dx, dw_dy, dw_dz};
                    if (upwind) {
                        ns_upwind_conv_derivs(field->u, field->v, field->w, idx,
                                              idx - 1, idx + 1, idx - nx, idx + nx,
                                              idx - stride_z, idx + stride_z,
                                              dx, dy, inv_dz, &cd);
                    }

                    double conv_u = u * cd.du_dx + v * cd.du_dy + w * cd.du_dz;
                    double conv_v = u * cd.dv_dx + v * cd.dv_dy + w * cd.dv_dz;
                    double conv_w = u * cd.dw_dx + v * cd.dw_dy + w * cd.dw_dz;

                    double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                    double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                    double d2u_dz2 = (field->u[idx + stride_z] - 2.0 * u + field->u[idx - stride_z]) * inv_dz2;
                    double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                    double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);
                    double d2v_dz2 = (field->v[idx + stride_z] - 2.0 * v + field->v[idx - stride_z]) * inv_dz2;
                    double d2w_dx2 = (field->w[idx + 1] - 2.0 * w + field->w[idx - 1]) / (dx * dx);
                    double d2w_dy2 = (field->w[idx + nx] - 2.0 * w + field->w[idx - nx]) / (dy * dy);
                    double d2w_dz2 = (field->w[idx + stride_z] - 2.0 * w + field->w[idx - stride_z]) * inv_dz2;

                    double visc_u, visc_v, visc_w;
                    if (!turb_on) {
                        visc_u = nu * (d2u_dx2 + d2u_dy2 + d2u_dz2);
                        visc_v = nu * (d2v_dx2 + d2v_dy2 + d2v_dz2);
                        visc_w = nu * (d2w_dx2 + d2w_dy2 + d2w_dz2);
                    } else {
                        /* Turbulent: conservative face-averaged effective viscosity
                         * div((nu + nu_t) grad u). The z-term keeps the laminar nu
                         * (turbulence is 2D-only; it vanishes when nz == 1). */
                        const double* nu_t = field->nu_t;
                        double nu_xp = nu + 0.5 * (nu_t[idx] + nu_t[idx + 1]);
                        double nu_xm = nu + 0.5 * (nu_t[idx] + nu_t[idx - 1]);
                        double nu_yp = nu + 0.5 * (nu_t[idx] + nu_t[idx + nx]);
                        double nu_ym = nu + 0.5 * (nu_t[idx] + nu_t[idx - nx]);

                        visc_u = (nu_xp * (field->u[idx + 1] - u) -
                                  nu_xm * (u - field->u[idx - 1])) / (dx * dx) +
                                 (nu_yp * (field->u[idx + nx] - u) -
                                  nu_ym * (u - field->u[idx - nx])) / (dy * dy) +
                                 nu * d2u_dz2;
                        visc_v = (nu_xp * (field->v[idx + 1] - v) -
                                  nu_xm * (v - field->v[idx - 1])) / (dx * dx) +
                                 (nu_yp * (field->v[idx + nx] - v) -
                                  nu_ym * (v - field->v[idx - nx])) / (dy * dy) +
                                 nu * d2v_dz2;
                        visc_w = (nu_xp * (field->w[idx + 1] - w) -
                                  nu_xm * (w - field->w[idx - 1])) / (dx * dx) +
                                 (nu_yp * (field->w[idx + nx] - w) -
                                  nu_ym * (w - field->w[idx - nx])) / (dy * dy) +
                                 nu * d2w_dz2;
                    }

                    double source_u = 0.0;
                    double source_v = 0.0;
                    double source_w = 0.0;
                    double z_coord = (nz > 1 && grid->z) ? grid->z[kk] : 0.0;
                    if (params->source_func) {
                        params->source_func(grid->x[i], grid->y[j], z_coord, iter * dt,
                                            params->source_context,
                                            &source_u, &source_v, &source_w);
                    } else if (params->source_amplitude_u > 0) {
                        source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]) *
                                   exp(-params->source_decay_rate * iter * dt);
                        source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]) *
                                   exp(-params->source_decay_rate * iter * dt);
                    }

                    // Boussinesq buoyancy source (no-op when beta == 0)
                    energy_compute_buoyancy(field->T[idx], params,
                                            &source_u, &source_v, &source_w);

                    u_star[idx] = u + dt * (-conv_u + visc_u + source_u);
                    v_star[idx] = v + dt * (-conv_v + visc_v + source_v);
                    w_star[idx] = w + dt * (-conv_w + visc_w + source_w);

                    /* An implicit scheme clamps after its solve instead */
                    if (!implicit_visc) {
                        u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                        v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
                        w_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, w_star[idx]));
                    }
                }
            }
        }

        /* Copy boundary values from field to star arrays */
        copy_boundary_velocities_3d(u_star, v_star, w_star,
                                    field->u, field->v, field->w, nx, ny, nz);

        /* Implicit viscous term, on the OpenMP CG solver: replace each explicit
         * increment by the solution of (I - theta*nu*dt*lap) delta = increment.
         * rhs is free until the pressure solve below overwrites it. */
        if (implicit_visc) {
            const double* cur[3] = {field->u, field->v, field->w};
            double* star[3] = {u_star, v_star, w_star};
            for (int c = 0; c < 3; c++) {
                cfd_status_t visc_status = implicit_viscous_component_omp(
                    viscous, cur[c], star[c], rhs, delta, nx, ny, total,
                    k_start, k_end, stride_z);
                if (visc_status != CFD_SUCCESS) {
                    cfd_free(u_star); cfd_free(v_star); cfd_free(w_star);
                    cfd_free(p_new); cfd_free(p_temp); cfd_free(rhs);
                    cfd_free(T_energy_ws); cfd_free(turb_ws); cfd_free(delta);
                    return visc_status;
                }
            }
        }

        /* STEP 2: Pressure Poisson equation */
        double rho = field->rho[0] < 1e-10 ? 1.0 : field->rho[0];

        for (size_t kk = k_start; kk < k_end; kk++) {
            int j;
#pragma omp parallel for schedule(static)
            for (j = 1; j < (int)ny - 1; j++) {
                for (int i = 1; i < (int)nx - 1; i++) {
                    size_t idx = kk * stride_z + IDX_2D(i, j, nx);
                    double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                    double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                    double dw_star_dz = (w_star[idx + stride_z] - w_star[idx - stride_z]) * inv_2dz;
                    rhs[idx] = (rho / dt) * (du_star_dx + dv_star_dy + dw_star_dz);
                }
            }
        }

        /* Neumann compatibility projection, but only when the operator is actually
         * singular. With every face zero-gradient the constants are its nullspace,
         * so the RHS must have zero interior mean -- div(u*) only nearly does, and
         * the remainder is removed here rather than assumed away. With a face
         * prescribed the operator is nonsingular and shifting the RHS would change
         * the answer instead of making it exist. */
        if (poisson_walls_are_singular(&params->pressure_bc, nz)) {
            mg_subtract_interior_mean_omp(rhs, nx, ny, nz);
        }

        /* Parallel Poisson solve on the solver this projection owns, which was
         * built on the OpenMP backend at init -- never a scalar one. */
        poisson_solver_stats_t pstats = poisson_solver_stats_default();
        cfd_status_t poisson_status =
            poisson_solver_solve(pressure, p_new, p_temp, rhs, &pstats);

        if (poisson_status != CFD_SUCCESS) {
            cfd_free(u_star);
            cfd_free(v_star);
            cfd_free(w_star);
            cfd_free(p_new);
            cfd_free(p_temp);
            cfd_free(rhs);
            cfd_free(T_energy_ws);
            cfd_free(turb_ws);
            cfd_free(delta);
            return poisson_status;
        }

        /* STEP 3: Corrector — project velocities with pressure gradient */
        double dt_over_rho = dt / rho;
        for (size_t kk = k_start; kk < k_end; kk++) {
            int j;
#pragma omp parallel for schedule(static)
            for (j = 1; j < (int)ny - 1; j++) {
                for (int i = 1; i < (int)nx - 1; i++) {
                    size_t idx = kk * stride_z + IDX_2D(i, j, nx);
                    double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                    double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);
                    double dp_dz = (p_new[idx + stride_z] - p_new[idx - stride_z]) * inv_2dz;

                    field->u[idx] = u_star[idx] - dt_over_rho * dp_dx;
                    field->v[idx] = v_star[idx] - dt_over_rho * dp_dy;
                    field->w[idx] = w_star[idx] - dt_over_rho * dp_dz;

                    field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                    field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
                    field->w[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->w[idx]));
                }
            }
        }

        memcpy(field->p, p_new, total * sizeof(double));

        /* Energy equation: advance temperature after velocity correction */
        {
            cfd_status_t energy_status = energy_step_explicit_omp_with_workspace(
                field, grid, params, dt, iter * dt, T_energy_ws, total);
            if (energy_status != CFD_SUCCESS) {
                cfd_free(u_star); cfd_free(v_star); cfd_free(w_star);
                cfd_free(p_new); cfd_free(p_temp); cfd_free(rhs);
                cfd_free(T_energy_ws); cfd_free(turb_ws); cfd_free(delta);
                return energy_status;
            }
        }

        /* Apply configured thermal BCs to temperature field */
        cfd_status_t bc_status = energy_apply_thermal_bcs(field, params);
        if (bc_status != CFD_SUCCESS) {
            cfd_free(u_star); cfd_free(v_star); cfd_free(w_star);
            cfd_free(p_new); cfd_free(p_temp); cfd_free(rhs);
            cfd_free(T_energy_ws); cfd_free(turb_ws); cfd_free(delta);
            return bc_status;
        }

        /* Copy boundary velocity values from star arrays (which have caller's BCs) */
        copy_boundary_velocities_3d(field->u, field->v, field->w,
                                    u_star, v_star, w_star, nx, ny, nz);

        /* Turbulence transport: advance k-eps/SA with the corrected velocity,
         * then apply turbulence BCs (including wall functions) */
        {
            cfd_status_t turb_status = turbulence_step_explicit_omp_with_workspace(
                field, grid, params, dt, iter * dt, turb_ws,
                turb_on ? TURB_WORKSPACE_SIZE(total) : 0);
            if (turb_status == CFD_SUCCESS) {
                turb_status = turbulence_apply_bcs(field, grid, params);
            }
            if (turb_status != CFD_SUCCESS) {
                cfd_free(u_star); cfd_free(v_star); cfd_free(w_star);
                cfd_free(p_new); cfd_free(p_temp); cfd_free(rhs);
                cfd_free(T_energy_ws); cfd_free(turb_ws); cfd_free(delta);
                return turb_status;
            }
        }

        /* Check for NaN/Inf values (parallelized) */
        int has_nan = 0;
        ptrdiff_t total_int = (ptrdiff_t)total;
        ptrdiff_t ii;
#pragma omp parallel for reduction(| : has_nan) schedule(static)
        for (ii = 0; ii < total_int; ii++) {
            if (!isfinite(field->u[ii]) || !isfinite(field->v[ii]) ||
                !isfinite(field->w[ii]) || !isfinite(field->p[ii])) {
                has_nan = 1;
            }
        }
        if (has_nan) {
            cfd_free(u_star);
            cfd_free(v_star);
            cfd_free(w_star);
            cfd_free(p_new);
            cfd_free(p_temp);
            cfd_free(rhs);
            cfd_free(T_energy_ws);
            cfd_free(turb_ws);
            cfd_free(delta);
            return CFD_ERROR_DIVERGED;
        }
    }

    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(w_star);
    cfd_free(p_new);
    cfd_free(p_temp);
    cfd_free(rhs);
    cfd_free(T_energy_ws);
    cfd_free(turb_ws);
    cfd_free(delta);
    return CFD_SUCCESS;
}
