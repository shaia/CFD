/**
 * @file solver_rk2.c
 * @brief RK2 (Heun's method) time integration for 2D Navier-Stokes
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
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Physical stability limits */
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DIVERGENCE_LIMIT        10.0

#define PRESSURE_UPDATE_FACTOR 0.1

/* ============================================================================
 * RHS EVALUATION
 * ============================================================================ */

/**
 * Compute the right-hand side of the semi-discrete Navier-Stokes equations.
 *
 * For each interior point, computes:
 *   du/dt = -u*du/dx - v*du/dy - (1/rho)*dp/dx + nu*(d2u/dx2 + d2u/dy2) + source_u
 *   dv/dt = -u*dv/dx - v*dv/dy - (1/rho)*dp/dy + nu*(d2v/dx2 + d2v/dy2) + source_v
 *   dp/dt = -0.1 * rho * (du/dx + dv/dy)
 *
 * @param u, v, p    Input state arrays (read-only)
 * @param rho        Density array (read-only)
 * @param rhs_u      Output: du/dt  (must be pre-zeroed for boundary points)
 * @param rhs_v      Output: dv/dt
 * @param rhs_p      Output: dp/dt
 * @param grid       Computational grid
 * @param params     Solver parameters (viscosity, sources)
 * @param nx, ny     Grid dimensions
 * @param iter       Current iteration (for time-decaying source terms)
 * @param dt         Time step (for source term decay)
 */
static void compute_rhs(const double* u, const double* v, const double* p,
                         const double* rho,
                         double* rhs_u, double* rhs_v, double* rhs_p,
                         const grid* grid, const ns_solver_params_t* params,
                         size_t nx, size_t ny, int iter, double dt) {
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = (j * nx) + i;

            /* Safety checks */
            if (rho[idx] <= 1e-10) {
                rhs_u[idx] = 0.0;
                rhs_v[idx] = 0.0;
                rhs_p[idx] = 0.0;
                continue;
            }
            if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) {
                rhs_u[idx] = 0.0;
                rhs_v[idx] = 0.0;
                rhs_p[idx] = 0.0;
                continue;
            }

            /* Periodic stencil indices — avoids relying on ghost cells,
             * which is critical for preserving RK2 temporal order.
             * Ghost cell values may be stale during intermediate RK stages. */
            size_t il = (i > 1)      ? idx - 1  : j * nx + (nx - 2);
            size_t ir = (i < nx - 2) ? idx + 1  : j * nx + 1;
            size_t jd = (j > 1)      ? idx - nx : (ny - 2) * nx + i;
            size_t ju = (j < ny - 2) ? idx + nx : nx + i;

            /* First derivatives (central differences) */
            double du_dx = (u[ir] - u[il]) / (2.0 * grid->dx[i]);
            double du_dy = (u[ju] - u[jd]) / (2.0 * grid->dy[j]);
            double dv_dx = (v[ir] - v[il]) / (2.0 * grid->dx[i]);
            double dv_dy = (v[ju] - v[jd]) / (2.0 * grid->dy[j]);

            /* Pressure gradients */
            double dp_dx = (p[ir] - p[il]) / (2.0 * grid->dx[i]);
            double dp_dy = (p[ju] - p[jd]) / (2.0 * grid->dy[j]);

            /* Second derivatives (viscous terms) */
            double d2u_dx2 = (u[ir] - 2.0 * u[idx] + u[il]) /
                             (grid->dx[i] * grid->dx[i]);
            double d2u_dy2 = (u[ju] - 2.0 * u[idx] + u[jd]) /
                             (grid->dy[j] * grid->dy[j]);
            double d2v_dx2 = (v[ir] - 2.0 * v[idx] + v[il]) /
                             (grid->dx[i] * grid->dx[i]);
            double d2v_dy2 = (v[ju] - 2.0 * v[idx] + v[jd]) /
                             (grid->dy[j] * grid->dy[j]);

            /* Kinematic viscosity */
            double nu = params->mu / fmax(rho[idx], 1e-10);
            nu = fmin(nu, 1.0);

            /* Clamp derivatives */
            du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
            du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
            dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
            dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
            dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
            dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
            d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT,
                           fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
            d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT,
                           fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
            d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT,
                           fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
            d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT,
                           fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));

            /* Source terms */
            double source_u = 0.0, source_v = 0.0;
            compute_source_terms(grid->x[i], grid->y[j], iter, dt,
                                 params, &source_u, &source_v);

            /* RHS for u-momentum */
            double du = -u[idx] * du_dx - v[idx] * du_dy
                        - dp_dx / rho[idx]
                        + nu * (d2u_dx2 + d2u_dy2)
                        + source_u;

            /* RHS for v-momentum */
            double dv = -u[idx] * dv_dx - v[idx] * dv_dy
                        - dp_dy / rho[idx]
                        + nu * (d2v_dx2 + d2v_dy2)
                        + source_v;

            rhs_u[idx] = du;
            rhs_v[idx] = dv;

            /* Simplified pressure RHS (divergence-based) */
            double divergence = du_dx + dv_dy;
            divergence = fmax(-MAX_DIVERGENCE_LIMIT,
                              fmin(MAX_DIVERGENCE_LIMIT, divergence));
            rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
        }
    }
}

/* ============================================================================
 * RK2 SOLVER
 * ============================================================================ */

cfd_status_t rk2_impl(flow_field* field, const grid* grid,
                       const ns_solver_params_t* params) {
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;
    size_t bytes = size * sizeof(double);

    /* Allocate working arrays:
     *   k1_u/v/p : Stage 1 derivatives
     *   k2_u/v/p : Stage 2 derivatives
     *   u0/v0/p0 : Saved state Q^n (needed for final combination)
     */
    double* k1_u = (double*)cfd_calloc(size, sizeof(double));
    double* k1_v = (double*)cfd_calloc(size, sizeof(double));
    double* k1_p = (double*)cfd_calloc(size, sizeof(double));
    double* k2_u = (double*)cfd_calloc(size, sizeof(double));
    double* k2_v = (double*)cfd_calloc(size, sizeof(double));
    double* k2_p = (double*)cfd_calloc(size, sizeof(double));
    double* u0 = (double*)cfd_calloc(size, sizeof(double));
    double* v0 = (double*)cfd_calloc(size, sizeof(double));
    double* p0 = (double*)cfd_calloc(size, sizeof(double));

    if (!k1_u || !k1_v || !k1_p || !k2_u || !k2_v || !k2_p ||
        !u0 || !v0 || !p0) {
        cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_p);
        cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_p);
        cfd_free(u0); cfd_free(v0); cfd_free(p0);
        return CFD_ERROR_NOMEM;
    }

    double dt = params->dt;
    cfd_status_t status = CFD_SUCCESS;

    for (int iter = 0; iter < params->max_iter; iter++) {
        /* Save Q^n */
        memcpy(u0, field->u, bytes);
        memcpy(v0, field->v, bytes);
        memcpy(p0, field->p, bytes);

        /* ---- Stage 1: k1 = RHS(Q^n) ---- */
        memset(k1_u, 0, bytes);
        memset(k1_v, 0, bytes);
        memset(k1_p, 0, bytes);

        compute_rhs(field->u, field->v, field->p, field->rho,
                     k1_u, k1_v, k1_p,
                     grid, params, nx, ny, iter, dt);

        /* ---- Intermediate: field = Q^n + dt * k1 ---- */
        for (size_t k = 0; k < size; k++) {
            field->u[k] = u0[k] + dt * k1_u[k];
            field->v[k] = v0[k] + dt * k1_v[k];
            field->p[k] = p0[k] + dt * k1_p[k];

            field->u[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[k]));
            field->v[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[k]));
        }

        /* NOTE: Do NOT apply BCs between RK stages. The ghost cells carry
         * zero-derivative evolution (k1[ghost]=0), which is consistent with
         * the semi-discrete ODE system. Applying BCs here would modify the
         * intermediate state outside the ODE trajectory and reduce RK2 to
         * first-order temporal accuracy. */

        /* ---- Stage 2: k2 = RHS(Q_pred) ---- */
        memset(k2_u, 0, bytes);
        memset(k2_v, 0, bytes);
        memset(k2_p, 0, bytes);

        compute_rhs(field->u, field->v, field->p, field->rho,
                     k2_u, k2_v, k2_p,
                     grid, params, nx, ny, iter, dt);

        /* ---- Final update: Q^{n+1} = Q^n + (dt/2)*(k1 + k2) ---- */
        double half_dt = 0.5 * dt;
        for (size_t k = 0; k < size; k++) {
            field->u[k] = u0[k] + half_dt * (k1_u[k] + k2_u[k]);
            field->v[k] = v0[k] + half_dt * (k1_v[k] + k2_v[k]);
            field->p[k] = p0[k] + half_dt * (k1_p[k] + k2_p[k]);

            field->u[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[k]));
            field->v[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[k]));
        }

        /* Apply BCs to final state only (after the full RK2 step).
         * This updates ghost cells for the next step's k1 evaluation. */
        apply_boundary_conditions(field, grid);

        /* NaN / Inf check */
        for (size_t k = 0; k < size; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) ||
                !isfinite(field->p[k])) {
                status = CFD_ERROR_DIVERGED;
                goto cleanup;
            }
        }
    }

cleanup:
    cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_p);
    cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_p);
    cfd_free(u0); cfd_free(v0); cfd_free(p0);

    return status;
}
