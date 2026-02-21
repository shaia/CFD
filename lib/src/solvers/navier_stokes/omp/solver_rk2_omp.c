/**
 * @file solver_rk2_omp.c
 * @brief RK2 (Heun's method) time integration - OpenMP parallelized
 *
 * Same algorithm as scalar RK2 but with OpenMP-parallelized loops.
 * Achieves O(dt^2) temporal accuracy with multi-threaded execution.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"

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

/* Physical stability limits (same as scalar RK2) */
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DIVERGENCE_LIMIT        10.0
#define PRESSURE_UPDATE_FACTOR      0.1

/* ============================================================================
 * RHS EVALUATION (OpenMP parallelized)
 * ============================================================================ */

static void compute_rhs_omp(const double* u, const double* v, const double* p,
                             const double* rho,
                             double* rhs_u, double* rhs_v, double* rhs_p,
                             const grid* grid, const ns_solver_params_t* params,
                             size_t nx, size_t ny, int iter, double dt) {
    ptrdiff_t ny_int = (ptrdiff_t)ny;
    ptrdiff_t nx_int = (ptrdiff_t)nx;
    ptrdiff_t j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (ptrdiff_t i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;

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
             * critical for preserving RK2 temporal order. */
            size_t il = ((size_t)i > 1)      ? idx - 1  : (size_t)j * nx + (nx - 2);
            size_t ir = ((size_t)i < nx - 2) ? idx + 1  : (size_t)j * nx + 1;
            size_t jd = ((size_t)j > 1)      ? idx - nx : (ny - 2) * nx + (size_t)i;
            size_t ju = ((size_t)j < ny - 2) ? idx + nx : nx + (size_t)i;

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
            rhs_u[idx] = -u[idx] * du_dx - v[idx] * du_dy
                         - dp_dx / rho[idx]
                         + nu * (d2u_dx2 + d2u_dy2)
                         + source_u;

            /* RHS for v-momentum */
            rhs_v[idx] = -u[idx] * dv_dx - v[idx] * dv_dy
                         - dp_dy / rho[idx]
                         + nu * (d2v_dx2 + d2v_dy2)
                         + source_v;

            /* Simplified pressure RHS (divergence-based) */
            double divergence = du_dx + dv_dy;
            divergence = fmax(-MAX_DIVERGENCE_LIMIT,
                              fmin(MAX_DIVERGENCE_LIMIT, divergence));
            rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
        }
    }
}

/* ============================================================================
 * RK2 OMP SOLVER
 * ============================================================================ */

cfd_status_t rk2_omp_impl(flow_field* field, const grid* grid,
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
     *   u0/v0/p0 : Saved state Q^n
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
    ptrdiff_t size_int = (ptrdiff_t)size;
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

        compute_rhs_omp(field->u, field->v, field->p, field->rho,
                         k1_u, k1_v, k1_p,
                         grid, params, nx, ny, iter, dt);

        /* ---- Intermediate: field = Q^n + dt * k1 ---- */
        {
            ptrdiff_t k;
            OMP_FOR_SIMD
            for (k = 0; k < size_int; k++) {
                field->u[k] = u0[k] + dt * k1_u[k];
                field->v[k] = v0[k] + dt * k1_v[k];
                field->p[k] = p0[k] + dt * k1_p[k];

                field->u[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[k]));
                field->v[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[k]));
            }
        }

        /* NOTE: Do NOT apply BCs between RK stages. The ghost cells carry
         * zero-derivative evolution (k1[ghost]=0), which is consistent with
         * the semi-discrete ODE system. Applying BCs here would reduce RK2
         * to first-order temporal accuracy. */

        /* ---- Stage 2: k2 = RHS(Q_pred) ---- */
        memset(k2_u, 0, bytes);
        memset(k2_v, 0, bytes);
        memset(k2_p, 0, bytes);

        compute_rhs_omp(field->u, field->v, field->p, field->rho,
                         k2_u, k2_v, k2_p,
                         grid, params, nx, ny, iter, dt);

        /* ---- Final update: Q^{n+1} = Q^n + (dt/2)*(k1 + k2) ---- */
        {
            double half_dt = 0.5 * dt;
            ptrdiff_t k;
            OMP_FOR_SIMD
            for (k = 0; k < size_int; k++) {
                field->u[k] = u0[k] + half_dt * (k1_u[k] + k2_u[k]);
                field->v[k] = v0[k] + half_dt * (k1_v[k] + k2_v[k]);
                field->p[k] = p0[k] + half_dt * (k1_p[k] + k2_p[k]);

                field->u[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[k]));
                field->v[k] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[k]));
            }
        }

        /* Apply BCs to final state only (after the full RK2 step) */
        apply_boundary_conditions(field, grid);

        /* NaN / Inf check (parallelized) */
        {
            int has_nan = 0;
            ptrdiff_t k;
#pragma omp parallel for reduction(| : has_nan) schedule(static)
            for (k = 0; k < size_int; k++) {
                if (!isfinite(field->u[k]) || !isfinite(field->v[k]) ||
                    !isfinite(field->p[k])) {
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
    cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_p);
    cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_p);
    cfd_free(u0); cfd_free(v0); cfd_free(p0);

    return status;
}

#endif /* CFD_ENABLE_OPENMP */
