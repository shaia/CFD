/**
 * @file solver_rk2_omp.c
 * @brief RK2 (Heun's method) time integration - OpenMP parallelized
 *
 * Same algorithm as scalar RK2 but with OpenMP-parallelized loops.
 * Achieves O(dt^2) temporal accuracy with multi-threaded execution.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing bit-identical results to the 2D code path.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
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

static void compute_rhs_omp(const double* u, const double* v, const double* w,
                             const double* p, const double* rho,
                             double* rhs_u, double* rhs_v, double* rhs_w,
                             double* rhs_p,
                             const grid* grid, const ns_solver_params_t* params,
                             size_t nx, size_t ny, size_t nz,
                             size_t stride_z, size_t k_start, size_t k_end,
                             double inv_2dz, double inv_dz2,
                             int iter, double dt) {
    ptrdiff_t ny_int = (ptrdiff_t)ny;
    ptrdiff_t nx_int = (ptrdiff_t)nx;

    for (size_t k = k_start; k < k_end; k++) {
        ptrdiff_t j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (ptrdiff_t i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

                /* Safety checks */
                if (rho[idx] <= 1e-10) {
                    rhs_u[idx] = 0.0;
                    rhs_v[idx] = 0.0;
                    rhs_w[idx] = 0.0;
                    rhs_p[idx] = 0.0;
                    continue;
                }
                if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) {
                    rhs_u[idx] = 0.0;
                    rhs_v[idx] = 0.0;
                    rhs_w[idx] = 0.0;
                    rhs_p[idx] = 0.0;
                    continue;
                }

                /* Periodic stencil indices in x and y — avoids relying on ghost
                 * cells, critical for preserving RK2 temporal order. */
                size_t il = ((size_t)i > 1)      ? idx - 1  : k * stride_z + IDX_2D(nx - 2, (size_t)j, nx);
                size_t ir = ((size_t)i < nx - 2) ? idx + 1  : k * stride_z + IDX_2D(1, (size_t)j, nx);
                size_t jd = ((size_t)j > 1)      ? idx - nx : k * stride_z + IDX_2D((size_t)i, ny - 2, nx);
                size_t ju = ((size_t)j < ny - 2) ? idx + nx : k * stride_z + IDX_2D((size_t)i, 1, nx);

                /* Periodic stencil indices in z.
                 * When nz==1: k=0, stride_z=0, so kd=ku=idx → z-terms vanish. */
                size_t kd = (k > 1)      ? idx - stride_z
                                         : (nz - 2) * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                size_t ku = (k < nz - 2) ? idx + stride_z
                                         : 1 * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

                /* First derivatives (central differences) */
                double du_dx = (u[ir] - u[il]) / (2.0 * grid->dx[i]);
                double du_dy = (u[ju] - u[jd]) / (2.0 * grid->dy[j]);
                double du_dz = (u[ku] - u[kd]) * inv_2dz;

                double dv_dx = (v[ir] - v[il]) / (2.0 * grid->dx[i]);
                double dv_dy = (v[ju] - v[jd]) / (2.0 * grid->dy[j]);
                double dv_dz = (v[ku] - v[kd]) * inv_2dz;

                double dw_dx = (w[ir] - w[il]) / (2.0 * grid->dx[i]);
                double dw_dy = (w[ju] - w[jd]) / (2.0 * grid->dy[j]);
                double dw_dz = (w[ku] - w[kd]) * inv_2dz;

                /* Pressure gradients */
                double dp_dx = (p[ir] - p[il]) / (2.0 * grid->dx[i]);
                double dp_dy = (p[ju] - p[jd]) / (2.0 * grid->dy[j]);
                double dp_dz = (p[ku] - p[kd]) * inv_2dz;

                /* Second derivatives (viscous terms) */
                double d2u_dx2 = (u[ir] - 2.0 * u[idx] + u[il]) / (grid->dx[i] * grid->dx[i]);
                double d2u_dy2 = (u[ju] - 2.0 * u[idx] + u[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2u_dz2 = (u[ku] - 2.0 * u[idx] + u[kd]) * inv_dz2;

                double d2v_dx2 = (v[ir] - 2.0 * v[idx] + v[il]) / (grid->dx[i] * grid->dx[i]);
                double d2v_dy2 = (v[ju] - 2.0 * v[idx] + v[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2v_dz2 = (v[ku] - 2.0 * v[idx] + v[kd]) * inv_dz2;

                double d2w_dx2 = (w[ir] - 2.0 * w[idx] + w[il]) / (grid->dx[i] * grid->dx[i]);
                double d2w_dy2 = (w[ju] - 2.0 * w[idx] + w[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2w_dz2 = (w[ku] - 2.0 * w[idx] + w[kd]) * inv_dz2;

                /* Kinematic viscosity */
                double nu = params->mu / fmax(rho[idx], 1e-10);
                nu = fmin(nu, 1.0);

                /* Clamp first derivatives */
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

                /* Clamp second derivatives */
                d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
                d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
                d2u_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dz2));
                d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
                d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));
                d2v_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dz2));
                d2w_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dx2));
                d2w_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dy2));
                d2w_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dz2));

                /* Source terms */
                double source_u = 0.0, source_v = 0.0, source_w = 0.0;
                double z_coord = (nz > 1 && grid->z) ? grid->z[k] : 0.0;
                compute_source_terms(grid->x[i], grid->y[j], z_coord, iter, dt,
                                     params, &source_u, &source_v, &source_w);

                /* RHS for u-momentum */
                rhs_u[idx] = -u[idx] * du_dx - v[idx] * du_dy - w[idx] * du_dz
                             - dp_dx / rho[idx]
                             + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
                             + source_u;

                /* RHS for v-momentum */
                rhs_v[idx] = -u[idx] * dv_dx - v[idx] * dv_dy - w[idx] * dv_dz
                             - dp_dy / rho[idx]
                             + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
                             + source_v;

                /* RHS for w-momentum */
                rhs_w[idx] = -u[idx] * dw_dx - v[idx] * dw_dy - w[idx] * dw_dz
                             - dp_dz / rho[idx]
                             + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2)
                             + source_w;

                /* Simplified pressure RHS (divergence-based) */
                double divergence = du_dx + dv_dy + dw_dz;
                divergence = fmax(-MAX_DIVERGENCE_LIMIT,
                                  fmin(MAX_DIVERGENCE_LIMIT, divergence));
                rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
            }
        }
    }
}

/* ============================================================================
 * RK2 OMP SOLVER
 * ============================================================================ */

cfd_status_t rk2_omp_impl(flow_field* field, const grid* grid,
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

    if (!k1_u || !k1_v || !k1_w || !k1_p ||
        !k2_u || !k2_v || !k2_w || !k2_p ||
        !u0 || !v0 || !w0 || !p0) {
        cfd_free(k1_u); cfd_free(k1_v); cfd_free(k1_w); cfd_free(k1_p);
        cfd_free(k2_u); cfd_free(k2_v); cfd_free(k2_w); cfd_free(k2_p);
        cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);
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

        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho,
                         k1_u, k1_v, k1_w, k1_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* ---- Intermediate: field = Q^n + dt * k1 ---- */
        {
            ptrdiff_t kk;
            OMP_FOR_SIMD
            for (kk = 0; kk < total_int; kk++) {
                field->u[kk] = u0[kk] + dt * k1_u[kk];
                field->v[kk] = v0[kk] + dt * k1_v[kk];
                field->w[kk] = w0[kk] + dt * k1_w[kk];
                field->p[kk] = p0[kk] + dt * k1_p[kk];

                field->u[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[kk]));
                field->v[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[kk]));
                field->w[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[kk]));
            }
        }

        /* NOTE: Do NOT apply BCs between RK stages. The ghost cells carry
         * zero-derivative evolution (k1[ghost]=0), which is consistent with
         * the semi-discrete ODE system. Applying BCs here would reduce RK2
         * to first-order temporal accuracy. */

        /* ---- Stage 2: k2 = RHS(Q_pred) ---- */
        memset(k2_u, 0, bytes);
        memset(k2_v, 0, bytes);
        memset(k2_w, 0, bytes);
        memset(k2_p, 0, bytes);

        compute_rhs_omp(field->u, field->v, field->w, field->p, field->rho,
                         k2_u, k2_v, k2_w, k2_p,
                         grid, params, nx, ny, nz,
                         stride_z, k_start, k_end, inv_2dz, inv_dz2,
                         iter, dt);

        /* ---- Final update: Q^{n+1} = Q^n + (dt/2)*(k1 + k2) ---- */
        {
            double half_dt = 0.5 * dt;
            ptrdiff_t kk;
            OMP_FOR_SIMD
            for (kk = 0; kk < total_int; kk++) {
                field->u[kk] = u0[kk] + half_dt * (k1_u[kk] + k2_u[kk]);
                field->v[kk] = v0[kk] + half_dt * (k1_v[kk] + k2_v[kk]);
                field->w[kk] = w0[kk] + half_dt * (k1_w[kk] + k2_w[kk]);
                field->p[kk] = p0[kk] + half_dt * (k1_p[kk] + k2_p[kk]);

                field->u[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->u[kk]));
                field->v[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->v[kk]));
                field->w[kk] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[kk]));
            }
        }

        /* Apply BCs to final state only (after the full RK2 step) */
        apply_boundary_conditions(field, grid);

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
    cfd_free(u0); cfd_free(v0); cfd_free(w0); cfd_free(p0);

    return status;
}

#endif /* CFD_ENABLE_OPENMP */
