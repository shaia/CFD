#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include "../boundary_copy_utils.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical limits
#define MAX_VELOCITY 100.0

cfd_status_t solve_projection_method_omp(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params) {
    if (!field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
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

    double* u_star = (double*)cfd_calloc(total, sizeof(double));
    double* v_star = (double*)cfd_calloc(total, sizeof(double));
    double* w_star = (double*)cfd_calloc(total, sizeof(double));
    double* p_new = (double*)cfd_calloc(total, sizeof(double));
    double* p_temp = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));

    if (!u_star || !v_star || !w_star || !p_new || !p_temp || !rhs) {
        cfd_free(u_star);
        cfd_free(v_star);
        cfd_free(w_star);
        cfd_free(p_new);
        cfd_free(p_temp);
        cfd_free(rhs);
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

                    double conv_u = u * du_dx + v * du_dy + w * du_dz;
                    double conv_v = u * dv_dx + v * dv_dy + w * dv_dz;
                    double conv_w = u * dw_dx + v * dw_dy + w * dw_dz;

                    double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                    double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                    double d2u_dz2 = (field->u[idx + stride_z] - 2.0 * u + field->u[idx - stride_z]) * inv_dz2;
                    double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                    double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);
                    double d2v_dz2 = (field->v[idx + stride_z] - 2.0 * v + field->v[idx - stride_z]) * inv_dz2;
                    double d2w_dx2 = (field->w[idx + 1] - 2.0 * w + field->w[idx - 1]) / (dx * dx);
                    double d2w_dy2 = (field->w[idx + nx] - 2.0 * w + field->w[idx - nx]) / (dy * dy);
                    double d2w_dz2 = (field->w[idx + stride_z] - 2.0 * w + field->w[idx - stride_z]) * inv_dz2;

                    double visc_u = nu * (d2u_dx2 + d2u_dy2 + d2u_dz2);
                    double visc_v = nu * (d2v_dx2 + d2v_dy2 + d2v_dz2);
                    double visc_w = nu * (d2w_dx2 + d2w_dy2 + d2w_dz2);

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

                    u_star[idx] = u + dt * (-conv_u + visc_u + source_u);
                    v_star[idx] = v + dt * (-conv_v + visc_v + source_v);
                    w_star[idx] = w + dt * (-conv_w + visc_w + source_w);

                    u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                    v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
                    w_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, w_star[idx]));
                }
            }
        }

        /* Copy boundary values from field to star arrays */
        copy_boundary_velocities_3d(u_star, v_star, w_star,
                                    field->u, field->v, field->w, nx, ny, nz);

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

        /* Use OMP CG for parallelized Poisson solve */
        int poisson_iters = poisson_solve_3d(p_new, p_temp, rhs, nx, ny, nz,
                                             dx, dy, dz, POISSON_SOLVER_CG_OMP);

        if (poisson_iters < 0) {
            cfd_free(u_star);
            cfd_free(v_star);
            cfd_free(w_star);
            cfd_free(p_new);
            cfd_free(p_temp);
            cfd_free(rhs);
            return CFD_ERROR_MAX_ITER;
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

        /* Copy boundary velocity values from star arrays (which have caller's BCs) */
        copy_boundary_velocities_3d(field->u, field->v, field->w,
                                    u_star, v_star, w_star, nx, ny, nz);

        /* Check for NaN/Inf values (parallelized) */
        int has_nan = 0;
        int limit = (int)total;
        int ii;
#pragma omp parallel for reduction(| : has_nan) schedule(static)
        for (ii = 0; ii < limit; ii++) {
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
            return CFD_ERROR_DIVERGED;
        }
    }

    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(w_star);
    cfd_free(p_new);
    cfd_free(p_temp);
    cfd_free(rhs);
    return CFD_SUCCESS;
}
