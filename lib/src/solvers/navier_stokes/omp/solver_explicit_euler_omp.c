#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include "cfd/solvers/navier_stokes_solver.h"
#include "../boundary_copy_utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical stability limits (same as cpu solver)
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DIVERGENCE_LIMIT        10.0
#define DT_CONSERVATIVE_LIMIT       0.0001
#define UPDATE_LIMIT                1.0
#define PRESSURE_UPDATE_FACTOR      0.1

// Internal OpenMP explicit Euler implementation
cfd_status_t explicit_euler_omp_impl(flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params) {
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    size_t nz = field->nz;
    if (nz > 1 && grid->dz) {
        for (size_t kk = 1; kk < nz - 1; kk++) {
            if (fabs(grid->dz[kk] - grid->dz[0]) > 1e-14) {
                return CFD_ERROR_INVALID;
            }
        }
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t plane = nx * ny;
    size_t total = plane * nz;

    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start  = (nz > 1) ? 1 : 0;
    size_t k_end    = (nz > 1) ? (nz - 1) : 1;
    double inv_2dz  = (nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    double inv_dz2  = (nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    // Allocate temporary arrays
    double* u_new = (double*)cfd_calloc(total, sizeof(double));
    double* v_new = (double*)cfd_calloc(total, sizeof(double));
    double* w_new = (double*)cfd_calloc(total, sizeof(double));
    double* p_new = (double*)cfd_calloc(total, sizeof(double));

    if (!u_new || !v_new || !w_new || !p_new) {
        cfd_free(u_new);
        cfd_free(v_new);
        cfd_free(w_new);
        cfd_free(p_new);
        return CFD_ERROR_NOMEM;
    }

    // Initialize with current values
    memcpy(u_new, field->u, total * sizeof(double));
    memcpy(v_new, field->v, total * sizeof(double));
    memcpy(w_new, field->w, total * sizeof(double));
    memcpy(p_new, field->p, total * sizeof(double));

    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

    for (int iter = 0; iter < params->max_iter; iter++) {
        for (size_t kk = k_start; kk < k_end; kk++) {
            int j;
#pragma omp parallel for schedule(static)
            for (j = 1; j < (int)ny - 1; j++) {
                for (int i = 1; i < (int)nx - 1; i++) {
                    size_t idx = (kk * stride_z) + IDX_2D(i, j, nx);

                    // Derivatives
                    double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
                    double du_dy =
                        (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * grid->dy[j]);
                    double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
                    double dv_dy =
                        (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * grid->dy[j]);

                    double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
                    double dp_dy =
                        (field->p[idx + nx] - field->p[idx - nx]) / (2.0 * grid->dy[j]);

                    double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) /
                                     (grid->dx[i] * grid->dx[i]);
                    double d2u_dy2 =
                        (field->u[idx + nx] - 2.0 * field->u[idx] + field->u[idx - nx]) /
                        (grid->dy[j] * grid->dy[j]);
                    double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) /
                                     (grid->dx[i] * grid->dx[i]);
                    double d2v_dy2 =
                        (field->v[idx + nx] - 2.0 * field->v[idx] + field->v[idx - nx]) /
                        (grid->dy[j] * grid->dy[j]);

                    double du_dz = (field->u[idx + stride_z] - field->u[idx - stride_z]) * inv_2dz;
                    double dv_dz = (field->v[idx + stride_z] - field->v[idx - stride_z]) * inv_2dz;
                    double dw_dx = (field->w[idx + 1] - field->w[idx - 1]) / (2.0 * grid->dx[i]);
                    double dw_dy = (field->w[idx + nx] - field->w[idx - nx]) / (2.0 * grid->dy[j]);
                    double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) * inv_2dz;
                    double dp_dz = (field->p[idx + stride_z] - field->p[idx - stride_z]) * inv_2dz;

                    double d2u_dz2 = (field->u[idx + stride_z] - 2.0 * field->u[idx] + field->u[idx - stride_z]) * inv_dz2;
                    double d2v_dz2 = (field->v[idx + stride_z] - 2.0 * field->v[idx] + field->v[idx - stride_z]) * inv_dz2;
                    double d2w_dx2 = (field->w[idx + 1] - 2.0 * field->w[idx] + field->w[idx - 1]) / (grid->dx[i] * grid->dx[i]);
                    double d2w_dy2 = (field->w[idx + nx] - 2.0 * field->w[idx] + field->w[idx - nx]) / (grid->dy[j] * grid->dy[j]);
                    double d2w_dz2 = (field->w[idx + stride_z] - 2.0 * field->w[idx] + field->w[idx - stride_z]) * inv_dz2;

                    if (field->rho[idx] <= 1e-10) {
                        continue;
                    }

                    double nu = params->mu / fmax(field->rho[idx], 1e-10);
                    nu = fmin(nu, 1.0);

                    // Source terms
                    double source_u = 0.0;
                    double source_v = 0.0;
                    double source_w = 0.0;
                    double z_coord = (nz > 1 && grid->z) ? grid->z[kk] : 0.0;
                    if (params->source_func) {
                        params->source_func(grid->x[i], grid->y[j], z_coord,
                                            iter * conservative_dt,
                                            params->source_context,
                                            &source_u, &source_v, &source_w);
                    } else {
                        source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]) *
                                   exp(-params->source_decay_rate * iter * conservative_dt);
                        source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]) *
                                   exp(-params->source_decay_rate * iter * conservative_dt);
                    }

                    // Update u
                    double du = conservative_dt * (-field->u[idx] * du_dx - field->v[idx] * du_dy
                                                   - field->w[idx] * du_dz
                                                   - dp_dx / fmax(field->rho[idx], 1e-10)
                                                   + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2) + source_u);

                    // Update v
                    double dv = conservative_dt * (-field->u[idx] * dv_dx - field->v[idx] * dv_dy
                                                   - field->w[idx] * dv_dz
                                                   - dp_dy / fmax(field->rho[idx], 1e-10)
                                                   + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2) + source_v);

                    // Update w
                    double dw = conservative_dt * (-field->u[idx] * dw_dx - field->v[idx] * dw_dy
                                                   - field->w[idx] * dw_dz
                                                   - dp_dz / fmax(field->rho[idx], 1e-10)
                                                   + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2) + source_w);

                    du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
                    dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));
                    dw = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dw));

                    u_new[idx] = field->u[idx] + du;
                    v_new[idx] = field->v[idx] + dv;

                    // Limit velocity
                    u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_new[idx]));
                    v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_new[idx]));
                    w_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, field->w[idx] + dw));

                    // Pressure update
                    double divergence = du_dx + dv_dy + dw_dz;
                    divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));

                    double dp =
                        -PRESSURE_UPDATE_FACTOR * conservative_dt * field->rho[idx] * divergence;
                    dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));
                    p_new[idx] = field->p[idx] + dp;
                }
            }
        }

        // Copy back (implicit barrier at end of parallel for)
        memcpy(field->u, u_new, total * sizeof(double));
        memcpy(field->v, v_new, total * sizeof(double));
        memcpy(field->w, w_new, total * sizeof(double));
        memcpy(field->p, p_new, total * sizeof(double));

        // Store caller-set boundary values before apply_boundary_conditions overwrites them,
        // then restore them afterward.
        copy_boundary_velocities_3d(u_new, v_new, w_new,
                                    field->u, field->v, field->w, nx, ny, nz);
        apply_boundary_conditions(field, grid);
        copy_boundary_velocities_3d(field->u, field->v, field->w,
                                    u_new, v_new, w_new, nx, ny, nz);

        // Check for NaN/Inf values
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
            cfd_free(u_new);
            cfd_free(v_new);
            cfd_free(w_new);
            cfd_free(p_new);
            cfd_set_error(CFD_ERROR_DIVERGED,
                          "NaN/Inf detected in explicit_euler_omp step");
            return CFD_ERROR_DIVERGED;
        }
    }

    cfd_free(u_new);
    cfd_free(v_new);
    cfd_free(w_new);
    cfd_free(p_new);

    return CFD_SUCCESS;
}
