/**
 * @file energy_solver.c
 * @brief Scalar CPU implementation of the energy equation solver
 *
 * Solves: dT/dt + u*nabla(T) = alpha * nabla^2(T) + Q
 * using explicit Euler time integration and central differences.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing identical results to a 2D code path.
 */

#include "cfd/solvers/energy_solver.h"
#include "../energy_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

cfd_status_t energy_step_explicit_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* T_workspace, size_t workspace_size) {
    /* Skip when energy equation is disabled */
    if (params->alpha <= 0.0) {
        return CFD_SUCCESS;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;
    size_t plane = nx * ny;
    size_t total = plane * nz;
    double alpha = params->alpha;

    /* Validate uniform dz for 3D (energy solver uses constant z-spacing) */
    if (nz > 1) {
        if (!grid->dz) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "energy_solver: missing dz for 3D energy solve");
            return CFD_ERROR_INVALID;
        }
        const double dz0 = grid->dz[0];
        const double tol = 1e-12 * fmax(1.0, fabs(dz0));
        for (size_t k = 1; k < nz - 1; k++) {
            if (fabs(grid->dz[k] - dz0) > tol) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver: non-uniform dz not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
    }

    /* Branch-free 3D constants */
    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start  = (nz > 1) ? 1 : 0;
    size_t k_end    = (nz > 1) ? (nz - 1) : 1;
    double inv_2dz  = (nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    double inv_dz2  = (nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    /* Use caller's workspace or allocate internally */
    int owns_buffer = 0;
    double* T_new;
    if (T_workspace && workspace_size >= total) {
        T_new = T_workspace;
    } else {
        T_new = (double*)cfd_calloc(total, sizeof(double));
        if (!T_new) {
            return CFD_ERROR_NOMEM;
        }
        owns_buffer = 1;
    }
    memcpy(T_new, field->T, total * sizeof(double));

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

                double T_c = field->T[idx];
                double u_c = field->u[idx];
                double v_c = field->v[idx];
                double w_c = field->w[idx];

                double dx = grid->dx[i];
                double dy = grid->dy[j];

                /* Advection: u * dT/dx + v * dT/dy + w * dT/dz */
                double dT_dx = (field->T[idx + 1] - field->T[idx - 1]) / (2.0 * dx);
                double dT_dy = (field->T[idx + nx] - field->T[idx - nx]) / (2.0 * dy);
                double dT_dz = (field->T[idx + stride_z] - field->T[idx - stride_z]) * inv_2dz;

                double advection = u_c * dT_dx + v_c * dT_dy + w_c * dT_dz;

                /* Diffusion: alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2) */
                double d2T_dx2 = (field->T[idx + 1] - 2.0 * T_c + field->T[idx - 1]) / (dx * dx);
                double d2T_dy2 = (field->T[idx + nx] - 2.0 * T_c + field->T[idx - nx]) / (dy * dy);
                double d2T_dz2 = (field->T[idx + stride_z] - 2.0 * T_c +
                                   field->T[idx - stride_z]) * inv_dz2;

                double diffusion = alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2);

                /* Heat source term */
                double Q = 0.0;
                if (params->heat_source_func) {
                    double x = grid->x[i];
                    double y = grid->y[j];
                    double z = (nz > 1 && grid->z) ? grid->z[k] : 0.0;
                    Q = params->heat_source_func(x, y, z, time,
                                                  params->heat_source_context);
                }

                /* Explicit Euler update */
                double dT = dt * (-advection + diffusion + Q);
                T_new[idx] = T_c + dT;
            }
        }
    }

    /* Check for NaN/Inf */
    for (size_t n = 0; n < total; n++) {
        if (!isfinite(T_new[n])) {
            cfd_set_error(CFD_ERROR_DIVERGED,
                          "NaN/Inf detected in energy_step_explicit");
            if (owns_buffer) cfd_free(T_new);
            return CFD_ERROR_DIVERGED;
        }
    }

    memcpy(field->T, T_new, total * sizeof(double));
    if (owns_buffer) cfd_free(T_new);

    return CFD_SUCCESS;
}

cfd_status_t energy_step_explicit(flow_field* field, const grid* grid,
                                   const ns_solver_params_t* params,
                                   double dt, double time) {
    return energy_step_explicit_with_workspace(field, grid, params,
                                                dt, time, NULL, 0);
}

void energy_compute_buoyancy(double T_local, const ns_solver_params_t* params,
                              double* source_u, double* source_v,
                              double* source_w) {
    if (params->beta == 0.0) {
        return;
    }

    double dT = T_local - params->T_ref;
    *source_u += -params->beta * dT * params->gravity[0];
    *source_v += -params->beta * dT * params->gravity[1];
    *source_w += -params->beta * dT * params->gravity[2];
}
