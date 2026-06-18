/**
 * @file energy_solver_omp.c
 * @brief OpenMP-parallelized energy equation solver (advection-diffusion step)
 *
 * Solves: dT/dt + u*nabla(T) = alpha * nabla^2(T) + Q
 * using explicit Euler time integration and central differences.
 *
 * Same numerics as the scalar reference (energy/cpu/energy_solver.c); the
 * interior stencil loop is parallelized over j. Read/write separation
 * (reads field->T, writes T_new) makes the loop race-free. Buoyancy and
 * thermal BC application are shared with the scalar backend.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing identical results to a 2D code path.
 */

#include "../energy_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <omp.h>
#include <string.h>

cfd_status_t energy_step_explicit_omp_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* T_workspace, size_t workspace_size) {
    if (!field || !grid || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_omp: field, grid, and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (!field->T) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_omp: missing temperature field");
        return CFD_ERROR_INVALID;
    }

    /* Skip when energy equation is disabled */
    if (params->alpha <= 0.0) {
        return CFD_SUCCESS;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;

    if (!grid->dx || !grid->dy || nx < 3 || ny < 3) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_omp: grid too small or missing dx/dy");
        return CFD_ERROR_INVALID;
    }
    size_t plane = nx * ny;
    size_t total = plane * nz;
    double alpha = params->alpha;

    /* Validate uniform spacing (central-difference stencil assumes it) */
    const double dx0 = grid->dx[0];
    const double dy0 = grid->dy[0];
    {
        const double tol_x = 1e-12 * fmax(1.0, fabs(dx0));
        for (size_t i = 1; i < nx - 1; i++) {
            if (fabs(grid->dx[i] - dx0) > tol_x) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_omp: non-uniform dx not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
        const double tol_y = 1e-12 * fmax(1.0, fabs(dy0));
        for (size_t j = 1; j < ny - 1; j++) {
            if (fabs(grid->dy[j] - dy0) > tol_y) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_omp: non-uniform dy not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
    }
    if (nz > 1) {
        if (!grid->dz) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "energy_solver_omp: missing dz for 3D energy solve");
            return CFD_ERROR_INVALID;
        }
        const double dz0 = grid->dz[0];
        const double tol_z = 1e-12 * fmax(1.0, fabs(dz0));
        for (size_t k = 1; k < nz - 1; k++) {
            if (fabs(grid->dz[k] - dz0) > tol_z) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_omp: non-uniform dz not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
    }

    /* Precomputed constants for uniform-grid stencil */
    double inv_2dx = 1.0 / (2.0 * dx0);
    double inv_2dy = 1.0 / (2.0 * dy0);
    double inv_dx2 = 1.0 / (dx0 * dx0);
    double inv_dy2 = 1.0 / (dy0 * dy0);

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
        size_t k_offset = k * stride_z;
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (int i = 1; i < (int)nx - 1; i++) {
                size_t idx = k_offset + IDX_2D((size_t)i, (size_t)j, nx);

                double T_c = field->T[idx];
                double u_c = field->u[idx];
                double v_c = field->v[idx];
                double w_c = field->w[idx];

                /* Advection: u * dT/dx + v * dT/dy + w * dT/dz */
                double dT_dx = (field->T[idx + 1] - field->T[idx - 1]) * inv_2dx;
                double dT_dy = (field->T[idx + nx] - field->T[idx - nx]) * inv_2dy;
                double dT_dz = (field->T[idx + stride_z] - field->T[idx - stride_z]) * inv_2dz;

                double advection = u_c * dT_dx + v_c * dT_dy + w_c * dT_dz;

                /* Diffusion: alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2) */
                double d2T_dx2 = (field->T[idx + 1] - 2.0 * T_c + field->T[idx - 1]) * inv_dx2;
                double d2T_dy2 = (field->T[idx + nx] - 2.0 * T_c + field->T[idx - nx]) * inv_dy2;
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
    int has_nan = 0;
    ptrdiff_t total_int = (ptrdiff_t)total;
    ptrdiff_t n;
#pragma omp parallel for reduction(| : has_nan) schedule(static)
    for (n = 0; n < total_int; n++) {
        if (!isfinite(T_new[n])) {
            has_nan = 1;
        }
    }
    if (has_nan) {
        cfd_set_error(CFD_ERROR_DIVERGED,
                      "NaN/Inf detected in energy_step_explicit_omp");
        if (owns_buffer) cfd_free(T_new);
        return CFD_ERROR_DIVERGED;
    }

    memcpy(field->T, T_new, total * sizeof(double));
    if (owns_buffer) cfd_free(T_new);

    return CFD_SUCCESS;
}
