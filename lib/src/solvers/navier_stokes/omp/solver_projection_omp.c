#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

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
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;

    double* u_star = (double*)cfd_calloc(size, sizeof(double));
    double* v_star = (double*)cfd_calloc(size, sizeof(double));
    double* p_new = (double*)cfd_calloc(size, sizeof(double));
    double* p_temp = (double*)cfd_calloc(size, sizeof(double));
    double* rhs = (double*)cfd_calloc(size, sizeof(double));

    if (!u_star || !v_star || !p_new || !p_temp || !rhs) {
        cfd_free(u_star);
        cfd_free(v_star);
        cfd_free(p_new);
        cfd_free(p_temp);
        cfd_free(rhs);
        return CFD_ERROR_NOMEM;
    }

    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(p_new, field->p, size * sizeof(double));

    for (int iter = 0; iter < params->max_iter; iter++) {
        // STEP 1: Predictor
        // Use static scheduling for uniform grid load balancing
        int i, j, k;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = (j * nx) + i;

                double u = field->u[idx];
                double v = field->v[idx];

                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

                double conv_u = (u * du_dx) + (v * du_dy);
                double conv_v = (u * dv_dx) + (v * dv_dy);

                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);

                double visc_u = nu * (d2u_dx2 + d2u_dy2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2);

                double source_u = 0.0;
                double source_v = 0.0;
                if (params->source_amplitude_u > 0) {
                    source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]) *
                               exp(-params->source_decay_rate * iter * dt);
                    source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]) *
                               exp(-params->source_decay_rate * iter * dt);
                }

                u_star[idx] = u + (dt * (-conv_u + visc_u + source_u));
                v_star[idx] = v + (dt * (-conv_v + visc_v + source_v));

                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
            }
        }

        // Copy boundary values from field to u_star/v_star
        // This preserves whatever BCs the caller set (Dirichlet for cavity, etc.)
        // Bottom and top boundaries (j = 0 and j = ny-1)
        for (i = 0; i < (int)nx; i++) {
            u_star[i] = field->u[i];
            v_star[i] = field->v[i];
            u_star[(ny - 1) * nx + i] = field->u[(ny - 1) * nx + i];
            v_star[(ny - 1) * nx + i] = field->v[(ny - 1) * nx + i];
        }
        // Left and right boundaries (i = 0 and i = nx-1)
        for (j = 0; j < (int)ny; j++) {
            u_star[j * nx] = field->u[j * nx];
            v_star[j * nx] = field->v[j * nx];
            u_star[j * nx + nx - 1] = field->u[j * nx + nx - 1];
            v_star[j * nx + nx - 1] = field->v[j * nx + nx - 1];
        }

        // STEP 2: Pressure
        double rho = field->rho[0] < 1e-10 ? 1.0 : field->rho[0];

#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = (j * nx) + i;
                double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                rhs[idx] = (rho / dt) * (du_star_dx + dv_star_dy);
            }
        }

        int poisson_iters = poisson_solve(p_new, p_temp, rhs, nx, ny, dx, dy,
                                          POISSON_SOLVER_REDBLACK_OMP);

        if (poisson_iters < 0) {
            // Poisson solver didn't converge - use simple pressure update as fallback
            int k;
#pragma omp parallel for schedule(static)
            for (k = 0; k < (int)size; k++) {
                p_new[k] = field->p[k] - (0.1 * dt * rhs[k]);
            }
        }

// STEP 3: Corrector
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = (j * nx) + i;
                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);

                field->u[idx] = u_star[idx] - ((dt / rho) * dp_dx);
                field->v[idx] = v_star[idx] - ((dt / rho) * dp_dy);

                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
            }
        }

        memcpy(field->p, p_new, size * sizeof(double));

        // Copy boundary velocity values from u_star (which has caller's BCs)
        // to field to ensure boundary conditions are preserved
        for (i = 0; i < (int)nx; i++) {
            field->u[i] = u_star[i];
            field->v[i] = v_star[i];
            field->u[(ny - 1) * nx + i] = u_star[(ny - 1) * nx + i];
            field->v[(ny - 1) * nx + i] = v_star[(ny - 1) * nx + i];
        }
        for (j = 1; j < (int)ny - 1; j++) {
            field->u[j * nx] = u_star[j * nx];
            field->v[j * nx] = v_star[j * nx];
            field->u[j * nx + nx - 1] = u_star[j * nx + nx - 1];
            field->v[j * nx + nx - 1] = v_star[j * nx + nx - 1];
        }

        // Check for NaN
        for (k = 0; k < (int)size; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                cfd_free(u_star);
                cfd_free(v_star);
                cfd_free(p_new);
                cfd_free(p_temp);
                cfd_free(rhs);
                return CFD_ERROR_DIVERGED;
            }
        }
    }

    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(p_new);
    cfd_free(p_temp);
    cfd_free(rhs);
    return CFD_SUCCESS;
}
