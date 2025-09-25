#include "solver.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include "vtk_output.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Stable version of Navier-Stokes solver
void solve_navier_stokes_stable(FlowField* field, const Grid* grid, const SolverParams* params) {
    // Check for minimum grid size
    if (field->nx < 3 || field->ny < 3) {
        // Grid too small for finite differences, do nothing
        return;
    }

    // Allocate temporary arrays
    double* u_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* v_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* p_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* rho_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* T_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));

    // Initialize with current values
    memcpy(u_new, field->u, field->nx * field->ny * sizeof(double));
    memcpy(v_new, field->v, field->nx * field->ny * sizeof(double));
    memcpy(p_new, field->p, field->nx * field->ny * sizeof(double));
    memcpy(rho_new, field->rho, field->nx * field->ny * sizeof(double));
    memcpy(T_new, field->T, field->nx * field->ny * sizeof(double));

    // Use conservative time step
    double dt = fmin(params->dt, 0.0001);
    double min_dx = grid->dx[0];
    double min_dy = grid->dy[0];
    for (size_t i = 0; i < field->nx - 1; i++) {
        min_dx = fmin(min_dx, grid->dx[i]);
    }
    for (size_t j = 0; j < field->ny - 1; j++) {
        min_dy = fmin(min_dy, grid->dy[j]);
    }

    // CFL condition for stability
    double max_velocity = 0.0;
    for (size_t i = 0; i < field->nx * field->ny; i++) {
        double vel_mag = sqrt(field->u[i]*field->u[i] + field->v[i]*field->v[i]);
        max_velocity = fmax(max_velocity, vel_mag);
    }

    if (max_velocity > 1e-10) {
        double cfl_dt = 0.2 * fmin(min_dx, min_dy) / max_velocity;
        dt = fmin(dt, cfl_dt);
    }

    // Main iteration loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Update only interior points
        for (size_t j = 1; j < field->ny - 1; j++) {
            for (size_t i = 1; i < field->nx - 1; i++) {
                size_t idx = j * field->nx + i;

                // Safety check for current values
                if (!isfinite(field->u[idx]) || !isfinite(field->v[idx]) ||
                    !isfinite(field->p[idx]) || field->rho[idx] <= 0.0) {
                    continue;
                }

                // Compute gradients with safety checks
                double dx = grid->dx[i];
                double dy = grid->dy[j];

                if (dx <= 1e-10 || dy <= 1e-10) continue;

                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) / (2.0 * dy);

                // Pressure gradients
                double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * dx);
                double dp_dy = (field->p[idx + field->nx] - field->p[idx - field->nx]) / (2.0 * dy);

                // Second derivatives for diffusion
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + field->nx] - 2.0 * field->u[idx] + field->u[idx - field->nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + field->nx] - 2.0 * field->v[idx] + field->v[idx - field->nx]) / (dy * dy);

                // Kinematic viscosity with safety check
                double nu = params->mu / fmax(field->rho[idx], 1e-10);
                nu = fmin(nu, 1.0); // Limit viscosity

                // Conservative velocity updates
                double u_conv = field->u[idx] * du_dx + field->v[idx] * du_dy;
                double v_conv = field->u[idx] * dv_dx + field->v[idx] * dv_dy;

                double u_diff = nu * (d2u_dx2 + d2u_dy2);
                double v_diff = nu * (d2v_dx2 + d2v_dy2);

                double u_pres = dp_dx / field->rho[idx];
                double v_pres = dp_dy / field->rho[idx];

                // Limit all terms to prevent instability
                u_conv = fmax(-10.0, fmin(10.0, u_conv));
                v_conv = fmax(-10.0, fmin(10.0, v_conv));
                u_diff = fmax(-10.0, fmin(10.0, u_diff));
                v_diff = fmax(-10.0, fmin(10.0, v_diff));
                u_pres = fmax(-10.0, fmin(10.0, u_pres));
                v_pres = fmax(-10.0, fmin(10.0, v_pres));

                // Update velocities
                u_new[idx] = field->u[idx] + dt * (-u_conv - u_pres + u_diff);
                v_new[idx] = field->v[idx] + dt * (-v_conv - v_pres + v_diff);

                // Limit velocity magnitudes
                u_new[idx] = fmax(-10.0, fmin(10.0, u_new[idx]));
                v_new[idx] = fmax(-10.0, fmin(10.0, v_new[idx]));

                // Simple pressure update (diffusion only)
                double div = du_dx + dv_dy;
                div = fmax(-10.0, fmin(10.0, div));

                double p_correction = -0.1 * dt * field->rho[idx] * div;
                p_correction = fmax(-1.0, fmin(1.0, p_correction));

                p_new[idx] = field->p[idx] + p_correction;

                // Keep density and temperature unchanged
                rho_new[idx] = field->rho[idx];
                T_new[idx] = field->T[idx];
            }
        }

        // Copy new solution back
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));
        memcpy(field->rho, rho_new, field->nx * field->ny * sizeof(double));
        memcpy(field->T, T_new, field->nx * field->ny * sizeof(double));

        // Apply boundary conditions
        apply_boundary_conditions(field, grid);

        // Check for NaN/Inf and stop if found
        int has_nan = 0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            if (!isfinite(field->u[i]) || !isfinite(field->v[i]) || !isfinite(field->p[i])) {
                has_nan = 1;
                break;
            }
        }

        if (has_nan) {
            printf("Warning: NaN detected in iteration %d, stopping solver\n", iter);
            break;
        }

        // Output every 100 iterations
        if (iter % 100 == 0) {
            ensure_directory_exists("../../artifacts");
            ensure_directory_exists("../../artifacts/output");
            char filename[256];
#ifdef _WIN32
            sprintf_s(filename, sizeof(filename), "..\\..\\artifacts\\output\\output_stable_%d.vtk", iter);
#else
            sprintf(filename, "../../artifacts/output/output_stable_%d.vtk", iter);
#endif
            write_vtk_output(filename, "pressure", field->p, field->nx, field->ny,
                           grid->xmin, grid->xmax, grid->ymin, grid->ymax);
        }
    }

    // Free temporary arrays
    cfd_free(u_new);
    cfd_free(v_new);
    cfd_free(p_new);
    cfd_free(rho_new);
    cfd_free(T_new);
}