#include "solver.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include "vtk_output.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to initialize SolverParams with default values
SolverParams solver_params_default(void) {
    SolverParams params = {
        .dt = DEFAULT_TIME_STEP,
        .cfl = DEFAULT_CFL_NUMBER,
        .gamma = DEFAULT_GAMMA,
        .mu = DEFAULT_VISCOSITY,
        .k = DEFAULT_THERMAL_CONDUCTIVITY,
        .max_iter = DEFAULT_MAX_ITERATIONS,
        .tolerance = DEFAULT_TOLERANCE,
        .source_amplitude_u = DEFAULT_SOURCE_AMPLITUDE_U,
        .source_amplitude_v = DEFAULT_SOURCE_AMPLITUDE_V,
        .source_decay_rate = DEFAULT_SOURCE_DECAY_RATE,
        .pressure_coupling = DEFAULT_PRESSURE_COUPLING
    };
    return params;
}
FlowField* flow_field_create(size_t nx, size_t ny) {
    FlowField* field = (FlowField*)cfd_malloc(sizeof(FlowField));
    
    field->nx = nx;
    field->ny = ny;
    
    // Allocate memory for flow variables
    field->u = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->v = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->p = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->rho = (double*)cfd_calloc(nx * ny, sizeof(double));
    field->T = (double*)cfd_calloc(nx * ny, sizeof(double));
    
    return field;
}

void flow_field_destroy(FlowField* field) {
    if (field != NULL) {
        cfd_free(field->u);
        cfd_free(field->v);
        cfd_free(field->p);
        cfd_free(field->rho);
        cfd_free(field->T);
        cfd_free(field);
    }
}

void initialize_flow_field(FlowField* field, const Grid* grid) {
    // Initialize with a more stable flow field
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = grid->x[i];
            double y = grid->y[j];

            // Set initial conditions with more interesting flow
            field->u[idx] = 1.0 + 0.1 * sin(M_PI * y);  // Slightly varying u-velocity
            field->v[idx] = 0.05 * sin(2.0 * M_PI * x); // Small v-velocity variation
            field->p[idx] = 1.0;  // Reference pressure
            field->rho[idx] = 1.0;  // Reference density
            field->T[idx] = 300.0;  // Reference temperature (K)

            // Add a pressure perturbation for interesting flow
            double cx = 1.0, cy = 0.5;  // Center of perturbation
            double r = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (r < 0.2) {
                field->p[idx] += 0.1 * exp(-r * r / 0.02);
                // Adjust velocities based on pressure gradient
                double dp_dx = -0.1 * 2.0 * (x - cx) / 0.02 * exp(-r * r / 0.02);
                double dp_dy = -0.1 * 2.0 * (y - cy) / 0.02 * exp(-r * r / 0.02);
                field->u[idx] += -0.1 * dp_dx;  // Simple pressure-velocity coupling
                field->v[idx] += -0.1 * dp_dy;
            }
        }
    }
}

void compute_time_step(FlowField* field, const Grid* grid, SolverParams* params) {
    double max_speed = 0.0;
    double dx_min = grid->dx[0];
    double dy_min = grid->dy[0];

    // Find minimum grid spacing
    for (size_t i = 0; i < grid->nx - 1; i++) {
        dx_min = min_double(dx_min, grid->dx[i]);
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        dy_min = min_double(dy_min, grid->dy[j]);
    }

    // Find maximum wave speed
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double u_speed = fabs(field->u[idx]);
            double v_speed = fabs(field->v[idx]);
            double sound_speed = sqrt(params->gamma * field->p[idx] / field->rho[idx]);
            double local_speed = sqrt(u_speed * u_speed + v_speed * v_speed) + sound_speed;
            max_speed = max_double(max_speed, local_speed);
        }
    }

    // Prevent division by zero and ensure reasonable time step
    if (max_speed < 1e-10) {
        max_speed = 1.0;  // Use default if speeds are too small
    }

    // Compute time step based on CFL condition with safety factor
    double dt_cfl = params->cfl * min_double(dx_min, dy_min) / max_speed;

    // Limit time step to reasonable bounds
    double dt_max = 0.01;  // Maximum allowed time step
    double dt_min = 1e-6;  // Minimum allowed time step

    params->dt = max_double(dt_min, min_double(dt_max, dt_cfl));
}

void apply_boundary_conditions(FlowField* field, const Grid* grid) {
    // Apply periodic boundary conditions in x-direction
    for (size_t j = 0; j < field->ny; j++) {
        field->u[j * field->nx + 0] = field->u[j * field->nx + field->nx - 2];
        field->v[j * field->nx + 0] = field->v[j * field->nx + field->nx - 2];
        field->p[j * field->nx + 0] = field->p[j * field->nx + field->nx - 2];
        field->rho[j * field->nx + 0] = field->rho[j * field->nx + field->nx - 2];
        field->T[j * field->nx + 0] = field->T[j * field->nx + field->nx - 2];

        field->u[j * field->nx + field->nx - 1] = field->u[j * field->nx + 1];
        field->v[j * field->nx + field->nx - 1] = field->v[j * field->nx + 1];
        field->p[j * field->nx + field->nx - 1] = field->p[j * field->nx + 1];
        field->rho[j * field->nx + field->nx - 1] = field->rho[j * field->nx + 1];
        field->T[j * field->nx + field->nx - 1] = field->T[j * field->nx + 1];
    }

    // Apply periodic boundary conditions in y-direction (instead of walls)
    for (size_t i = 0; i < field->nx; i++) {
        // Bottom boundary = top interior
        field->u[i] = field->u[(field->ny - 2) * field->nx + i];
        field->v[i] = field->v[(field->ny - 2) * field->nx + i];
        field->p[i] = field->p[(field->ny - 2) * field->nx + i];
        field->rho[i] = field->rho[(field->ny - 2) * field->nx + i];
        field->T[i] = field->T[(field->ny - 2) * field->nx + i];

        // Top boundary = bottom interior
        size_t top_idx = (field->ny - 1) * field->nx + i;
        field->u[top_idx] = field->u[field->nx + i];
        field->v[top_idx] = field->v[field->nx + i];
        field->p[top_idx] = field->p[field->nx + i];
        field->rho[top_idx] = field->rho[field->nx + i];
        field->T[top_idx] = field->T[field->nx + i];
    }
}

// Helper function to compute source terms consistently across all solvers
void compute_source_terms(double x, double y, int iter, double dt, const SolverParams* params,
                         double* source_u, double* source_v) {
    *source_u = params->source_amplitude_u * sin(M_PI * y) * exp(-params->source_decay_rate * iter * dt);
    *source_v = params->source_amplitude_v * sin(2.0 * M_PI * x) * exp(-params->source_decay_rate * iter * dt);
}

void solve_navier_stokes(FlowField* field, const Grid* grid, const SolverParams* params) {
    // Check for minimum grid size - prevent crashes on small grids
    if (field->nx < 3 || field->ny < 3) {
        return; // Skip solver for grids too small for finite differences
    }

    // Allocate temporary arrays for the solution update
    double* u_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* v_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* p_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* rho_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* T_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));

    // Initialize with current values to prevent uninitialized memory
    memcpy(u_new, field->u, field->nx * field->ny * sizeof(double));
    memcpy(v_new, field->v, field->nx * field->ny * sizeof(double));
    memcpy(p_new, field->p, field->nx * field->ny * sizeof(double));
    memcpy(rho_new, field->rho, field->nx * field->ny * sizeof(double));
    memcpy(T_new, field->T, field->nx * field->ny * sizeof(double));

    // Use conservative time step to prevent instabilities
    double conservative_dt = fmin(params->dt, 0.0001);

    // Main time-stepping loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        
        // Update solution using explicit Euler method
        for (size_t j = 1; j < field->ny - 1; j++) {
            for (size_t i = 1; i < field->nx - 1; i++) {
                size_t idx = j * field->nx + i;

                // Compute spatial derivatives
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) /
                             (2.0 * grid->dx[i]);
                double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) /
                             (2.0 * grid->dy[j]);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) /
                             (2.0 * grid->dx[i]);
                double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) /
                             (2.0 * grid->dy[j]);

                // Pressure gradients
                double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) /
                             (2.0 * grid->dx[i]);
                double dp_dy = (field->p[idx + field->nx] - field->p[idx - field->nx]) /
                             (2.0 * grid->dy[j]);

                // Second derivatives for viscous terms
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) /
                               (grid->dx[i] * grid->dx[i]);
                double d2u_dy2 = (field->u[idx + field->nx] - 2.0 * field->u[idx] + field->u[idx - field->nx]) /
                               (grid->dy[j] * grid->dy[j]);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) /
                               (grid->dx[i] * grid->dx[i]);
                double d2v_dy2 = (field->v[idx + field->nx] - 2.0 * field->v[idx] + field->v[idx - field->nx]) /
                               (grid->dy[j] * grid->dy[j]);

                // Safety checks to prevent division by zero
                if (field->rho[idx] <= 1e-10) continue;
                if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) continue;

                // Viscosity coefficient (kinematic viscosity = dynamic viscosity / density) with safety
                double nu = params->mu / fmax(field->rho[idx], 1e-10);
                nu = fmin(nu, 1.0);  // Limit maximum viscosity

                // Limit derivatives to prevent instabilities
                du_dx = fmax(-100.0, fmin(100.0, du_dx));
                du_dy = fmax(-100.0, fmin(100.0, du_dy));
                dv_dx = fmax(-100.0, fmin(100.0, dv_dx));
                dv_dy = fmax(-100.0, fmin(100.0, dv_dy));
                dp_dx = fmax(-100.0, fmin(100.0, dp_dx));
                dp_dy = fmax(-100.0, fmin(100.0, dp_dy));
                d2u_dx2 = fmax(-1000.0, fmin(1000.0, d2u_dx2));
                d2u_dy2 = fmax(-1000.0, fmin(1000.0, d2u_dy2));
                d2v_dx2 = fmax(-1000.0, fmin(1000.0, d2v_dx2));
                d2v_dy2 = fmax(-1000.0, fmin(1000.0, d2v_dy2));

                // Conservative velocity updates with limited changes
                double du = conservative_dt * (
                    -field->u[idx] * du_dx - field->v[idx] * du_dy  // Convection
                    - dp_dx / field->rho[idx]                        // Pressure gradient
                    + nu * (d2u_dx2 + d2u_dy2)                      // Viscous diffusion
                );

                double dv = conservative_dt * (
                    -field->u[idx] * dv_dx - field->v[idx] * dv_dy  // Convection
                    - dp_dy / field->rho[idx]                        // Pressure gradient
                    + nu * (d2v_dx2 + d2v_dy2)                      // Viscous diffusion
                );

                // Limit velocity changes
                du = fmax(-1.0, fmin(1.0, du));
                dv = fmax(-1.0, fmin(1.0, dv));

                u_new[idx] = field->u[idx] + du;
                v_new[idx] = field->v[idx] + dv;

                // Limit velocity magnitudes
                u_new[idx] = fmax(-100.0, fmin(100.0, u_new[idx]));
                v_new[idx] = fmax(-100.0, fmin(100.0, v_new[idx]));

                // Simplified stable pressure update
                double divergence = du_dx + dv_dy;
                divergence = fmax(-10.0, fmin(10.0, divergence));

                double dp = -0.1 * conservative_dt * field->rho[idx] * divergence;
                dp = fmax(-1.0, fmin(1.0, dp));  // Limit pressure changes
                p_new[idx] = field->p[idx] + dp;

                // Keep density and temperature constant for this simplified model
                rho_new[idx] = field->rho[idx];
                T_new[idx] = field->T[idx];
            }
        }
        
        // Copy new solution to old solution
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));
        memcpy(field->rho, rho_new, field->nx * field->ny * sizeof(double));
        memcpy(field->T, T_new, field->nx * field->ny * sizeof(double));
        
        // Apply boundary conditions
        apply_boundary_conditions(field, grid);

        // Check for NaN/Inf values and stop if found
        int has_nan = 0;
        for (size_t k = 0; k < field->nx * field->ny; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                has_nan = 1;
                break;
            }
        }

        if (has_nan) {
            printf("Warning: NaN/Inf detected in iteration %d, stopping solver\n", iter);
            break;
        }

        // Output solution every 100 iterations
        if (iter % 100 == 0) {
            ensure_directory_exists("../../output");
            ensure_directory_exists("..\\..\\artifacts\\output");
            char filename[256];
#ifdef _WIN32
            sprintf_s(filename, sizeof(filename), "..\\..\\artifacts\\output\\output_%d.vtk", iter);
#else
            sprintf(filename, "..\\..\\artifacts\\output\\output_%d.vtk", iter);
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

