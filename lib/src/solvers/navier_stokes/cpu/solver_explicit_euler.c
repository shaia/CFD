#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"


#include "cfd/solvers/navier_stokes_solver.h"
#include <math.h>
#include <stdio.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical stability limits for numerical computation
#define MAX_DERIVATIVE_LIMIT        100.0   // Maximum allowed first derivative magnitude (1/s)
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0  // Maximum allowed second derivative magnitude (1/s²)
#define MAX_VELOCITY_LIMIT          100.0   // Maximum allowed velocity magnitude (m/s)
#define MAX_DIVERGENCE_LIMIT        10.0    // Maximum allowed velocity divergence (1/s)

// Initial condition constants
#define INIT_U_BASE   1.0
#define INIT_U_VAR    0.1
#define INIT_V_VAR    0.05
#define INIT_PRESSURE 1.0
#define INIT_DENSITY  1.0
#define INIT_TEMP     300.0

// Perturbation constants
#define PERTURB_CENTER_X    1.0
#define PERTURB_CENTER_Y    0.5
#define PERTURB_RADIUS      0.2
#define PERTURB_WIDTH_SQ    0.02
#define PERTURB_MAG         0.1
#define PERTURB_GRAD_FACTOR 2.0

// Time stepping and stability constants
#define VELOCITY_EPSILON      1e-20
#define SPEED_EPSILON         1e-10
#define DT_MAX_LIMIT          0.01
#define DT_MIN_LIMIT          1e-6
#define DT_CONSERVATIVE_LIMIT 0.0001

// Update limits
#define UPDATE_LIMIT           1.0
#define PRESSURE_UPDATE_FACTOR 0.1

// Helper function to initialize ns_solver_params_t with default values
ns_solver_params_t ns_solver_params_default(void) {
    ns_solver_params_t params = {.dt = DEFAULT_TIME_STEP,
                            .cfl = DEFAULT_CFL_NUMBER,
                            .gamma = DEFAULT_GAMMA,
                            .mu = DEFAULT_VISCOSITY,
                            .k = DEFAULT_THERMAL_CONDUCTIVITY,
                            .max_iter = DEFAULT_MAX_ITERATIONS,
                            .tolerance = DEFAULT_TOLERANCE,
                            .source_amplitude_u = DEFAULT_SOURCE_AMPLITUDE_U,
                            .source_amplitude_v = DEFAULT_SOURCE_AMPLITUDE_V,
                            .source_decay_rate = DEFAULT_SOURCE_DECAY_RATE,
                            .pressure_coupling = DEFAULT_PRESSURE_COUPLING};
    return params;
}
flow_field* flow_field_create_3d(size_t nx, size_t ny, size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "Flow field dimensions must be positive");
        return NULL;
    }

    flow_field* field = (flow_field*)cfd_calloc(1, sizeof(flow_field));
    if (field == NULL) {
        return NULL;
    }

    field->nx = nx;
    field->ny = ny;
    field->nz = nz;

    size_t total = nx * ny * nz;

    // Allocate 32-byte aligned memory for flow variables (optimized for SIMD operations)
    field->u = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->v = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->w = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->p = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->rho = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->T = (double*)cfd_aligned_calloc(total, sizeof(double));

    if (!field->u || !field->v || !field->w || !field->p || !field->rho || !field->T) {
        flow_field_destroy(field);
        return NULL;
    }

    return field;
}

flow_field* flow_field_create(size_t nx, size_t ny) {
    return flow_field_create_3d(nx, ny, 1);
}

void flow_field_destroy(flow_field* field) {
    if (field != NULL) {
        cfd_aligned_free(field->u);
        cfd_aligned_free(field->v);
        cfd_aligned_free(field->w);
        cfd_aligned_free(field->p);
        cfd_aligned_free(field->rho);
        cfd_aligned_free(field->T);
        cfd_free(field);
    }
}

void initialize_flow_field(flow_field* field, const grid* grid) {
    // Initialize with a more stable flow field
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = IDX_2D(i, j, field->nx);
            double x = grid->x[i];
            double y = grid->y[j];

            // Set initial conditions with more interesting flow
            field->u[idx] =
                INIT_U_BASE + (INIT_U_VAR * sin(M_PI * y));    // Slightly varying u-velocity
            field->v[idx] = INIT_V_VAR * sin(2.0 * M_PI * x);  // Small v-velocity variation
            field->p[idx] = INIT_PRESSURE;                     // Reference pressure
            field->rho[idx] = INIT_DENSITY;                    // Reference density
            field->T[idx] = INIT_TEMP;                         // Reference temperature (K)

            // Add a pressure perturbation for interesting flow
            double cx = PERTURB_CENTER_X, cy = PERTURB_CENTER_Y;  // Center of perturbation
            double r = sqrt(((x - cx) * (x - cx)) + ((y - cy) * (y - cy)));
            if (r < PERTURB_RADIUS) {
                field->p[idx] += PERTURB_MAG * exp(-r * r / PERTURB_WIDTH_SQ);
                // Adjust velocities based on pressure gradient
                double dp_dx = -PERTURB_MAG * PERTURB_GRAD_FACTOR * (x - cx) / PERTURB_WIDTH_SQ *
                               exp(-r * r / PERTURB_WIDTH_SQ);
                double dp_dy = -PERTURB_MAG * PERTURB_GRAD_FACTOR * (y - cy) / PERTURB_WIDTH_SQ *
                               exp(-r * r / PERTURB_WIDTH_SQ);
                field->u[idx] += -PERTURB_MAG * dp_dx;  // Simple pressure-velocity coupling
                field->v[idx] += -PERTURB_MAG * dp_dy;
            }
        }
    }
}

void compute_time_step(flow_field* field, const grid* grid, ns_solver_params_t* params) {
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
            size_t idx = IDX_2D(i, j, field->nx);
            double u_speed = fabs(field->u[idx]);
            double v_speed = fabs(field->v[idx]);
            double sound_speed = sqrt(params->gamma * field->p[idx] / field->rho[idx]);

            // Optimized velocity magnitude calculation - avoid sqrt when possible
            double vel_mag_sq = (u_speed * u_speed) + (v_speed * v_speed);
            double vel_mag = (vel_mag_sq > VELOCITY_EPSILON) ? sqrt(vel_mag_sq) : 0.0;
            double local_speed = vel_mag + sound_speed;
            max_speed = max_double(max_speed, local_speed);
        }
    }

    // Prevent division by zero and ensure reasonable time step
    if (max_speed < SPEED_EPSILON) {
        max_speed = 1.0;  // Use default if speeds are too small
    }

    // Include z-direction in CFL if 3D
    double dmin = min_double(dx_min, dy_min);
    if (grid->nz > 1 && grid->dz) {
        double dz_min = grid->dz[0];
        for (size_t k = 0; k < grid->nz - 1; k++) {
            dz_min = min_double(dz_min, grid->dz[k]);
        }
        dmin = min_double(dmin, dz_min);
    }

    // Compute time step based on CFL condition with safety factor
    double dt_cfl = params->cfl * dmin / max_speed;

    // Limit time step to reasonable bounds
    double dt_max = DT_MAX_LIMIT;  // Maximum allowed time step
    double dt_min = DT_MIN_LIMIT;  // Minimum allowed time step

    params->dt = max_double(dt_min, min_double(dt_max, dt_cfl));
}

void apply_boundary_conditions(flow_field* field, const grid* grid) {
    (void)grid;
    // Apply periodic boundary conditions in x-direction
    for (size_t j = 0; j < field->ny; j++) {
        field->u[IDX_2D(0, j, field->nx)] = field->u[IDX_2D(field->nx - 2, j, field->nx)];
        field->v[IDX_2D(0, j, field->nx)] = field->v[IDX_2D(field->nx - 2, j, field->nx)];
        field->p[IDX_2D(0, j, field->nx)] = field->p[IDX_2D(field->nx - 2, j, field->nx)];
        field->rho[IDX_2D(0, j, field->nx)] = field->rho[IDX_2D(field->nx - 2, j, field->nx)];
        field->T[IDX_2D(0, j, field->nx)] = field->T[IDX_2D(field->nx - 2, j, field->nx)];

        field->u[IDX_2D(field->nx - 1, j, field->nx)] = field->u[IDX_2D(1, j, field->nx)];
        field->v[IDX_2D(field->nx - 1, j, field->nx)] = field->v[IDX_2D(1, j, field->nx)];
        field->p[IDX_2D(field->nx - 1, j, field->nx)] = field->p[IDX_2D(1, j, field->nx)];
        field->rho[IDX_2D(field->nx - 1, j, field->nx)] = field->rho[IDX_2D(1, j, field->nx)];
        field->T[IDX_2D(field->nx - 1, j, field->nx)] = field->T[IDX_2D(1, j, field->nx)];
    }

    // Apply periodic boundary conditions in y-direction (instead of walls)
    for (size_t i = 0; i < field->nx; i++) {
        // Bottom boundary = top interior
        field->u[i] = field->u[IDX_2D(i, field->ny - 2, field->nx)];
        field->v[i] = field->v[IDX_2D(i, field->ny - 2, field->nx)];
        field->p[i] = field->p[IDX_2D(i, field->ny - 2, field->nx)];
        field->rho[i] = field->rho[IDX_2D(i, field->ny - 2, field->nx)];
        field->T[i] = field->T[IDX_2D(i, field->ny - 2, field->nx)];

        // Top boundary = bottom interior
        size_t top_idx = IDX_2D(i, field->ny - 1, field->nx);
        field->u[top_idx] = field->u[field->nx + i];
        field->v[top_idx] = field->v[field->nx + i];
        field->p[top_idx] = field->p[field->nx + i];
        field->rho[top_idx] = field->rho[field->nx + i];
        field->T[top_idx] = field->T[field->nx + i];
    }
}

// Helper function to compute source terms consistently across all solvers
void compute_source_terms(double x, double y, int iter, double dt, const ns_solver_params_t* params,
                          double* source_u, double* source_v) {
    // Use custom source function if provided
    if (params->source_func) {
        double t = iter * dt;
        params->source_func(x, y, t, params->source_context, source_u, source_v);
        return;
    }

    // Default source term implementation
    *source_u =
        params->source_amplitude_u * sin(M_PI * y) * exp(-params->source_decay_rate * iter * dt);
    *source_v = params->source_amplitude_v * sin(2.0 * M_PI * x) *
                exp(-params->source_decay_rate * iter * dt);
}

// Internal explicit Euler implementation
// This is called by the solver registry - not part of public API
cfd_status_t explicit_euler_impl(flow_field* field, const grid* grid, const ns_solver_params_t* params) {
    // Check for minimum grid size - prevent crashes on small grids
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;  // Skip solver for grids too small for finite differences
    }

    // Allocate temporary arrays for the solution update
    double* u_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* v_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* p_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* rho_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* t_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));

    if (!u_new || !v_new || !p_new || !rho_new || !t_new) {
        cfd_free(u_new);
        cfd_free(v_new);
        cfd_free(p_new);
        cfd_free(rho_new);
        cfd_free(t_new);
        return CFD_ERROR_NOMEM;
    }

    // Initialize with current values to prevent uninitialized memory
    memcpy(u_new, field->u, field->nx * field->ny * sizeof(double));
    memcpy(v_new, field->v, field->nx * field->ny * sizeof(double));
    memcpy(p_new, field->p, field->nx * field->ny * sizeof(double));
    memcpy(rho_new, field->rho, field->nx * field->ny * sizeof(double));
    memcpy(t_new, field->T, field->nx * field->ny * sizeof(double));

    // Use conservative time step to prevent instabilities
    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

    // Main time-stepping loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Update solution using explicit Euler method
        for (size_t j = 1; j < field->ny - 1; j++) {
            for (size_t i = 1; i < field->nx - 1; i++) {
                size_t idx = IDX_2D(i, j, field->nx);

                // Compute spatial derivatives
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
                double du_dy =
                    (field->u[idx + field->nx] - field->u[idx - field->nx]) / (2.0 * grid->dy[j]);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
                double dv_dy =
                    (field->v[idx + field->nx] - field->v[idx - field->nx]) / (2.0 * grid->dy[j]);

                // Pressure gradients
                double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
                double dp_dy =
                    (field->p[idx + field->nx] - field->p[idx - field->nx]) / (2.0 * grid->dy[j]);

                // Second derivatives for viscous terms
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) /
                                 (grid->dx[i] * grid->dx[i]);
                double d2u_dy2 =
                    (field->u[idx + field->nx] - 2.0 * field->u[idx] + field->u[idx - field->nx]) /
                    (grid->dy[j] * grid->dy[j]);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) /
                                 (grid->dx[i] * grid->dx[i]);
                double d2v_dy2 =
                    (field->v[idx + field->nx] - 2.0 * field->v[idx] + field->v[idx - field->nx]) /
                    (grid->dy[j] * grid->dy[j]);

                // Safety checks to prevent division by zero
                if (field->rho[idx] <= 1e-10) {
                    continue;
                }
                if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) {
                    continue;
                }

                // Viscosity coefficient (kinematic viscosity = dynamic viscosity / density) with
                // safety
                double nu = params->mu / fmax(field->rho[idx], 1e-10);
                nu = fmin(nu, 1.0);  // Limit maximum viscosity

                // Limit derivatives to prevent instabilities
                du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
                du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
                dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
                dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
                dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
                dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
                d2u_dx2 =
                    fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
                d2u_dy2 =
                    fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
                d2v_dx2 =
                    fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
                d2v_dy2 =
                    fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));

                // Source terms to maintain flow (prevents decay)
                double x = grid->x[i];
                double y = grid->y[j];
                double source_u, source_v;
                compute_source_terms(x, y, iter, conservative_dt, params, &source_u, &source_v);

                // Conservative velocity updates with limited changes
                double du =
                    conservative_dt * (-field->u[idx] * du_dx - field->v[idx] * du_dy  // Convection
                                       - dp_dx / field->rho[idx]   // Pressure gradient
                                       + nu * (d2u_dx2 + d2u_dy2)  // Viscous diffusion
                                       + source_u                  // Source term
                                      );

                double dv =
                    conservative_dt * (-field->u[idx] * dv_dx - field->v[idx] * dv_dy  // Convection
                                       - dp_dy / field->rho[idx]   // Pressure gradient
                                       + nu * (d2v_dx2 + d2v_dy2)  // Viscous diffusion
                                       + source_v                  // Source term
                                      );

                // Limit velocity changes
                du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
                dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));

                u_new[idx] = field->u[idx] + du;
                v_new[idx] = field->v[idx] + dv;

                // Limit velocity magnitudes
                u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_new[idx]));
                v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_new[idx]));

                // Simplified stable pressure update
                double divergence = du_dx + dv_dy;
                divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));

                double dp =
                    -PRESSURE_UPDATE_FACTOR * conservative_dt * field->rho[idx] * divergence;
                dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));  // Limit pressure changes
                p_new[idx] = field->p[idx] + dp;

                // Keep density and temperature constant for this simplified model
                rho_new[idx] = field->rho[idx];
                t_new[idx] = field->T[idx];
            }
        }

        // Copy new solution to old solution
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));
        memcpy(field->rho, rho_new, field->nx * field->ny * sizeof(double));
        memcpy(field->T, t_new, field->nx * field->ny * sizeof(double));

        // Copy caller-set boundary values to preserve them (e.g., lid-driven cavity BCs)
        // We store boundaries before apply_boundary_conditions overwrites them
        size_t nx = field->nx;
        size_t ny = field->ny;

        // Store boundary values from u_new/v_new (which have the correct caller BCs)
        for (size_t i = 0; i < nx; i++) {
            // Bottom and top boundaries
            u_new[i] = field->u[i];
            v_new[i] = field->v[i];
            u_new[IDX_2D(i, ny - 1, nx)] = field->u[IDX_2D(i, ny - 1, nx)];
            v_new[IDX_2D(i, ny - 1, nx)] = field->v[IDX_2D(i, ny - 1, nx)];
        }
        for (size_t jj = 0; jj < ny; jj++) {
            // Left and right boundaries
            u_new[IDX_2D(0, jj, nx)] = field->u[IDX_2D(0, jj, nx)];
            v_new[IDX_2D(0, jj, nx)] = field->v[IDX_2D(0, jj, nx)];
            u_new[IDX_2D(nx - 1, jj, nx)] = field->u[IDX_2D(nx - 1, jj, nx)];
            v_new[IDX_2D(nx - 1, jj, nx)] = field->v[IDX_2D(nx - 1, jj, nx)];
        }

        // Apply boundary conditions (this applies periodic BCs)
        apply_boundary_conditions(field, grid);

        // Restore caller-set velocity boundary values
        for (size_t i = 0; i < nx; i++) {
            field->u[i] = u_new[i];
            field->v[i] = v_new[i];
            field->u[IDX_2D(i, ny - 1, nx)] = u_new[IDX_2D(i, ny - 1, nx)];
            field->v[IDX_2D(i, ny - 1, nx)] = v_new[IDX_2D(i, ny - 1, nx)];
        }
        for (size_t jj = 0; jj < ny; jj++) {
            field->u[IDX_2D(0, jj, nx)] = u_new[IDX_2D(0, jj, nx)];
            field->v[IDX_2D(0, jj, nx)] = v_new[IDX_2D(0, jj, nx)];
            field->u[IDX_2D(nx - 1, jj, nx)] = u_new[IDX_2D(nx - 1, jj, nx)];
            field->v[IDX_2D(nx - 1, jj, nx)] = v_new[IDX_2D(nx - 1, jj, nx)];
        }

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
    }

    // Free temporary arrays
    cfd_free(u_new);
    cfd_free(v_new);
    cfd_free(p_new);
    cfd_free(rho_new);
    cfd_free(t_new);

    return CFD_SUCCESS;
}
