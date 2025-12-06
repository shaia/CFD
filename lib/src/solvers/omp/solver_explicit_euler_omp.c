#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <omp.h>

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
void explicit_euler_omp_impl(FlowField* field, const Grid* grid, const SolverParams* params) {
    if (field->nx < 3 || field->ny < 3) {
        return;
    }

    // Allocate temporary arrays
    double* u_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* v_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));
    double* p_new = (double*)cfd_calloc(field->nx * field->ny, sizeof(double));

    if (!u_new || !v_new || !p_new) {
        cfd_free(u_new);
        cfd_free(v_new);
        cfd_free(p_new);
        return;
    }

    // Initialize with current values
    memcpy(u_new, field->u, field->nx * field->ny * sizeof(double));
    memcpy(v_new, field->v, field->nx * field->ny * sizeof(double));
    memcpy(p_new, field->p, field->nx * field->ny * sizeof(double));

    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

    for (int iter = 0; iter < params->max_iter; iter++) {
        // Parallelize the main loop
        int i, j;
        #pragma omp parallel for private(i) schedule(static)
        for (j = 1; j < (int)field->ny - 1; j++) {
            for (i = 1; i < (int)field->nx - 1; i++) {
                size_t idx = j * field->nx + i;

                // Derivatives
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
                double du_dy = (field->u[idx + field->nx] - field->u[idx - field->nx]) / (2.0 * grid->dy[j]);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
                double dv_dy = (field->v[idx + field->nx] - field->v[idx - field->nx]) / (2.0 * grid->dy[j]);

                double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
                double dp_dy = (field->p[idx + field->nx] - field->p[idx - field->nx]) / (2.0 * grid->dy[j]);

                double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) / (grid->dx[i] * grid->dx[i]);
                double d2u_dy2 = (field->u[idx + field->nx] - 2.0 * field->u[idx] + field->u[idx - field->nx]) / (grid->dy[j] * grid->dy[j]);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) / (grid->dx[i] * grid->dx[i]);
                double d2v_dy2 = (field->v[idx + field->nx] - 2.0 * field->v[idx] + field->v[idx - field->nx]) / (grid->dy[j] * grid->dy[j]);

                if (field->rho[idx] <= 1e-10) continue;

                double nu = params->mu / fmax(field->rho[idx], 1e-10);
                nu = fmin(nu, 1.0);

                // Source terms
                double source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]) * exp(-params->source_decay_rate * iter * conservative_dt);
                double source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]) * exp(-params->source_decay_rate * iter * conservative_dt);

                // Update u
                double du = conservative_dt * (
                    -field->u[idx] * du_dx - field->v[idx] * du_dy
                    - dp_dx / field->rho[idx]
                    + nu * (d2u_dx2 + d2u_dy2)
                    + source_u
                );

                // Update v
                double dv = conservative_dt * (
                    -field->u[idx] * dv_dx - field->v[idx] * dv_dy
                    - dp_dy / field->rho[idx]
                    + nu * (d2v_dx2 + d2v_dy2)
                    + source_v
                );

                du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
                dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));

                u_new[idx] = field->u[idx] + du;
                v_new[idx] = field->v[idx] + dv;
                
                // Limit velocity
                u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_new[idx]));
                v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_new[idx]));

                // Pressure update
                double divergence = du_dx + dv_dy;
                divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
                
                double dp = -PRESSURE_UPDATE_FACTOR * conservative_dt * field->rho[idx] * divergence;
                dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));
                p_new[idx] = field->p[idx] + dp;
            }
        }

        // Copy back (implicit barrier at end of parallel for)
        memcpy(field->u, u_new, field->nx * field->ny * sizeof(double));
        memcpy(field->v, v_new, field->nx * field->ny * sizeof(double));
        memcpy(field->p, p_new, field->nx * field->ny * sizeof(double));

        // Boundary conditions - applied sequentially in this implementation (O(N) vs O(N^2))
        apply_boundary_conditions(field, grid);

        // Check for NaN/Inf values
        int has_nan = 0;
        int k;
        int limit = (int)(field->nx * field->ny);
        #pragma omp parallel for reduction(|:has_nan)
        for (k = 0; k < limit; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                has_nan = 1;
            }
        }

        if (has_nan) {
            printf("Warning: NaN/Inf detected in iteration %d, stopping solver\n", iter);
            break;
        }
    }

    cfd_free(u_new);
    cfd_free(v_new);
    cfd_free(p_new);
}
