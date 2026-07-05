/**
 * Turbulent Channel Flow Example (RANS + wall functions)
 *
 * Fully-developed turbulent channel flow at a prescribed friction Reynolds
 * number Re_tau, driven by a constant streamwise body force f_x = u_tau^2/delta
 * (so the exact steady friction velocity is u_tau = 1 in these units):
 *   - periodic in x, log-law wall-function walls at y = 0 and y = 2*delta
 *   - RANS closure: standard k-epsilon or Spalart-Allmaras
 *
 * This example demonstrates:
 *   - Enabling a turbulence model via params.turb_model
 *   - Configuring wall-function walls via params.turb_bc (BC_TYPE_NOSLIP faces)
 *   - Initializing the turbulence fields with turbulence_init_uniform
 *   - Comparing the computed u+ profile against the log law
 *
 * Uses the direct solver interface (registry + solver_step) with a fixed time
 * step; run_simulation_step is not used because it overrides params.dt.
 *
 * Usage: turbulent_channel [model]
 *   model = "ke" (k-epsilon, default) or "sa" (Spalart-Allmaras)
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/grid.h"
#include "cfd/io/vtk_output.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KAPPA 0.41
#define LOG_B 5.2

/* Periodic in x, no-slip walls in y (boundary values set directly; the
 * projection solver preserves caller-set boundary values). */
static void apply_channel_bc(flow_field* field) {
    size_t nx = field->nx, ny = field->ny;
    for (size_t j = 0; j < ny; j++) {
        field->u[j * nx] = field->u[j * nx + (nx - 2)];
        field->v[j * nx] = field->v[j * nx + (nx - 2)];
        field->u[j * nx + (nx - 1)] = field->u[j * nx + 1];
        field->v[j * nx + (nx - 1)] = field->v[j * nx + 1];
    }
    for (size_t i = 0; i < nx; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->u[(ny - 1) * nx + i] = 0.0;
        field->v[(ny - 1) * nx + i] = 0.0;
    }
}

/* Constant streamwise body force f_x = u_tau^2/delta = 1 */
static void channel_body_force(double x, double y, double z, double t, void* ctx,
                               double* su, double* sv, double* sw) {
    (void)x; (void)y; (void)z; (void)t; (void)ctx;
    *su = 1.0;
    *sv = 0.0;
    *sw = 0.0;
}

static double compute_ke(const flow_field* field) {
    double ke = 0.0;
    size_t total = field->nx * field->ny;
    for (size_t n = 0; n < total; n++) {
        ke += field->u[n] * field->u[n] + field->v[n] * field->v[n];
    }
    return 0.5 * ke;
}

int main(int argc, char* argv[]) {
    /* Parameters (see tests/validation/test_turbulent_channel.c) */
    size_t nx = 16, ny = 21;
    double Lx = 4.0, H = 2.0, delta = 1.0;
    double Re_tau = 395.0;
    double nu = 1.0 / Re_tau;
    double dt = 0.002;
    int max_steps = 40000;
    int min_steps = 5000;
    int print_interval = 2000;
    double steady_tol = 1e-6;

    turbulence_model_t model = TURB_MODEL_K_EPSILON;
    const char* model_name = "k-epsilon";
    if (argc > 1 && strcmp(argv[1], "sa") == 0) {
        model = TURB_MODEL_SPALART_ALLMARAS;
        model_name = "Spalart-Allmaras";
    }

    printf("Turbulent Channel Flow (RANS + wall functions)\n");
    printf("===============================================\n");
    printf("Grid:       %zu x %zu\n", nx, ny);
    printf("Re_tau:     %.1f\n", Re_tau);
    printf("Model:      %s\n", model_name);
    printf("First-node y+: %.1f (wall-function target: 30-100)\n\n",
           (H / (double)(ny - 1)) / nu);

    cfd_init();

    grid* g = grid_create(nx, ny, 1, 0.0, Lx, 0.0, H, 0.0, 0.0);
    flow_field* field = flow_field_create(nx, ny, 1);
    if (!g || !field) {
        fprintf(stderr, "Failed to create grid/field\n");
        return 1;
    }
    grid_initialize_uniform(g);

    /* Initial condition: plug profile near the expected bulk velocity */
    double u_bulk0 = 15.0;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double y = g->y[j];
            int at_wall = (j == 0 || j == ny - 1);
            field->u[idx] = at_wall ? 0.0
                : u_bulk0 * (1.0 - pow(fabs(y - delta), 8.0));
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.max_iter = 1;
    params.mu = nu; /* rho = 1: dynamic == kinematic viscosity */
    params.source_func = channel_body_force;
    params.turb_model = model;
    params.turb_bc.bottom = BC_TYPE_NOSLIP; /* wall-function wall */
    params.turb_bc.top = BC_TYPE_NOSLIP;    /* left/right stay PERIODIC */

    /* Turbulence initial condition: ~5% intensity of the bulk velocity */
    double k0 = 1.5 * pow(0.05 * u_bulk0, 2.0);
    double eps0 = pow(0.09, 0.75) * pow(k0, 1.5) / (0.07 * delta);
    if (turbulence_init_uniform(field, &params, k0, eps0, 3.0 * nu) != CFD_SUCCESS) {
        fprintf(stderr, "Failed to initialize turbulence fields\n");
        return 1;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!slv || solver_init(slv, g, &params) != CFD_SUCCESS) {
        fprintf(stderr, "Failed to create projection solver\n");
        return 1;
    }

    /* March to steady state (kinetic-energy residual) */
    printf("Running simulation...\n");
    double prev_ke = compute_ke(field);
    int step = 0;
    for (step = 0; step < max_steps; step++) {
        apply_channel_bc(field);

        ns_solver_stats_t stats;
        cfd_status_t status = solver_step(slv, field, g, &params, &stats);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d (status=%d)\n", step, status);
            break;
        }

        double ke = compute_ke(field);
        double residual = fabs(ke - prev_ke) / (prev_ke + 1e-10);
        prev_ke = ke;

        if (step % print_interval == 0) {
            printf("  Step %6d: KE residual = %.2e, time = %.2f s\n",
                   step, residual, step * dt);
        }
        if (residual < steady_tol && step > min_steps) {
            printf("  Converged at step %d (KE residual %.2e)\n", step, residual);
            break;
        }
    }
    apply_channel_bc(field);

    /* Report u+ vs the log law along the bottom half of the channel */
    size_t i_mid = nx / 2;
    double y_p = g->y[1] - g->y[0];
    double u_p = fabs(field->u[nx + i_mid]);
    double u_tau = turbulence_wall_u_tau(u_p, y_p, nu);

    printf("\nRecovered u_tau = %.4f (exact force balance: 1.0000)\n\n", u_tau);
    printf("  %8s  %8s  %8s  %8s\n", "y+", "u+", "log-law", "err");
    for (size_t j = 1; j <= (ny - 1) / 2; j++) {
        double yplus = g->y[j] / nu; /* u_tau = 1 in these units */
        double u_plus = field->u[j * nx + i_mid] / u_tau;
        double u_log = log(yplus) / KAPPA + LOG_B;
        printf("  %8.1f  %8.2f  %8.2f  %7.1f%%\n",
               yplus, u_plus, u_log, 100.0 * fabs(u_plus - u_log) / u_log);
    }

    /* Write a final VTK snapshot (includes k, eps, nu_tilde, nu_t scalars) */
    write_vtk_flow_field("turbulent_channel.vtk", field, nx, ny, 1,
                         0.0, Lx, 0.0, H, 0.0, 0.0);
    printf("\nWrote turbulent_channel.vtk (open in ParaView to inspect nu_t/k).\n");

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_finalize();
    return 0;
}
