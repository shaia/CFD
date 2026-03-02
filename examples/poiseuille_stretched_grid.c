/**
 * Poiseuille Flow with Stretched Grids
 *
 * Validates the pressure-driven channel flow (Poiseuille flow) against the
 * analytical parabolic velocity profile, comparing uniform vs stretched grids.
 * Stretched grids cluster points near the walls where velocity gradients are
 * steepest, improving accuracy without increasing total grid points.
 *
 * This example demonstrates:
 *   - grid_initialize_stretched() with different beta values
 *   - Inlet/outlet boundary conditions (parabolic inlet, zero-gradient outlet)
 *   - No-slip walls and Neumann pressure BCs
 *   - Derived field statistics
 *   - Analytical solution comparison with L2 error
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/**
 * Analytical Poiseuille flow: u(y) = U_max * 4 * y/H * (1 - y/H)
 * where H is channel height and U_max is centerline velocity.
 */
static double analytical_u(double y, double H, double U_max) {
    return U_max * 4.0 * (y / H) * (1.0 - y / H);
}

static void apply_channel_bcs(flow_field* field, size_t nx, size_t ny,
                               double U_max) {
    /* Parabolic inlet on left boundary */
    bc_inlet_config_t inlet = bc_inlet_config_parabolic(U_max);
    bc_inlet_set_edge(&inlet, BC_EDGE_LEFT);
    bc_apply_inlet(field->u, field->v, nx, ny, &inlet);

    /* Zero-gradient outlet on right boundary */
    bc_outlet_config_t outlet = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&outlet, BC_EDGE_RIGHT);
    bc_apply_outlet_velocity(field->u, field->v, nx, ny, &outlet);

    /* No-slip walls on top and bottom */
    /* Top wall */
    for (size_t i = 0; i < nx; i++) {
        field->u[(ny - 1) * nx + i] = 0.0;
        field->v[(ny - 1) * nx + i] = 0.0;
    }
    /* Bottom wall */
    for (size_t i = 0; i < nx; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
    }

    /* Neumann BC for pressure */
    bc_apply_neumann(field->p, nx, ny);
}

typedef struct {
    double l2_error;
    double min_dy;
    double max_dy;
    double dy_ratio;
} case_result_t;

static case_result_t run_case(size_t nx, size_t ny, double beta,
                               double nu, double dt, int steps,
                               double L, double H, double U_max) {
    case_result_t result = {0};

    grid* g = grid_create(nx, ny, 1, 0.0, L, 0.0, H, 0.0, 0.0);
    if (!g) { result.l2_error = -1.0; return result; }

    if (beta <= 0.0) {
        grid_initialize_uniform(g);
    } else {
        grid_initialize_stretched(g, beta);
    }

    /* Compute grid spacing statistics */
    result.min_dy = g->dy[0];
    result.max_dy = g->dy[0];
    for (size_t j = 0; j < ny - 1; j++) {
        if (g->dy[j] < result.min_dy) result.min_dy = g->dy[j];
        if (g->dy[j] > result.max_dy) result.max_dy = g->dy[j];
    }
    result.dy_ratio = result.max_dy / result.min_dy;

    /* Adapt dt for stability: CFL requires dt < min_dy^2 / (4*nu) */
    double dt_stable = 0.25 * result.min_dy * result.min_dy / nu;
    if (dt < dt_stable) dt_stable = dt;
    dt = dt_stable;

    flow_field* field = flow_field_create(nx, ny, 1);
    if (!field) { grid_destroy(g); result.l2_error = -1.0; return result; }
    initialize_flow_field(field, g);

    /* Initialize with approximate Poiseuille profile */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field->u[j * nx + i] = analytical_u(g->y[j], H, U_max);
        }
    }

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        flow_field_destroy(field);
        grid_destroy(g);
        result.l2_error = -1.0;
        return result;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver) {
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        result.l2_error = -1.0;
        return result;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = nu;
    params.dt = dt;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    cfd_status_t status = solver_init(solver, g, &params);
    if (status != CFD_SUCCESS) {
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        result.l2_error = -1.0;
        return result;
    }

    /* Time-stepping */
    ns_solver_stats_t stats_out;
    for (int step = 0; step < steps; step++) {
        apply_channel_bcs(field, nx, ny, U_max);
        stats_out = ns_solver_stats_default();
        solver_step(solver, field, g, &params, &stats_out);
    }

    /* Compute L2 error at outlet (last interior column) */
    double sum_sq = 0.0;
    size_t count = 0;
    size_t i_out = nx - 2;
    for (size_t j = 1; j < ny - 1; j++) {
        double u_num = field->u[j * nx + i_out];
        double u_ana = analytical_u(g->y[j], H, U_max);
        double diff = u_num - u_ana;
        sum_sq += diff * diff;
        count++;
    }
    result.l2_error = sqrt(sum_sq / (double)count);

    /* Print field statistics */
    derived_fields* df = derived_fields_create(nx, ny, 1);
    if (df) {
        derived_fields_compute_statistics(df, field);
        printf("    Field stats: u=[%.3f, %.3f], v=[%.3e, %.3e]\n",
               df->u_stats.min_val, df->u_stats.max_val,
               df->v_stats.min_val, df->v_stats.max_val);
        derived_fields_destroy(df);
    }

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

int main(void) {
    printf("Poiseuille Flow with Stretched Grids\n");
    printf("====================================\n\n");

    size_t nx = 40, ny = 32;
    double L = 4.0, H = 1.0;
    double U_max = 1.0;
    double Re = 100.0;
    double nu = U_max * H / Re;
    double dt = 0.0005;
    int steps = 500;

    printf("Channel: %.0f x %.0f, Grid: %zu x %zu\n", L, H, nx, ny);
    printf("Re = %.0f, nu = %.6f, U_max = %.1f\n", Re, nu, U_max);
    printf("Steps: %d, dt = %.4f\n\n", steps, dt);

    double betas[] = {0.0, 1.5, 2.0};
    const char* labels[] = {"Uniform (beta=0)", "Mild (beta=1.5)", "Strong (beta=2.0)"};
    case_result_t results[3];

    for (int c = 0; c < 3; c++) {
        printf("  Case %d: %s\n", c + 1, labels[c]);
        results[c] = run_case(nx, ny, betas[c], nu, dt, steps, L, H, U_max);
        if (results[c].l2_error < 0) {
            printf("    FAILED\n");
            continue;
        }
        double eff_dt = dt;
        double dt_stable = 0.25 * results[c].min_dy * results[c].min_dy / nu;
        if (dt_stable < eff_dt) eff_dt = dt_stable;
        printf("    dy range: %.5f - %.5f (ratio: %.1f), eff_dt=%.2e\n",
               results[c].min_dy, results[c].max_dy, results[c].dy_ratio, eff_dt);
        printf("    L2 error vs analytical: %.3e\n\n", results[c].l2_error);
    }

    /* Summary table */
    printf("--- Summary ---\n");
    printf("  %-22s  %10s  %10s  %8s  %10s\n",
           "Grid Type", "min(dy)", "max(dy)", "Ratio", "L2 Error");
    for (int c = 0; c < 3; c++) {
        if (results[c].l2_error >= 0) {
            printf("  %-22s  %10.5f  %10.5f  %8.1f  %10.3e\n",
                   labels[c], results[c].min_dy, results[c].max_dy,
                   results[c].dy_ratio, results[c].l2_error);
        }
    }

    printf("\nNote: Stretched grids cluster points near walls where gradients are\n");
    printf("steepest. The current solver uses uniform-grid stencils, so accuracy\n");
    printf("gains from stretching require non-uniform stencil support.\n");
    return 0;
}
