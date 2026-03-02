/**
 * Taylor-Green Vortex with Periodic Boundary Conditions
 *
 * Simulates the 2D Taylor-Green vortex, an exact decaying solution of
 * the Navier-Stokes equations on a periodic domain [0, 2pi]^2.
 *
 * The example tracks velocity decay over time and compares against the
 * analytical solution, then runs the same problem with different solver
 * types and grid resolutions.
 *
 * This example demonstrates:
 *   - Periodic boundary conditions (bc_apply_periodic macro)
 *   - Three NS solver types: projection, rk2, explicit_euler
 *   - Analytical solution comparison (velocity decay)
 *   - Grid refinement showing error reduction with resolution
 */

#include "cfd/api/simulation_api.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TG_U0  1.0    /* Initial velocity amplitude */
#define TG_NU  0.01   /* Kinematic viscosity */

static void init_taylor_green(flow_field* field, const grid* g) {
    size_t nx = field->nx, ny = field->ny;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = g->x[i];
            double y = g->y[j];
            size_t idx = IDX_2D(i, j, nx);
            field->u[idx] = TG_U0 * cos(x) * sin(y);
            field->v[idx] = -TG_U0 * sin(x) * cos(y);
            field->p[idx] = -(TG_U0 * TG_U0 / 4.0) * (cos(2.0 * x) + cos(2.0 * y));
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

static void apply_periodic_bcs(flow_field* field) {
    size_t nx = field->nx, ny = field->ny;
    bc_apply_periodic(field->u, nx, ny);
    bc_apply_periodic(field->v, nx, ny);
    bc_apply_periodic(field->p, nx, ny);
}

static double compute_max_u(const flow_field* field) {
    size_t nx = field->nx, ny = field->ny;
    double max_u = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double u = fabs(field->u[IDX_2D(i, j, nx)]);
            if (u > max_u) max_u = u;
        }
    }
    return max_u;
}

static double compute_kinetic_energy(const flow_field* field) {
    size_t nx = field->nx, ny = field->ny;
    double ke = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double u = field->u[idx];
            double v = field->v[idx];
            ke += 0.5 * (u * u + v * v);
            count++;
        }
    }
    return ke / (double)count;
}

typedef struct {
    double l2_error;
    double final_max_u;
    double final_ke;
} tg_result_t;

/**
 * Run a Taylor-Green vortex case and return metrics.
 */
static tg_result_t run_case(size_t n, const char* solver_type,
                             double dt, double t_final) {
    tg_result_t res = {-1.0, 0.0, 0.0};
    double domain = 2.0 * M_PI;

    simulation_data* sim = init_simulation_with_solver(
        n, n, 1,
        0.0, domain, 0.0, domain, 0.0, 0.0,
        solver_type);

    if (!sim) return res;

    sim->params.dt = dt;
    sim->params.mu = TG_NU;
    sim->params.source_amplitude_u = 0.0;
    sim->params.source_amplitude_v = 0.0;

    init_taylor_green(sim->field, sim->grid);
    apply_periodic_bcs(sim->field);

    /* Time-step to t_final */
    double t = 0.0;
    while (t < t_final - 1e-12) {
        double step_dt = dt;
        if (t + step_dt > t_final) step_dt = t_final - t;
        sim->params.dt = step_dt;

        apply_periodic_bcs(sim->field);
        ns_solver_stats_t stats = ns_solver_stats_default();
        cfd_status_t status = solver_step(sim->solver, sim->field, sim->grid, &sim->params, &stats);
        if (status != CFD_SUCCESS) {
            free_simulation(sim);
            return res;  /* l2_error stays -1.0 → reported as FAILED */
        }
        t += step_dt;
    }

    /* Compute L2 error against analytical solution */
    size_t nx = n, ny = n;
    double decay = exp(-2.0 * TG_NU * t_final);
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double x = sim->grid->x[i];
            double y = sim->grid->y[j];
            size_t idx = IDX_2D(i, j, nx);
            double u_exact = TG_U0 * cos(x) * sin(y) * decay;
            double v_exact = -TG_U0 * sin(x) * cos(y) * decay;
            double du = sim->field->u[idx] - u_exact;
            double dv = sim->field->v[idx] - v_exact;
            sum_sq += (du * du) + (dv * dv);
            count += 2;
        }
    }
    res.l2_error = sqrt(sum_sq / (double)count);
    res.final_max_u = compute_max_u(sim->field);
    res.final_ke = compute_kinetic_energy(sim->field);

    free_simulation(sim);
    return res;
}

int main(void) {
    printf("Taylor-Green Vortex with Periodic BCs\n");
    printf("=====================================\n");
    printf("Domain: [0, 2pi]^2 | nu = %.4f | U0 = %.1f\n", TG_NU, TG_U0);
    printf("Analytical: u decays as exp(-2*nu*t)\n\n");

    /* === Part 1: Velocity Decay Tracking ===
     * Run with projection solver and print velocity/KE at intervals */
    size_t n = 32;
    double dt = 5e-4;
    double t_end = 0.5;
    int print_every = 100;

    printf("Part 1: Velocity Decay (Projection, %zux%zu, dt=%.0e)\n", n, n, dt);
    printf("  %-8s  %10s  %10s  %10s  %10s\n",
           "Time", "max|u|", "Analytical", "KE", "KE_exact");

    simulation_data* sim = init_simulation_with_solver(
        n, n, 1,
        0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI, 0.0, 0.0,
        NS_SOLVER_TYPE_PROJECTION);

    if (!sim) {
        printf("  Failed to create simulation\n");
        return 1;
    }

    sim->params.dt = dt;
    sim->params.mu = TG_NU;
    sim->params.source_amplitude_u = 0.0;
    sim->params.source_amplitude_v = 0.0;

    init_taylor_green(sim->field, sim->grid);
    apply_periodic_bcs(sim->field);

    int num_steps = (int)(t_end / dt);
    for (int step = 0; step <= num_steps; step++) {
        if (step % print_every == 0) {
            double t = step * dt;
            double decay = exp(-2.0 * TG_NU * t);
            double u_max = compute_max_u(sim->field);
            double ke = compute_kinetic_energy(sim->field);
            double ke_exact = 0.25 * TG_U0 * TG_U0 * decay * decay;
            printf("  t=%-6.3f  %10.6f  %10.6f  %10.6f  %10.6f\n",
                   t, u_max, TG_U0 * decay, ke, ke_exact);
        }

        if (step < num_steps) {
            apply_periodic_bcs(sim->field);
            ns_solver_stats_t stats = ns_solver_stats_default();
            cfd_status_t status = solver_step(sim->solver, sim->field, sim->grid,
                                              &sim->params, &stats);
            if (status != CFD_SUCCESS) {
                fprintf(stderr, "  Error: solver_step failed at step %d\n", step);
                free_simulation(sim);
                return 1;
            }
        }
    }
    free_simulation(sim);

    /* === Part 2: Solver Type Comparison ===
     * Run the same problem with three different solver types */
    printf("\nPart 2: Solver Comparison (%zux%zu, dt=%.0e, T=%.1f)\n",
           n, n, dt, t_end);
    printf("  %-20s  %12s  %10s\n", "Solver", "L2 Error", "max|u|");

    const char* solver_types[] = {
        NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_RK2,
        NS_SOLVER_TYPE_EXPLICIT_EULER
    };
    const char* solver_labels[] = {
        "Projection",
        "RK2 (Heun)",
        "Explicit Euler"
    };

    for (int s = 0; s < 3; s++) {
        tg_result_t res = run_case(n, solver_types[s], dt, t_end);
        if (res.l2_error < 0) {
            printf("  %-20s  %12s\n", solver_labels[s], "FAILED");
        } else {
            printf("  %-20s  %12.3e  %10.6f\n",
                   solver_labels[s], res.l2_error, res.final_max_u);
        }
    }

    /* === Part 3: Grid Refinement ===
     * Show that finer grids reduce error */
    printf("\nPart 3: Grid Refinement (Explicit Euler, dt=%.0e, T=%.1f)\n",
           dt, t_end);
    printf("  %-12s  %12s\n", "Resolution", "L2 Error");

    size_t resolutions[] = {16, 32, 64};
    for (int r = 0; r < 3; r++) {
        tg_result_t res = run_case(resolutions[r],
                                    NS_SOLVER_TYPE_EXPLICIT_EULER, dt, t_end);
        if (res.l2_error < 0) {
            printf("  %3zu x %-3zu     FAILED\n", resolutions[r], resolutions[r]);
        } else {
            printf("  %3zu x %-3zu     %.3e\n", resolutions[r], resolutions[r],
                   res.l2_error);
        }
    }

    printf("\nFiner grids produce smaller errors against the analytical solution.\n");
    return 0;
}
