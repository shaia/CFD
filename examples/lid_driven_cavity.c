/**
 * Lid-Driven Cavity Example (Simulation API)
 *
 * Classic CFD benchmark using the high-level simulation API.
 * A square cavity with:
 *   - Top wall (lid) moving at constant velocity u=1
 *   - All other walls stationary (no-slip)
 *
 * This example demonstrates:
 *   - Using the simulation API with the projection method solver
 *   - Setting Dirichlet BCs for the lid-driven cavity problem
 *   - Automatic VTK output via the output registry
 *   - Configuring Reynolds number from the command line
 *
 * Usage: lid_driven_cavity [Re]
 *   Re = Reynolds number (default: 100)
 */

#include "cfd/api/simulation_api.h"
#include "cfd/boundary/boundary_conditions.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void apply_cavity_bc(flow_field* field, double lid_velocity) {
    bc_dirichlet_values_t u_bc = {
        .left = 0.0,
        .right = 0.0,
        .top = lid_velocity,
        .bottom = 0.0
    };
    bc_dirichlet_values_t v_bc = {
        .left = 0.0,
        .right = 0.0,
        .top = 0.0,
        .bottom = 0.0
    };

    bc_apply_dirichlet_velocity(field->u, field->v,
                                field->nx, field->ny, &u_bc, &v_bc);
    bc_apply_neumann(field->p, field->nx, field->ny);
}

int main(int argc, char* argv[]) {
    /* Parameters */
    size_t nx = 64, ny = 64;
    double L = 1.0;
    double U = 1.0;              /* Lid velocity */
    double Re = 100.0;           /* Reynolds number */
    int max_steps = 5000;
    int print_interval = 500;

    /* Parse command-line Re */
    if (argc > 1) {
        double arg = atof(argv[1]);
        if (arg > 0) Re = arg;
    }

    double nu = U * L / Re;

    printf("Lid-Driven Cavity (Simulation API)\n");
    printf("===================================\n");
    printf("Grid:       %zu x %zu\n", nx, ny);
    printf("Re:         %.1f\n", Re);
    printf("Viscosity:  %.6f\n", nu);
    printf("Solver:     projection\n\n");

    /* Create simulation with projection method */
    simulation_data* sim = init_simulation_with_solver(
        nx, ny, 0.0, L, 0.0, L, NS_SOLVER_TYPE_PROJECTION);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation\n");
        return 1;
    }

    /* Configure solver parameters */
    sim->params.mu = nu;
    sim->params.dt = 0.001;
    sim->params.max_iter = 1;
    sim->params.source_amplitude_u = 0.0;  /* No artificial forcing for cavity flow */
    sim->params.source_amplitude_v = 0.0;

    /* Configure output */
    simulation_set_output_dir(sim, "output");
    simulation_set_run_prefix(sim, "lid_cavity");
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 500, "vmag");
    simulation_register_output(sim, OUTPUT_VELOCITY, 500, "velocity");

    /* Apply initial BCs */
    apply_cavity_bc(sim->field, U);

    /* Time-stepping loop */
    printf("Running simulation...\n");
    for (int step = 0; step <= max_steps; step++) {
        apply_cavity_bc(sim->field, U);

        cfd_status_t status = run_simulation_step(sim);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d (status=%d)\n",
                    step, status);
            break;
        }

        simulation_write_outputs(sim, step);

        if (step % print_interval == 0) {
            const ns_solver_stats_t* stats = simulation_get_stats(sim);
            printf("  Step %5d: max|u| = %.6f, time = %.3f s\n",
                   step, stats->max_velocity, sim->current_time);
        }
    }

    printf("\nSimulation completed!\n");
    printf("Tip: open VTK files in ParaView to visualize the cavity vortex.\n");

    free_simulation(sim);
    return 0;
}
