/**
 * Lid-Driven Cavity Example (Direct Solver API)
 *
 * Same physics as lid_driven_cavity.c but using the mid-level solver
 * registry API for full control over grid, flow field, solver, and output.
 *
 * This example demonstrates:
 *   - Creating grid and flow_field manually
 *   - Using the solver registry to create a projection solver
 *   - Applying Dirichlet BCs explicitly each time step
 *   - Monitoring solver statistics (max velocity, CFL, timing)
 *   - Manual VTK output with write_vtk_flow_field()
 *
 * Usage: lid_driven_cavity_direct [Re]
 *   Re = Reynolds number (default: 100)
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/io/vtk_output.h"
#include "cfd/solvers/navier_stokes_solver.h"

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
    int output_interval = 500;
    int print_interval = 500;

    /* Parse command-line Re */
    if (argc > 1) {
        double arg = atof(argv[1]);
        if (arg > 0) Re = arg;
    }

    double nu = U * L / Re;

    printf("Lid-Driven Cavity (Direct Solver API)\n");
    printf("======================================\n");
    printf("Grid:       %zu x %zu\n", nx, ny);
    printf("Re:         %.1f\n", Re);
    printf("Viscosity:  %.6f\n", nu);
    printf("Solver:     projection\n\n");

    /* Create grid and flow field */
    grid* g = grid_create(nx, ny, 0.0, L, 0.0, L);
    if (!g) {
        fprintf(stderr, "Failed to allocate grid\n");
        return 1;
    }
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    if (!field) {
        fprintf(stderr, "Failed to allocate flow field\n");
        grid_destroy(g);
        return 1;
    }
    initialize_flow_field(field, g);

    /* Create solver via registry */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver) {
        fprintf(stderr, "Failed to create projection solver\n");
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        return 1;
    }

    /* Configure solver parameters */
    ns_solver_params_t params = ns_solver_params_default();
    params.mu = nu;
    params.dt = 0.001;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;  /* No artificial forcing for cavity flow */
    params.source_amplitude_v = 0.0;

    cfd_status_t status = solver_init(solver, g, &params);
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Solver init failed (status=%d)\n", status);
        solver_destroy(solver);
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        return 1;
    }

    /* Configure output directory */
    cfd_set_output_base_dir("output");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "lid_cavity_direct", nx, ny);
    printf("Output directory: %s\n\n", run_dir);

    /* Apply initial BCs */
    apply_cavity_bc(field, U);

    /* Time-stepping loop */
    printf("Running simulation...\n");
    for (int step = 0; step <= max_steps; step++) {
        apply_cavity_bc(field, U);

        ns_solver_stats_t stats_out = ns_solver_stats_default();
        status = solver_step(solver, field, g, &params, &stats_out);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d (status=%d)\n",
                    step, status);
            break;
        }

        /* Write VTK output at intervals */
        if (step % output_interval == 0) {
            char filename[512];
#ifdef _WIN32
            snprintf(filename, sizeof(filename), "%s\\cavity_%05d.vtk",
                     run_dir, step);
#else
            snprintf(filename, sizeof(filename), "%s/cavity_%05d.vtk",
                     run_dir, step);
#endif
            write_vtk_flow_field(filename, field, nx, ny,
                                 g->xmin, g->xmax, g->ymin, g->ymax);
        }

        /* Print progress */
        if (step % print_interval == 0) {
            printf("  Step %5d: max|u| = %.6f, CFL = %.4f, "
                   "time = %.2f ms\n",
                   step, stats_out.max_velocity, stats_out.cfl_number,
                   stats_out.elapsed_time_ms);
        }
    }

    printf("\nSimulation completed!\n");
    printf("Output: %s/cavity_*.vtk\n", run_dir);
    printf("Tip: open in ParaView to visualize the cavity vortex.\n");

    /* Cleanup */
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);

    return 0;
}
