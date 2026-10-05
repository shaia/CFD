/**
 * NSSolver Selection Example
 *
 * This example demonstrates the new pluggable solver architecture:
 * - Listing available solvers
 * - Creating simulations with specific solver types
 * - Switching solvers at runtime
 * - Accessing solver statistics
 */

#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"


#include "cfd/io/vtk_output.h"
#include <stdio.h>


// Grid parameters
#define NX   100
#define NY   50
#define XMIN 0.0
#define XMAX 2.0
#define YMIN 0.0
#define YMAX 1.0

// Number of time steps for each solver
#define NUM_STEPS 50

void print_separator(void) {
    printf("\n========================================\n");
}

void print_solver_info(const struct NSSolver* solver) {
    if (!solver) {
        printf("  NSSolver: (legacy/default)\n");
        return;
    }

    printf("  Name: %s\n", solver->name);
    printf("  Description: %s\n", solver->description);
    printf("  Version: %s\n", solver->version);
    printf("  Capabilities: ");

    if (solver->capabilities & NS_SOLVER_CAP_INCOMPRESSIBLE) {
        printf("incompressible ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_COMPRESSIBLE) {
        printf("compressible ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_TRANSIENT) {
        printf("transient ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_STEADY_STATE) {
        printf("steady-state ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_SIMD) {
        printf("SIMD ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_PARALLEL) {
        printf("parallel ");
    }
    if (solver->capabilities & NS_SOLVER_CAP_GPU) {
        printf("GPU ");
    }

    printf("\n");
}

void print_stats(const ns_solver_stats_t* stats) {
    if (!stats) {
        return;
    }

    printf("  Iterations: %d\n", stats->iterations);
    printf("  Max velocity: %.4f\n", stats->max_velocity);
    printf("  Max pressure: %.4f\n", stats->max_pressure);
    printf("  Elapsed time: %.2f ms\n", stats->elapsed_time_ms);
}

// The library's message for a failure, falling back to the status name
static const char* failure_reason(cfd_status_t status) {
    const char* reason = cfd_get_last_error();
    return reason ? reason : cfd_get_error_string(status);
}

// Returns the number of solvers that could run here but failed
int run_solver_comparison(void) {
    print_separator();
    printf("SOLVER COMPARISON TEST\n");
    print_separator();

    int failures = 0;

    // List available solvers
    const char* solver_names[10];
    int num_solvers = simulation_list_solvers(solver_names, 10);

    printf("\nAvailable solvers (%d):\n", num_solvers);
    for (int i = 0; i < num_solvers; i++) {
        printf("  %d. %s\n", i + 1, solver_names[i]);
    }

    // The list can name solvers this build does not contain (the GPU ones without
    // CUDA), so check the registry before treating a missing solver as a failure.
    struct NSSolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    // Test each solver
    for (int i = 0; i < num_solvers; i++) {
        const char* solver_type = solver_names[i];

        print_separator();
        printf("Testing solver: %s\n", solver_type);
        print_separator();

        if (!cfd_registry_has(registry, solver_type)) {
            printf("  SKIPPED: not built into this library\n");
            continue;
        }

        // Create simulation with this solver. A backend this build or CPU lacks
        // refuses at init with CFD_ERROR_UNSUPPORTED; anything else is a failure.
        cfd_clear_error();
        simulation_data* sim =
            init_simulation_with_solver(NX, NY, 1, XMIN, XMAX, YMIN, YMAX, 0.0, 0.0, solver_type);
        if (!sim) {
            cfd_status_t status = cfd_get_last_status();
            if (status == CFD_ERROR_UNSUPPORTED) {
                printf("  SKIPPED: %s\n", failure_reason(status));
            } else {
                printf("  ERROR: Failed to create simulation: %s\n", failure_reason(status));
                failures++;
            }
            continue;
        }
        sim->params.dt = 0.005; /* the step this example has always run at */

        // Print solver info
        struct NSSolver* solver = simulation_get_solver(sim);
        print_solver_info(solver);

        // Set run prefix for this solver test
        simulation_set_run_prefix(sim, solver_type);

        // Configure output directory
        simulation_set_output_dir(sim, "../../artifacts");

        // Register output at end of simulation only
        simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, NUM_STEPS, "solver_test");

        // Run simulation
        printf("\nRunning %d steps...\n", NUM_STEPS);
        cfd_status_t status = CFD_SUCCESS;
        for (int step = 0; step <= NUM_STEPS; step++) {
            status = run_simulation_step(sim);
            if (status != CFD_SUCCESS) {
                printf("  ERROR: step %d failed: %s\n", step, failure_reason(status));
                failures++;
                break;
            }
            simulation_write_outputs(sim, step);
        }

        if (status == CFD_SUCCESS) {
            // Print final statistics
            printf("\nFinal statistics:\n");
            print_stats(simulation_get_stats(sim));
            printf("\nOutput written automatically\n");
        }

        // Cleanup
        free_simulation(sim);
    }

    cfd_registry_destroy(registry);
    return failures;
}

// Returns the number of failures
int run_dynamic_solver_switch(void) {
    print_separator();
    printf("DYNAMIC SOLVER SWITCHING\n");
    print_separator();

    // Start with default solver (explicit_euler)
    printf("\n1. Creating simulation with default solver...\n");
    simulation_data* sim = init_simulation(NX, NY, 1, XMIN, XMAX, YMIN, YMAX, 0.0, 0.0);
    if (!sim) {
        printf("  ERROR: Failed to create simulation: %s\n",
               failure_reason(cfd_get_last_status()));
        return 1;
    }
    sim->params.dt = 0.005; /* the step this example has always run at */
    simulation_set_run_prefix(sim, "dynamic_switch");
    simulation_set_output_dir(sim, "../../artifacts");

    // Register output every 10 steps
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");

    struct NSSolver* solver = simulation_get_solver(sim);
    print_solver_info(solver);

    int step_counter = 0;
    int failures = 0;

    // Run a few steps
    printf("\nRunning 10 steps with default solver...\n");
    for (int step = 0; step < 10; step++, step_counter++) {
        cfd_status_t status = run_simulation_step(sim);
        if (status != CFD_SUCCESS) {
            printf("  ERROR: step %d failed: %s\n", step, failure_reason(status));
            free_simulation(sim);
            return 1;
        }
        simulation_write_outputs(sim, step_counter);
    }

    // Switch to optimized solver. It needs AVX2 compiled in and supported by the CPU;
    // check that first, since a solver switched in whose backend is missing fails at
    // its first step rather than at the switch.
    printf("\n2. Switching to optimized solver...\n");
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        printf("  SKIPPED: the AVX2 backend is not available in this build or on this CPU\n");
    } else if (simulation_set_solver_by_name(sim, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED) == 0) {
        solver = simulation_get_solver(sim);
        print_solver_info(solver);

        // Run more steps
        printf("\nRunning 10 more steps with optimized solver...\n");
        cfd_status_t status = CFD_SUCCESS;
        for (int step = 0; step < 10; step++, step_counter++) {
            status = run_simulation_step(sim);
            if (status != CFD_SUCCESS) {
                printf("  ERROR: step %d failed: %s\n", step, failure_reason(status));
                failures++;
                break;
            }
            simulation_write_outputs(sim, step_counter);
        }

        if (status == CFD_SUCCESS) {
            printf("\nStatistics after optimized solver:\n");
            print_stats(simulation_get_stats(sim));
        }
    } else {
        printf("  ERROR: Failed to switch solver: %s\n", failure_reason(cfd_get_last_status()));
        failures++;
    }

    printf("\nOutput written automatically at regular intervals\n");

    free_simulation(sim);
    return failures;
}

// Returns the number of failures
int run_direct_solver_usage(void) {
    print_separator();
    printf("DIRECT SOLVER API USAGE\n");
    print_separator();

    struct NSSolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    // Create solver directly
    printf("\nCreating solver directly via cfd_solver_create()...\n");
    struct NSSolver* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    if (!solver) {
        printf("  ERROR: Failed to create solver: %s\n", failure_reason(cfd_get_last_status()));
        cfd_registry_destroy(registry);
        return 1;
    }

    print_solver_info(solver);

    // Create grid and flow field manually
    grid* grid = grid_create(NX, NY, 1, XMIN, XMAX, YMIN, YMAX, 0.0, 0.0);
    grid_initialize_uniform(grid);

    flow_field* field = flow_field_create(NX, NY, 1);
    initialize_flow_field(field, grid);

    ns_solver_params_t params = ns_solver_params_default();
    params.max_iter = 1;
    params.dt = 0.005;

    int failures = 0;

    // Initialize solver
    cfd_status_t status = solver_init(solver, grid, &params);
    printf("\nSolver init status: %d\n", status);
    if (status != CFD_SUCCESS) {
        printf("  ERROR: Solver init failed: %s\n", failure_reason(status));
        failures++;
        goto cleanup;
    }

    // Run steps directly
    printf("\nRunning 20 steps using direct solver API...\n");
    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < 20; step++) {
        status = solver_step(solver, field, grid, &params, &stats);
        if (status != CFD_SUCCESS) {
            printf("  ERROR: step %d failed: %s\n", step, failure_reason(status));
            failures++;
            goto cleanup;
        }

        if (step % 5 == 0) {
            printf("  Step %d: max_vel=%.4f, max_p=%.4f, time=%.2fms\n", step, stats.max_velocity,
                   stats.max_pressure, stats.elapsed_time_ms);
        }
    }

    // Write output using VTK functions directly. make_output_path() only composes
    // "{base}/output/<file>", so create the directory first: write_vtk_flow_field()
    // returns no status to check.
    char output_dir[512];
    make_output_path(output_dir, sizeof(output_dir), "");
    if (!ensure_directory_exists(output_dir)) {
        printf("  ERROR: could not create output directory %s\n", output_dir);
        failures++;
        goto cleanup;
    }
    char output_path[512];
    make_output_path(output_path, sizeof(output_path), "direct_api_test.vtk");
    write_vtk_flow_field(output_path, field, NX, NY, 1, XMIN, XMAX, YMIN, YMAX, 0.0, 0.0);
    printf("\nOutput written to: %s\n", output_path);

cleanup:
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(grid);
    cfd_registry_destroy(registry);
    return failures;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("CFD Framework - NSSolver Selection Example\n");
    printf("=========================================\n");


    // Run demonstrations
    int failures = run_solver_comparison();
    failures += run_dynamic_solver_switch();
    failures += run_direct_solver_usage();

    print_separator();
    if (failures > 0) {
        printf("FAILED: %d solver run(s) failed; see the messages above.\n", failures);
        print_separator();
        return 1;
    }
    printf("All tests completed!\n");
    print_separator();

    return 0;
}
