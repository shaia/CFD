/**
 * Solver Selection Example
 *
 * This example demonstrates the new pluggable solver architecture:
 * - Listing available solvers
 * - Creating simulations with specific solver types
 * - Switching solvers at runtime
 * - Accessing solver statistics
 */

#include "cfd/core/cfd_status.h"
#include "cfd/api/simulation_api.h"
#include "cfd/solvers/solver_interface.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"
#include "cfd/core/logging.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/math_utils.h"

#include "cfd/io/vtk_output.h"
#include <stdio.h>
#include <string.h>


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

void print_solver_info(const Solver* solver) {
    if (!solver) {
        printf("  Solver: (legacy/default)\n");
        return;
    }

    printf("  Name: %s\n", solver->name);
    printf("  Description: %s\n", solver->description);
    printf("  Version: %s\n", solver->version);
    printf("  Capabilities: ");

    if (solver->capabilities & SOLVER_CAP_INCOMPRESSIBLE)
        printf("incompressible ");
    if (solver->capabilities & SOLVER_CAP_COMPRESSIBLE)
        printf("compressible ");
    if (solver->capabilities & SOLVER_CAP_TRANSIENT)
        printf("transient ");
    if (solver->capabilities & SOLVER_CAP_STEADY_STATE)
        printf("steady-state ");
    if (solver->capabilities & SOLVER_CAP_SIMD)
        printf("SIMD ");
    if (solver->capabilities & SOLVER_CAP_PARALLEL)
        printf("parallel ");
    if (solver->capabilities & SOLVER_CAP_GPU)
        printf("GPU ");

    printf("\n");
}

void print_stats(const SolverStats* stats) {
    if (!stats)
        return;

    printf("  Iterations: %d\n", stats->iterations);
    printf("  Max velocity: %.4f\n", stats->max_velocity);
    printf("  Max pressure: %.4f\n", stats->max_pressure);
    printf("  Elapsed time: %.2f ms\n", stats->elapsed_time_ms);
}

void run_solver_comparison(void) {
    print_separator();
    printf("SOLVER COMPARISON TEST\n");
    print_separator();

    // List available solvers
    const char* solver_names[10];
    int num_solvers = simulation_list_solvers(solver_names, 10);

    printf("\nAvailable solvers (%d):\n", num_solvers);
    for (int i = 0; i < num_solvers; i++) {
        printf("  %d. %s\n", i + 1, solver_names[i]);
    }

    // Test each solver
    for (int i = 0; i < num_solvers; i++) {
        const char* solver_type = solver_names[i];

        print_separator();
        printf("Testing solver: %s\n", solver_type);
        print_separator();

        // Create simulation with this solver
        SimulationData* sim =
            init_simulation_with_solver(NX, NY, XMIN, XMAX, YMIN, YMAX, solver_type);
        if (!sim) {
            printf("  ERROR: Failed to create simulation\n");
            continue;
        }

        // Print solver info
        Solver* solver = simulation_get_solver(sim);
        print_solver_info(solver);

        // Set run prefix for this solver test
        simulation_set_run_prefix(sim, solver_type);

        // Register output at end of simulation only
        simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, NUM_STEPS, "solver_test");

        // Run simulation
        printf("\nRunning %d steps...\n", NUM_STEPS);
        for (int step = 0; step <= NUM_STEPS; step++) {
            run_simulation_step(sim);
            simulation_write_outputs(sim, step);
        }

        // Print final statistics
        printf("\nFinal statistics:\n");
        print_stats(simulation_get_stats(sim));
        printf("\nOutput written automatically\n");

        // Cleanup
        free_simulation(sim);
    }
}

void run_dynamic_solver_switch(void) {
    print_separator();
    printf("DYNAMIC SOLVER SWITCHING\n");
    print_separator();

    // Start with default solver (explicit_euler)
    printf("\n1. Creating simulation with default solver...\n");
    SimulationData* sim = init_simulation(NX, NY, XMIN, XMAX, YMIN, YMAX);
    simulation_set_run_prefix(sim, "dynamic_switch");

    // Register output every 10 steps
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");

    Solver* solver = simulation_get_solver(sim);
    print_solver_info(solver);

    int step_counter = 0;

    // Run a few steps
    printf("\nRunning 10 steps with default solver...\n");
    for (int step = 0; step < 10; step++, step_counter++) {
        run_simulation_step(sim);
        simulation_write_outputs(sim, step_counter);
    }

    // Switch to optimized solver
    printf("\n2. Switching to optimized solver...\n");
    if (simulation_set_solver_by_name(sim, SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED) == 0) {
        solver = simulation_get_solver(sim);
        print_solver_info(solver);

        // Run more steps
        printf("\nRunning 10 more steps with optimized solver...\n");
        for (int step = 0; step < 10; step++, step_counter++) {
            run_simulation_step(sim);
            simulation_write_outputs(sim, step_counter);
        }

        printf("\nStatistics after optimized solver:\n");
        print_stats(simulation_get_stats(sim));
    } else {
        printf("  ERROR: Failed to switch solver\n");
    }

    printf("\nOutput written automatically at regular intervals\n");

    free_simulation(sim);
}

void run_direct_solver_usage(void) {
    print_separator();
    printf("DIRECT SOLVER API USAGE\n");
    print_separator();

    // Create solver directly
    printf("\nCreating solver directly via solver_create()...\n");
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    if (!solver) {
        printf("  ERROR: Failed to create solver\n");
        return;
    }

    print_solver_info(solver);

    // Create grid and flow field manually
    Grid* grid = grid_create(NX, NY, XMIN, XMAX, YMIN, YMAX);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(NX, NY);
    initialize_flow_field(field, grid);

    SolverParams params = solver_params_default();
    params.max_iter = 1;
    params.dt = 0.005;

    // Initialize solver
    cfd_status_t status = solver_init(solver, grid, &params);
    printf("\nSolver init status: %d\n", status);

    // Run steps directly
    printf("\nRunning 20 steps using direct solver API...\n");
    SolverStats stats = solver_stats_default();

    for (int step = 0; step < 20; step++) {
        status = solver_step(solver, field, grid, &params, &stats);

        if (step % 5 == 0) {
            printf("  Step %d: max_vel=%.4f, max_p=%.4f, time=%.2fms\n", step, stats.max_velocity,
                   stats.max_pressure, stats.elapsed_time_ms);
        }
    }

    // Write output using VTK functions directly
    char output_path[512];
    make_output_path(output_path, sizeof(output_path), "direct_api_test.vtk");
    write_vtk_flow_field(output_path, field, NX, NY, XMIN, XMAX, YMIN, YMAX);
    printf("\nOutput written to: %s\n", output_path);

    // Cleanup
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(grid);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("CFD Framework - Solver Selection Example\n");
    printf("=========================================\n");

    // Configure output directory (optional - defaults to ../../artifacts)
    simulation_set_output_dir("../../artifacts");

    // Run demonstrations
    run_solver_comparison();
    run_dynamic_solver_switch();
    run_direct_solver_usage();

    print_separator();
    printf("All tests completed!\n");
    print_separator();

    return 0;
}
