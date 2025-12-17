#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"


#include "cfd/io/vtk_output.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Example demonstrating how to customize source term parameters
 * for different flow control scenarios using the modern solver interface
 */

// Helper function to calculate max velocity
double calculate_max_velocity(const flow_field* field, size_t nx, size_t ny) {
    double max_vel = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        double vel_mag = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
        if (vel_mag > max_vel) {
            max_vel = vel_mag;
        }
    }
    return max_vel;
}

// Helper function to run simulation with given parameters
void run_simulation_case(struct NSSolver* solver, flow_field* field, grid* grid,
                         ns_solver_params_t* params, int steps) {
    initialize_flow_field(field, grid);

    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int step = 0; step < steps; step++) {
        solver_step(solver, field, grid, params, &stats);
    }
}

int main(int argc, char* argv[]) {
    printf("CFD Simulation - Custom Source Terms Example\n");
    printf("=============================================\n\n");

    // Simulation domain
    size_t nx = 50, ny = 25;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    // Create grid and flow field
    grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    flow_field* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Create solver using modern interface
    struct NSSolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    struct NSSolver* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    if (!solver) {
        fprintf(stderr, "Failed to create solver\n");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(grid);
        return 1;
    }

    // Configure output directory
    cfd_set_output_base_dir("../../artifacts");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "source_terms", nx, ny);
    printf("Output directory: %s\n\n", run_dir);

    // Example 1: Default parameters
    printf("1. Running simulation with DEFAULT source term parameters:\n");
    ns_solver_params_t params_default = ns_solver_params_default();
    printf("   - Source amplitude U: %.3f\n", params_default.source_amplitude_u);
    printf("   - Source amplitude V: %.3f\n", params_default.source_amplitude_v);
    printf("   - Source decay rate:  %.3f\n", params_default.source_decay_rate);
    printf("   - Pressure coupling:  %.3f\n", params_default.pressure_coupling);

    // Initialize solver
    solver_init(solver, grid, &params_default);

    // Run simulation
    run_simulation_case(solver, field, grid, &params_default, 10);

    // Save default case
    write_vtk_output("..\\..\\artifacts\\output\\default_source_terms.vtk", "u_velocity", field->u,
                     nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: default_source_terms.vtk\n\n");

    // Example 2: High energy injection (stronger sources)
    printf("2. Running simulation with HIGH ENERGY source terms:\n");
    ns_solver_params_t params_high_energy = ns_solver_params_default();

    // Customize source term parameters for high energy injection
    params_high_energy.source_amplitude_u = 0.3;   // 3x stronger U source
    params_high_energy.source_amplitude_v = 0.15;  // 3x stronger V source
    params_high_energy.source_decay_rate = 0.05;   // Slower decay (more persistent)
    params_high_energy.pressure_coupling = 0.15;   // Stronger pressure coupling

    printf("   - Source amplitude U: %.3f (3x stronger)\n", params_high_energy.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (3x stronger)\n", params_high_energy.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (slower decay)\n", params_high_energy.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (stronger)\n", params_high_energy.pressure_coupling);

    // Run with high energy parameters
    run_simulation_case(solver, field, grid, &params_high_energy, 10);

    // Save high energy case
    write_vtk_output("..\\..\\artifacts\\output\\high_energy_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: high_energy_source_terms.vtk\n\n");

    // Example 3: Low energy injection (weaker sources)
    printf("3. Running simulation with LOW ENERGY source terms:\n");
    ns_solver_params_t params_low_energy = ns_solver_params_default();

    // Customize source term parameters for low energy injection
    params_low_energy.source_amplitude_u = 0.03;   // 30% of default
    params_low_energy.source_amplitude_v = 0.015;  // 30% of default
    params_low_energy.source_decay_rate = 0.2;     // Faster decay
    params_low_energy.pressure_coupling = 0.05;    // Weaker pressure coupling

    printf("   - Source amplitude U: %.3f (30%% of default)\n",
           params_low_energy.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (30%% of default)\n",
           params_low_energy.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (faster decay)\n", params_low_energy.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (weaker)\n", params_low_energy.pressure_coupling);

    // Run with low energy parameters
    run_simulation_case(solver, field, grid, &params_low_energy, 10);

    // Save low energy case
    write_vtk_output("..\\..\\artifacts\\output\\low_energy_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: low_energy_source_terms.vtk\n\n");

    // Example 4: Asymmetric flow (different U and V sources)
    printf("4. Running simulation with ASYMMETRIC source terms:\n");
    ns_solver_params_t params_asymmetric = ns_solver_params_default();

    // Create asymmetric flow pattern
    params_asymmetric.source_amplitude_u = 0.2;   // Strong horizontal flow
    params_asymmetric.source_amplitude_v = 0.01;  // Weak vertical flow
    params_asymmetric.source_decay_rate = 0.08;   // Medium decay
    params_asymmetric.pressure_coupling = 0.12;   // Medium coupling

    printf("   - Source amplitude U: %.3f (strong horizontal)\n",
           params_asymmetric.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (weak vertical)\n", params_asymmetric.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (medium decay)\n", params_asymmetric.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (medium)\n", params_asymmetric.pressure_coupling);

    // Run with asymmetric parameters
    run_simulation_case(solver, field, grid, &params_asymmetric, 10);

    // Save asymmetric case
    write_vtk_output("..\\..\\artifacts\\output\\asymmetric_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: asymmetric_source_terms.vtk\n\n");

    // Calculate and display flow statistics for comparison
    printf("Flow Statistics Comparison:\n");
    printf("===========================\n");

    // Re-run each case briefly to get statistics
    double max_velocities[4];
    const char* case_names[] = {"Default", "High Energy", "Low Energy", "Asymmetric"};
    ns_solver_params_t* all_params[] = {&params_default, &params_high_energy, &params_low_energy,
                                   &params_asymmetric};

    for (int case_idx = 0; case_idx < 4; case_idx++) {
        run_simulation_case(solver, field, grid, all_params[case_idx], 5);
        max_velocities[case_idx] = calculate_max_velocity(field, nx, ny);

        printf("%s case: Max velocity = %.4f m/s\n", case_names[case_idx],
               max_velocities[case_idx]);
    }

    printf("\nConclusion:\n");
    printf("===========\n");
    printf("The source term parameters allow fine control over:\n");
    printf("- Flow energy level (amplitude parameters)\n");
    printf("- Flow persistence (decay rate)\n");
    printf("- Pressure-velocity coupling strength\n");
    printf("- Flow directionality (U vs V amplitudes)\n\n");

    printf("Users can customize these parameters in their code:\n");
    printf("  NSSolverRegistry* registry = cfd_registry_create();\n");
    printf("  cfd_registry_register_defaults(registry);\n");
    printf("  Solver* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);\n");
    printf("  SolverParams params = ns_solver_params_default();\n");
    printf("  params.source_amplitude_u = 0.2;  // Custom value\n");
    printf("  params.source_amplitude_v = 0.1;  // Custom value\n");
    printf("  solver_init(solver, grid, &params);\n");
    printf("  solver_step(solver, field, grid, &params, &stats);\n\n");

    printf("All output files saved to ..\\..\\artifacts\\output\\\n");
    printf("Use visualization tools to compare the different cases.\n");

    // Clean up
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(grid);
    cfd_registry_destroy(registry);
}
