#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h"
#include "solver.h"
#include "utils.h"
#include "vtk_output.h"

/**
 * Example demonstrating how to customize source term parameters
 * for different flow control scenarios
 */

// Helper function to calculate max velocity
double calculate_max_velocity(const FlowField* field, size_t nx, size_t ny) {
    double max_vel = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        double vel_mag = sqrt(field->u[i]*field->u[i] + field->v[i]*field->v[i]);
        if (vel_mag > max_vel) max_vel = vel_mag;
    }
    return max_vel;
}

int main(int argc, char* argv[]) {
    printf("CFD Simulation - Custom Source Terms Example\n");
    printf("=============================================\n\n");

    // Simulation domain
    size_t nx = 50, ny = 25;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Create output directory
    ensure_directory_exists("../../output");
    ensure_directory_exists("..\\..\\artifacts\\output");

    // Example 1: Default parameters
    printf("1. Running simulation with DEFAULT source term parameters:\n");
    SolverParams params_default = solver_params_default();
    printf("   - Source amplitude U: %.3f\n", params_default.source_amplitude_u);
    printf("   - Source amplitude V: %.3f\n", params_default.source_amplitude_v);
    printf("   - Source decay rate:  %.3f\n", params_default.source_decay_rate);
    printf("   - Pressure coupling:  %.3f\n", params_default.pressure_coupling);

    // Reset field for clean start
    initialize_flow_field(field, grid);
    params_default.max_iter = 10;  // Short run for demonstration
    solve_navier_stokes(field, grid, &params_default);

    // Save default case
    write_vtk_output("..\\..\\artifacts\\output\\default_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: default_source_terms.vtk\n\n");

    // Example 2: High energy injection (stronger sources)
    printf("2. Running simulation with HIGH ENERGY source terms:\n");
    SolverParams params_high_energy = solver_params_default();

    // Customize source term parameters for high energy injection
    params_high_energy.source_amplitude_u = 0.3;   // 3x stronger U source
    params_high_energy.source_amplitude_v = 0.15;  // 3x stronger V source
    params_high_energy.source_decay_rate = 0.05;   // Slower decay (more persistent)
    params_high_energy.pressure_coupling = 0.15;   // Stronger pressure coupling
    params_high_energy.max_iter = 10;

    printf("   - Source amplitude U: %.3f (3x stronger)\n", params_high_energy.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (3x stronger)\n", params_high_energy.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (slower decay)\n", params_high_energy.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (stronger)\n", params_high_energy.pressure_coupling);

    // Reset field and run with high energy parameters
    initialize_flow_field(field, grid);
    solve_navier_stokes(field, grid, &params_high_energy);

    // Save high energy case
    write_vtk_output("..\\..\\artifacts\\output\\high_energy_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: high_energy_source_terms.vtk\n\n");

    // Example 3: Low energy injection (weaker sources)
    printf("3. Running simulation with LOW ENERGY source terms:\n");
    SolverParams params_low_energy = solver_params_default();

    // Customize source term parameters for low energy injection
    params_low_energy.source_amplitude_u = 0.03;   // 30% of default
    params_low_energy.source_amplitude_v = 0.015;  // 30% of default
    params_low_energy.source_decay_rate = 0.2;     // Faster decay
    params_low_energy.pressure_coupling = 0.05;    // Weaker pressure coupling
    params_low_energy.max_iter = 10;

    printf("   - Source amplitude U: %.3f (30%% of default)\n", params_low_energy.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (30%% of default)\n", params_low_energy.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (faster decay)\n", params_low_energy.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (weaker)\n", params_low_energy.pressure_coupling);

    // Reset field and run with low energy parameters
    initialize_flow_field(field, grid);
    solve_navier_stokes(field, grid, &params_low_energy);

    // Save low energy case
    write_vtk_output("..\\..\\artifacts\\output\\low_energy_source_terms.vtk", "u_velocity",
                     field->u, nx, ny, xmin, xmax, ymin, ymax);
    printf("   Output saved to: low_energy_source_terms.vtk\n\n");

    // Example 4: Asymmetric flow (different U and V sources)
    printf("4. Running simulation with ASYMMETRIC source terms:\n");
    SolverParams params_asymmetric = solver_params_default();

    // Create asymmetric flow pattern
    params_asymmetric.source_amplitude_u = 0.2;    // Strong horizontal flow
    params_asymmetric.source_amplitude_v = 0.01;   // Weak vertical flow
    params_asymmetric.source_decay_rate = 0.08;    // Medium decay
    params_asymmetric.pressure_coupling = 0.12;    // Medium coupling
    params_asymmetric.max_iter = 10;

    printf("   - Source amplitude U: %.3f (strong horizontal)\n", params_asymmetric.source_amplitude_u);
    printf("   - Source amplitude V: %.3f (weak vertical)\n", params_asymmetric.source_amplitude_v);
    printf("   - Source decay rate:  %.3f (medium decay)\n", params_asymmetric.source_decay_rate);
    printf("   - Pressure coupling:  %.3f (medium)\n", params_asymmetric.pressure_coupling);

    // Reset field and run with asymmetric parameters
    initialize_flow_field(field, grid);
    solve_navier_stokes(field, grid, &params_asymmetric);

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
    SolverParams* all_params[] = {&params_default, &params_high_energy, &params_low_energy, &params_asymmetric};

    for (int case_idx = 0; case_idx < 4; case_idx++) {
        initialize_flow_field(field, grid);
        all_params[case_idx]->max_iter = 5;  // Quick run for stats
        solve_navier_stokes(field, grid, all_params[case_idx]);
        max_velocities[case_idx] = calculate_max_velocity(field, nx, ny);

        printf("%s case: Max velocity = %.4f m/s\n",
               case_names[case_idx], max_velocities[case_idx]);
    }

    printf("\nConclusion:\n");
    printf("===========\n");
    printf("The source term parameters allow fine control over:\n");
    printf("- Flow energy level (amplitude parameters)\n");
    printf("- Flow persistence (decay rate)\n");
    printf("- Pressure-velocity coupling strength\n");
    printf("- Flow directionality (U vs V amplitudes)\n\n");

    printf("Users can customize these parameters in their code:\n");
    printf("  SolverParams params = solver_params_default();\n");
    printf("  params.source_amplitude_u = 0.2;  // Custom value\n");
    printf("  params.source_amplitude_v = 0.1;  // Custom value\n");
    printf("  // ... then use params in solve_navier_stokes()\n\n");

    printf("All output files saved to ..\\..\\artifacts\\output\\\n");
    printf("Use visualization tools to compare the different cases.\n");

    // Clean up
    flow_field_destroy(field);
    grid_destroy(grid);

    return 0;
}