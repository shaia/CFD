#include "simulation_api.h"
#include "utils.h"
#include <stdio.h>

int main() {
    printf("CFD Framework - Velocity Visualization Example\n");
    printf("==============================================\n");

    // Define grid and domain parameters
    size_t nx = 50, ny = 25;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    printf("Grid size: %zu x %zu\n", nx, ny);
    printf("Domain: [%.1f, %.1f] x [%.1f, %.1f]\n", xmin, xmax, ymin, ymax);

    // Configure output (optional - defaults to "../../artifacts")
    simulation_set_output_dir("../../artifacts");

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    if (!sim_data) {
        printf("Failed to initialize simulation\n");
        return 1;
    }

    // Set run prefix for organized output directories
    simulation_set_run_prefix(sim_data, "velocity_viz");

    // Register outputs - specify what to output and how often
    simulation_register_output(sim_data, OUTPUT_PRESSURE, 10, "velocity_magnitude");
    simulation_register_output(sim_data, OUTPUT_VELOCITY, 10, "velocity_vectors");
    simulation_register_output(sim_data, OUTPUT_FULL_FIELD, 10, "flow_field");

    // Run simulation with automatic output
    int max_steps = 100;

    printf("\nRunning simulation with automatic output...\n");
    printf("Registered outputs:\n");
    printf("  - velocity_magnitude (every 10 steps)\n");
    printf("  - velocity_vectors (every 10 steps)\n");
    printf("  - flow_field (every 10 steps)\n");

    for (int step = 0; step < max_steps; step++) {
        // Run simulation step
        run_simulation_step(sim_data);

        // Automatically write all registered outputs
        simulation_write_outputs(sim_data, step);

        if (step % 10 == 0) {
            printf("Step %3d: Output written\n", step);
        }
    }

    // Cleanup
    free_simulation(sim_data);

    printf("\nSimulation completed successfully!\n");
    printf("\nGenerated visualization files automatically in timestamped directory\n");
    printf("  - velocity_magnitude_*.vtk  : Velocity magnitude contours\n");
    printf("  - velocity_vectors_*.vtk    : Velocity vector field\n");
    printf(
        "  - flow_field_*.vtk          : Complete flow field (vectors + magnitude + pressure)\n");
    printf("\nUse ParaView, VisIt, or Python visualization scripts to view the results.\n");

    return 0;
}