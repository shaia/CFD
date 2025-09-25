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

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    if (!sim_data) {
        printf("Failed to initialize simulation\n");
        return 1;
    }

    // Create output directory
    ensure_directory_exists("output");
    ensure_directory_exists("artifacts\\output");

    // Run simulation with velocity visualization
    int max_steps = 100;
    int output_interval = 10;

    printf("\nRunning simulation with velocity visualization...\n");
    printf("Output interval: every %d steps\n", output_interval);

    for (int step = 0; step < max_steps; step++) {
        // Run simulation step
        run_simulation_step(sim_data);

        // Output visualization data at intervals
        if (step % output_interval == 0) {
            char filename[256];

            // 1. Velocity magnitude (original functionality)
            snprintf(filename, sizeof(filename), "artifacts\\output\\velocity_magnitude_%03d.vtk", step);
            write_simulation_to_vtk(sim_data, filename);

            // 2. Velocity vectors only
            snprintf(filename, sizeof(filename), "artifacts\\output\\velocity_vectors_%03d.vtk", step);
            write_velocity_vectors_to_vtk(sim_data, filename);

            // 3. Complete flow field (vectors + magnitude + pressure)
            snprintf(filename, sizeof(filename), "artifacts\\output\\flow_field_%03d.vtk", step);
            write_flow_field_to_vtk(sim_data, filename);

            printf("Step %3d: Output written to VTK files\n", step);
        }
    }

    // Final comprehensive output
    printf("\nWriting final comprehensive flow field visualization...\n");
    write_flow_field_to_vtk(sim_data, "artifacts\\output\\final_flow_field.vtk");
    write_velocity_vectors_to_vtk(sim_data, "artifacts\\output\\final_velocity_vectors.vtk");

    // Cleanup
    free_simulation(sim_data);

    printf("\nSimulation completed successfully!\n");
    printf("\nGenerated visualization files:\n");
    printf("  - velocity_magnitude_*.vtk  : Velocity magnitude contours\n");
    printf("  - velocity_vectors_*.vtk    : Velocity vector field\n");
    printf("  - flow_field_*.vtk          : Complete flow field (vectors + magnitude + pressure)\n");
    printf("  - final_*.vtk               : Final state visualization\n");
    printf("\nUse ParaView, VisIt, or Python visualization scripts to view the results.\n");

    return 0;
}