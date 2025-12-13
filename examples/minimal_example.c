/**
 * Minimal CFD Example
 *
 * The simplest possible example showing how to use the CFD library.
 * Perfect for getting started and understanding the basic API.
 */

#include "cfd/api/simulation_api.h"
#include <stdio.h>

int main() {
    printf("Minimal CFD Library Example\n");
    printf("===========================\n");


    // Step 2: Initialize simulation
    size_t nx = 50, ny = 25;
    simulation_data* sim = init_simulation(nx, ny, 0.0, 1.0, 0.0, 0.5);

    if (!sim) {
        printf("Error: Failed to initialize simulation\n");
        return 1;
    }

    printf("✓ Simulation initialized (%zux%zu grid)\n", nx, ny);
    printf("✓ Domain: [0,1] x [0,0.5]\n");

    // Configure output directory (optional)
    simulation_set_output_dir(sim, "../../artifacts");
    printf("✓ Output directory: ../../artifacts\n");

    // Step 3: Set run prefix for organized output
    simulation_set_run_prefix(sim, "minimal");

    // Step 4: Register automatic output every 5 steps
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 5, "velocity_mag");
    printf("✓ Registered velocity magnitude output every 5 steps\n");

    // Step 5: Run simulation with automatic output
    printf("\nRunning simulation...\n");

    for (int step = 0; step < 10; step++) {
        run_simulation_step(sim);
        simulation_write_outputs(sim, step);

        if (step % 5 == 0) {
            printf("  Step %d completed\n", step);
        }
    }

    // Step 6: Cleanup
    free_simulation(sim);

    printf("\n✓ Simulation completed successfully!\n");
    printf("\nNext steps:\n");
    printf("- Visualize VTK files with ParaView, VisIt, or Python\n");
    printf("- Try the other examples for more advanced features\n");

    return 0;
}