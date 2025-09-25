/**
 * Minimal CFD Example
 *
 * The simplest possible example showing how to use the CFD library.
 * Perfect for getting started and understanding the basic API.
 */

#include <stdio.h>
#include "simulation_api.h"
#include "utils.h"

int main() {
    printf("Minimal CFD Library Example\n");
    printf("===========================\n");

    // Step 1: Initialize simulation
    // Parameters: nx, ny, xmin, xmax, ymin, ymax
    SimulationData* sim = init_simulation(50, 25, 0.0, 1.0, 0.0, 0.5);

    if (!sim) {
        printf("Error: Failed to initialize simulation\n");
        return 1;
    }

    printf("✓ Simulation initialized (50x25 grid)\n");
    printf("✓ Domain: [0,1] x [0,0.5]\n");

    // Step 2: Ensure output directory exists
    ensure_directory_exists("../../output");
    ensure_directory_exists("..\\..\\artifacts\\output");

    // Step 3: Run a few simulation steps
    printf("\nRunning simulation...\n");

    for (int step = 0; step < 10; step++) {
        run_simulation_step(sim);

        if (step % 5 == 0) {  // Output every 5 steps
            char filename[256];
            snprintf(filename, sizeof(filename), "..\\..\\artifacts\\output\\minimal_step_%02d.vtk", step);
            write_simulation_to_vtk(sim, filename);
            printf("  Step %d completed, saved: %s\n", step, filename);
        }
    }

    // Step 4: Cleanup
    free_simulation(sim);

    printf("\n✓ Simulation completed successfully!\n");
    printf("✓ Check output/ directory for VTK files\n");
    printf("\nNext steps:\n");
    printf("- Visualize VTK files with ParaView, VisIt, or Python\n");
    printf("- Try the other examples for more advanced features\n");

    return 0;
}