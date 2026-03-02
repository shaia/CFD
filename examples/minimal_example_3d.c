/**
 * Minimal 3D CFD Example
 *
 * Demonstrates 3D simulation using the CFD library.
 * Creates a small 16x16x16 grid and runs a few time steps.
 */

#include "cfd/api/simulation_api.h"
#include <stdio.h>

int main() {
    printf("Minimal 3D CFD Library Example\n");
    printf("==============================\n");

    /* Initialize 3D simulation: 16x16x16 grid on unit cube */
    size_t nx = 16, ny = 16, nz = 16;
    simulation_data* sim = init_simulation(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (!sim) {
        printf("Error: Failed to initialize simulation\n");
        return 1;
    }

    printf("  Simulation initialized (%zux%zux%zu grid)\n", nx, ny, nz);
    printf("  Domain: [0,1] x [0,1] x [0,1]\n");

    /* Configure output directory */
    simulation_set_output_dir(sim, "../../artifacts");
    simulation_set_run_prefix(sim, "minimal_3d");

    /* Register velocity magnitude output every 5 steps */
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 5, "velocity_mag");
    printf("  Registered velocity magnitude output every 5 steps\n");

    /* Run simulation */
    printf("\nRunning 3D simulation...\n");

    for (int step = 0; step < 10; step++) {
        run_simulation_step(sim);
        simulation_write_outputs(sim, step);

        if (step % 5 == 0) {
            printf("  Step %d completed\n", step);
        }
    }

    /* Cleanup */
    free_simulation(sim);

    printf("\n  Simulation completed successfully!\n");
    printf("\nThis demonstrates 3D support with nz > 1.\n");
    printf("All solver backends (CPU, SIMD, OMP) work identically\n");
    printf("in 3D using the branch-free stride_z=0 pattern for 2D.\n");

    return 0;
}
