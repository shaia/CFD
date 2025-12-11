#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    // Initialize simulation parameters
    size_t nx = 100, ny = 50;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 0.5;

    printf("Starting CFD simulation...\n");
    printf("Grid size: %zu x %zu\n", nx, ny);


    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Configure output directory (optional)
    simulation_set_output_dir(sim_data, "../../artifacts");
    simulation_set_run_prefix(sim_data, "basic_sim");

    // Register automatic output every 100 steps
    simulation_register_output(sim_data, OUTPUT_VELOCITY_MAGNITUDE, 100, NULL);

    // Run the simulation
    for (int iter = 0; iter < sim_data->params.max_iter; iter++) {
        run_simulation_step(sim_data);

        // Automatically write registered outputs
        simulation_write_outputs(sim_data, iter);

        if (iter % 100 == 0) {
            printf("Iteration %d: Output written\n", iter);
        }
    }

    printf("\nSimulation completed.\n");

    // Clean up
    free_simulation(sim_data);

    return 0;
}