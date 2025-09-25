#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "grid.h"
#include "solver.h"
#include "utils.h"
#include "simulation_api.h"

void run_simulation(SimulationData* sim_data) {
    printf("Starting CFD simulation...\n");

    for (int iter = 0; iter < sim_data->params.max_iter; iter++) {
        run_simulation_step(sim_data);

        // Optionally write output every 100 iterations
        if (iter % 100 == 0) {
            ensure_directory_exists("../../output");
            ensure_directory_exists("..\\..\\artifacts\\output");
            char filename[256];
            snprintf(filename, sizeof(filename), "..\\..\\artifacts\\output\\output_optimized_%d.vtk", iter);
            write_simulation_to_vtk(sim_data, filename);
        }
    }

    printf("\nSimulation completed.\n");
}

int main(int argc, char* argv[]) {
    // Initialize simulation parameters
    size_t nx = 100, ny = 50;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 0.5;

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Run the simulation
    run_simulation(sim_data);

    // Clean up
    free_simulation(sim_data);

    return 0;
}