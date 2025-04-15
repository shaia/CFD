#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "grid.h"
#include "solver.h"
#include "utils.h"
#include "simulation_api.c"

int main(int argc, char* argv[]) {
    // Initialize simulation
    size_t nx = 100, ny = 50;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 0.5;
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    printf("Starting CFD simulation...\n");

    // Main simulation loop
    int iter = 0;
    while (iter < sim_data->params.max_iter) {
        run_simulation_step(sim_data);

        // Optionally write output every 100 iterations
        if (iter % 100 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "output_optimized_%d.vtk", iter);
            write_simulation_to_vtk(sim_data, filename);
        }

        iter++;
    }

    printf("\nSimulation completed.\n");

    // Clean up
    free_simulation(sim_data);
    return 0;
}