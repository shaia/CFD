#include "grid.h"
#include "solver.h"
#include "vtk_output.h"
#include <stddef.h>
#include <math.h>

// Structure to hold simulation data
typedef struct {
    Grid* grid;
    FlowField* field;
    SolverParams params;
} SimulationData;

// Initialize simulation data
SimulationData* init_simulation(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax) {
    SimulationData* sim_data = (SimulationData*)cfd_malloc(sizeof(SimulationData));

    // Create and initialize grid
    sim_data->grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(sim_data->grid);

    // Create flow field
    sim_data->field = flow_field_create(nx, ny);
    initialize_flow_field(sim_data->field, sim_data->grid);

    // Set solver parameters
    sim_data->params = (SolverParams){
        .dt = 0.001,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = 1.789e-5,
        .k = 0.0242,
        .max_iter = 1000,
        .tolerance = 1e-6
    };

    return sim_data;
}

// Run simulation step
void run_simulation_step(SimulationData* sim_data) {
    compute_time_step(sim_data->field, sim_data->grid, &sim_data->params);
    solve_navier_stokes_optimized(sim_data->field, sim_data->grid, &sim_data->params);
}

// Write simulation data to VTK file
void write_simulation_to_vtk(SimulationData* sim_data, const char* filename) {
    // Combine u and v into a single velocity magnitude array
    size_t nx = sim_data->grid->nx;
    size_t ny = sim_data->grid->ny;
    double* velocity_magnitude = (double*)malloc(nx * ny * sizeof(double));

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double u = sim_data->field->u[idx];
            double v = sim_data->field->v[idx];
            velocity_magnitude[idx] = sqrt(u * u + v * v);
        }
    }

    // Write the velocity magnitude to the VTK file
    write_vtk_output(filename, "velocity_magnitude", velocity_magnitude, 
                     nx, ny,
                     sim_data->grid->xmin, sim_data->grid->xmax,
                     sim_data->grid->ymin, sim_data->grid->ymax);

    free(velocity_magnitude);
}

// Free simulation data
void free_simulation(SimulationData* sim_data) {
    flow_field_destroy(sim_data->field);
    grid_destroy(sim_data->grid);
    cfd_free(sim_data);
}