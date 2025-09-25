#include "simulation_api.h"
#include "solver.h"
#include "grid.h"
#include "vtk_output.h"
#include "utils.h"
#include <math.h>

// SimulationData struct is defined in simulation_api.h

// Initialize simulation data
SimulationData* init_simulation(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax) {
    SimulationData* sim_data = (SimulationData*)cfd_malloc(sizeof(SimulationData));

    // Create and initialize grid
    sim_data->grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(sim_data->grid);

    // Create flow field
    sim_data->field = flow_field_create(nx, ny);
    initialize_flow_field(sim_data->field, sim_data->grid);

    // Initialize solver parameters with defaults, then customize for animation
    sim_data->params = solver_params_default();

    // Customize parameters for animation
    sim_data->params.dt = 0.001;      // Will be computed dynamically
    sim_data->params.cfl = 0.2;       // More conservative CFL number
    sim_data->params.mu = 0.01;       // Increased viscosity for stable visual flow
    sim_data->params.max_iter = 1;    // Single iteration per time step for animation

    return sim_data;
}

// Run simulation step
void run_simulation_step(SimulationData* sim_data) {
    // Use fixed time step for animation stability
    sim_data->params.dt = 0.005;  // Fixed small time step
    solve_navier_stokes(sim_data->field, sim_data->grid, &sim_data->params);
}

// Write simulation data to VTK file (original - pressure only)
void write_simulation_to_vtk(SimulationData* sim_data, const char* filename) {
    double* velocity_magnitude = calculate_velocity_magnitude(sim_data->field, sim_data->grid->nx, sim_data->grid->ny);

    write_vtk_output(filename, "velocity_magnitude", velocity_magnitude,
                     sim_data->grid->nx, sim_data->grid->ny,
                     sim_data->grid->xmin, sim_data->grid->xmax,
                     sim_data->grid->ymin, sim_data->grid->ymax);

    free(velocity_magnitude);
}

// Write velocity vectors to VTK file
void write_velocity_vectors_to_vtk(SimulationData* sim_data, const char* filename) {
    write_vtk_vector_output(filename, "velocity",
                           sim_data->field->u, sim_data->field->v,
                           sim_data->grid->nx, sim_data->grid->ny,
                           sim_data->grid->xmin, sim_data->grid->xmax,
                           sim_data->grid->ymin, sim_data->grid->ymax);
}

// Write complete flow field (velocity vectors + magnitude + pressure) to VTK file
void write_flow_field_to_vtk(SimulationData* sim_data, const char* filename) {
    write_vtk_flow_field(filename,
                        sim_data->field,
                        sim_data->grid->nx, sim_data->grid->ny,
                        sim_data->grid->xmin, sim_data->grid->xmax,
                        sim_data->grid->ymin, sim_data->grid->ymax);
}

// Calculate velocity magnitude field (use for visualization/output)
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny) {
    double* velocity_magnitude = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude[i] = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }

    return velocity_magnitude;
}

// Optimized version that avoids sqrt for performance-critical comparisons
// Use this when you only need to compare magnitudes or compute statistics
double* calculate_velocity_magnitude_squared(const FlowField* field, size_t nx, size_t ny) {
    double* velocity_magnitude_sq = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude_sq[i] = field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }

    return velocity_magnitude_sq;
}
// Free simulation data
void free_simulation(SimulationData* sim_data) {
    flow_field_destroy(sim_data->field);
    grid_destroy(sim_data->grid);
    cfd_free(sim_data);
}

// Configuration functions for easy output directory management
void set_output_directory(const char* path) {
    cfd_set_artifacts_path(path);
}

void reset_output_directory(void) {
    cfd_reset_artifacts_path();
}

void set_default_output_mode(output_path_mode_t mode) {
    // Map simulation API enum to utils enum
    switch (mode) {
        case OUTPUT_CURRENT_DIR:
            cfd_set_default_path_mode(CFD_PATH_CURRENT_DIR);
            break;
        case OUTPUT_TEMP_DIR:
            cfd_set_default_path_mode(CFD_PATH_TEMP_DIR);
            break;
        case OUTPUT_RELATIVE_BUILD:
            cfd_set_default_path_mode(CFD_PATH_RELATIVE_BUILD);
            break;
    }
}