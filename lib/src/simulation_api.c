#include "solver.h"
#include "grid.h"
#include "vtk_output.h"
#include "utils.h"
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

    // Set solver parameters for transient animation
    sim_data->params = (SolverParams){
        .dt = 0.001,      // Will be computed dynamically
        .cfl = 0.2,       // More conservative CFL number
        .gamma = 1.4,
        .mu = 0.01,       // Increased viscosity for stable visual flow
        .k = 0.0242,
        .max_iter = 1,    // Single iteration per time step for animation
        .tolerance = 1e-6,

        // Source term parameters for energy maintenance
        .source_amplitude_u = 0.1,    // Default amplitude of u-velocity source term
        .source_amplitude_v = 0.05,   // Default amplitude of v-velocity source term
        .source_decay_rate = 0.1,     // Default decay rate for source terms over time
        .pressure_coupling = 0.1      // Default coupling coefficient for pressure update
    };

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