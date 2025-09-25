#ifndef SIMULATION_API_H
#define SIMULATION_API_H

#include "grid.h"
#include "solver.h"

// Structure to hold simulation data
typedef struct {
    Grid* grid;
    FlowField* field;
    SolverParams params;
} SimulationData;

// Initialize simulation data
SimulationData* init_simulation(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax);

// Run simulation step
void run_simulation_step(SimulationData* sim_data);

// Write simulation data to VTK file
void write_simulation_to_vtk(SimulationData* sim_data, const char* filename);

// Write velocity vectors to VTK file
void write_velocity_vectors_to_vtk(SimulationData* sim_data, const char* filename);

// Write complete flow field to VTK file
void write_flow_field_to_vtk(SimulationData* sim_data, const char* filename);

// Calculate velocity magnitude (returns sqrt of u^2 + v^2)
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny);

// Calculate velocity magnitude squared (for performance-critical comparisons)
double* calculate_velocity_magnitude_squared(const FlowField* field, size_t nx, size_t ny);
// Free simulation data
void free_simulation(SimulationData* sim_data);

// Configuration functions
void set_output_directory(const char* path);
void reset_output_directory(void);

// Default path options for when no custom path is set
typedef enum {
    OUTPUT_CURRENT_DIR,     // "./output" (default - creates output in current directory)
    OUTPUT_TEMP_DIR,        // System temp directory (e.g., /tmp/cfd_output)
    OUTPUT_RELATIVE_BUILD   // "../../artifacts" (for build tree compatibility)
} output_path_mode_t;

void set_default_output_mode(output_path_mode_t mode);

#endif // SIMULATION_API_H