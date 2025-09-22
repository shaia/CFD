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

// Free simulation data
void free_simulation(SimulationData* sim_data);

#endif // SIMULATION_API_H