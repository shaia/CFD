#ifndef CFD_SOLVER_H
#define CFD_SOLVER_H

#include "grid.h"

// Flow field structure to store solution variables
typedef struct {
    double* u;          // x-velocity component
    double* v;          // y-velocity component
    double* p;          // pressure
    double* rho;        // density
    double* T;          // temperature
    size_t nx;          // number of points in x-direction
    size_t ny;          // number of points in y-direction
} FlowField;

// Solver parameters
typedef struct {
    double dt;          // time step
    double cfl;         // Courant-Friedrichs-Lewy number
    double gamma;       // specific heat ratio
    double mu;          // viscosity
    double k;           // thermal conductivity
    int max_iter;       // maximum number of iterations
    double tolerance;   // convergence tolerance

    // Source term parameters for energy maintenance
    double source_amplitude_u;    // Amplitude of u-velocity source term
    double source_amplitude_v;    // Amplitude of v-velocity source term
    double source_decay_rate;     // Decay rate for source terms over time
    double pressure_coupling;     // Coupling coefficient for pressure update
} SolverParams;

// Function declarations
FlowField* flow_field_create(size_t nx, size_t ny);
void flow_field_destroy(FlowField* field);
void initialize_flow_field(FlowField* field, const Grid* grid);
void solve_navier_stokes(FlowField* field, const Grid* grid, const SolverParams* params);
void solve_navier_stokes_optimized(FlowField* field, const Grid* grid, const SolverParams* params);
void apply_boundary_conditions(FlowField* field, const Grid* grid);
void compute_time_step(FlowField* field, const Grid* grid, SolverParams* params);
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny);

#endif // CFD_SOLVER_H