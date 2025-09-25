#ifndef CFD_SOLVER_H
#define CFD_SOLVER_H

#include "grid.h"

// Default solver parameters
#define DEFAULT_TIME_STEP 0.001            // Default time step size
#define DEFAULT_CFL_NUMBER 0.2             // Default Courant-Friedrichs-Lewy number
#define DEFAULT_GAMMA 1.4                  // Default specific heat ratio
#define DEFAULT_VISCOSITY 0.01             // Default dynamic viscosity
#define DEFAULT_THERMAL_CONDUCTIVITY 0.0242 // Default thermal conductivity
#define DEFAULT_MAX_ITERATIONS 100         // Default maximum number of iterations
#define DEFAULT_TOLERANCE 1e-6             // Default convergence tolerance

// Default source term parameters
#define DEFAULT_SOURCE_AMPLITUDE_U 0.1    // Default amplitude of u-velocity source term
#define DEFAULT_SOURCE_AMPLITUDE_V 0.05   // Default amplitude of v-velocity source term
#define DEFAULT_SOURCE_DECAY_RATE 0.1     // Default decay rate for source terms over time
#define DEFAULT_PRESSURE_COUPLING 0.1     // Default coupling coefficient for pressure update

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
void solve_navier_stokes_stable(FlowField* field, const Grid* grid, const SolverParams* params);
void apply_boundary_conditions(FlowField* field, const Grid* grid);
void compute_time_step(FlowField* field, const Grid* grid, SolverParams* params);
void compute_source_terms(double x, double y, int iter, double dt, const SolverParams* params,
                         double* source_u, double* source_v);
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny);

// Helper function to initialize SolverParams with default values
SolverParams solver_params_default(void);

#endif // CFD_SOLVER_H