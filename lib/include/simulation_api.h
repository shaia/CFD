#ifndef SIMULATION_API_H
#define SIMULATION_API_H

#include "grid.h"
#include "solver_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// SIMULATION STRUCTURE
//=============================================================================

// Forward declaration for output registry
typedef struct OutputRegistry OutputRegistry;

// Main simulation state containing grid, flow field, solver, and parameters
typedef struct {
    Grid* grid;               // Computational grid
    FlowField* field;         // Flow variables (u, v, p, rho, T)
    SolverParams params;      // Solver parameters (dt, CFL, viscosity, etc.)
    Solver* solver;           // Active solver (NULL = default solver)
    SolverStats last_stats;   // Statistics from last solve step
    OutputRegistry* outputs;  // Registered output configurations (internal)
    char* run_prefix;         // Run directory prefix (NULL = default "sim")
    double current_time;      // Current simulation time (accumulated dt)
} SimulationData;

//=============================================================================
// INITIALIZATION & CLEANUP
//=============================================================================

// Initialize simulation with default solver
SimulationData* init_simulation(size_t nx, size_t ny, double xmin, double xmax, double ymin,
                                double ymax);

// Initialize simulation with specific solver type
SimulationData* init_simulation_with_solver(size_t nx, size_t ny, double xmin, double xmax,
                                            double ymin, double ymax, const char* solver_type);

// Free all simulation resources
void free_simulation(SimulationData* sim_data);

//=============================================================================
// SOLVER MANAGEMENT
//=============================================================================

// Set solver for existing simulation (takes ownership)
void simulation_set_solver(SimulationData* sim_data, Solver* solver);

// Set solver by type name (e.g., "explicit_euler", "projection")
int simulation_set_solver_by_name(SimulationData* sim_data, const char* solver_type);

// Get current solver (returns NULL if using default)
Solver* simulation_get_solver(SimulationData* sim_data);

// List all available solver types
int simulation_list_solvers(const char** names, int max_count);

// Check if specific solver type is available
int simulation_has_solver(const char* solver_type);

//=============================================================================
// SIMULATION EXECUTION
//=============================================================================

// Run a single simulation time step
void run_simulation_step(SimulationData* sim_data);

// Run simulation until convergence or max iterations
// Uses the solver's solve() method instead of step()
void run_simulation_solve(SimulationData* sim_data);

// Get statistics from last solve (iterations, residuals, etc.)
const SolverStats* simulation_get_stats(const SimulationData* sim_data);

//=============================================================================
// OUTPUT CONTROL
//=============================================================================

// Field types for output
typedef enum {
    // VTK outputs (3D visualization)
    OUTPUT_PRESSURE,    // Pressure/velocity magnitude field (VTK)
    OUTPUT_VELOCITY,    // Velocity vector field (VTK)
    OUTPUT_FULL_FIELD,  // Complete flow field (VTK)

    // CSV outputs (data analysis)
    OUTPUT_CSV_TIMESERIES,  // Time series: step, time, max_vel, max_p, residual, etc.
    OUTPUT_CSV_CENTERLINE,  // Centerline profile: x or y, u, v, p along domain center
    OUTPUT_CSV_STATISTICS   // Global statistics: min/max/avg for all fields per step
} OutputFieldType;

// Output configuration for automatic file generation
typedef struct {
    OutputFieldType field_type;  // What to output
    int interval;                // Output every N steps (0 = disabled)
    const char* prefix;          // Optional filename prefix (NULL = use field type name)
} OutputConfig;

//=============================================================================
// OUTPUT REGISTRATION (Registry/Injection Pattern)
//=============================================================================

// Register output for automatic generation
// Example: simulation_register_output(sim, OUTPUT_PRESSURE, 10, "pressure");
//          Automatically writes pressure_000.vtk, pressure_010.vtk, etc.
void simulation_register_output(SimulationData* sim_data, OutputFieldType field_type, int interval,
                                const char* prefix);

// Clear all registered outputs
void simulation_clear_outputs(SimulationData* sim_data);

// Set base output directory (default: "../../artifacts")
// This is the only manual configuration needed - everything else is automatic
void simulation_set_output_dir(const char* base_dir);

// Set run name prefix (default: "sim")
// Creates directories like: {base_dir}/output/{prefix}_{grid}_{timestamp}/
void simulation_set_run_prefix(SimulationData* sim_data, const char* prefix);

//=============================================================================
// AUTOMATIC OUTPUT GENERATION
//=============================================================================

// Automatically write all registered outputs for current step
// Handles directory creation, file naming, and output writing internally
void simulation_write_outputs(SimulationData* sim_data, int step);

#ifdef __cplusplus
}
#endif

#endif  // SIMULATION_API_H
