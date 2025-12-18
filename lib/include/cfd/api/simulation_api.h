#ifndef SIMULATION_API_H
#define SIMULATION_API_H

#include "cfd/cfd_export.h"

#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// SIMULATION STRUCTURE
//=============================================================================

// Forward declaration for output registry
typedef struct OutputRegistry output_registry;

// Main simulation state containing grid, flow field, solver, and parameters
typedef struct {
    grid* grid;                 // Computational grid
    flow_field* field;          // Flow variables (u, v, p, rho, T)
    ns_solver_params_t params;       // Solver parameters (dt, CFL, viscosity, etc.)
    ns_solver_t* solver;             // Active solver (NULL = default solver)
    ns_solver_registry_t* registry;  // Solver registry (context-bound)
    ns_solver_stats_t last_stats;    // Statistics from last solve step
    output_registry* outputs;   // Registered output configurations (internal)
    char* run_prefix;           // Run directory prefix (NULL = default "sim")
    double current_time;        // Current simulation time (accumulated dt)
    char output_base_dir[512];  // Base output directory
} simulation_data;

//=============================================================================
// INITIALIZATION & CLEANUP
//=============================================================================

// Initialize simulation with default solver
CFD_LIBRARY_EXPORT simulation_data* init_simulation(size_t nx, size_t ny, double xmin, double xmax,
                                                    double ymin, double ymax);

// Initialize simulation with specific solver type
CFD_LIBRARY_EXPORT simulation_data* init_simulation_with_solver(size_t nx, size_t ny, double xmin,
                                                                double xmax, double ymin,
                                                                double ymax,
                                                                const char* solver_type);

// Free all simulation resources
CFD_LIBRARY_EXPORT void free_simulation(simulation_data* sim_data);

//=============================================================================
// SOLVER MANAGEMENT
//=============================================================================

// Set solver for existing simulation (takes ownership)
CFD_LIBRARY_EXPORT void simulation_set_solver(simulation_data* sim_data, ns_solver_t* solver);

// Set solver by type name (e.g., "explicit_euler", "projection")
CFD_LIBRARY_EXPORT int simulation_set_solver_by_name(simulation_data* sim_data,
                                                     const char* solver_type);

// Get current solver (returns NULL if using default)
CFD_LIBRARY_EXPORT ns_solver_t* simulation_get_solver(simulation_data* sim_data);

// List all available solver types
CFD_LIBRARY_EXPORT int simulation_list_solvers(const char** names, int max_count);

// Check if specific solver type is available
CFD_LIBRARY_EXPORT int simulation_has_solver(const char* solver_type);

//=============================================================================
// SIMULATION EXECUTION
//=============================================================================

// Run a single simulation time step
CFD_LIBRARY_EXPORT void run_simulation_step(simulation_data* sim_data);

// Run simulation until convergence or max iterations
// Uses the solver's solve() method instead of step()
CFD_LIBRARY_EXPORT void run_simulation_solve(simulation_data* sim_data);

// Get statistics from last solve (iterations, residuals, etc.)
CFD_LIBRARY_EXPORT const ns_solver_stats_t* simulation_get_stats(const simulation_data* sim_data);

//=============================================================================
// OUTPUT CONTROL
//=============================================================================

// Field types for automatic output (registry-based)
typedef enum {
    // VTK outputs (3D visualization)
    OUTPUT_VELOCITY_MAGNITUDE,  // Velocity magnitude scalar field (VTK) - derived from u,v
    OUTPUT_VELOCITY,            // Velocity vector field (VTK)
    OUTPUT_FULL_FIELD,          // Complete flow field (VTK)

    // CSV outputs (data analysis)
    OUTPUT_CSV_TIMESERIES,  // Time series: step, time, max_vel, max_p, residual, etc.
    OUTPUT_CSV_CENTERLINE,  // Centerline profile: x or y, u, v, p along domain center
    OUTPUT_CSV_STATISTICS   // Global statistics: min/max/avg for all fields per step
} output_field_type;

// Output configuration for automatic file generation
typedef struct {
    output_field_type field_type;  // What to output
    int interval;                // Output every N steps (0 = disabled)
    const char* prefix;          // Optional filename prefix (NULL = use field type name)
} output_config;

//=============================================================================
// OUTPUT REGISTRATION (Registry/Injection Pattern)
//=============================================================================

// Register output for automatic generation
// Example: simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "velocity_mag");
//          Automatically writes velocity_mag_000.vtk, velocity_mag_010.vtk, etc.
CFD_LIBRARY_EXPORT void simulation_register_output(simulation_data* sim_data,
                                                   output_field_type field_type, int interval,
                                                   const char* prefix);

// Clear all registered outputs
CFD_LIBRARY_EXPORT void simulation_clear_outputs(simulation_data* sim_data);

// Set base output directory (default: "../../artifacts")
// This is the only manual configuration needed - everything else is automatic
CFD_LIBRARY_EXPORT void simulation_set_output_dir(simulation_data* sim_data, const char* base_dir);

// Set run name prefix (default: "sim")
// Creates directories like: {base_dir}/output/{prefix}_{grid}_{timestamp}/
CFD_LIBRARY_EXPORT void simulation_set_run_prefix(simulation_data* sim_data, const char* prefix);

//=============================================================================
// AUTOMATIC OUTPUT GENERATION
//=============================================================================

// Automatically write all registered outputs for current step
// Handles directory creation, file naming, and output writing internally
CFD_LIBRARY_EXPORT void simulation_write_outputs(simulation_data* sim_data, int step);

#ifdef __cplusplus
}
#endif

#endif  // SIMULATION_API_H
