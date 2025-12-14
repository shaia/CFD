#ifndef CFD_SOLVER_INTERFACE_H
#define CFD_SOLVER_INTERFACE_H

#include "cfd/cfd_export.h"

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// FLOW FIELD TYPES AND PARAMETERS
//=============================================================================

// Default solver parameters
#define DEFAULT_TIME_STEP            0.001   // Default time step size
#define DEFAULT_CFL_NUMBER           0.2     // Default Courant-Friedrichs-Lewy number
#define DEFAULT_GAMMA                1.4     // Default specific heat ratio
#define DEFAULT_VISCOSITY            0.01    // Default dynamic viscosity
#define DEFAULT_THERMAL_CONDUCTIVITY 0.0242  // Default thermal conductivity
#define DEFAULT_MAX_ITERATIONS       100     // Default maximum number of iterations
#define DEFAULT_TOLERANCE            1e-6    // Default convergence tolerance

// Default source term parameters
#define DEFAULT_SOURCE_AMPLITUDE_U 0.1   // Default amplitude of u-velocity source term
#define DEFAULT_SOURCE_AMPLITUDE_V 0.05  // Default amplitude of v-velocity source term
#define DEFAULT_SOURCE_DECAY_RATE  0.1   // Default decay rate for source terms over time
#define DEFAULT_PRESSURE_COUPLING  0.1   // Default coupling coefficient for pressure update

// Flow field structure to store solution variables
typedef struct {
    double* u;    // x-velocity component
    double* v;    // y-velocity component
    double* p;    // pressure
    double* rho;  // density
    double* T;    // temperature
    size_t nx;    // number of points in x-direction
    size_t ny;    // number of points in y-direction
} flow_field;

// Solver parameters
typedef struct {
    double dt;         // time step
    double cfl;        // Courant-Friedrichs-Lewy number
    double gamma;      // specific heat ratio
    double mu;         // viscosity
    double k;          // thermal conductivity
    int max_iter;      // maximum number of iterations
    double tolerance;  // convergence tolerance

    // Source term parameters for energy maintenance
    double source_amplitude_u;  // Amplitude of u-velocity source term
    double source_amplitude_v;  // Amplitude of v-velocity source term
    double source_decay_rate;   // Decay rate for source terms over time
    double pressure_coupling;   // Coupling coefficient for pressure update
} solver_params;

//=============================================================================
// PLUGGABLE SOLVER INTERFACE
//=============================================================================

// Forward declaration
typedef struct Solver solver;

/**
 * Solver capability flags
 * Used to describe what features a solver supports
 */
typedef enum {
    SOLVER_CAP_NONE = 0,
    SOLVER_CAP_INCOMPRESSIBLE = (1 << 0),  // Supports incompressible flow
    SOLVER_CAP_COMPRESSIBLE = (1 << 1),    // Supports compressible flow
    SOLVER_CAP_STEADY_STATE = (1 << 2),    // Supports steady-state solving
    SOLVER_CAP_TRANSIENT = (1 << 3),       // Supports transient solving
    SOLVER_CAP_SIMD = (1 << 4),            // Uses SIMD optimizations
    SOLVER_CAP_PARALLEL = (1 << 5),        // Supports parallel execution
    SOLVER_CAP_GPU = (1 << 6),             // Supports GPU acceleration
} solver_capabilities;

/**
 * Solver status codes - Deprecated, use cfd_status_t
 */
// typedef enum { ... } SolverStatus; // Removed in favor of cfd_status_t

/**
 * Solver statistics - filled after each solve step
 */
typedef struct {
    int iterations;          // Number of iterations performed
    double residual;         // Final residual norm
    double max_velocity;     // Maximum velocity magnitude
    double max_pressure;     // Maximum pressure
    double cfl_number;       // Actual CFL number used
    double elapsed_time_ms;  // Wall clock time for solve
    cfd_status_t status;     // Status of the solve
} solver_stats;

/**
 * Solver context - opaque pointer for solver-specific data
 * Each solver implementation can define its own context structure
 */
typedef void* solver_context;

/**
 * Function pointer types for solver operations
 */

// Initialize solver context (allocate internal buffers, etc.)
// Returns CFD_SUCCESS on success, or error code on failure (e.g., CFD_ERROR_NOMEM)
typedef cfd_status_t (*solver_init_func)(solver* solver, const grid* grid,
                                         const solver_params* params);

// Destroy solver context (free internal buffers)
typedef void (*solver_destroy_func)(solver* solver);

// Perform one time step
// Returns CFD_SUCCESS on success, CFD_ERROR_DIVERGED if solution diverges,
// or error code on failure
typedef cfd_status_t (*solver_step_func)(solver* solver, flow_field* field, const grid* grid,
                                         const solver_params* params, solver_stats* stats);

// Perform multiple iterations until convergence or max_iter
// Returns CFD_SUCCESS on success, CFD_ERROR_DIVERGED if solution diverges,
// or error code on failure
typedef cfd_status_t (*solver_solve_func)(solver* solver, flow_field* field, const grid* grid,
                                          const solver_params* params, solver_stats* stats);

// Apply boundary conditions (can be overridden by specific solvers)
typedef void (*solver_boundary_func)(solver* solver, flow_field* field, const grid* grid);

// Compute stable time step based on CFL condition
typedef double (*solver_compute_dt_func)(solver* solver, const flow_field* field, const grid* grid,
                                         const solver_params* params);

// Get solver name
typedef const char* (*solver_get_name_func)(const solver* solver);

// Get solver description
typedef const char* (*solver_get_description_func)(const solver* solver);

// Get solver capabilities
typedef solver_capabilities (*solver_get_capabilities_func)(const solver* solver);

/**
 * Solver interface structure
 * This is the main polymorphic solver type
 */
struct Solver {
    // Identification
    const char* name;
    const char* description;
    const char* version;
    solver_capabilities capabilities;

    // Solver-specific context (internal state, buffers, etc.)
    solver_context context;

    // Function pointers for solver operations
    solver_init_func init;
    solver_destroy_func destroy;
    solver_step_func step;
    solver_solve_func solve;
    solver_boundary_func apply_boundary;
    solver_compute_dt_func compute_dt;

    // Optional metadata functions
    solver_get_name_func get_name;
    solver_get_description_func get_description;
    solver_get_capabilities_func get_capabilities;
};

/**
 * Solver creation and management functions
 */

// Opaque handle for solver registry
typedef struct SolverRegistry solver_registry;

/**
 * Registry Management
 */

// Create a new solver registry
CFD_LIBRARY_EXPORT solver_registry* cfd_registry_create(void);

// Destroy a solver registry
CFD_LIBRARY_EXPORT void cfd_registry_destroy(solver_registry* registry);

// Register default built-in solvers
CFD_LIBRARY_EXPORT void cfd_registry_register_defaults(solver_registry* registry);

/**
 * Solver Creation
 */

// Create a new solver instance from the registry
CFD_LIBRARY_EXPORT solver* cfd_solver_create(solver_registry* registry, const char* type_name);

// Destroy a solver and free all resources
CFD_LIBRARY_EXPORT void solver_destroy(solver* solver);

// Initialize a solver for a specific grid configuration
CFD_LIBRARY_EXPORT cfd_status_t solver_init(solver* solver, const grid* grid,
                                            const solver_params* params);

// Perform a single time step
CFD_LIBRARY_EXPORT cfd_status_t solver_step(solver* solver, flow_field* field, const grid* grid,
                                            const solver_params* params, solver_stats* stats);

// Solve until convergence or max iterations
CFD_LIBRARY_EXPORT cfd_status_t solver_solve(solver* solver, flow_field* field, const grid* grid,
                                             const solver_params* params, solver_stats* stats);

// Apply boundary conditions
CFD_LIBRARY_EXPORT void solver_apply_boundary(solver* solver, flow_field* field, const grid* grid);

// Compute stable time step
CFD_LIBRARY_EXPORT double solver_compute_dt(solver* solver, const flow_field* field,
                                            const grid* grid, const solver_params* params);

/**
 * Registry Operations
 */

// Solver factory function type - creates a new solver instance
typedef solver* (*solver_factory_func)(void);

// Register a new solver type
CFD_LIBRARY_EXPORT int cfd_registry_register(solver_registry* registry, const char* type_name,
                                             solver_factory_func factory);

// Unregister a solver type
CFD_LIBRARY_EXPORT int cfd_registry_unregister(solver_registry* registry, const char* type_name);

// Get list of available solver types (returns count, fills names array)
CFD_LIBRARY_EXPORT int cfd_registry_list(solver_registry* registry, const char** names,
                                         int max_count);

// Check if a solver type is available
CFD_LIBRARY_EXPORT int cfd_registry_has(solver_registry* registry, const char* type_name);

// Get description for a solver type
CFD_LIBRARY_EXPORT const char* cfd_registry_get_description(solver_registry* registry,
                                                            const char* type_name);

/**
 * Standard Built-in Solver Types
 */
#define SOLVER_TYPE_EXPLICIT_EULER           "explicit_euler"
#define SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED "explicit_euler_optimized"
#define SOLVER_TYPE_EXPLICIT_EULER_OMP       "explicit_euler_omp"
#define SOLVER_TYPE_EXPLICIT_EULER_GPU       "explicit_euler_gpu"
#define SOLVER_TYPE_PROJECTION               "projection"
#define SOLVER_TYPE_PROJECTION_OPTIMIZED     "projection_optimized"
#define SOLVER_TYPE_PROJECTION_OMP           "projection_omp"
#define SOLVER_TYPE_PROJECTION_JACOBI_GPU    "projection_jacobi_gpu"

/**
 * Helper to initialize SolverStats with default values
 */
static inline solver_stats solver_stats_default(void) {
    solver_stats stats = {.iterations = 0,
                          .residual = 0.0,
                          .max_velocity = 0.0,
                          .max_pressure = 0.0,
                          .cfl_number = 0.0,
                          .elapsed_time_ms = 0.0,
                          .status = CFD_SUCCESS};
    return stats;
}

//=============================================================================
// FLOW FIELD UTILITY FUNCTIONS
// Low-level functions for flow field management and operations
//=============================================================================

// Flow field memory management
CFD_LIBRARY_EXPORT flow_field* flow_field_create(size_t nx, size_t ny);
CFD_LIBRARY_EXPORT void flow_field_destroy(flow_field* field);

// Flow field initialization and operations
CFD_LIBRARY_EXPORT void initialize_flow_field(flow_field* field, const grid* grid);
CFD_LIBRARY_EXPORT void apply_boundary_conditions(flow_field* field, const grid* grid);

// Source term computation
CFD_LIBRARY_EXPORT void compute_source_terms(double x, double y, int iter, double dt,
                                             const solver_params* params, double* source_u,
                                             double* source_v);

// Time step computation
CFD_LIBRARY_EXPORT void compute_time_step(flow_field* field, const grid* grid,
                                          solver_params* params);

// Helper function to initialize SolverParams with default values
CFD_LIBRARY_EXPORT solver_params solver_params_default(void);

#ifdef __cplusplus
}
#endif

#endif  // CFD_SOLVER_INTERFACE_H
