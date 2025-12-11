#ifndef CFD_SOLVER_INTERFACE_H
#define CFD_SOLVER_INTERFACE_H

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"


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
} FlowField;

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
} SolverParams;

//=============================================================================
// PLUGGABLE SOLVER INTERFACE
//=============================================================================

// Forward declaration
typedef struct Solver Solver;

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
} SolverCapabilities;

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
} SolverStats;

/**
 * Solver context - opaque pointer for solver-specific data
 * Each solver implementation can define its own context structure
 */
typedef void* SolverContext;

/**
 * Function pointer types for solver operations
 */

// Initialize solver context (allocate internal buffers, etc.)
// Returns CFD_SUCCESS on success, or error code on failure (e.g., CFD_ERROR_NOMEM)
typedef cfd_status_t (*SolverInitFunc)(Solver* solver, const Grid* grid,
                                       const SolverParams* params);

// Destroy solver context (free internal buffers)
typedef void (*SolverDestroyFunc)(Solver* solver);

// Perform one time step
// Returns CFD_SUCCESS on success, CFD_ERROR_DIVERGED if solution diverges,
// or error code on failure
typedef cfd_status_t (*SolverStepFunc)(Solver* solver, FlowField* field, const Grid* grid,
                                       const SolverParams* params, SolverStats* stats);

// Perform multiple iterations until convergence or max_iter
// Returns CFD_SUCCESS on success, CFD_ERROR_DIVERGED if solution diverges,
// or error code on failure
typedef cfd_status_t (*SolverSolveFunc)(Solver* solver, FlowField* field, const Grid* grid,
                                        const SolverParams* params, SolverStats* stats);

// Apply boundary conditions (can be overridden by specific solvers)
typedef void (*SolverBoundaryFunc)(Solver* solver, FlowField* field, const Grid* grid);

// Compute stable time step based on CFL condition
typedef double (*SolverComputeDtFunc)(Solver* solver, const FlowField* field, const Grid* grid,
                                      const SolverParams* params);

// Get solver name
typedef const char* (*SolverGetNameFunc)(const Solver* solver);

// Get solver description
typedef const char* (*SolverGetDescriptionFunc)(const Solver* solver);

// Get solver capabilities
typedef SolverCapabilities (*SolverGetCapabilitiesFunc)(const Solver* solver);

/**
 * Solver interface structure
 * This is the main polymorphic solver type
 */
struct Solver {
    // Identification
    const char* name;
    const char* description;
    const char* version;
    SolverCapabilities capabilities;

    // Solver-specific context (internal state, buffers, etc.)
    SolverContext context;

    // Function pointers for solver operations
    SolverInitFunc init;
    SolverDestroyFunc destroy;
    SolverStepFunc step;
    SolverSolveFunc solve;
    SolverBoundaryFunc apply_boundary;
    SolverComputeDtFunc compute_dt;

    // Optional metadata functions
    SolverGetNameFunc get_name;
    SolverGetDescriptionFunc get_description;
    SolverGetCapabilitiesFunc get_capabilities;
};

/**
 * Solver creation and management functions
 */

// Create a new solver by type name (e.g., "explicit_euler", "projection", "simple")
Solver* solver_create(const char* type_name);

// Destroy a solver and free all resources
void solver_destroy(Solver* solver);

// Initialize a solver for a specific grid configuration
cfd_status_t solver_init(Solver* solver, const Grid* grid, const SolverParams* params);

// Perform a single time step
cfd_status_t solver_step(Solver* solver, FlowField* field, const Grid* grid,
                         const SolverParams* params, SolverStats* stats);

// Solve until convergence or max iterations
cfd_status_t solver_solve(Solver* solver, FlowField* field, const Grid* grid,
                          const SolverParams* params, SolverStats* stats);

// Apply boundary conditions
void solver_apply_boundary(Solver* solver, FlowField* field, const Grid* grid);

// Compute stable time step
double solver_compute_dt(Solver* solver, const FlowField* field, const Grid* grid,
                         const SolverParams* params);

/**
 * Solver registry functions
 */

// Solver factory function type - creates a new solver instance
typedef Solver* (*SolverFactoryFunc)(void);

// Register a new solver type
int solver_registry_register(const char* type_name, SolverFactoryFunc factory);

// Unregister a solver type
int solver_registry_unregister(const char* type_name);

// Get list of available solver types (returns count, fills names array)
int solver_registry_list(const char** names, int max_count);

// Check if a solver type is available
int solver_registry_has(const char* type_name);

// Get description for a solver type
const char* solver_registry_get_description(const char* type_name);
// Solver Types
#define SOLVER_TYPE_EXPLICIT_EULER           "explicit_euler"
#define SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED "explicit_euler_optimized"
#define SOLVER_TYPE_PROJECTION               "projection"
#define SOLVER_TYPE_PROJECTION_OPTIMIZED     "projection_optimized"

#define SOLVER_TYPE_EXPLICIT_EULER_GPU    "explicit_euler_gpu"
#define SOLVER_TYPE_PROJECTION_JACOBI_GPU "projection_jacobi_gpu"

#define SOLVER_TYPE_EXPLICIT_EULER_OMP "explicit_euler_omp"
#define SOLVER_TYPE_PROJECTION_OMP     "projection_omp"

// Future solver types (placeholders)
#define SOLVER_TYPE_SIMPLE "simple"
#define SOLVER_TYPE_LBM    "lbm"

/**
 * Initialize all built-in solvers in the registry
 * Call this once at program startup
 */
void solver_registry_init(void);

/**
 * Cleanup the solver registry
 * Call this at program shutdown
 */
void solver_registry_cleanup(void);

/**
 * Helper to initialize SolverStats with default values
 */
static inline SolverStats solver_stats_default(void) {
    SolverStats stats = {.iterations = 0,
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
FlowField* flow_field_create(size_t nx, size_t ny);
void flow_field_destroy(FlowField* field);

// Flow field initialization and operations
void initialize_flow_field(FlowField* field, const Grid* grid);
void apply_boundary_conditions(FlowField* field, const Grid* grid);

// Source term computation
void compute_source_terms(double x, double y, int iter, double dt, const SolverParams* params,
                          double* source_u, double* source_v);

// Time step computation
void compute_time_step(FlowField* field, const Grid* grid, SolverParams* params);

// Helper function to initialize SolverParams with default values
SolverParams solver_params_default(void);

#ifdef __cplusplus
}
#endif

#endif  // CFD_SOLVER_INTERFACE_H
