/**
 * @file navier_stokes_solver.h
 * @brief Navier-Stokes flow solver interface
 *
 * This module provides a pluggable interface for time-stepping solvers
 * of the incompressible Navier-Stokes equations:
 *   - Momentum: du/dt + (u.nabla)u = -nabla(p) + nu*nabla^2(u)
 *   - Continuity: nabla.u = 0
 *
 * Features:
 * - Multiple solver methods (Explicit Euler, Projection)
 * - Multiple backends (Scalar, SIMD, OpenMP, GPU)
 * - Pluggable solver registry
 * - Statistics reporting
 */

#ifndef CFD_NAVIER_STOKES_SOLVER_H
#define CFD_NAVIER_STOKES_SOLVER_H

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

/**
 * Flow field structure to store solution variables
 */
typedef struct {
    double* u;    /**< x-velocity component */
    double* v;    /**< y-velocity component */
    double* p;    /**< pressure */
    double* rho;  /**< density */
    double* T;    /**< temperature */
    size_t nx;    /**< number of points in x-direction */
    size_t ny;    /**< number of points in y-direction */
} flow_field;

/**
 * Navier-Stokes solver parameters
 */
typedef struct {
    double dt;         /**< time step */
    double cfl;        /**< Courant-Friedrichs-Lewy number */
    double gamma;      /**< specific heat ratio */
    double mu;         /**< viscosity */
    double k;          /**< thermal conductivity */
    int max_iter;      /**< maximum number of iterations */
    double tolerance;  /**< convergence tolerance */

    /* Source term parameters for energy maintenance */
    double source_amplitude_u;  /**< Amplitude of u-velocity source term */
    double source_amplitude_v;  /**< Amplitude of v-velocity source term */
    double source_decay_rate;   /**< Decay rate for source terms over time */
    double pressure_coupling;   /**< Coupling coefficient for pressure update */
} ns_solver_params_t;


//=============================================================================
// PLUGGABLE SOLVER INTERFACE
//=============================================================================

/* Forward declaration */
typedef struct NSSolver ns_solver_t;

/**
 * NSSolver capability flags
 * Used to describe what features a solver supports
 */
typedef enum {
    NS_SOLVER_CAP_NONE = 0,
    NS_SOLVER_CAP_INCOMPRESSIBLE = (1 << 0),  /**< Supports incompressible flow */
    NS_SOLVER_CAP_COMPRESSIBLE = (1 << 1),    /**< Supports compressible flow */
    NS_SOLVER_CAP_STEADY_STATE = (1 << 2),    /**< Supports steady-state solving */
    NS_SOLVER_CAP_TRANSIENT = (1 << 3),       /**< Supports transient solving */
    NS_SOLVER_CAP_SIMD = (1 << 4),            /**< Uses SIMD optimizations */
    NS_SOLVER_CAP_PARALLEL = (1 << 5),        /**< Supports parallel execution */
    NS_SOLVER_CAP_GPU = (1 << 6),             /**< Supports GPU acceleration */
} ns_solver_capabilities_t;


/**
 * NSSolver statistics - filled after each solve step
 */
typedef struct {
    int iterations;          /**< Number of iterations performed */
    double residual;         /**< Final residual norm */
    double max_velocity;     /**< Maximum velocity magnitude */
    double max_pressure;     /**< Maximum pressure */
    double cfl_number;       /**< Actual CFL number used */
    double elapsed_time_ms;  /**< Wall clock time for solve */
    cfd_status_t status;     /**< Status of the solve */
} ns_solver_stats_t;

/**
 * NSSolver context - opaque pointer for solver-specific data
 * Each solver implementation can define its own context structure
 */
typedef void* ns_solver_context_t;

/**
 * Function pointer types for solver operations
 */

/** Initialize solver context (allocate internal buffers, etc.) */
typedef cfd_status_t (*ns_solver_init_func)(ns_solver_t* solver, const grid* grid,
                                            const ns_solver_params_t* params);

/** Destroy solver context (free internal buffers) */
typedef void (*ns_solver_destroy_func)(ns_solver_t* solver);

/** Perform one time step */
typedef cfd_status_t (*ns_solver_step_func)(ns_solver_t* solver, flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params, ns_solver_stats_t* stats);

/** Perform multiple iterations until convergence or max_iter */
typedef cfd_status_t (*ns_solver_solve_func)(ns_solver_t* solver, flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params, ns_solver_stats_t* stats);

/** Apply boundary conditions (can be overridden by specific solvers) */
typedef void (*ns_solver_boundary_func)(ns_solver_t* solver, flow_field* field, const grid* grid);

/** Compute stable time step based on CFL condition */
typedef double (*ns_solver_compute_dt_func)(ns_solver_t* solver, const flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params);

/** Get solver name */
typedef const char* (*ns_solver_get_name_func)(const ns_solver_t* solver);

/** Get solver description */
typedef const char* (*ns_solver_get_description_func)(const ns_solver_t* solver);

/** Get solver capabilities */
typedef ns_solver_capabilities_t (*ns_solver_get_capabilities_func)(const ns_solver_t* solver);

/**
 * Navier-Stokes solver interface structure
 * This is the main polymorphic solver type
 */
struct NSSolver {
    /* Identification */
    const char* name;
    const char* description;
    const char* version;
    ns_solver_capabilities_t capabilities;

    /* NSSolver-specific context (internal state, buffers, etc.) */
    ns_solver_context_t context;

    /* Function pointers for solver operations */
    ns_solver_init_func init;
    ns_solver_destroy_func destroy;
    ns_solver_step_func step;
    ns_solver_solve_func solve;
    ns_solver_boundary_func apply_boundary;
    ns_solver_compute_dt_func compute_dt;

    /* Optional metadata functions */
    ns_solver_get_name_func get_name;
    ns_solver_get_description_func get_description;
    ns_solver_get_capabilities_func get_capabilities;
};

/**
 * NSSolver creation and management functions
 */

/* Opaque handle for solver registry */
typedef struct NSSolverRegistry ns_solver_registry_t;

/**
 * Registry Management
 */

/** Create a new solver registry */
CFD_LIBRARY_EXPORT ns_solver_registry_t* cfd_registry_create(void);

/** Destroy a solver registry */
CFD_LIBRARY_EXPORT void cfd_registry_destroy(ns_solver_registry_t* registry);

/** Register default built-in solvers */
CFD_LIBRARY_EXPORT void cfd_registry_register_defaults(ns_solver_registry_t* registry);

/**
 * NSSolver Creation
 */

/** Create a new solver instance from the registry */
CFD_LIBRARY_EXPORT ns_solver_t* cfd_solver_create(ns_solver_registry_t* registry, const char* type_name);

/** Destroy a solver and free all resources */
CFD_LIBRARY_EXPORT void solver_destroy(ns_solver_t* solver);

/** Initialize a solver for a specific grid configuration */
CFD_LIBRARY_EXPORT cfd_status_t solver_init(ns_solver_t* solver, const grid* grid,
                                            const ns_solver_params_t* params);

/** Perform a single time step */
CFD_LIBRARY_EXPORT cfd_status_t solver_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params, ns_solver_stats_t* stats);

/** Solve until convergence or max iterations */
CFD_LIBRARY_EXPORT cfd_status_t solver_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params, ns_solver_stats_t* stats);

/** Apply boundary conditions */
CFD_LIBRARY_EXPORT void solver_apply_boundary(ns_solver_t* solver, flow_field* field, const grid* grid);

/** Compute stable time step */
CFD_LIBRARY_EXPORT double solver_compute_dt(ns_solver_t* solver, const flow_field* field,
                                            const grid* grid, const ns_solver_params_t* params);

/**
 * Registry Operations
 */

/** NSSolver factory function type - creates a new solver instance */
typedef ns_solver_t* (*ns_solver_factory_func)(void);

/** Register a new solver type */
CFD_LIBRARY_EXPORT int cfd_registry_register(ns_solver_registry_t* registry, const char* type_name,
                                             ns_solver_factory_func factory);

/** Unregister a solver type */
CFD_LIBRARY_EXPORT int cfd_registry_unregister(ns_solver_registry_t* registry, const char* type_name);

/** Get list of available solver types (returns count, fills names array) */
CFD_LIBRARY_EXPORT int cfd_registry_list(ns_solver_registry_t* registry, const char** names,
                                         int max_count);

/** Check if a solver type is available */
CFD_LIBRARY_EXPORT int cfd_registry_has(ns_solver_registry_t* registry, const char* type_name);

/** Get description for a solver type */
CFD_LIBRARY_EXPORT const char* cfd_registry_get_description(ns_solver_registry_t* registry,
                                                            const char* type_name);

/**
 * Standard Built-in NSSolver Types
 */
#define NS_SOLVER_TYPE_EXPLICIT_EULER           "explicit_euler"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED "explicit_euler_optimized"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_OMP       "explicit_euler_omp"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_GPU       "explicit_euler_gpu"
#define NS_SOLVER_TYPE_PROJECTION               "projection"
#define NS_SOLVER_TYPE_PROJECTION_OPTIMIZED     "projection_optimized"
#define NS_SOLVER_TYPE_PROJECTION_OMP           "projection_omp"
#define NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU    "projection_jacobi_gpu"

/**
 * Helper to initialize solver stats with default values
 */
static inline ns_solver_stats_t ns_solver_stats_default(void) {
    /* Use individual assignments for NVCC compatibility */
    ns_solver_stats_t stats;
    stats.iterations = 0;
    stats.residual = 0.0;
    stats.max_velocity = 0.0;
    stats.max_pressure = 0.0;
    stats.cfl_number = 0.0;
    stats.elapsed_time_ms = 0.0;
    stats.status = CFD_SUCCESS;
    return stats;
}

//=============================================================================
// FLOW FIELD UTILITY FUNCTIONS
// Low-level functions for flow field management and operations
//=============================================================================

/** Flow field memory management */
CFD_LIBRARY_EXPORT flow_field* flow_field_create(size_t nx, size_t ny);
CFD_LIBRARY_EXPORT void flow_field_destroy(flow_field* field);

/** Flow field initialization and operations */
CFD_LIBRARY_EXPORT void initialize_flow_field(flow_field* field, const grid* grid);
CFD_LIBRARY_EXPORT void apply_boundary_conditions(flow_field* field, const grid* grid);

/** Source term computation */
CFD_LIBRARY_EXPORT void compute_source_terms(double x, double y, int iter, double dt,
                                             const ns_solver_params_t* params, double* source_u,
                                             double* source_v);

/** Time step computation */
CFD_LIBRARY_EXPORT void compute_time_step(flow_field* field, const grid* grid,
                                          ns_solver_params_t* params);

/** Helper function to initialize solver params with default values */
CFD_LIBRARY_EXPORT ns_solver_params_t ns_solver_params_default(void);

#ifdef __cplusplus
}
#endif

#endif  /* CFD_NAVIER_STOKES_SOLVER_H */
