/**
 * @file poisson_solver.h
 * @brief Poisson pressure equation solver interface
 *
 * This module provides a pluggable interface for iterative Poisson solvers,
 * used for solving the pressure Poisson equation in pressure projection:
 *   nabla^2 p = rhs  (where rhs is typically divergence of intermediate velocity)
 *
 * Features:
 * - Multiple solver methods (Jacobi, SOR, Red-Black SOR)
 * - Multiple backends (Scalar, SIMD, OpenMP, GPU)
 * - Runtime configurable parameters (tolerance, max_iter, omega)
 * - Statistics reporting (iterations, residual, timing)
 * - Convenience functions for common use cases
 *
 * Usage:
 * @code
 * poisson_solver_t* solver = poisson_solver_create(
 *     POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_AUTO);
 *
 * poisson_solver_params_t params = poisson_solver_params_default();
 * params.tolerance = 1e-8;
 *
 * poisson_solver_init(solver, nx, ny, dx, dy, &params);
 *
 * poisson_solver_stats_t stats = poisson_solver_stats_default();
 * poisson_solver_solve(solver, p, p_temp, rhs, &stats);
 *
 * poisson_solver_destroy(solver);
 * @endcode
 */

#ifndef CFD_POISSON_SOLVER_H
#define CFD_POISSON_SOLVER_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TYPE ENUMERATIONS
 * ============================================================================ */

/**
 * Poisson solver method types
 */
typedef enum {
    POISSON_METHOD_JACOBI,        /**< Jacobi iteration (fully parallelizable) */
    POISSON_METHOD_GAUSS_SEIDEL,  /**< Gauss-Seidel iteration */
    POISSON_METHOD_SOR,           /**< Successive Over-Relaxation */
    POISSON_METHOD_REDBLACK_SOR,  /**< Red-Black SOR (parallelizable) */
    POISSON_METHOD_CG,            /**< Conjugate Gradient (for SPD systems) */
    POISSON_METHOD_BICGSTAB,      /**< BiCGSTAB (future) */
    POISSON_METHOD_MULTIGRID      /**< Multigrid (future) */
} poisson_solver_method_t;

/**
 * Poisson solver backend types (execution strategy)
 */
typedef enum {
    POISSON_BACKEND_AUTO,     /**< Auto-select best available */
    POISSON_BACKEND_SCALAR,   /**< Scalar CPU implementation */
    POISSON_BACKEND_OMP,      /**< OpenMP parallelized */
    POISSON_BACKEND_SIMD,     /**< SIMD + OpenMP with runtime detection (AVX2/NEON) */
    POISSON_BACKEND_GPU       /**< CUDA GPU (future) */
} poisson_solver_backend_t;

/**
 * Convergence status
 */
typedef enum {
    POISSON_CONVERGED = 0,   /**< Converged within tolerance */
    POISSON_MAX_ITER = 1,    /**< Reached max iterations without converging */
    POISSON_DIVERGED = 2,    /**< Solution diverged (residual increased) */
    POISSON_STAGNATED = 3,   /**< Residual stagnated (no progress) */
    POISSON_ERROR = -1       /**< Error occurred */
} poisson_solver_status_t;

/**
 * Preconditioner types for iterative solvers
 */
typedef enum {
    POISSON_PRECOND_NONE = 0,   /**< No preconditioning (default) */
    POISSON_PRECOND_JACOBI = 1  /**< Diagonal (Jacobi) preconditioning */
} poisson_precond_type_t;

/* ============================================================================
 * PARAMETERS AND STATISTICS
 * ============================================================================ */

/**
 * Parameters for Poisson solver configuration
 */
typedef struct {
    double tolerance;          /**< Relative convergence tolerance (default: 1e-6) */
    double absolute_tolerance; /**< Absolute tolerance (default: 1e-10) */
    int max_iterations;        /**< Maximum iterations (default: 1000) */
    double omega;              /**< SOR relaxation parameter (default: 1.5, range: 1.0-2.0) */
    int check_interval;        /**< Check convergence every N iterations (default: 1) */
    bool verbose;              /**< Print iteration progress (default: false) */
    poisson_precond_type_t preconditioner; /**< Preconditioner type (default: NONE) */
} poisson_solver_params_t;

/**
 * Statistics from a Poisson solve operation
 */
typedef struct {
    poisson_solver_status_t status; /**< Convergence status */
    int iterations;                 /**< Iterations performed */
    double initial_residual;        /**< Residual at start */
    double final_residual;          /**< Residual at end */
    double elapsed_time_ms;         /**< Wall clock time in milliseconds */
} poisson_solver_stats_t;

/**
 * Initialize parameters with default values
 *
 * Default values:
 * - tolerance: 1e-6
 * - absolute_tolerance: 1e-10
 * - max_iterations: 1000
 * - omega: 1.5
 * - check_interval: 1
 * - verbose: false
 */
CFD_LIBRARY_EXPORT poisson_solver_params_t poisson_solver_params_default(void);

/**
 * Initialize statistics with default values
 */
CFD_LIBRARY_EXPORT poisson_solver_stats_t poisson_solver_stats_default(void);

/* ============================================================================
 * POISSON SOLVER INTERFACE
 * ============================================================================ */

/* Forward declaration */
typedef struct poisson_solver poisson_solver_t;

/* Opaque context for solver-specific data */
typedef void* poisson_solver_context_t;

/**
 * Function pointer types for Poisson solver operations
 */

/** Initialize solver for given problem size */
typedef cfd_status_t (*poisson_solver_init_func)(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params);

/** Destroy solver and free resources */
typedef void (*poisson_solver_destroy_func)(poisson_solver_t* solver);

/**
 * Solve the Poisson equation: nabla^2 x = rhs
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector (in/out, initial guess on input)
 * @param x_temp Temporary buffer (required for Jacobi, optional for SOR)
 * @param rhs Right-hand side vector
 * @param stats Output statistics (may be NULL)
 */
typedef cfd_status_t (*poisson_solver_solve_func)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats);

/**
 * Perform a single iteration
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector (in/out)
 * @param x_temp Temporary buffer
 * @param rhs Right-hand side vector
 * @param residual Output: current residual norm (may be NULL)
 */
typedef cfd_status_t (*poisson_solver_iterate_func)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual);

/** Apply boundary conditions to solution */
typedef void (*poisson_solver_apply_bc_func)(
    poisson_solver_t* solver,
    double* x);

/**
 * Poisson solver interface structure
 *
 * Follows the pattern from navier_stokes_solver.h with function pointers
 * for polymorphic dispatch.
 */
struct poisson_solver {
    /* Identification */
    const char* name;                     /**< Solver name (e.g., "jacobi_simd") */
    const char* description;              /**< Human-readable description */
    poisson_solver_method_t method;       /**< Poisson solver method type */
    poisson_solver_backend_t backend;     /**< Backend type */

    /* Problem parameters (set during init) */
    size_t nx;                            /**< Grid points in x */
    size_t ny;                            /**< Grid points in y */
    size_t nz;                            /**< Grid points in z (1 for 2D) */
    double dx;                            /**< Grid spacing in x */
    double dy;                            /**< Grid spacing in y */
    double dz;                            /**< Grid spacing in z (0.0 for 2D) */

    /* Solver parameters */
    poisson_solver_params_t params;       /**< Solver configuration */

    /* Internal context */
    poisson_solver_context_t context;     /**< Solver-specific state */

    /* Function pointers */
    poisson_solver_init_func init;        /**< Initialize solver */
    poisson_solver_destroy_func destroy;  /**< Destroy solver */
    poisson_solver_solve_func solve;      /**< Solve to convergence */
    poisson_solver_iterate_func iterate;  /**< Single iteration */
    poisson_solver_apply_bc_func apply_bc; /**< Apply boundary conditions */
};

/* ============================================================================
 * SOLVER LIFECYCLE
 * ============================================================================ */

/**
 * Create a Poisson solver by method and backend
 *
 * @param method Solver method (Jacobi, SOR, Red-Black SOR, etc.)
 * @param backend Backend type (AUTO, SCALAR, SIMD, OMP, GPU)
 * @return New solver instance, or NULL on error
 *
 * If backend is AUTO, the best available backend is selected:
 * 1. SIMD if available (AVX2)
 * 2. Scalar otherwise
 */
CFD_LIBRARY_EXPORT poisson_solver_t* poisson_solver_create(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend);

/**
 * Initialize solver for a specific problem
 *
 * @param solver Poisson solver instance from poisson_solver_create()
 * @param nx Grid points in x direction
 * @param ny Grid points in y direction
 * @param dx Grid spacing in x direction
 * @param dy Grid spacing in y direction
 * @param params Solver parameters (NULL for defaults)
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t poisson_solver_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params);

/**
 * Destroy solver and free all resources
 *
 * @param solver Poisson solver to destroy (NULL is safe)
 */
CFD_LIBRARY_EXPORT void poisson_solver_destroy(poisson_solver_t* solver);

/* ============================================================================
 * SOLVER OPERATIONS
 * ============================================================================ */

/**
 * Solve the Poisson equation: nabla^2 x = rhs
 *
 * Iterates until convergence or max_iterations is reached.
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector (in/out, initial guess on input)
 * @param x_temp Temporary buffer (required for Jacobi, may be NULL for SOR)
 * @param rhs Right-hand side vector
 * @param stats Output statistics (may be NULL)
 * @return CFD_SUCCESS on convergence, CFD_ERROR_MAX_ITER if not converged
 */
CFD_LIBRARY_EXPORT cfd_status_t poisson_solver_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats);

/**
 * Perform a single iteration
 *
 * Useful for custom iteration control or monitoring.
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector (in/out)
 * @param x_temp Temporary buffer
 * @param rhs Right-hand side vector
 * @param residual Output: current residual norm (may be NULL)
 * @return CFD_SUCCESS on success
 */
CFD_LIBRARY_EXPORT cfd_status_t poisson_solver_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual);

/**
 * Compute residual norm: ||nabla^2 x - rhs||_inf
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector
 * @param rhs Right-hand side vector
 * @return Maximum absolute residual
 */
CFD_LIBRARY_EXPORT double poisson_solver_compute_residual(
    poisson_solver_t* solver,
    const double* x,
    const double* rhs);

/**
 * Apply boundary conditions to solution
 *
 * @param solver Initialized Poisson solver
 * @param x Solution vector
 */
CFD_LIBRARY_EXPORT void poisson_solver_apply_bc(
    poisson_solver_t* solver,
    double* x);

/* ============================================================================
 * BACKEND SELECTION
 * ============================================================================ */

/**
 * Get the currently active default backend
 */
CFD_LIBRARY_EXPORT poisson_solver_backend_t poisson_solver_get_backend(void);

/**
 * Get the name of the current backend
 *
 * @return Backend name string ("auto", "scalar", "simd", "omp", "gpu")
 */
CFD_LIBRARY_EXPORT const char* poisson_solver_get_backend_name(void);

/**
 * Set the default backend for new solvers
 *
 * @param backend Backend to use
 * @return true if backend is available and was set
 */
CFD_LIBRARY_EXPORT bool poisson_solver_set_backend(poisson_solver_backend_t backend);

/**
 * Check if a backend is available
 *
 * @param backend Backend to check
 * @return true if backend is available
 */
CFD_LIBRARY_EXPORT bool poisson_solver_backend_available(poisson_solver_backend_t backend);

/* ============================================================================
 * SOLVER TYPE NAMES
 * ============================================================================ */

#define POISSON_SOLVER_TYPE_JACOBI_SCALAR     "jacobi_scalar"
#define POISSON_SOLVER_TYPE_JACOBI_SIMD       "jacobi_simd"
#define POISSON_SOLVER_TYPE_SOR_SCALAR        "sor_scalar"
#define POISSON_SOLVER_TYPE_REDBLACK_SCALAR   "redblack_scalar"
#define POISSON_SOLVER_TYPE_REDBLACK_OMP      "redblack_omp"
#define POISSON_SOLVER_TYPE_REDBLACK_SIMD     "redblack_simd"
#define POISSON_SOLVER_TYPE_CG_SCALAR         "cg_scalar"
#define POISSON_SOLVER_TYPE_CG_OMP            "cg_omp"
#define POISSON_SOLVER_TYPE_CG_SIMD           "cg_simd"
#define POISSON_SOLVER_TYPE_BICGSTAB_SCALAR   "bicgstab_scalar"
#define POISSON_SOLVER_TYPE_BICGSTAB_SIMD     "bicgstab_simd"

/* ============================================================================
 * CONVENIENCE API
 * ============================================================================ */

/**
 * Poisson solver type presets
 *
 * Convenience enum for common solver configurations.
 * All SIMD backends use runtime CPU detection (AVX2/NEON).
 */
typedef enum {
    POISSON_SOLVER_SOR_SCALAR = 0,     /**< SOR method with scalar backend */
    POISSON_SOLVER_JACOBI_SIMD = 1,    /**< Jacobi method with SIMD backend (runtime detection) */
    POISSON_SOLVER_REDBLACK_SIMD = 2,  /**< Red-Black SOR with SIMD backend (runtime detection) */
    POISSON_SOLVER_REDBLACK_OMP = 3,   /**< Red-Black SOR with OpenMP backend */
    POISSON_SOLVER_REDBLACK_SCALAR = 4, /**< Red-Black SOR with scalar backend (always available) */
    POISSON_SOLVER_CG_SCALAR = 5,      /**< Conjugate Gradient with scalar backend (always available) */
    POISSON_SOLVER_CG_SIMD = 6,        /**< Conjugate Gradient with SIMD backend (runtime detection) */
    POISSON_SOLVER_CG_OMP = 7          /**< Conjugate Gradient with OpenMP backend */
} poisson_solver_type;

/** Default Poisson solver - uses runtime SIMD detection */
#define DEFAULT_POISSON_SOLVER POISSON_SOLVER_REDBLACK_SIMD

/**
 * Unified Poisson solver function
 *
 * Convenience function that internally uses the poisson_solver interface.
 *
 * @param p Pressure field (in/out)
 * @param p_temp Temporary buffer
 * @param rhs Right-hand side (divergence)
 * @param nx Grid points in x
 * @param ny Grid points in y
 * @param dx Grid spacing in x
 * @param dy Grid spacing in y
 * @param solver_type Solver type preset
 * @return Number of iterations if converged, -1 if not converged
 */
CFD_LIBRARY_EXPORT int poisson_solve(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy,
    poisson_solver_type solver_type);

/**
 * Direct solver functions
 *
 * These provide direct access to specific solver implementations.
 */
CFD_LIBRARY_EXPORT int poisson_solve_sor_scalar(
    double* p, const double* rhs,
    size_t nx, size_t ny, double dx, double dy);

/**
 * SIMD solver functions with runtime CPU detection (AVX2/NEON)
 *
 * These functions automatically select the best SIMD implementation
 * (AVX2 on x86-64, NEON on ARM64) at runtime.
 */
CFD_LIBRARY_EXPORT int poisson_solve_jacobi_simd(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy);

CFD_LIBRARY_EXPORT int poisson_solve_redblack_simd(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy);

/**
 * Check if SIMD backend is available at runtime
 *
 * @return true if SIMD (AVX2 or NEON) with OpenMP is available
 */
CFD_LIBRARY_EXPORT bool poisson_solver_simd_available(void);

/**
 * Get the name of the detected SIMD architecture for solvers
 *
 * @return "avx2", "neon", or "none"
 */
CFD_LIBRARY_EXPORT const char* poisson_solver_get_simd_arch_name(void);

#ifdef __cplusplus
}
#endif

#endif /* CFD_POISSON_SOLVER_H */
