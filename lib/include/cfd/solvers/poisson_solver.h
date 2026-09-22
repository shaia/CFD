/**
 * @file poisson_solver.h
 * @brief Poisson pressure equation solver interface
 *
 * This module provides a pluggable interface for iterative Poisson solvers,
 * used for solving the pressure Poisson equation in pressure projection:
 *   nabla^2 p = rhs  (where rhs is typically divergence of intermediate velocity)
 *
 * Features:
 * - Stationary methods (Jacobi, Gauss-Seidel, SOR, Red-Black SOR), Krylov
 *   methods (CG, BiCGSTAB, GMRES) and geometric multigrid
 * - Multiple backends (Scalar, SIMD, OpenMP, GPU)
 * - Parameters grouped by the method that reads them; a group the chosen method
 *   does not read is refused at init rather than ignored
 * - Statistics reporting (iterations, residual, timing)
 * - Convenience functions for common use cases
 *
 * The walls come first. Every solver here defaults to zero-gradient on all
 * faces, which leaves the constants in the operator's nullspace: the solution is
 * pinned only up to an additive constant, and the right-hand side must have zero
 * interior mean or the system has no solution at all. The Krylov methods refuse
 * one that does not -- see poisson_solver_solve(). Two ways out, and the choice
 * belongs to the physics: remove the mean (a pressure defined up to a constant,
 * which is what projection wants), or prescribe a value on a face through
 * params.walls (a pressure-driven channel, an outlet).
 *
 * Usage:
 * @code
 * poisson_solver_t* solver = poisson_solver_create(
 *     POISSON_METHOD_CG, POISSON_BACKEND_AUTO);
 * if (!solver) { ... }  // this method has no such backend; no silent fallback
 *
 * poisson_solver_params_t params = poisson_solver_params_default();
 * params.tolerance = 1e-8;
 * // params.walls stays zero-gradient here, so make the rhs compatible with it:
 * poisson_make_rhs_compatible(rhs, nx, ny, 1);
 *
 * // nz = 1, dz = 0 in 2D. Refuses a configuration it cannot honour -- check it.
 * cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
 * if (status != CFD_SUCCESS) { puts(cfd_get_last_error()); ... }
 *
 * poisson_solver_stats_t stats = poisson_solver_stats_default();
 * status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
 * printf("%d iterations, %s
", stats.iterations,
 *        poisson_solver_status_string(stats.status));
 *
 * poisson_solver_destroy(solver);
 * @endcode
 *
 * For a one-off solve, poisson_solve() does the same in a single call. It builds
 * a solver per call, so in a loop own one as above.
 */

#ifndef CFD_POISSON_SOLVER_H
#define CFD_POISSON_SOLVER_H

#include "cfd/boundary/boundary_conditions.h"
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
    POISSON_METHOD_GAUSS_SEIDEL,  /**< Gauss-Seidel: SOR at omega = 1 (params.omega is ignored) */
    POISSON_METHOD_SOR,           /**< Successive Over-Relaxation */
    POISSON_METHOD_REDBLACK_SOR,  /**< Red-Black SOR (parallelizable) */
    POISSON_METHOD_CG,            /**< Conjugate Gradient (for SPD systems) */
    POISSON_METHOD_BICGSTAB,      /**< BiCGSTAB (for non-symmetric systems) */
    POISSON_METHOD_GMRES,         /**< Restarted GMRES(m) (for non-symmetric systems) */
    POISSON_METHOD_MULTIGRID      /**< Geometric multigrid (V/W/F cycles, scalar and OpenMP backends) */
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
    POISSON_DIVERGED = 2,    /**< Solution diverged (the residual is no longer finite) */
    POISSON_STAGNATED = 3,   /**< Residual stagnated (no progress) */
    POISSON_INCOMPATIBLE_RHS = 4, /**< Singular walls, but the rhs has a nonzero interior mean,
                                       so the system has no solution (see poisson_walls_are_singular) */
    POISSON_ERROR = -1       /**< Error occurred */
} poisson_solver_status_t;

/**
 * Preconditioner types for iterative solvers
 */
typedef enum {
    POISSON_PRECOND_NONE = 0,      /**< No preconditioning (default) */
    POISSON_PRECOND_JACOBI = 1,    /**< Diagonal (Jacobi) preconditioning */
    POISSON_PRECOND_MULTIGRID = 2  /**< One geometric-multigrid V-cycle per apply
                                        (scalar and OpenMP CG; grid dims must be 2^k+1) */
} poisson_precond_type_t;

/**
 * Multigrid cycle type (POISSON_METHOD_MULTIGRID only)
 */
typedef enum {
    MG_CYCLE_V = 0,   /**< V-cycle (default) */
    MG_CYCLE_W = 1,   /**< W-cycle (two coarse-grid visits per level, more robust) */
    MG_CYCLE_F = 2    /**< F-cycle / full multigrid (FMG first pass, then V-cycles) */
} mg_cycle_type_t;

/**
 * Multigrid smoother type (POISSON_METHOD_MULTIGRID only)
 */
typedef enum {
    MG_SMOOTHER_REDBLACK_GS = 0,  /**< Red-Black Gauss-Seidel, omega=1 (default) */
    MG_SMOOTHER_JACOBI = 1        /**< Weighted Jacobi, omega=2/3 */
} mg_smoother_type_t;

/**
 * Multigrid boundary-condition mode (POISSON_METHOD_MULTIGRID only)
 *
 * MG_BC_NEUMANN matches the default zero-gradient behavior of all other
 * Poisson solvers (solution defined up to an additive constant; the RHS
 * should have zero interior mean for full convergence).
 * MG_BC_DIRICHLET holds the caller-supplied boundary values of x fixed on
 * the finest grid and uses homogeneous zero boundaries for coarse-level
 * corrections (supports inhomogeneous Dirichlet data).
 */
typedef enum {
    MG_BC_NEUMANN = 0,   /**< Zero-gradient BCs (default; matches other solvers) */
    MG_BC_DIRICHLET = 1  /**< Fixed boundary values on finest level */
} mg_bc_type_t;

/* ============================================================================
 * WALLS
 * ============================================================================ */

/**
 * Boundary condition on one face of the Poisson domain.
 *
 * POISSON_WALL_ZERO_GRADIENT (0) is the operator every solver here has always
 * inverted by default, so zero-initialization is fully backward compatible.
 *
 * Deliberately not bc_type_t: that enum's zero value is BC_TYPE_PERIODIC, a
 * condition the Poisson solvers do not implement, so a zero-initialized config
 * would name walls that cannot be honoured.
 */
typedef enum {
    POISSON_WALL_ZERO_GRADIENT = 0, /**< Neumann dp/dn = 0 (default) */
    POISSON_WALL_DIRICHLET     = 1  /**< Prescribed value from poisson_walls_t.values */
} poisson_wall_t;

/**
 * Per-face walls for the Poisson operator.
 *
 * Zero-initialization is all-zero-gradient. Face names mirror
 * ns_thermal_bc_config_t: front is the z = nz-1 plane and back the z = 0 plane,
 * and both are ignored when nz == 1.
 *
 * A face marked POISSON_WALL_DIRICHLET takes its value from the matching member
 * of `values`; 0 there is the homogeneous Dirichlet wall. Spatially varying wall
 * data is what the apply_bc hook is for, and the two cannot be combined --
 * poisson_solver_init() rejects that with CFD_ERROR_INVALID.
 */
typedef struct {
    poisson_wall_t left;   /**< x = 0 face */
    poisson_wall_t right;  /**< x = Lx face */
    poisson_wall_t bottom; /**< y = 0 face */
    poisson_wall_t top;    /**< y = Ly face */
    poisson_wall_t front;  /**< z = Lz face (3D only) */
    poisson_wall_t back;   /**< z = 0 face (3D only) */
    bc_dirichlet_values_t values; /**< Prescribed value per DIRICHLET face */
} poisson_walls_t;

/**
 * Subtract the interior mean of a right-hand side, in place.
 *
 * Makes it compatible with the singular all-zero-gradient operator, which is what
 * poisson_solver_solve() requires there. Do NOT call this when any face is
 * prescribed: that operator is nonsingular, and shifting the rhs then changes the
 * answer rather than making it exist.
 *
 * Does nothing for a NULL rhs or a grid with no interior (nx or ny below 3, or
 * nz below 3 in 3D): there is no interior mean to remove. That is silent because
 * there is nothing to report -- such a grid has no solve to make compatible, and
 * poisson_solver_solve() rejects it on its own terms. A caller that reaches for
 * this after POISSON_INCOMPATIBLE_RHS on a degenerate grid should read the
 * refusal as being about the grid, not the rhs.
 */
CFD_LIBRARY_EXPORT void poisson_make_rhs_compatible(double* rhs, size_t nx, size_t ny, size_t nz);

/** All faces zero-gradient: the default operator. */
CFD_LIBRARY_EXPORT poisson_walls_t poisson_walls_default(void);

/**
 * Whether every face is zero-gradient, i.e. the operator the solvers default to.
 *
 * Exported beside poisson_walls_are_singular() because the Navier-Stokes side
 * asks the same question when it decides whether a projection backend can
 * honour params.pressure_bc, and a second open-coded copy of the six-face
 * comparison is one more place to forget a face.
 */
CFD_LIBRARY_EXPORT bool poisson_walls_are_default(const poisson_walls_t* walls);

/**
 * Whether every face carries a value this operator can express.
 *
 * False for anything outside poisson_wall_t. Worth asking because the readers of
 * this enum disagree off-enum: the halo routines treat any non-DIRICHLET value
 * as zero-gradient, while poisson_walls_are_singular() calls a face prescribed
 * unless it is exactly ZERO_GRADIENT -- so a stray value builds the singular
 * operator while reporting the system as nonsingular. A caller that accepts
 * walls from a file or an untrusted struct should check before configuring them.
 */
CFD_LIBRARY_EXPORT bool poisson_walls_are_legal(const poisson_walls_t* walls);

/** Every face the same type, Dirichlet faces all at `value`. */
CFD_LIBRARY_EXPORT poisson_walls_t poisson_walls_uniform(poisson_wall_t type, double value);

/**
 * Whether the operator these walls describe has the constants as a nullspace.
 *
 * True only when every face that exists on this grid is zero-gradient, in which
 * case the system is solvable only for a right-hand side of zero interior mean;
 * poisson_solver_solve() rejects an incompatible one with CFD_ERROR_INVALID.
 * A single Dirichlet face makes the operator nonsingular and admits any rhs --
 * and means a caller must NOT mean-subtract, which would change the answer.
 *
 * nz is required because the z-faces do not exist on a 2D grid.
 */
CFD_LIBRARY_EXPORT bool poisson_walls_are_singular(const poisson_walls_t* walls, size_t nz);

/* ============================================================================
 * PARAMETERS AND STATISTICS
 * ============================================================================ */

/**
 * Parameters owned by the SOR family: SOR, Gauss-Seidel, Red-Black SOR.
 */
typedef struct {
    double omega; /**< Relaxation factor. 0 = automatic, at or just below the optimum
                       for the grid and the walls in use.
                       POISSON_METHOD_GAUSS_SEIDEL is the one documented exception to
                       the rule that a parameter is either honoured or refused: it IS
                       SOR at omega = 1, so it accepts a value here and relaxes with 1
                       regardless. Use POISSON_METHOD_SOR to choose omega yourself. */
} poisson_sor_params_t;

/**
 * Parameters owned by the Krylov family: CG, BiCGSTAB, GMRES.
 */
typedef struct {
    poisson_precond_type_t preconditioner; /**< 0 = NONE. Honoured by CG and GMRES on
                                                the CPU backends. BiCGSTAB implements no
                                                preconditioner on any backend, and no GPU
                                                solver implements one at all; both refuse
                                                a non-NONE value rather than ignore it. */
    int restart;                           /**< GMRES(m) restart length; 0 = auto (30).
                                                Must stay 0 for CG and BiCGSTAB. */
} poisson_krylov_params_t;

/**
 * Parameters owned by a multigrid hierarchy: the POISSON_METHOD_MULTIGRID solver,
 * and the hierarchy built for krylov.preconditioner == POISSON_PRECOND_MULTIGRID.
 *
 * As a preconditioner the first three are not the caller's. A V-cycle with
 * weighted-Jacobi smoothing, Dirichlet coarse corrections and an equal pre/post
 * sweep count is what makes one apply a symmetric operator, and CG minimising
 * over a Krylov space depends on that; poisson_solver_init() refuses a value for
 * any of them there rather than overriding it silently. The last four are yours
 * in both roles.
 */
typedef struct {
    mg_cycle_type_t cycle;       /**< 0 = MG_CYCLE_V; fixed when preconditioning */
    mg_smoother_type_t smoother; /**< 0 = whatever this context requires */
    mg_bc_type_t bc;             /**< 0 = whatever this context requires */
    int pre_smooth;              /**< Pre-smoothing sweeps per level (0 = 2) */
    int post_smooth;             /**< Post-smoothing sweeps per level (0 = 2); must
                                      equal pre_smooth when preconditioning */
    int coarse_max_iter;         /**< Smoother sweeps on the coarsest grid (0 = 50) */
    int max_levels;              /**< Max grid levels (0 = coarsen as far as possible) */
} poisson_multigrid_params_t;

/**
 * Parameters for Poisson solver configuration.
 *
 * The first block applies to every method. The three groups below it are owned by
 * one method family each: a group that is not all-zero and is not owned by the
 * method being initialized is a configuration error, not a no-op, and
 * poisson_solver_init() says so. That is the whole point of the grouping -- a flat
 * struct cannot tell "the caller set an SOR knob on a CG solve" from "the caller
 * left it alone".
 */
typedef struct {
    /* The operator being inverted. Not an algorithm choice: every method either
     * honours these walls or refuses the configuration at init. */
    poisson_walls_t walls;     /**< Per-face walls (zero-init = all zero-gradient) */

    /* Convergence control, every method. */
    double tolerance;          /**< Relative convergence tolerance (default: 1e-6) */
    double absolute_tolerance; /**< Absolute tolerance (default: 1e-10) */
    int max_iterations;        /**< Maximum iterations (default: 5000) */
    int check_interval;        /**< Check convergence every N iterations. Must be at
                                    least 1: poisson_solver_init() returns
                                    CFD_ERROR_INVALID for 0, which would otherwise
                                    reach iter % 0. Default 1. */
    bool verbose;              /**< Print iteration progress (default: false) */

    /**
     * Helmholtz shift sigma (default: 0 = pure Poisson, bit-identical to prior
     * releases). The solved equation is
     *
     *     nabla^2 x - sigma*x = rhs
     *
     * Common to every method, like walls above, because it describes the
     * operator rather than the algorithm used to invert it -- even though only
     * the scalar CG solver implements it today, which poisson_solver_init()
     * refuses rather than ignores.
     *
     * For implicit diffusion, (I - nu*dt*nabla^2)u = b rearranges to sigma =
     * 1/(nu*dt) with rhs[i] = -b[i]/(nu*dt). Note the minus sign on the rhs.
     * The caller must ensure nu*dt > 0; sigma = 1/(nu*dt) is otherwise infinite
     * and rejected below.
     *
     * sigma > 0 makes the operator strictly diagonally dominant, so it is better
     * conditioned than the pure Poisson problem: cond = 1 + 8d with d = nu*dt/h^2,
     * and CG iterations go as sqrt(1 + 8d).
     *
     * It also removes the Neumann nullspace, so a shifted solve has a unique
     * solution and does NOT require a compatible (zero-mean) rhs -- the
     * zero-interior-mean rule that zero-gradient walls otherwise impose does not
     * apply, and poisson_solver_solve() does not enforce it here. Callers that
     * mean-subtract for the pure-Neumann pressure solve must not do so with a
     * shift set; poisson_walls_are_singular() reports the same thing for walls.
     *
     * Negative or non-finite values are rejected with CFD_ERROR_INVALID at init:
     * a negative shift is the indefinite Helmholtz operator, which breaks CG's
     * SPD requirement and standard geometric multigrid.
     *
     * Note that absolute_tolerance is not invariant under the 1/(nu*dt) rhs
     * scaling; the relative tolerance is.
     */
    double helmholtz_shift;

    /* Owned by exactly one method family each. */
    poisson_sor_params_t       sor;
    poisson_krylov_params_t    krylov;
    poisson_multigrid_params_t multigrid;
} poisson_solver_params_t;

/**
 * Everything needed to describe a Poisson solve: what to run, where, and how.
 *
 * Obtain one from poisson_solver_config_preset() and edit it. That is the whole
 * point of a preset here -- it is a starting point, not a terminal selector, so
 * "the accurate settings, but on OpenMP, with a prescribed outlet" is three
 * assignments rather than a combination somebody has to have enumerated in
 * advance.
 */
typedef struct {
    poisson_solver_method_t  method;
    poisson_solver_backend_t backend;
    poisson_solver_params_t  params;
} poisson_solver_config_t;

/**
 * Named starting points, by intent.
 *
 * A preset names a backend only where AUTO would pick one that cannot run it:
 * AUTO resolves to SIMD wherever AVX2 or NEON is present, and the two multigrid
 * presets have no SIMD backend, so those name POISSON_BACKEND_SCALAR and the
 * rest leave POISSON_BACKEND_AUTO. Choosing another is a single assignment on
 * the config either way. Nor does any preset name a method it does not need to
 * -- GMRES and the GPU need no enumerator here, just cfg.method and cfg.backend.
 */
typedef enum {
    /** Conjugate Gradient at 1e-6. Works on any grid with an interior. */
    POISSON_PRESET_DEFAULT = 0,

    /** Conjugate Gradient at 1e-10, with the iteration budget to reach it. */
    POISSON_PRESET_ACCURATE,

    /** BiCGSTAB, for an operator CG cannot invert. */
    POISSON_PRESET_NONSYMMETRIC,

    /**
     * Red-Black SOR: the stationary option.
     *
     * Exempt from the compatibility rule below, because the stationary methods
     * relax towards a solution modulo a drifting constant rather than chasing
     * the nullspace component. That makes it the right choice for an rhs you
     * cannot make compatible, and for use as a smoother.
     *
     * For a fixed number of sweeps rather than a convergence request, set
     * params.tolerance = 0 and params.max_iterations yourself; the solve then
     * runs the budget out and reports CFD_ERROR_MAX_ITER, which is the honest
     * answer to "did it converge?" when you never asked it to.
     */
    POISSON_PRESET_SMOOTHER,

    /**
     * Geometric multigrid. Requires 2^k+1 points per active dimension.
     *
     * Also exempt: the cycle removes the interior mean on each coarse level, so
     * it too converges modulo a constant rather than being derailed by one.
     */
    POISSON_PRESET_MULTIGRID,

    /**
     * CG preconditioned by one multigrid V-cycle. Requires 2^k+1 dimensions.
     *
     * The outer method is CG, so this one IS subject to the compatibility rule.
     */
    POISSON_PRESET_MULTIGRID_PCG
} poisson_preset_t;

/**
 * The config a preset stands for.
 *
 * Note which presets are subject to the zero-interior-mean rule: DEFAULT,
 * ACCURATE, NONSYMMETRIC and MULTIGRID_PCG, whose outer method is Krylov.
 * SMOOTHER and MULTIGRID are not. With the default zero-gradient walls the
 * operator is singular, so a right-hand side with a nonzero interior mean
 * describes a system with no solution; a Krylov solve refuses it with
 * CFD_ERROR_INVALID and POISSON_INCOMPATIBLE_RHS rather than chasing the
 * nullspace component -- see poisson_make_rhs_compatible().
 *
 * Switching between presets therefore changes whether a given rhs is legal.
 * That is a difference in the methods, not a policy: it is worth knowing which
 * side of it a preset falls on before swapping one for another.
 */
CFD_LIBRARY_EXPORT poisson_solver_config_t poisson_solver_config_preset(poisson_preset_t preset);

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
 * Common to every method:
 * - tolerance: 1e-6, absolute_tolerance: 1e-10
 * - max_iterations: 5000, check_interval: 1
 * - verbose: false
 * - walls: all faces POISSON_WALL_ZERO_GRADIENT
 *
 * Per method group, all zero, which each owning method reads as its own default:
 * - sor.omega: 0 (automatic, at or just below the optimum for this grid and these walls)
 * - krylov.preconditioner: POISSON_PRECOND_NONE, krylov.restart: 0 (= 30)
 * - multigrid: MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,
 *   pre_smooth/post_smooth 0 (= 2), coarse_max_iter 0 (= 50), max_levels 0 (= auto)
 *
 * A group belonging to another method must stay zero: poisson_solver_init()
 * refuses a group the chosen method does not read, rather than ignoring it.
 */
CFD_LIBRARY_EXPORT poisson_solver_params_t poisson_solver_params_default(void);

/**
 * Initialize statistics with default values
 */
CFD_LIBRARY_EXPORT poisson_solver_stats_t poisson_solver_stats_default(void);

/**
 * A short, human-readable name for a convergence status.
 *
 * For the message on a CFD_ERROR_* return, use cfd_get_last_error() instead --
 * that is the sentence naming the fix. This one names the outcome a solve
 * reported in poisson_solver_stats_t.status, for a log line or a table.
 *
 * @param status Any poisson_solver_status_t value
 * @return A static string, never NULL; "unknown" for a value outside the enum
 */
CFD_LIBRARY_EXPORT const char* poisson_solver_status_string(poisson_solver_status_t status);

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
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
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
    poisson_solver_apply_bc_func apply_bc; /**< Apply boundary conditions. NULL: zero-gradient walls copied
                                                from the interior, for which the SOR solvers relax the points
                                                beside a wall and choose omega. To hold other wall values,
                                                install a function here before poisson_solver_init() rather
                                                than writing the walls between iterations. Multigrid and every
                                                GPU solver apply their walls themselves and never call this, so
                                                their init rejects a function here with CFD_ERROR_UNSUPPORTED
                                                rather than accepting one that would have no effect.
                                                A function here must prescribe wall values that do not depend on
                                                the interior: the Krylov solvers (CG, BiCGSTAB, GMRES) apply only
                                                its homogeneous part, a zero halo, to their search directions,
                                                since an inhomogeneous condition there would make the operator
                                                affine rather than linear. That is exactly what lifting an
                                                inhomogeneous Dirichlet problem onto a homogeneous one needs, and
                                                it makes the operator nonsingular. Leaving this NULL gives the
                                                zero-gradient walls, which are their own homogeneous form and are
                                                applied throughout; that operator is singular (the constants are
                                                its nullspace), so the rhs must have zero interior mean. */

    /**
     * Walls a solver installs for itself. Library-internal: do not assign to it.
     *
     * Multigrid needs its own boundary routine, because the default here is
     * zero-gradient and would corrupt a field held in MG_BC_DIRICHLET mode. It
     * used to install that routine into apply_bc above, which meant a caller
     * following the documented workflow -- create, assign apply_bc, init --
     * silently overwrote it, and since the cycle applies its boundaries
     * internally the caller's function then had no effect on the solve at all.
     *
     * Keeping the two apart means apply_bc above is the caller's and nothing
     * else, so everything that asks "did the caller prescribe walls?" gets an
     * honest answer. poisson_solver_apply_bc() prefers the caller's function and
     * falls back to this one; a solver that sets this rejects a caller function
     * at init rather than choosing between them.
     */
    poisson_solver_apply_bc_func internal_apply_bc;
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
 * @param nz Grid points in z direction (1 for 2D)
 * @param dx Grid spacing in x direction
 * @param dy Grid spacing in y direction
 * @param dz Grid spacing in z direction (0.0 for 2D)
 * @param params Solver parameters (NULL for defaults)
 * @return CFD_SUCCESS, or a refusal -- see below
 *
 * This is where a configuration this solver cannot honour is refused, rather
 * than being accepted and then ignored during the solve:
 *
 * - CFD_ERROR_INVALID if a parameter group the method does not read is set
 *   (params.sor on a Krylov method, params.krylov on SOR, params.multigrid on a
 *   CG that is not multigrid-preconditioned), if params.krylov.restart is set on
 *   anything but GMRES, if a multigrid preconditioner is given a cycle, smoother,
 *   boundary mode or unequal pre/post sweep count of its own (each would cost the
 *   inner cycle the symmetry CG depends on; its sweep counts, coarse_max_iter and
 *   max_levels are yours), if check_interval is below 1, if params.walls
 *   collides with a custom apply_bc, or if helmholtz_shift is negative or
 *   non-finite (a negative shift is the indefinite Helmholtz operator, which is
 *   not SPD).
 * - CFD_ERROR_UNSUPPORTED if this method or backend cannot implement what was
 *   asked: any preconditioner on BiCGSTAB or on a GPU backend (neither
 *   implements one), a multigrid preconditioner outside scalar and OpenMP CG,
 *   prescribed faces outside the CPU Krylov solvers, a nonzero helmholtz_shift
 *   outside scalar CG (or on scalar CG with a multigrid preconditioner), or a
 *   caller's apply_bc on multigrid or on any GPU solver (both apply their walls
 *   themselves and never call it).
 *
 * Install a custom apply_bc before calling this, not after: omega resolution
 * reads it. cfd_get_last_error() carries the sentence naming the fix.
 */
CFD_LIBRARY_EXPORT cfd_status_t poisson_solver_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
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
 * @return CFD_SUCCESS on convergence, CFD_ERROR_MAX_ITER if not converged.
 *         Jacobi, SOR, Red-Black SOR and multigrid, on every backend, return
 *         CFD_ERROR_DIVERGED when the residual is not finite at the start or at a
 *         convergence check; the Krylov solvers (CG, BiCGSTAB, GMRES) do not report
 *         a non-finite residual as divergence.
 *
 *         CFD_ERROR_INVALID, with stats->status == POISSON_INCOMPATIBLE_RHS, when
 *         the walls make the operator singular and rhs has a nonzero interior
 *         mean -- the system then has no solution, and no number of iterations
 *         would find one. poisson_make_rhs_compatible() is the fix. Only the
 *         Krylov methods check this; the stationary ones run a fixed sweep and
 *         are not asked to converge.
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
 * Useful for custom iteration control or monitoring. Wall values other than the
 * default zero-gradient copy belong in solver->apply_bc, which every iteration
 * applies after its sweep (see struct poisson_solver).
 *
 * Unlike poisson_solver_solve(), this never checks the rhs against singular
 * walls. Its callers are the stationary solvers, which that check exempts
 * anyway, and one iteration of anything is a step rather than an answer -- there
 * is nothing here to refuse. Drive a Krylov method through this and you take on
 * the compatibility question yourself.
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
 * @return Maximum absolute residual, or NaN if any point's residual is NaN
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
#define POISSON_SOLVER_TYPE_JACOBI_OMP        "jacobi_omp"
#define POISSON_SOLVER_TYPE_JACOBI_SIMD       "jacobi_simd"
#define POISSON_SOLVER_TYPE_SOR_SCALAR        "sor_scalar"
#define POISSON_SOLVER_TYPE_SOR_SIMD          "sor_simd"
#define POISSON_SOLVER_TYPE_REDBLACK_SCALAR   "redblack_scalar"
#define POISSON_SOLVER_TYPE_REDBLACK_OMP      "redblack_omp"
#define POISSON_SOLVER_TYPE_REDBLACK_SIMD     "redblack_simd"
#define POISSON_SOLVER_TYPE_CG_SCALAR         "cg_scalar"
#define POISSON_SOLVER_TYPE_CG_OMP            "cg_omp"
#define POISSON_SOLVER_TYPE_CG_SIMD           "cg_simd"
#define POISSON_SOLVER_TYPE_BICGSTAB_SCALAR   "bicgstab_scalar"
#define POISSON_SOLVER_TYPE_BICGSTAB_OMP      "bicgstab_omp"
#define POISSON_SOLVER_TYPE_BICGSTAB_SIMD     "bicgstab_simd"
#define POISSON_SOLVER_TYPE_BICGSTAB_GPU      "bicgstab_gpu"
#define POISSON_SOLVER_TYPE_GMRES_SCALAR      "gmres_scalar"
#define POISSON_SOLVER_TYPE_GMRES_OMP         "gmres_omp"
#define POISSON_SOLVER_TYPE_GMRES_SIMD        "gmres_simd"
#define POISSON_SOLVER_TYPE_MG_SCALAR         "multigrid_scalar"
#define POISSON_SOLVER_TYPE_MG_OMP            "multigrid_omp"

/* ============================================================================
 * CONVENIENCE API
 * ============================================================================ */

/**
 * Solve a Poisson problem in one call: create, init, solve, destroy.
 *
 * 2D is nz = 1 and dz = 0.0. `config` NULL means
 * poisson_solver_config_preset(POISSON_PRESET_DEFAULT).
 *
 * Builds a solver per call, so in a loop own a poisson_solver_t instead and
 * call poisson_solver_solve() on it -- that is what the projection solvers do,
 * and it matters most for the multigrid modes, whose hierarchy would otherwise
 * be rebuilt every time.
 *
 * Remember that the default walls are zero-gradient, which makes the operator
 * singular for every preset but POISSON_PRESET_SMOOTHER: the rhs must then have
 * zero interior mean, or this returns CFD_ERROR_INVALID with
 * stats->status == POISSON_INCOMPATIBLE_RHS. poisson_make_rhs_compatible() is
 * the fix; prescribing a wall value on one face through config->params.walls is
 * the other.
 *
 * @param x       Solution field (in/out; the initial guess on entry)
 * @param x_temp  Scratch of the same size, or NULL where the method allows it
 * @param rhs     Right-hand side
 * @param stats   Optional; receives iterations, residuals and the solve status
 * @return CFD_SUCCESS, or CFD_ERROR_MAX_ITER / CFD_ERROR_DIVERGED /
 *         CFD_ERROR_INVALID / CFD_ERROR_UNSUPPORTED / CFD_ERROR_NOMEM.
 *         cfd_get_last_error() carries the detail.
 */
CFD_LIBRARY_EXPORT cfd_status_t poisson_solve(
    double* x, double* x_temp, const double* rhs,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_config_t* config,
    poisson_solver_stats_t* stats);

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
