/**
 * @file lid_driven_cavity_common.h
 * @brief Shared utilities for lid-driven cavity validation tests
 *
 * Contains Ghia et al. reference data, test context management,
 * and common simulation utilities.
 */

#ifndef LID_DRIVEN_CAVITY_COMMON_H
#define LID_DRIVEN_CAVITY_COMMON_H

#include "cavity_reference_data.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "solver_availability.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/**
 * Test configuration - reduce iterations for faster CI runs.
 * Set CAVITY_FULL_VALIDATION=1 for comprehensive validation.
 */
#ifndef CAVITY_FULL_VALIDATION
#define CAVITY_FULL_VALIDATION 0
#endif

#if CAVITY_FULL_VALIDATION
#define FAST_STEPS      5000
#define MEDIUM_STEPS    10000
#define FULL_STEPS      20000
#define FINE_DT         0.0005
#else
/* Fast mode for CI - uses fewer iterations
 * At Re=100 with 33x33 grid using scalar CG for the pressure solve:
 *   - 5000 steps achieves RMS < 0.10
 * Using SIMD backends achieves RMS < 0.05.
 * Using dt=0.0005 for stability. */
#define FAST_STEPS      2000
#define MEDIUM_STEPS    3000
#define FULL_STEPS      5000
#define FINE_DT         0.0005
#endif

/* Explicit Euler solvers have internal DT_CONSERVATIVE_LIMIT = 0.0001
 * which caps the actual time step to 0.0001 regardless of FINE_DT.
 * They need 5x more steps to achieve equivalent simulation time. */
#define EULER_FULL_STEPS   25000
#define EULER_MEDIUM_STEPS 15000
#define EULER_FAST_STEPS   10000

/* Steady-state detection.
 *
 * Steady means the velocity field has stopped changing: the harness stops once
 * max |u^{n+1} - u^n| / (dt * U_lid), over u and v at every node, falls below
 * CAVITY_STEADY_TOL. It is a rate per unit time, so it means the same thing at
 * every dt and for every solver.
 *
 * It replaces |d(ln KE)/dt|, which passes through zero wherever kinetic energy
 * has a turning point. The cavity's KE overshoots before it settles, so the old
 * test fired at the top of the overshoot: a 129x129 Re=1000 run stopped at
 * t = 45.2 with u at the centre still 7.4e-4 from the value it settles to by
 * t ~ 100 -- as large as the differences between grids, which made the
 * Richardson order of that quantity measure where each grid happened to stop.
 * A pointwise residual cannot vanish while any part of the field still moves.
 *
 * 1e-6: the core relaxes at about 0.07 per unit time at Re=1000, so what is left
 * when the test fires is ~1e-5 -- two orders below the finest grid differences.
 * CAVITY_MIN_SETTLE_TIME keeps the test from firing during the first moments
 * after the lid starts. */
#define CAVITY_STEADY_TOL       1e-6
#define CAVITY_MIN_SETTLE_TIME  1.0

/* Domain configuration for unit square cavity [0,1] x [0,1] */
#define CAVITY_DOMAIN_XMIN  0.0
#define CAVITY_DOMAIN_XMAX  1.0
#define CAVITY_DOMAIN_YMIN  0.0
#define CAVITY_DOMAIN_YMAX  1.0

/* Initial field values */
#define CAVITY_INITIAL_DENSITY      1.0
#define CAVITY_INITIAL_TEMPERATURE  300.0
#define CAVITY_INITIAL_PRESSURE     0.0
#define CAVITY_INITIAL_VELOCITY     0.0

/* ============================================================================
 * GHIA ET AL. REFERENCE DATA (1982)
 * ============================================================================
 * Canonical Ghia reference data is defined in cavity_reference_data.h:
 *   - GHIA_Y_COORDS, GHIA_X_COORDS: coordinate arrays
 *   - GHIA_U_RE100, GHIA_V_RE100: velocity profiles at Re=100
 *   - GHIA_NUM_POINTS: number of sample points (17)
 */

typedef struct {
    const double* coords;
    const double* values;
    size_t n;
} ghia_profile_t;

/* ============================================================================
 * TEST CONTEXT
 * ============================================================================ */

typedef struct {
    grid* g;
    flow_field* field;
    size_t nx;
    size_t ny;
    double* u_prev; /* velocity at the start of the step, for the steady test */
    double* v_prev;
} cavity_context_t;

static inline cavity_context_t* cavity_context_create(size_t nx, size_t ny) {
    cavity_context_t* ctx = malloc(sizeof(cavity_context_t));
    if (!ctx) return NULL;

    ctx->nx = nx;
    ctx->ny = ny;
    ctx->g = grid_create(nx, ny, 1,
                         CAVITY_DOMAIN_XMIN, CAVITY_DOMAIN_XMAX,
                         CAVITY_DOMAIN_YMIN, CAVITY_DOMAIN_YMAX, 0.0, 0.0);
    ctx->field = flow_field_create(nx, ny, 1);
    ctx->u_prev = malloc(nx * ny * sizeof(double));
    ctx->v_prev = malloc(nx * ny * sizeof(double));

    if (!ctx->g || !ctx->field || !ctx->u_prev || !ctx->v_prev) {
        if (ctx->g) grid_destroy(ctx->g);
        if (ctx->field) flow_field_destroy(ctx->field);
        free(ctx->u_prev);
        free(ctx->v_prev);
        free(ctx);
        return NULL;
    }

    grid_initialize_uniform(ctx->g);

    /* Initialize field to quiescent state */
    size_t total = nx * ny;
    for (size_t i = 0; i < total; i++) {
        ctx->field->u[i] = CAVITY_INITIAL_VELOCITY;
        ctx->field->v[i] = CAVITY_INITIAL_VELOCITY;
        ctx->field->p[i] = CAVITY_INITIAL_PRESSURE;
        ctx->field->rho[i] = CAVITY_INITIAL_DENSITY;
        ctx->field->T[i] = CAVITY_INITIAL_TEMPERATURE;
    }

    return ctx;
}

static inline void cavity_context_destroy(cavity_context_t* ctx) {
    if (!ctx) return;
    if (ctx->g) grid_destroy(ctx->g);
    if (ctx->field) flow_field_destroy(ctx->field);
    free(ctx->u_prev);
    free(ctx->v_prev);
    free(ctx);
}

/* ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================ */

static inline void apply_cavity_bc(flow_field* field, double lid_velocity) {
    bc_dirichlet_values_t u_bc = {.left = 0.0, .right = 0.0, .top = lid_velocity, .bottom = 0.0};
    bc_dirichlet_values_t v_bc = {.left = 0.0, .right = 0.0, .top = 0.0, .bottom = 0.0};

    bc_apply_dirichlet_velocity(field->u, field->v, field->nx, field->ny, &u_bc, &v_bc);
    bc_apply_neumann(field->p, field->nx, field->ny);
}

/* ============================================================================
 * SIMULATION UTILITIES
 * ============================================================================ */

typedef struct {
    int steps_completed;
    double sim_time;        /* physical time reached, from the solver's real dt */
    double final_residual;
    double max_velocity;
    int converged;
    int blew_up;
} simulation_result_t;

/**
 * Unified simulation result structure
 *
 * Contains all fields needed by different test scenarios:
 * - Basic simulation info (success, error messages)
 * - Center point values (for architecture consistency tests)
 * - Min/max values (for stability and sanity checks)
 * - Convergence tracking (for regression tests)
 */
typedef struct {
    /* Status */
    int success;
    int solver_unavailable;  /* 1 if solver/backend not compiled in, 0 otherwise */
    char error_msg[256];

    /* Center point values */
    double u_at_center;
    double v_at_center;

    /* Field statistics */
    double max_velocity;
    double u_min;
    double kinetic_energy;

    /* Convergence tracking */
    int steps_completed;
    double sim_time;        /* physical time reached, from the solver's real dt */
    double final_residual;
    int converged;
} cavity_sim_result_t;

/* Field analysis helper functions (must be defined before cavity_run_with_solver) */

static inline double compute_kinetic_energy(const flow_field* field) {
    double ke = 0.0;
    size_t total = field->nx * field->ny;
    for (size_t i = 0; i < total; i++) {
        ke += 0.5 * field->rho[i] * (field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }
    return ke;
}

static inline double find_max_velocity(const flow_field* field) {
    double max_vmag = 0.0;
    size_t total = field->nx * field->ny;
    for (size_t i = 0; i < total; i++) {
        double vmag = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
        if (vmag > max_vmag) max_vmag = vmag;
    }
    return max_vmag;
}

static inline int check_field_finite(const flow_field* field) {
    size_t total = field->nx * field->ny;
    for (size_t i = 0; i < total; i++) {
        if (!isfinite(field->u[i]) || !isfinite(field->v[i]) ||
            !isfinite(field->p[i]) || !isfinite(field->rho[i])) {
            return 0;
        }
    }
    return 1;
}

/* Remember the velocity a step starts from; call after the lid BC, before the step. */
static inline void cavity_save_velocity(cavity_context_t* ctx) {
    size_t total = ctx->nx * ctx->ny;
    memcpy(ctx->u_prev, ctx->field->u, total * sizeof(double));
    memcpy(ctx->v_prev, ctx->field->v, total * sizeof(double));
}

/* max |u^{n+1} - u^n| / (dt * U_lid) over u and v: the steady-state residual. */
static inline double cavity_steady_residual(const cavity_context_t* ctx, double dt,
                                            double lid_velocity) {
    size_t total = ctx->nx * ctx->ny;
    double change = 0.0;
    for (size_t i = 0; i < total; i++) {
        double du = fabs(ctx->field->u[i] - ctx->u_prev[i]);
        double dv = fabs(ctx->field->v[i] - ctx->v_prev[i]);
        change = fmax(change, fmax(du, dv));
    }
    return change / (dt * fabs(lid_velocity));
}

/**
 * Run cavity simulation with specified solver type
 *
 * This is the unified base function for all cavity simulation tests.
 * It handles solver creation, time stepping, and result extraction.
 *
 * @param solver_type  Solver type string (e.g., NS_SOLVER_TYPE_PROJECTION)
 * @param nx, ny       Grid dimensions
 * @param reynolds     Reynolds number
 * @param lid_velocity Lid velocity (typically 1.0)
 * @param max_steps    Maximum number of time steps
 * @param dt           Time step size
 * @return Simulation result with all extracted data
 */
static inline cavity_sim_result_t cavity_run_with_solver(
    const char* solver_type,
    size_t nx, size_t ny,
    double reynolds, double lid_velocity,
    int max_steps, double dt)
{
    cavity_sim_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    /* Create context */
    cavity_context_t* ctx = cavity_context_create(nx, ny);
    if (!ctx) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create context");
        return result;
    }

    double L = ctx->g->xmax - ctx->g->xmin;
    double nu = lid_velocity * L / reynolds;

    ns_solver_params_t params = {
        .dt = dt,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = nu,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = validation_create_solver(registry, solver_type,
                                                   &result.solver_unavailable,
                                                   result.error_msg, sizeof(result.error_msg));
    if (!solver) {
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        return result;
    }

    cfd_status_t init_status = solver_init(solver, ctx->g, &params);
    if (init_status != CFD_SUCCESS) {
        if (init_status == CFD_ERROR_UNSUPPORTED) {
            result.solver_unavailable = 1;
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver '%s' backend not available (not compiled)", solver_type);
        } else {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver '%s' init failed with status %d", solver_type, init_status);
        }
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        return result;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_velocity);
        cavity_save_velocity(ctx);
        cfd_status_t step_status = solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (step_status != CFD_SUCCESS) {
            const char* err_str = cfd_get_error_string(step_status);
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver step failed at step %d: %s (%d)", step, err_str, step_status);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            return result;
        }

        if (!check_field_finite(ctx->field)) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            return result;
        }

        /* Steady-state residual (see CAVITY_STEADY_TOL). dt_used, not params.dt:
         * the Euler solvers clamp their own step. */
        double dt_step = (stats.dt_used > 0.0) ? stats.dt_used : params.dt;
        result.final_residual = cavity_steady_residual(ctx, dt_step, lid_velocity);
        result.steps_completed = step + 1;
        result.sim_time = (double)(step + 1) * dt_step;

        if (result.sim_time > CAVITY_MIN_SETTLE_TIME &&
            result.final_residual < CAVITY_STEADY_TOL) {
            result.converged = 1;
            break;
        }
    }

    /* Extract results */
    size_t center_i = nx / 2;
    size_t center_j = ny / 2;
    size_t center_idx = center_j * nx + center_i;

    result.u_at_center = ctx->field->u[center_idx];
    result.v_at_center = ctx->field->v[center_idx];
    result.max_velocity = find_max_velocity(ctx->field);
    result.kinetic_energy = compute_kinetic_energy(ctx->field);

    /* Find u_min along vertical centerline */
    result.u_min = 1e10;
    for (size_t j = 0; j < ny; j++) {
        double u_val = ctx->field->u[j * nx + center_i];
        if (u_val < result.u_min) {
            result.u_min = u_val;
        }
    }

    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    cavity_context_destroy(ctx);

    return result;
}

/**
 * Run cavity simulation and return context for post-processing
 *
 * Like cavity_run_with_solver() but returns the context instead of destroying
 * it, allowing the caller to extract additional data (e.g., full profiles for
 * Ghia validation).
 *
 * IMPORTANT: Caller is responsible for calling cavity_context_destroy() on
 * the returned context when done.
 *
 * @param solver_type  Solver type string (e.g., NS_SOLVER_TYPE_PROJECTION)
 * @param nx, ny       Grid dimensions
 * @param reynolds     Reynolds number
 * @param lid_velocity Lid velocity (typically 1.0)
 * @param max_steps    Maximum number of time steps
 * @param dt           Time step size
 * @param pressure_solver  Pressure solve for the projection solvers
 *                         (NS_PRESSURE_SOLVER_DEFAULT = each backend's CG)
 * @param out_ctx      Output: pointer to store the context (NULL on failure)
 * @return Simulation result with basic status info
 */
static inline cavity_sim_result_t cavity_run_with_pressure_solver_ctx(
    const char* solver_type,
    size_t nx, size_t ny,
    double reynolds, double lid_velocity,
    int max_steps, double dt,
    ns_pressure_solver_t pressure_solver,
    cavity_context_t** out_ctx)
{
    cavity_sim_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';
    if (out_ctx) *out_ctx = NULL;

    /* Create context */
    cavity_context_t* ctx = cavity_context_create(nx, ny);
    if (!ctx) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create context");
        return result;
    }

    double L = ctx->g->xmax - ctx->g->xmin;
    double nu = lid_velocity * L / reynolds;

    ns_solver_params_t params = {
        .dt = dt,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = nu,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };
    params.pressure_solver = pressure_solver;

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = validation_create_solver(registry, solver_type,
                                                   &result.solver_unavailable,
                                                   result.error_msg, sizeof(result.error_msg));
    if (!solver) {
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        return result;
    }

    cfd_status_t init_status = solver_init(solver, ctx->g, &params);
    if (init_status != CFD_SUCCESS) {
        if (init_status == CFD_ERROR_UNSUPPORTED) {
            result.solver_unavailable = 1;
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver '%s' backend not available (not compiled)", solver_type);
        } else {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver '%s' init failed with status %d", solver_type, init_status);
        }
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        if (out_ctx) *out_ctx = NULL;
        return result;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_velocity);
        cavity_save_velocity(ctx);
        cfd_status_t step_status = solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (step_status != CFD_SUCCESS) {
            const char* err_str = cfd_get_error_string(step_status);
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver step failed at step %d: %s (%d)", step, err_str, step_status);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            if (out_ctx) *out_ctx = NULL;
            return result;
        }

        if (!check_field_finite(ctx->field)) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            if (out_ctx) *out_ctx = NULL;
            return result;
        }

        /* Steady-state residual (see CAVITY_STEADY_TOL). dt_used, not params.dt:
         * the Euler solvers clamp their own step. */
        double dt_step = (stats.dt_used > 0.0) ? stats.dt_used : params.dt;
        result.final_residual = cavity_steady_residual(ctx, dt_step, lid_velocity);
        result.steps_completed = step + 1;
        result.sim_time = (double)(step + 1) * dt_step;

        if (result.sim_time > CAVITY_MIN_SETTLE_TIME &&
            result.final_residual < CAVITY_STEADY_TOL) {
            result.converged = 1;
            break;
        }
    }

    /* Extract basic results */
    size_t center_i = nx / 2;
    size_t center_j = ny / 2;
    size_t center_idx = center_j * nx + center_i;

    result.u_at_center = ctx->field->u[center_idx];
    result.v_at_center = ctx->field->v[center_idx];
    result.max_velocity = find_max_velocity(ctx->field);
    result.kinetic_energy = compute_kinetic_energy(ctx->field);

    /* Find u_min along vertical centerline */
    result.u_min = 1e10;
    for (size_t j = 0; j < ny; j++) {
        double u_val = ctx->field->u[j * nx + center_i];
        if (u_val < result.u_min) {
            result.u_min = u_val;
        }
    }

    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);

    /* Return context to caller for additional post-processing */
    if (out_ctx) {
        *out_ctx = ctx;
    } else {
        cavity_context_destroy(ctx);
    }

    return result;
}

/** cavity_run_with_pressure_solver_ctx() with each backend's default pressure solve. */
static inline cavity_sim_result_t cavity_run_with_solver_ctx(
    const char* solver_type,
    size_t nx, size_t ny,
    double reynolds, double lid_velocity,
    int max_steps, double dt,
    cavity_context_t** out_ctx)
{
    return cavity_run_with_pressure_solver_ctx(solver_type, nx, ny, reynolds, lid_velocity,
                                               max_steps, dt, NS_PRESSURE_SOLVER_DEFAULT,
                                               out_ctx);
}

/**
 * Run cavity simulation with default projection solver (legacy interface)
 *
 * This function maintains backward compatibility with tests that pass
 * a pre-created context. It runs the simulation in-place, modifying
 * the context's field.
 *
 * Note: This function is less efficient than cavity_run_with_solver()
 * because it requires the caller to manage the context. Prefer using
 * cavity_run_with_solver() for new code.
 */
static inline simulation_result_t run_cavity_simulation(
    cavity_context_t* ctx, double reynolds, double lid_velocity,
    int max_steps, double dt)
{
    simulation_result_t result = {0, 1.0, 0.0, 0, 0};

    double L = ctx->g->xmax - ctx->g->xmin;
    double nu = lid_velocity * L / reynolds;

    ns_solver_params_t params = {
        .dt = dt,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = nu,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver) {
        cfd_registry_destroy(registry);
        result.blew_up = 1;
        return result;
    }

    cfd_status_t init_status = solver_init(solver, ctx->g, &params);
    if (init_status != CFD_SUCCESS) {
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        result.blew_up = 1;
        return result;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_velocity);
        cavity_save_velocity(ctx);
        cfd_status_t step_status = solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (step_status != CFD_SUCCESS) {
            result.blew_up = 1;
            break;
        }

        if (!check_field_finite(ctx->field)) {
            result.blew_up = 1;
            break;
        }

        /* Steady-state residual (see CAVITY_STEADY_TOL). dt_used, not params.dt:
         * the Euler solvers clamp their own step. */
        double dt_step = (stats.dt_used > 0.0) ? stats.dt_used : params.dt;
        result.final_residual = cavity_steady_residual(ctx, dt_step, lid_velocity);
        result.steps_completed = step + 1;
        result.sim_time = (double)(step + 1) * dt_step;

        if (result.sim_time > CAVITY_MIN_SETTLE_TIME &&
            result.final_residual < CAVITY_STEADY_TOL) {
            result.converged = 1;
            break;
        }
    }

    result.max_velocity = find_max_velocity(ctx->field);

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    return result;
}

/* ============================================================================
 * PROFILE ANALYSIS
 * ============================================================================ */

static inline double compute_profile_rms_error(
    const double* computed_coords, const double* computed_vals, size_t computed_n,
    const double* ref_coords, const double* ref_vals, size_t ref_n)
{
    double sum_sq_error = 0.0;
    int count = 0;

    for (size_t i = 0; i < ref_n; i++) {
        double coord = ref_coords[i];
        double computed = 0.0;

        for (size_t j = 0; j < computed_n - 1; j++) {
            if (coord >= computed_coords[j] && coord <= computed_coords[j + 1]) {
                double t = (coord - computed_coords[j]) / (computed_coords[j + 1] - computed_coords[j]);
                computed = computed_vals[j] + t * (computed_vals[j + 1] - computed_vals[j]);
                break;
            }
        }

        double error = computed - ref_vals[i];
        sum_sq_error += error * error;
        count++;
    }

    return sqrt(sum_sq_error / count);
}

#endif /* LID_DRIVEN_CAVITY_COMMON_H */
