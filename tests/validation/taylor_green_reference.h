/**
 * @file taylor_green_reference.h
 * @brief Taylor-Green vortex analytical solution and validation utilities
 *
 * The Taylor-Green vortex is an exact solution to the incompressible Navier-Stokes
 * equations. It provides a rigorous test for numerical solvers because the analytical
 * solution is known for all time.
 *
 * Analytical solution on domain [0, 2π] × [0, 2π]:
 *   u(x,y,t) = cos(x) * sin(y) * exp(-2νt)
 *   v(x,y,t) = -sin(x) * cos(y) * exp(-2νt)
 *   p(x,y,t) = -0.25 * (cos(2x) + cos(2y)) * exp(-4νt)
 *
 * Key properties:
 *   - Velocity decays as exp(-2νt)
 *   - Pressure decays as exp(-4νt)
 *   - Kinetic energy decays as exp(-4νt)
 *   - Divergence-free: ∇·u = 0
 *   - Vorticity decays as exp(-2νt)
 *
 * References:
 *   Taylor, G.I., Green, A.E. (1937). "Mechanism of the production of small eddies
 *   from large ones". Proc. R. Soc. Lond. A 158, 499-521.
 */

#ifndef TAYLOR_GREEN_REFERENCE_H
#define TAYLOR_GREEN_REFERENCE_H

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Domain for Taylor-Green vortex: [0, 2π] × [0, 2π] */
#define TG_DOMAIN_XMIN  0.0
#define TG_DOMAIN_XMAX  (2.0 * M_PI)
#define TG_DOMAIN_YMIN  0.0
#define TG_DOMAIN_YMAX  (2.0 * M_PI)

/* Default test parameters
 * Note: Use modest values for CI. Full validation uses larger grids/more steps. */
#define TG_DEFAULT_NU       0.01    /* Kinematic viscosity */
#define TG_DEFAULT_NX       32      /* Grid points in x */
#define TG_DEFAULT_NY       32      /* Grid points in y */
#define TG_DEFAULT_DT       0.001   /* Time step */
#define TG_DEFAULT_STEPS    200     /* Number of time steps (reduced for CI) */

/* Tolerances for validation */
#define TG_VELOCITY_DECAY_TOL   0.05    /* 5% tolerance on decay rate */
#define TG_PRESSURE_DECAY_TOL   0.10    /* 10% tolerance on pressure decay */
#define TG_KE_DECAY_TOL         0.05    /* 5% tolerance on kinetic energy decay */
#define TG_DIVERGENCE_TOL       0.1     /* Max acceptable divergence (projection method limit) */
#define TG_L2_ERROR_TOL         0.15    /* L2 error tolerance (relative to initial) */

/* ============================================================================
 * ANALYTICAL SOLUTION FUNCTIONS
 * ============================================================================ */

/**
 * Compute analytical u-velocity at (x, y, t)
 */
static inline double tg_analytical_u(double x, double y, double t, double nu) {
    return cos(x) * sin(y) * exp(-2.0 * nu * t);
}

/**
 * Compute analytical v-velocity at (x, y, t)
 */
static inline double tg_analytical_v(double x, double y, double t, double nu) {
    return -sin(x) * cos(y) * exp(-2.0 * nu * t);
}

/**
 * Compute analytical pressure at (x, y, t)
 */
static inline double tg_analytical_p(double x, double y, double t, double nu) {
    return -0.25 * (cos(2.0 * x) + cos(2.0 * y)) * exp(-4.0 * nu * t);
}

/**
 * Compute analytical vorticity at (x, y, t)
 * ω = ∂v/∂x - ∂u/∂y = -2 * cos(x) * cos(y) * exp(-2νt)
 */
static inline double tg_analytical_vorticity(double x, double y, double t, double nu) {
    return -2.0 * cos(x) * cos(y) * exp(-2.0 * nu * t);
}

/**
 * Compute analytical kinetic energy at time t
 * KE(t) = ∫∫ 0.5 * (u² + v²) dx dy = (2π)² / 4 * exp(-4νt) = π² * exp(-4νt)
 * For domain [0, 2π]²
 */
static inline double tg_analytical_ke(double t, double nu) {
    return M_PI * M_PI * exp(-4.0 * nu * t);
}

/**
 * Compute expected velocity decay factor at time t
 */
static inline double tg_velocity_decay_factor(double t, double nu) {
    return exp(-2.0 * nu * t);
}

/**
 * Compute expected pressure decay factor at time t
 */
static inline double tg_pressure_decay_factor(double t, double nu) {
    return exp(-4.0 * nu * t);
}

/* ============================================================================
 * TEST CONTEXT
 * ============================================================================ */

typedef struct {
    grid* g;
    flow_field* field;
    size_t nx;
    size_t ny;
    double nu;
} tg_context_t;

/**
 * Create Taylor-Green test context
 */
static inline tg_context_t* tg_context_create(size_t nx, size_t ny, double nu) {
    tg_context_t* ctx = malloc(sizeof(tg_context_t));
    if (!ctx) return NULL;

    ctx->nx = nx;
    ctx->ny = ny;
    ctx->nu = nu;

    ctx->g = grid_create(nx, ny,
                         TG_DOMAIN_XMIN, TG_DOMAIN_XMAX,
                         TG_DOMAIN_YMIN, TG_DOMAIN_YMAX);
    ctx->field = flow_field_create(nx, ny);

    if (!ctx->g || !ctx->field) {
        if (ctx->g) grid_destroy(ctx->g);
        if (ctx->field) flow_field_destroy(ctx->field);
        free(ctx);
        return NULL;
    }

    grid_initialize_uniform(ctx->g);

    return ctx;
}

/**
 * Destroy Taylor-Green test context
 */
static inline void tg_context_destroy(tg_context_t* ctx) {
    if (!ctx) return;
    if (ctx->g) grid_destroy(ctx->g);
    if (ctx->field) flow_field_destroy(ctx->field);
    free(ctx);
}

/**
 * Initialize flow field with Taylor-Green vortex at t=0
 */
static inline void tg_initialize_field(tg_context_t* ctx) {
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = ctx->g->x[i];
            double y = ctx->g->y[j];
            size_t idx = j * nx + i;

            ctx->field->u[idx] = tg_analytical_u(x, y, 0.0, ctx->nu);
            ctx->field->v[idx] = tg_analytical_v(x, y, 0.0, ctx->nu);
            ctx->field->p[idx] = tg_analytical_p(x, y, 0.0, ctx->nu);
            ctx->field->rho[idx] = 1.0;
            ctx->field->T[idx] = 300.0;
        }
    }
}

/* ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================
 * Taylor-Green vortex uses periodic boundary conditions
 */

static inline void tg_apply_bc(flow_field* field) {
    bc_apply_periodic(field->u, field->nx, field->ny);
    bc_apply_periodic(field->v, field->nx, field->ny);
    bc_apply_periodic(field->p, field->nx, field->ny);
}

/* ============================================================================
 * ERROR METRICS
 * ============================================================================ */

/**
 * Compute L2 error between numerical and analytical solution
 */
static inline double tg_compute_l2_error_u(const tg_context_t* ctx, double t) {
    double sum_sq_error = 0.0;
    double sum_sq_exact = 0.0;
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = ctx->g->x[i];
            double y = ctx->g->y[j];
            size_t idx = j * nx + i;

            double u_exact = tg_analytical_u(x, y, t, ctx->nu);
            double u_num = ctx->field->u[idx];
            double error = u_num - u_exact;

            sum_sq_error += error * error;
            sum_sq_exact += u_exact * u_exact;
        }
    }

    if (sum_sq_exact < 1e-15) return sqrt(sum_sq_error / (nx * ny));
    return sqrt(sum_sq_error / sum_sq_exact);
}

static inline double tg_compute_l2_error_v(const tg_context_t* ctx, double t) {
    double sum_sq_error = 0.0;
    double sum_sq_exact = 0.0;
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = ctx->g->x[i];
            double y = ctx->g->y[j];
            size_t idx = j * nx + i;

            double v_exact = tg_analytical_v(x, y, t, ctx->nu);
            double v_num = ctx->field->v[idx];
            double error = v_num - v_exact;

            sum_sq_error += error * error;
            sum_sq_exact += v_exact * v_exact;
        }
    }

    if (sum_sq_exact < 1e-15) return sqrt(sum_sq_error / (nx * ny));
    return sqrt(sum_sq_error / sum_sq_exact);
}

/**
 * Compute maximum velocity magnitude in the field
 */
static inline double tg_compute_max_velocity(const tg_context_t* ctx) {
    double max_vmag = 0.0;
    size_t total = ctx->nx * ctx->ny;

    for (size_t i = 0; i < total; i++) {
        double vmag = sqrt(ctx->field->u[i] * ctx->field->u[i] +
                          ctx->field->v[i] * ctx->field->v[i]);
        if (vmag > max_vmag) max_vmag = vmag;
    }
    return max_vmag;
}

/**
 * Compute kinetic energy of the flow field
 * KE = ∫∫ 0.5 * ρ * (u² + v²) dx dy
 * For discrete field: KE ≈ 0.5 * Σ (u² + v²) * dx * dy
 */
static inline double tg_compute_kinetic_energy(const tg_context_t* ctx) {
    double ke = 0.0;
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;
    double dx = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (nx - 1);
    double dy = (TG_DOMAIN_YMAX - TG_DOMAIN_YMIN) / (ny - 1);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double u = ctx->field->u[idx];
            double v = ctx->field->v[idx];
            ke += 0.5 * (u * u + v * v) * dx * dy;
        }
    }
    return ke;
}

/**
 * Compute maximum divergence of velocity field
 * This should be close to zero for incompressible flow
 */
static inline double tg_compute_max_divergence(const tg_context_t* ctx) {
    double max_div = 0.0;
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;
    double dx = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (nx - 1);
    double dy = (TG_DOMAIN_YMAX - TG_DOMAIN_YMIN) / (ny - 1);

    /* Compute divergence using central differences, skip boundaries */
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double du_dx = (ctx->field->u[idx + 1] - ctx->field->u[idx - 1]) / (2.0 * dx);
            double dv_dy = (ctx->field->v[idx + nx] - ctx->field->v[idx - nx]) / (2.0 * dy);
            double div = fabs(du_dx + dv_dy);
            if (div > max_div) max_div = div;
        }
    }
    return max_div;
}

/* ============================================================================
 * SIMULATION RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    /* Status */
    int success;
    char error_msg[256];

    /* Error metrics at final time */
    double l2_error_u;
    double l2_error_v;
    double max_divergence;

    /* Decay rate validation */
    double measured_velocity_decay;
    double expected_velocity_decay;
    double measured_ke_decay;
    double expected_ke_decay;

    /* Field statistics */
    double initial_ke;
    double final_ke;
    double initial_max_velocity;
    double final_max_velocity;

    /* Simulation info */
    int steps_completed;
    double final_time;
} tg_result_t;

/* ============================================================================
 * SIMULATION RUNNER
 * ============================================================================ */

/**
 * Run Taylor-Green vortex simulation with specified solver
 *
 * @param solver_type  Solver type string (e.g., NS_SOLVER_TYPE_PROJECTION)
 * @param nx, ny       Grid dimensions
 * @param nu           Kinematic viscosity
 * @param dt           Time step
 * @param max_steps    Number of time steps
 * @return Simulation result with validation metrics
 */
static inline tg_result_t tg_run_simulation(
    const char* solver_type,
    size_t nx, size_t ny,
    double nu, double dt, int max_steps)
{
    tg_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    /* Create context */
    tg_context_t* ctx = tg_context_create(nx, ny, nu);
    if (!ctx) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create context");
        return result;
    }

    /* Initialize with analytical solution at t=0 */
    tg_initialize_field(ctx);

    /* Record initial state */
    result.initial_ke = tg_compute_kinetic_energy(ctx);
    result.initial_max_velocity = tg_compute_max_velocity(ctx);

    /* Set up solver parameters */
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

    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver '%s' not available", solver_type);
        cfd_registry_destroy(registry);
        tg_context_destroy(ctx);
        return result;
    }

    solver_init(solver, ctx->g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    /* Run simulation */
    for (int step = 0; step < max_steps; step++) {
        tg_apply_bc(ctx->field);
        solver_step(solver, ctx->field, ctx->g, &params, &stats);

        /* Check for numerical blowup */
        double max_v = tg_compute_max_velocity(ctx);
        if (!isfinite(max_v) || max_v > 1e6) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            tg_context_destroy(ctx);
            return result;
        }

        result.steps_completed = step + 1;
    }

    /* Compute final time */
    result.final_time = max_steps * dt;

    /* Record final state */
    result.final_ke = tg_compute_kinetic_energy(ctx);
    result.final_max_velocity = tg_compute_max_velocity(ctx);

    /* Compute error metrics */
    result.l2_error_u = tg_compute_l2_error_u(ctx, result.final_time);
    result.l2_error_v = tg_compute_l2_error_v(ctx, result.final_time);
    result.max_divergence = tg_compute_max_divergence(ctx);

    /* Compute decay rates */
    /* Velocity decay: u(t) / u(0) should be exp(-2νt) */
    if (result.initial_max_velocity > 1e-10) {
        result.measured_velocity_decay = result.final_max_velocity / result.initial_max_velocity;
    } else {
        result.measured_velocity_decay = 0.0;
    }
    result.expected_velocity_decay = tg_velocity_decay_factor(result.final_time, nu);

    /* KE decay: KE(t) / KE(0) should be exp(-4νt) */
    if (result.initial_ke > 1e-10) {
        result.measured_ke_decay = result.final_ke / result.initial_ke;
    } else {
        result.measured_ke_decay = 0.0;
    }
    result.expected_ke_decay = tg_pressure_decay_factor(result.final_time, nu);  /* same as KE */

    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    tg_context_destroy(ctx);

    return result;
}

/* ============================================================================
 * PRINT UTILITIES
 * ============================================================================ */

static inline void tg_print_result(const tg_result_t* result, const char* solver_name) {
    printf("\n    %s Taylor-Green Vortex Validation:\n", solver_name);
    printf("      Steps: %d, Final time: %.4f\n", result->steps_completed, result->final_time);
    printf("\n      Velocity decay:\n");
    if (result->expected_velocity_decay > 1e-15) {
        printf("        Measured: %.6f, Expected: %.6f (error: %.2f%%)\n",
               result->measured_velocity_decay, result->expected_velocity_decay,
               100.0 * fabs(result->measured_velocity_decay - result->expected_velocity_decay) /
                   result->expected_velocity_decay);
    } else {
        printf("        Measured: %.6f, Expected: %.6f\n",
               result->measured_velocity_decay, result->expected_velocity_decay);
    }
    printf("\n      Kinetic energy decay:\n");
    if (result->expected_ke_decay > 1e-15) {
        printf("        Measured: %.6f, Expected: %.6f (error: %.2f%%)\n",
               result->measured_ke_decay, result->expected_ke_decay,
               100.0 * fabs(result->measured_ke_decay - result->expected_ke_decay) /
                   result->expected_ke_decay);
    } else {
        printf("        Measured: %.6f, Expected: %.6f\n",
               result->measured_ke_decay, result->expected_ke_decay);
    }
    printf("\n      L2 errors:\n");
    printf("        u: %.6f, v: %.6f\n", result->l2_error_u, result->l2_error_v);
    printf("\n      Max divergence: %.2e\n", result->max_divergence);
}

#endif /* TAYLOR_GREEN_REFERENCE_H */
