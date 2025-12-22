/**
 * @file lid_driven_cavity_common.h
 * @brief Shared utilities for lid-driven cavity validation tests
 *
 * Contains Ghia et al. reference data, test context management,
 * and common simulation utilities.
 */

#ifndef LID_DRIVEN_CAVITY_COMMON_H
#define LID_DRIVEN_CAVITY_COMMON_H

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
 * At Re=100 with 33x33 grid:
 *   - 5000 steps achieves RMS ~0.03 (good for CI)
 *   - 10000 steps achieves RMS ~0.01 (publication quality)
 * Using dt=0.0005 for stability */
#define FAST_STEPS      2000
#define MEDIUM_STEPS    3000
#define FULL_STEPS      5000
#define FINE_DT         0.0005
#endif

/* ============================================================================
 * GHIA ET AL. REFERENCE DATA (1982)
 * ============================================================================ */

typedef struct {
    const double* coords;
    const double* values;
    size_t n;
} ghia_profile_t;

/* y-coordinates for vertical centerline data */
static const double ghia_y_coords[] = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
};

/* u-velocity along vertical centerline at Re=100 */
static const double ghia_u_re100[] = {
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
    0.78871, 0.84123, 1.00000
};

/* x-coordinates for horizontal centerline data */
static const double ghia_x_coords[] = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000
};

/* v-velocity along horizontal centerline at Re=100 */
static const double ghia_v_re100[] = {
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
    0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
    -0.07391, -0.05906, 0.00000
};

#ifndef GHIA_NUM_POINTS
#define GHIA_NUM_POINTS (sizeof(ghia_y_coords) / sizeof(ghia_y_coords[0]))
#endif

/* ============================================================================
 * TEST CONTEXT
 * ============================================================================ */

typedef struct {
    grid* g;
    flow_field* field;
    size_t nx;
    size_t ny;
} cavity_context_t;

static inline cavity_context_t* cavity_context_create(size_t nx, size_t ny) {
    cavity_context_t* ctx = malloc(sizeof(cavity_context_t));
    if (!ctx) return NULL;

    ctx->nx = nx;
    ctx->ny = ny;
    ctx->g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    ctx->field = flow_field_create(nx, ny);

    if (!ctx->g || !ctx->field) {
        if (ctx->g) grid_destroy(ctx->g);
        if (ctx->field) flow_field_destroy(ctx->field);
        free(ctx);
        return NULL;
    }

    grid_initialize_uniform(ctx->g);

    /* Initialize field to quiescent state */
    size_t total = nx * ny;
    for (size_t i = 0; i < total; i++) {
        ctx->field->u[i] = 0.0;
        ctx->field->v[i] = 0.0;
        ctx->field->p[i] = 0.0;
        ctx->field->rho[i] = 1.0;
        ctx->field->T[i] = 300.0;
    }

    return ctx;
}

static inline void cavity_context_destroy(cavity_context_t* ctx) {
    if (!ctx) return;
    if (ctx->g) grid_destroy(ctx->g);
    if (ctx->field) flow_field_destroy(ctx->field);
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
    double final_residual;
    double max_velocity;
    int converged;
    int blew_up;
} simulation_result_t;

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

    solver_init(solver, ctx->g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    double prev_ke = compute_kinetic_energy(ctx->field);

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_velocity);
        solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (!check_field_finite(ctx->field)) {
            result.blew_up = 1;
            break;
        }

        double ke = compute_kinetic_energy(ctx->field);
        result.final_residual = fabs(ke - prev_ke) / (prev_ke + 1e-10);
        prev_ke = ke;
        result.steps_completed = step + 1;

        if (step > 100 && result.final_residual < 1e-8) {
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
