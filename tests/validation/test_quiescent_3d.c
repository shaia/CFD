/**
 * @file test_quiescent_3d.c
 * @brief 3D quiescent (zero-velocity) flow stability tests
 *
 * Verifies that a zero-velocity 3D field remains at rest — no spurious
 * velocities from BC application, solver artifacts, or 3D indexing bugs.
 *
 * Tests:
 *   - Zero field remains zero across multiple solvers
 *   - Uniform pressure field generates no spurious velocity
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define Q3D_NX  8
#define Q3D_NY  8
#define Q3D_NZ  8
#define Q3D_DT  0.001
#define Q3D_STEPS 50
#define Q3D_NU  0.01

/* Quiescent flow should produce velocities below machine epsilon level */
#define Q3D_VELOCITY_TOL  1e-10

/* ============================================================================
 * HELPERS
 * ============================================================================ */

static double q3d_max_velocity(const flow_field* field) {
    size_t total = field->nx * field->ny * field->nz;
    double max_v = 0.0;
    for (size_t i = 0; i < total; i++) {
        double u = fabs(field->u[i]);
        double v = fabs(field->v[i]);
        double w = field->w ? fabs(field->w[i]) : 0.0;
        double vmag = u > v ? u : v;
        if (w > vmag) vmag = w;
        if (vmag > max_v) max_v = vmag;
    }
    return max_v;
}

typedef struct {
    int success;
    int solver_unavailable;
    char error_msg[256];
    double max_velocity;
    int steps_completed;
} q3d_result_t;

static q3d_result_t q3d_run_zero_field(const char* solver_type) {
    q3d_result_t result = {0};
    result.success = 0;

    grid* g = grid_create(Q3D_NX, Q3D_NY, Q3D_NZ,
                          0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(Q3D_NX, Q3D_NY, Q3D_NZ);
    if (!g || !field) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create grid/field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }

    grid_initialize_uniform(g);

    /* Initialize to zero everywhere */
    size_t total = Q3D_NX * Q3D_NY * Q3D_NZ;
    for (size_t i = 0; i < total; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->w[i] = 0.0;
        field->p[i] = 0.0;
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }

    ns_solver_params_t params = {
        .dt = Q3D_DT,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = Q3D_NU,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create solver registry");
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver '%s' not available", solver_type);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    cfd_status_t init_status = solver_init(solver, g, &params);
    if (init_status == CFD_ERROR_UNSUPPORTED) {
        result.solver_unavailable = 1;
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver '%s' returned UNSUPPORTED", solver_type);
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }
    if (init_status != CFD_SUCCESS) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver init failed: %d", init_status);
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < Q3D_STEPS; step++) {
        cfd_status_t step_status = solver_step(solver, field, g, &params, &stats);
        if (step_status != CFD_SUCCESS) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver step failed at step %d: %d", step, step_status);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            flow_field_destroy(field);
            grid_destroy(g);
            return result;
        }
        result.steps_completed = step + 1;
    }

    result.max_velocity = q3d_max_velocity(field);
    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

/* ============================================================================
 * SETUP / TEARDOWN
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * TESTS
 * ============================================================================ */

void test_quiescent_3d_projection(void) {
    printf("\n    Testing 3D quiescent flow (projection)...\n");

    q3d_result_t result = q3d_run_zero_field(NS_SOLVER_TYPE_PROJECTION);

    if (result.solver_unavailable) {
        TEST_IGNORE_MESSAGE("Solver unavailable");
    }
    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Max velocity after %d steps: %.2e (tolerance: %.2e)\n",
           result.steps_completed, result.max_velocity, Q3D_VELOCITY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity < Q3D_VELOCITY_TOL,
        "Spurious velocity in quiescent 3D flow (projection)");
}

void test_quiescent_3d_explicit_euler(void) {
    printf("\n    Testing 3D quiescent flow (explicit_euler)...\n");

    q3d_result_t result = q3d_run_zero_field(NS_SOLVER_TYPE_EXPLICIT_EULER);

    if (result.solver_unavailable) {
        TEST_IGNORE_MESSAGE("Solver unavailable");
    }
    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Max velocity after %d steps: %.2e (tolerance: %.2e)\n",
           result.steps_completed, result.max_velocity, Q3D_VELOCITY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity < Q3D_VELOCITY_TOL,
        "Spurious velocity in quiescent 3D flow (explicit_euler)");
}

void test_quiescent_3d_rk2(void) {
    printf("\n    Testing 3D quiescent flow (rk2)...\n");

    q3d_result_t result = q3d_run_zero_field(NS_SOLVER_TYPE_RK2);

    if (result.solver_unavailable) {
        TEST_IGNORE_MESSAGE("Solver unavailable");
    }
    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Max velocity after %d steps: %.2e (tolerance: %.2e)\n",
           result.steps_completed, result.max_velocity, Q3D_VELOCITY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity < Q3D_VELOCITY_TOL,
        "Spurious velocity in quiescent 3D flow (rk2)");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_quiescent_3d_projection);
    RUN_TEST(test_quiescent_3d_explicit_euler);
    RUN_TEST(test_quiescent_3d_rk2);

    return UNITY_END();
}
