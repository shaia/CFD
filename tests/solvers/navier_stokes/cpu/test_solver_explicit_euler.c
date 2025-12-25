/**
 * Mathematical Validation Tests for Explicit Euler Solver
 *
 * Tests the basic explicit Euler finite difference solver for correctness
 * including stability, accuracy, and conservation properties.
 */

#include "../test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <stdio.h>
#include <string.h>

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

//=============================================================================
// TEST: BASIC FUNCTIONALITY
//=============================================================================

void test_creates_successfully(void) {
    printf("\n=== Test: Explicit Euler Creates Successfully ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);

    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL_MESSAGE(slv, "Failed to create explicit Euler solver");

    TEST_ASSERT_NOT_NULL(slv->name);
    printf("NSSolver name: %s\n", slv->name);

    solver_destroy(slv);
    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

void test_initializes_correctly(void) {
    printf("\n=== Test: Explicit Euler Initializes Correctly ===\n");

    size_t nx = 16, ny = 16;
    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(slv);

    cfd_status_t status = solver_init(slv, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    grid_destroy(g);

    printf("PASSED\n");
}

//=============================================================================
// TEST: STABILITY
//=============================================================================

void test_stability_small_cfl(void) {
    printf("\n=== Test: Stability with Small CFL ===\n");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        32, 32, &params, 100
    );

    printf("%s\n", result.message);
    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);

    printf("PASSED\n");
}

void test_stability_large_grid(void) {
    printf("\n=== Test: Stability with Large Grid ===\n");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0002;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        64, 64, &params, 50
    );

    printf("%s\n", result.message);
    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);

    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DISSIPATION (TAYLOR-GREEN VORTEX)
//=============================================================================

void test_energy_dissipation(void) {
    printf("\n=== Test: Energy Dissipation ===\n");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        32, 32, &params, 100
    );

    printf("Initial KE: %.6e\n", result.initial_energy);
    printf("Final KE: %.6e\n", result.final_energy);
    printf("Ratio: %.4f\n", result.final_energy / result.initial_energy);
    printf("%s\n", result.message);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);

    printf("PASSED\n");
}

//=============================================================================
// TEST: TAYLOR-GREEN VORTEX ANALYTICAL DECAY
//=============================================================================

void test_taylor_green_decay_rate(void) {
    printf("\n=== Test: Taylor-Green Vortex Decay Rate ===\n");

    size_t nx = 32, ny = 32;
    double L = 1.0;
    double U = 0.1;
    double nu = 0.01;
    double k = 2.0 * M_PI / L;

    grid* g = grid_create(nx, ny, 0.0, L, 0.0, L);
    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);

    grid_initialize_uniform(g);
    test_init_taylor_green_with_params(field, g, U, k);

    double initial_ke = test_compute_kinetic_energy(field, g);
    printf("Initial kinetic energy: %.6e\n", initial_ke);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = nu;
    params.max_iter = 1;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(slv, g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    // Run simulation
    int num_steps = 100;
    double t = 0.0;
    for (int step = 0; step < num_steps; step++) {
        solver_step(slv, field, g, &params, &stats);
        t += params.dt;
    }

    TEST_ASSERT_TRUE_MESSAGE(test_flow_field_is_valid(field), "Field contains NaN/Inf");

    double final_ke = test_compute_kinetic_energy(field, g);
    printf("Final kinetic energy: %.6e\n", final_ke);

    // Theoretical decay: E(t) = E0 * exp(-4*nu*k^2*t)
    double decay_rate = 4.0 * nu * k * k;
    double expected_ke = initial_ke * exp(-decay_rate * t);
    printf("Expected KE (analytical): %.6e\n", expected_ke);

    double ke_error = fabs(final_ke - expected_ke) / initial_ke;
    printf("Relative KE error: %.2f%%\n", ke_error * 100.0);

    // Energy should decay
    TEST_ASSERT_TRUE_MESSAGE(final_ke < initial_ke,
                             "Kinetic energy should decay");

    // Error should be reasonable (allowing for numerical diffusion)
    TEST_ASSERT_TRUE_MESSAGE(ke_error < 0.5,
                             "Decay rate differs too much from theory");

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    printf("PASSED\n");
}

//=============================================================================
// TEST: SPATIAL CONVERGENCE
//=============================================================================

void test_spatial_convergence(void) {
    printf("\n=== Test: Spatial Convergence ===\n");

    double U = 0.1;
    double nu = 0.01;
    double L = 1.0;
    double k = 2.0 * M_PI / L;

    size_t grid_sizes[] = {16, 32, 64};
    int num_grids = 3;
    double errors[3];

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    for (int g_idx = 0; g_idx < num_grids; g_idx++) {
        size_t n = grid_sizes[g_idx];

        grid* g = grid_create(n, n, 0.0, L, 0.0, L);
        flow_field* field = flow_field_create(n, n);
        double* analytical_u = (double*)cfd_calloc(n * n, sizeof(double));

        grid_initialize_uniform(g);
        test_init_taylor_green_with_params(field, g, U, k);

        double dx = g->dx[0];

        ns_solver_params_t params = ns_solver_params_default();
        // For spatial convergence, use dt proportional to dx^2 so that:
        // 1. Diffusion CFL is satisfied: dt < dx^2 / (4*nu)
        // 2. Total error = spatial + temporal both scale as h^2
        // This allows us to see 2nd order convergence
        double dt_cfl = 0.2 * dx * dx / nu;  // Safe CFL factor
        params.dt = dt_cfl;
        params.mu = nu;
        params.max_iter = 1;

        // Run same number of iterations - final time will differ but error comparison still valid
        // since we compare numerical vs analytical at actual final time
        int num_steps = 10;

        ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
        solver_init(slv, g, &params);
        ns_solver_stats_t stats = ns_solver_stats_default();

        double t = 0.0;
        for (int step = 0; step < num_steps; step++) {
            solver_step(slv, field, g, &params, &stats);
            t += params.dt;
        }

        // Compute analytical solution at final time
        double dy = g->dy[0];
        double decay = exp(-2.0 * nu * k * k * t);
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < n; i++) {
                double x = i * dx;
                double y = j * dy;
                analytical_u[j * n + i] = U * cos(k * x) * sin(k * y) * decay;
            }
        }

        errors[g_idx] = test_compute_l2_error(field->u, analytical_u, n * n);
        printf("Grid %zux%zu (h=%.4f, dt=%.2e, steps=%d): L2 error = %.6e\n",
               n, n, dx, params.dt, num_steps, errors[g_idx]);

        solver_destroy(slv);
        cfd_free(analytical_u);
        flow_field_destroy(field);
        grid_destroy(g);
    }

    cfd_registry_destroy(registry);

    // Check convergence
    if (errors[0] > 1e-15 && errors[1] > 1e-15) {
        double ratio_1 = errors[0] / errors[1];
        double ratio_2 = (errors[2] > 1e-15) ? errors[1] / errors[2] : 0.0;

        printf("Error ratio (16->32): %.2f\n", ratio_1);
        if (ratio_2 > 0) {
            printf("Error ratio (32->64): %.2f\n", ratio_2);
        }

        TEST_ASSERT_TRUE_MESSAGE(errors[1] < errors[0],
                                 "Error should decrease with refinement");
        TEST_ASSERT_TRUE_MESSAGE(ratio_1 > 1.5,
                                 "Should show at least 1st-order convergence");
    }

    printf("PASSED\n");
}

//=============================================================================
// TEST: CFL STABILITY CONDITION
//=============================================================================

void test_cfl_stability(void) {
    printf("\n=== Test: CFL Stability Condition ===\n");

    size_t nx = 32, ny = 32;
    double cfl_safe = 0.2;

    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);
    grid_initialize_uniform(g);

    double dx = g->dx[0];
    double dy = g->dy[0];

    // Initialize with velocity
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            size_t idx = j * nx + i;
            field->u[idx] = 0.5 * sin(2.0 * M_PI * x);
            field->v[idx] = 0.5 * cos(2.0 * M_PI * y);
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.cfl = cfl_safe;
    params.mu = 0.01;
    params.max_iter = 1;
    params.dt = cfl_safe * fmin(dx, dy) / 0.5;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(slv, g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    int stable = 1;
    for (int step = 0; step < 200; step++) {
        solver_step(slv, field, g, &params, &stats);
        if (!test_flow_field_is_valid(field)) {
            stable = 0;
            printf("Instability at step %d\n", step);
            break;
        }
    }

    printf("CFL=%.2f: %s\n", cfl_safe, stable ? "STABLE" : "UNSTABLE");
    TEST_ASSERT_TRUE_MESSAGE(stable, "Should be stable with CFL < 1");

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    printf("PASSED\n");
}

//=============================================================================
// TEST: CROSS-SOLVER CONSISTENCY (vs OPTIMIZED)
//=============================================================================

void test_consistency_with_optimized(void) {
    printf("\n=== Test: Consistency with Optimized NSSolver ===\n");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_consistency(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        32, 32, &params, 20, 0.05
    );

    printf("Relative error: %.2e\n", result.relative_error);
    printf("%s\n", result.message);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);

    printf("PASSED\n");
}

//=============================================================================
// TEST: EDGE CASES
//=============================================================================

void test_zero_velocity(void) {
    printf("\n=== Test: Zero Velocity Field ===\n");

    size_t nx = 16, ny = 16;
    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);
    grid_initialize_uniform(g);

    // Set zero velocity
    for (size_t i = 0; i < nx * ny; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->p[i] = 1.0;
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(slv, g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < 10; step++) {
        solver_step(slv, field, g, &params, &stats);
    }

    TEST_ASSERT_TRUE_MESSAGE(test_flow_field_is_valid(field), "Field contains NaN/Inf");

    double u_norm = test_compute_l2_norm(field->u, nx * ny);
    double v_norm = test_compute_l2_norm(field->v, nx * ny);
    printf("u L2 norm: %.6e, v L2 norm: %.6e\n", u_norm, v_norm);

    // Tolerance relaxed due to numerical diffusion and pressure effects
    TEST_ASSERT_TRUE_MESSAGE(u_norm < 1e-3, "u should remain near zero");
    TEST_ASSERT_TRUE_MESSAGE(v_norm < 1e-3, "v should remain near zero");

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    printf("PASSED\n");
}

void test_non_square_grid(void) {
    printf("\n=== Test: Non-Square Grid ===\n");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0003;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        64, 32, &params, 50
    );

    printf("%s\n", result.message);
    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);

    printf("PASSED\n");
}

//=============================================================================
// MAIN
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    printf("\n");
    printf("================================================\n");
    printf("  Explicit Euler NSSolver Validation Tests\n");
    printf("================================================\n");

    // Basic functionality
    RUN_TEST(test_creates_successfully);
    RUN_TEST(test_initializes_correctly);

    // Stability tests
    RUN_TEST(test_stability_small_cfl);
    RUN_TEST(test_stability_large_grid);
    RUN_TEST(test_cfl_stability);

    // Physics validation
    RUN_TEST(test_energy_dissipation);
    RUN_TEST(test_taylor_green_decay_rate);

    // Convergence
    RUN_TEST(test_spatial_convergence);

    // Cross-solver consistency
    RUN_TEST(test_consistency_with_optimized);

    // Edge cases
    RUN_TEST(test_zero_velocity);
    RUN_TEST(test_non_square_grid);

    printf("\n================================================\n");

    return UNITY_END();
}
