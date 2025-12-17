/**
 * Validation Tests for OpenMP-Parallelized Explicit Euler Solver
 *
 * Tests that OpenMP explicit Euler implementation produces results consistent
 * with the serial implementation.
 */

#include "../test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

//=============================================================================
// HELPER: Check if OpenMP solver is available
//=============================================================================

static int omp_solver_available(ns_solver_registry_t* registry, const char* type) {
    ns_solver_t* slv = cfd_solver_create(registry, type);
    if (slv == NULL) {
        return 0;
    }
    solver_destroy(slv);
    return 1;
}

//=============================================================================
// TEST: OPENMP EXPLICIT EULER CREATES
//=============================================================================

void test_omp_explicit_euler_creates(void) {
    printf("\n=== Test: OpenMP Explicit Euler Creates ===\n");

#ifdef _OPENMP
    printf("OpenMP available with max threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP not available in this build\n");
#endif

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);
    if (slv == NULL) {
        printf("OpenMP explicit Euler solver not available (expected if OpenMP disabled)\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    printf("NSSolver name: %s\n", slv->name ? slv->name : "NULL");

    solver_destroy(slv);
    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

//=============================================================================
// TEST: OPENMP VS SERIAL CONSISTENCY
//=============================================================================

void test_omp_serial_consistency(void) {
    printf("\n=== Test: OpenMP vs Serial Consistency ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP)) {
        printf("OpenMP solver not available, skipping consistency test\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_consistency(
        NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        32, 32, &params, 20, 0.01);  // 1% relative tolerance

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_l2_secondary);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: STABILITY WITH LARGE GRID
//=============================================================================

void test_omp_stability_large_grid(void) {
    printf("\n=== Test: OpenMP Stability with Large Grid ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP)) {
        printf("OpenMP solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0002;
    params.mu = 0.01;
    params.max_iter = 1;
    params.cfl = 0.2;

    // Larger grid to benefit from parallelization
    test_result result = test_run_stability(NS_SOLVER_TYPE_EXPLICIT_EULER_OMP, 128, 128, &params, 100);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("OMP solver stable for 100 steps on 128x128 grid\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: DETERMINISTIC RESULTS
//=============================================================================

void test_omp_deterministic_results(void) {
    printf("\n=== Test: OpenMP Deterministic Results ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP)) {
        printf("OpenMP solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    size_t nx = 32, ny = 32;
    size_t n = nx * ny;

    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);

    // Run same simulation twice
    flow_field* field1 = flow_field_create(nx, ny);
    flow_field* field2 = flow_field_create(nx, ny);

    test_init_taylor_green(field1, g);
    test_init_taylor_green(field2, g);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    ns_solver_t* slv1 = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);
    ns_solver_t* slv2 = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);

    solver_init(slv1, g, &params);
    solver_init(slv2, g, &params);

    ns_solver_stats_t stats1 = ns_solver_stats_default();
    ns_solver_stats_t stats2 = ns_solver_stats_default();

    // Run both with same number of steps
    for (int step = 0; step < 20; step++) {
        solver_step(slv1, field1, g, &params, &stats1);
        solver_step(slv2, field2, g, &params, &stats2);
    }

    // Results should be identical (deterministic)
    double u_diff = test_compute_l2_error(field1->u, field2->u, n);
    double v_diff = test_compute_l2_error(field1->v, field2->v, n);

    printf("Difference between two runs - u: %.6e, v: %.6e\n", u_diff, v_diff);

    // Results should be very close (allowing for floating-point non-determinism)
    TEST_ASSERT_TRUE_MESSAGE(u_diff < 1e-10, "OMP results should be deterministic");
    TEST_ASSERT_TRUE_MESSAGE(v_diff < 1e-10, "OMP results should be deterministic");

    solver_destroy(slv1);
    solver_destroy(slv2);
    cfd_registry_destroy(registry);
    flow_field_destroy(field1);
    flow_field_destroy(field2);
    grid_destroy(g);

    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DECAY
//=============================================================================

void test_omp_energy_decay(void) {
    printf("\n=== Test: OpenMP Energy Decay ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP)) {
        printf("OpenMP solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(NS_SOLVER_TYPE_EXPLICIT_EULER_OMP, 32, 32, &params, 50);

    printf("Initial kinetic energy: %.6e\n", result.initial_energy);
    printf("Final kinetic energy: %.6e\n", result.final_energy);
    printf("Energy ratio: %.4f\n", result.final_energy / result.initial_energy);

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
    printf("  OpenMP Explicit Euler NSSolver Tests\n");
    printf("================================================\n");

#ifdef _OPENMP
    printf("OpenMP enabled, max threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP not enabled in this build\n");
#endif

    RUN_TEST(test_omp_explicit_euler_creates);
    RUN_TEST(test_omp_serial_consistency);
    RUN_TEST(test_omp_stability_large_grid);
    RUN_TEST(test_omp_deterministic_results);
    RUN_TEST(test_omp_energy_decay);

    printf("\n================================================\n");

    return UNITY_END();
}
