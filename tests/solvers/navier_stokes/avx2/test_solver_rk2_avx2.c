/**
 * Validation Tests for AVX2-Optimized RK2 Solver
 *
 * Tests that AVX2 RK2 implementation produces results consistent
 * with the scalar RK2 implementation.
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

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

//=============================================================================
// HELPER: Check if solver is available (registered and init succeeds)
//=============================================================================

static int avx2_rk2_solver_available(ns_solver_registry_t* registry, const char* type) {
    ns_solver_t* slv = cfd_solver_create(registry, type);
    if (!slv) {
        return 0;
    }

    grid* g = grid_create(8, 8, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    ns_solver_params_t p = ns_solver_params_default();

    cfd_status_t st = solver_init(slv, g, &p);
    solver_destroy(slv);
    grid_destroy(g);

    return (st == CFD_SUCCESS);
}

//=============================================================================
// TEST: AVX2 RK2 CREATES
//=============================================================================

void test_avx2_rk2_creates(void) {
    printf("\n=== Test: AVX2 RK2 Creates ===\n");

#ifdef CFD_HAS_AVX2
    printf("AVX2 available in this build\n");
#else
    printf("AVX2 not available in this build\n");
#endif

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_RK2_OPTIMIZED);
    if (!slv) {
        printf("AVX2 RK2 solver not in registry\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    printf("Solver name: %s\n", slv->name ? slv->name : "NULL");

    /* Try to init with a small grid to verify AVX2 availability */
    grid* g = grid_create(8, 8, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    ns_solver_params_t params = ns_solver_params_default();

    cfd_status_t st = solver_init(slv, g, &params);
    if (st == CFD_ERROR_UNSUPPORTED) {
        printf("AVX2 RK2 solver not available (AVX2 disabled at compile time)\n");
        solver_destroy(slv);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);

    solver_destroy(slv);
    grid_destroy(g);
    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

//=============================================================================
// TEST: AVX2 VS SCALAR CONSISTENCY
//=============================================================================

void test_avx2_rk2_scalar_consistency(void) {
    printf("\n=== Test: AVX2 RK2 vs Scalar Consistency ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!avx2_rk2_solver_available(registry, NS_SOLVER_TYPE_RK2_OPTIMIZED)) {
        printf("AVX2 RK2 solver not available, skipping consistency test\n");
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
        NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OPTIMIZED,
        32, 32, &params, 20, 0.01);  /* 1% relative tolerance */

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_l2_secondary);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: STABILITY WITH LARGE GRID
//=============================================================================

void test_avx2_rk2_stability(void) {
    printf("\n=== Test: AVX2 RK2 Stability with Large Grid ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!avx2_rk2_solver_available(registry, NS_SOLVER_TYPE_RK2_OPTIMIZED)) {
        printf("AVX2 RK2 solver not available, skipping\n");
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

    test_result result = test_run_stability(NS_SOLVER_TYPE_RK2_OPTIMIZED, 128, 128, &params, 100);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("AVX2 RK2 solver stable for 100 steps on 128x128 grid\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DECAY
//=============================================================================

void test_avx2_rk2_energy_decay(void) {
    printf("\n=== Test: AVX2 RK2 Energy Decay ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!avx2_rk2_solver_available(registry, NS_SOLVER_TYPE_RK2_OPTIMIZED)) {
        printf("AVX2 RK2 solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(NS_SOLVER_TYPE_RK2_OPTIMIZED, 32, 32, &params, 50);

    printf("Initial kinetic energy: %.6e\n", result.initial_energy);
    printf("Final kinetic energy:   %.6e\n", result.final_energy);
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
    printf("  AVX2 RK2 NSSolver Tests\n");
    printf("================================================\n");

#ifdef CFD_HAS_AVX2
    printf("AVX2 enabled in this build\n");
#else
    printf("AVX2 not enabled in this build\n");
#endif

    RUN_TEST(test_avx2_rk2_creates);
    RUN_TEST(test_avx2_rk2_scalar_consistency);
    RUN_TEST(test_avx2_rk2_stability);
    RUN_TEST(test_avx2_rk2_energy_decay);

    printf("\n================================================\n");

    return UNITY_END();
}
