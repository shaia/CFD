/**
 * Validation Tests for SIMD-Optimized Explicit Euler Solver
 *
 * Tests that SIMD explicit Euler implementation produces results consistent
 * with the scalar implementation.
 */

#include "../test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"
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
// TEST: SIMD EXPLICIT EULER CREATES SUCCESSFULLY
//=============================================================================

void test_simd_explicit_euler_creates(void) {
    printf("\n=== Test: SIMD Explicit Euler Creates ===\n");

    solver_registry* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    TEST_ASSERT_NOT_NULL_MESSAGE(slv, "Failed to create SIMD explicit Euler solver");

    printf("Solver name: %s\n", slv->name ? slv->name : "NULL");

    solver_destroy(slv);
    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

//=============================================================================
// TEST: SIMD VS SCALAR CONSISTENCY
//=============================================================================

void test_simd_scalar_consistency(void) {
    printf("\n=== Test: SIMD vs Scalar Consistency ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;
    params.cfl = 0.2;

    test_result result = test_run_consistency(
        SOLVER_TYPE_EXPLICIT_EULER, SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        32, 32, &params, 20, 0.01);  // 1% relative tolerance

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_l2_secondary);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: SIMD STABILITY
//=============================================================================

void test_simd_stability(void) {
    printf("\n=== Test: SIMD Solver Stability ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;
    params.cfl = 0.2;

    test_result result = test_run_stability(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, 64, 64, &params, 200);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("SIMD solver remained stable for 200 steps\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: SIMD ALIGNMENT HANDLING
//=============================================================================

void test_simd_non_aligned_grid_size(void) {
    printf("\n=== Test: SIMD Non-Aligned Grid Size ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // Use grid size that's not a multiple of 4 (typical SIMD width)
    test_result result = test_run_stability(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, 33, 35, &params, 10);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("SIMD handles 33x35 grid correctly\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DECAY
//=============================================================================

void test_simd_energy_decay(void) {
    printf("\n=== Test: SIMD Energy Decay ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, 32, 32, &params, 50);

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
    printf("  SIMD Explicit Euler Solver Tests\n");
    printf("================================================\n");

    RUN_TEST(test_simd_explicit_euler_creates);
    RUN_TEST(test_simd_scalar_consistency);
    RUN_TEST(test_simd_stability);
    RUN_TEST(test_simd_non_aligned_grid_size);
    RUN_TEST(test_simd_energy_decay);

    printf("\n================================================\n");

    return UNITY_END();
}
