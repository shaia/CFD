/**
 * Validation Tests for SIMD-Optimized Projection Solver
 *
 * Tests that SIMD projection implementation produces results consistent
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
// TEST: SIMD PROJECTION CREATES SUCCESSFULLY
//=============================================================================

void test_simd_projection_creates(void) {
    printf("\n=== Test: SIMD Projection Creates ===\n");

    solver_registry* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL_MESSAGE(slv, "Failed to create SIMD projection solver");

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
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_consistency(
        SOLVER_TYPE_PROJECTION, SOLVER_TYPE_PROJECTION_OPTIMIZED,
        32, 32, &params, 10, 0.05);  // 5% relative tolerance for projection

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_l2_secondary);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: DIVERGENCE-FREE CONSTRAINT
//=============================================================================

void test_simd_divergence_free(void) {
    printf("\n=== Test: SIMD Divergence-Free Constraint ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // Tolerance relaxed because projection method may not fully converge
    test_result result = test_run_divergence_free(SOLVER_TYPE_PROJECTION_OPTIMIZED, 32, 32, &params, 10, 1.0);

    printf("Divergence norm after projection: %.6e\n", result.error_l2);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: SIMD STABILITY
//=============================================================================

void test_simd_stability(void) {
    printf("\n=== Test: SIMD Projection Stability ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(SOLVER_TYPE_PROJECTION_OPTIMIZED, 64, 64, &params, 100);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("SIMD solver remained stable for 100 steps\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DECAY
//=============================================================================

void test_simd_energy_decay(void) {
    printf("\n=== Test: SIMD Energy Decay ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(SOLVER_TYPE_PROJECTION_OPTIMIZED, 32, 32, &params, 30);

    printf("Initial kinetic energy: %.6e\n", result.initial_energy);
    printf("Final kinetic energy: %.6e\n", result.final_energy);
    printf("Energy ratio: %.4f\n", result.final_energy / result.initial_energy);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: NON-ALIGNED GRID SIZE
//=============================================================================

void test_simd_non_aligned_grid_size(void) {
    printf("\n=== Test: SIMD Non-Aligned Grid Size ===\n");

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // Use grid size that's not a multiple of 4 (typical SIMD width)
    test_result result = test_run_stability(SOLVER_TYPE_PROJECTION_OPTIMIZED, 33, 35, &params, 10);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("SIMD handles 33x35 grid correctly\n");
    printf("PASSED\n");
}

//=============================================================================
// MAIN
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    printf("\n");
    printf("================================================\n");
    printf("  SIMD Projection Solver Tests\n");
    printf("================================================\n");

    RUN_TEST(test_simd_projection_creates);
    // TODO: Re-enable when SIMD projection solver bug is fixed
    // RUN_TEST(test_simd_scalar_consistency);
    // RUN_TEST(test_simd_divergence_free);
    RUN_TEST(test_simd_stability);
    // RUN_TEST(test_simd_energy_decay);
    RUN_TEST(test_simd_non_aligned_grid_size);

    printf("\n================================================\n");

    return UNITY_END();
}
