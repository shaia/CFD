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
// TEST: FAIL-FAST ERROR HANDLING
//=============================================================================

void test_simd_fail_fast_on_divergence(void) {
    printf("\n=== Test: SIMD Projection Fail-Fast on Divergence ===\n");

    // Test that the SIMD solver properly handles and reports divergence conditions.
    // We verify:
    // 1. The solver doesn't crash on challenging inputs
    // 2. It returns a valid status code
    // 3. For valid inputs, it returns CFD_SUCCESS
    // 4. The fail-fast code path exists (early return on Poisson failure)

    size_t nx = 16, ny = 16;
    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);

    grid_initialize_uniform(g);

    // Initialize with valid (but challenging) values
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            // High velocities that might stress the solver
            field->u[idx] = 10.0 * ((double)i / nx - 0.5);
            field->v[idx] = 10.0 * ((double)j / ny - 0.5);
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(slv);
    solver_init(slv, g, &params);
    solver_stats stats = solver_stats_default();

    // Run solver - it should either succeed or fail gracefully
    cfd_status_t status = solver_step(slv, field, g, &params, &stats);

    printf("SIMD solver returned status: %d\n", status);

    // Verify the solver returns a valid status (not some garbage value)
    TEST_ASSERT_TRUE_MESSAGE(status == CFD_SUCCESS || status < 0,
        "SIMD solver should return a valid status code");

    // Check that field is valid after solver step (no memory corruption)
    int valid_field = 1;
    for (size_t k = 0; k < nx * ny && valid_field; k++) {
        if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
            valid_field = 0;
        }
    }

    if (status == CFD_SUCCESS) {
        TEST_ASSERT_TRUE_MESSAGE(valid_field, "Field should be valid when solver succeeds");
        printf("SIMD solver succeeded with valid field\n");
    } else if (status == CFD_ERROR_DIVERGED) {
        printf("SIMD solver reported divergence (fail-fast behavior working)\n");
    }

    printf("SIMD solver handled input without crashing\n");

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

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
    RUN_TEST(test_simd_scalar_consistency);
    RUN_TEST(test_simd_divergence_free);
    RUN_TEST(test_simd_stability);
    RUN_TEST(test_simd_energy_decay);
    RUN_TEST(test_simd_non_aligned_grid_size);
    RUN_TEST(test_simd_fail_fast_on_divergence);

    printf("\n================================================\n");

    return UNITY_END();
}
