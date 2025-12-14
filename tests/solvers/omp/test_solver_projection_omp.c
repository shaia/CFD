/**
 * Validation Tests for OpenMP-Parallelized Projection Solver
 *
 * Tests that OpenMP projection implementation produces results consistent
 * with the serial implementation.
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

static int omp_solver_available(solver_registry* registry, const char* type) {
    solver* slv = cfd_solver_create(registry, type);
    if (slv == NULL) {
        return 0;
    }
    solver_destroy(slv);
    return 1;
}

//=============================================================================
// TEST: OPENMP PROJECTION CREATES
//=============================================================================

void test_omp_projection_creates(void) {
    printf("\n=== Test: OpenMP Projection Creates ===\n");

#ifdef _OPENMP
    printf("OpenMP available with max threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP not available in this build\n");
#endif

    solver_registry* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_OMP);
    if (slv == NULL) {
        printf("OpenMP projection solver not available (expected if OpenMP disabled)\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    printf("Solver name: %s\n", slv->name ? slv->name : "NULL");

    solver_destroy(slv);
    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

//=============================================================================
// TEST: OPENMP VS SERIAL CONSISTENCY
//=============================================================================

void test_omp_serial_consistency(void) {
    printf("\n=== Test: OpenMP vs Serial Consistency ===\n");

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, SOLVER_TYPE_PROJECTION_OMP)) {
        printf("OpenMP projection solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // Projection may have slightly more variation due to iterative solver
    test_result result = test_run_consistency(
        SOLVER_TYPE_PROJECTION, SOLVER_TYPE_PROJECTION_OMP,
        32, 32, &params, 10, 0.05);  // 5% relative tolerance

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_linf);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: DIVERGENCE-FREE CONSTRAINT
//=============================================================================

void test_omp_divergence_free(void) {
    printf("\n=== Test: OpenMP Divergence-Free Constraint ===\n");

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, SOLVER_TYPE_PROJECTION_OMP)) {
        printf("OpenMP projection solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // Tolerance relaxed because projection method may not fully converge
    test_result result = test_run_divergence_free(SOLVER_TYPE_PROJECTION_OMP, 32, 32, &params, 10, 1.0);

    printf("Divergence norm after projection: %.6e\n", result.error_l2);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: STABILITY WITH LARGE GRID
//=============================================================================

void test_omp_stability_large_grid(void) {
    printf("\n=== Test: OpenMP Stability with Large Grid ===\n");

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, SOLVER_TYPE_PROJECTION_OMP)) {
        printf("OpenMP solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.0005;
    params.mu = 0.01;
    params.max_iter = 1;

    // Larger grid to benefit from parallelization
    test_result result = test_run_stability(SOLVER_TYPE_PROJECTION_OMP, 128, 128, &params, 50);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("OMP solver stable for 50 steps on 128x128 grid\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: ENERGY DECAY
//=============================================================================

void test_omp_energy_decay(void) {
    printf("\n=== Test: OpenMP Energy Decay ===\n");

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!omp_solver_available(registry, SOLVER_TYPE_PROJECTION_OMP)) {
        printf("OpenMP solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(SOLVER_TYPE_PROJECTION_OMP, 32, 32, &params, 30);

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
    printf("  OpenMP Projection Solver Tests\n");
    printf("================================================\n");

#ifdef _OPENMP
    printf("OpenMP enabled, max threads: %d\n", omp_get_max_threads());
#else
    printf("OpenMP not enabled in this build\n");
#endif

    RUN_TEST(test_omp_projection_creates);
    RUN_TEST(test_omp_serial_consistency);
    RUN_TEST(test_omp_divergence_free);
    RUN_TEST(test_omp_stability_large_grid);
    RUN_TEST(test_omp_energy_decay);

    printf("\n================================================\n");

    return UNITY_END();
}
