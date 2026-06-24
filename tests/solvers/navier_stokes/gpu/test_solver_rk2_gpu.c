/**
 * Mathematical Validation Tests for the GPU RK2 (Heun's method) Solver
 *
 * Verifies that the rk2_gpu backend matches the scalar rk2 reference within
 * tolerance (velocity-only and energy-enabled), stays stable, and dissipates
 * kinetic energy. All tests skip gracefully when CUDA is unavailable.
 */

#include "../test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <stdio.h>

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

static int gpu_solver_available(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        return 0;
    }
    cfd_registry_register_defaults(registry);
    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_RK2_GPU);
    int available = (slv != NULL);
    if (slv) {
        solver_destroy(slv);
    }
    cfd_registry_destroy(registry);
    return available;
}

void test_gpu_solver_creates(void) {
    printf("\n=== Test: RK2 GPU Solver Creates ===\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_RK2_GPU);
    if (slv == NULL) {
        printf("RK2 GPU solver not available (expected if CUDA not enabled)\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    printf("Solver name: %s\n", slv->name ? slv->name : "NULL");
    solver_destroy(slv);
    cfd_registry_destroy(registry);
    printf("PASSED\n");
}

void test_gpu_cpu_consistency(void) {
    printf("\n=== Test: RK2 GPU vs CPU Consistency ===\n");

    if (!gpu_is_available() || !gpu_solver_available()) {
        printf("RK2 GPU solver not available, skipping\n");
        TEST_PASS();
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // GPU restores caller-set boundaries from upload-time values instead of the
    // full host BC dispatch, so a relaxed tolerance (matching projection_gpu).
    test_result result = test_run_consistency(
        NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_GPU, 32, 32, &params, 10, 0.10);

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_l2_secondary);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

void test_gpu_cpu_consistency_energy(void) {
    printf("\n=== Test: RK2 GPU vs CPU Consistency (energy on) ===\n");

    if (!gpu_is_available() || !gpu_solver_available()) {
        printf("RK2 GPU solver not available, skipping\n");
        TEST_PASS();
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;
    params.alpha = 0.001;  // enable the energy equation (periodic thermal BCs by default)

    test_result result = test_run_consistency(
        NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_GPU, 32, 32, &params, 10, 0.10);

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

void test_gpu_stability(void) {
    printf("\n=== Test: RK2 GPU Stability ===\n");

    if (!gpu_is_available() || !gpu_solver_available()) {
        printf("RK2 GPU solver not available, skipping\n");
        TEST_PASS();
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(NS_SOLVER_TYPE_RK2_GPU, 64, 64, &params, 50);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("RK2 GPU solver stable for 50 steps\n");
    printf("PASSED\n");
}

void test_gpu_energy_decay(void) {
    printf("\n=== Test: RK2 GPU Energy Decay ===\n");

    if (!gpu_is_available() || !gpu_solver_available()) {
        printf("RK2 GPU solver not available, skipping\n");
        TEST_PASS();
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(NS_SOLVER_TYPE_RK2_GPU, 32, 32, &params, 30);

    printf("Initial kinetic energy: %.6e\n", result.initial_energy);
    printf("Final kinetic energy: %.6e\n", result.final_energy);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

int main(void) {
    UNITY_BEGIN();

    printf("\n");
    printf("================================================\n");
    printf("  GPU RK2 Solver Validation Tests\n");
    printf("================================================\n");

    RUN_TEST(test_gpu_solver_creates);
    RUN_TEST(test_gpu_cpu_consistency);
    RUN_TEST(test_gpu_cpu_consistency_energy);
    RUN_TEST(test_gpu_stability);
    RUN_TEST(test_gpu_energy_decay);

    printf("\n================================================\n");

    return UNITY_END();
}
