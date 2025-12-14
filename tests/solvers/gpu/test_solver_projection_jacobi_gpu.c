/**
 * Mathematical Validation Tests for GPU Projection Jacobi Solver
 *
 * Tests that GPU implementations produce correct results compared to
 * CPU implementations and verify GPU-specific correctness.
 */

#include "../test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_gpu.h"
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
// HELPER: Check if GPU solver is available
//=============================================================================

static int gpu_solver_available(solver_registry* registry) {
    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (slv == NULL) {
        return 0;
    }
    solver_destroy(slv);
    return 1;
}

//=============================================================================
// TEST: GPU AVAILABILITY CHECK
//=============================================================================

void test_gpu_availability(void) {
    printf("\n=== Test: GPU Availability Check ===\n");

    int available = gpu_is_available();
    printf("GPU available: %s\n", available ? "YES" : "NO");

    if (available) {
        gpu_device_info info;
        int num_devices = gpu_get_device_info(&info, 1);
        if (num_devices > 0 && info.is_available) {
            printf("GPU Name: %s\n", info.name);
            printf("Compute Capability: %d.%d\n", info.compute_capability_major,
                   info.compute_capability_minor);
            printf("Total Memory: %.2f GB\n", info.total_memory / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // This test passes regardless - we're just checking availability
    TEST_PASS();
    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU SOLVER CREATES
//=============================================================================

void test_gpu_solver_creates(void) {
    printf("\n=== Test: GPU Solver Creates ===\n");

    solver_registry* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (slv == NULL) {
        printf("GPU solver not available (expected if CUDA not enabled)\n");
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
// TEST: GPU VS CPU CONSISTENCY
//=============================================================================

void test_gpu_cpu_consistency(void) {
    printf("\n=== Test: GPU vs CPU Consistency ===\n");

    if (!gpu_is_available()) {
        printf("No GPU available, skipping consistency test\n");
        TEST_PASS();
        return;
    }

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!gpu_solver_available(registry)) {
        printf("GPU solver not registered, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    // GPU and CPU may have larger differences due to different algorithms
    // and floating-point order of operations
    test_result result = test_run_consistency(
        SOLVER_TYPE_PROJECTION, SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        32, 32, &params, 10, 0.10);  // 10% tolerance

    printf("L2 difference in u: %.6e (relative: %.2e)\n",
           result.error_l2, result.relative_error);
    printf("L2 difference in v: %.6e\n", result.error_linf);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU STABILITY
//=============================================================================

void test_gpu_stability(void) {
    printf("\n=== Test: GPU Solver Stability ===\n");

    if (!gpu_is_available()) {
        printf("No GPU available, skipping stability test\n");
        TEST_PASS();
        return;
    }

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!gpu_solver_available(registry)) {
        printf("GPU solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_stability(SOLVER_TYPE_PROJECTION_JACOBI_GPU, 64, 64, &params, 50);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("GPU solver stable for 50 steps\n");
    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU ENERGY DECAY
//=============================================================================

void test_gpu_energy_decay(void) {
    printf("\n=== Test: GPU Energy Decay ===\n");

    if (!gpu_is_available()) {
        printf("No GPU available, skipping energy test\n");
        TEST_PASS();
        return;
    }

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!gpu_solver_available(registry)) {
        printf("GPU solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_energy_decay(SOLVER_TYPE_PROJECTION_JACOBI_GPU, 32, 32, &params, 30);

    printf("Initial kinetic energy: %.6e\n", result.initial_energy);
    printf("Final kinetic energy: %.6e\n", result.final_energy);
    printf("Energy ratio: %.4f\n", result.final_energy / result.initial_energy);

    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU DIFFERENT GRID SIZES
//=============================================================================

void test_gpu_various_grid_sizes(void) {
    printf("\n=== Test: GPU Various Grid Sizes ===\n");

    if (!gpu_is_available()) {
        printf("No GPU available, skipping grid size test\n");
        TEST_PASS();
        return;
    }

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!gpu_solver_available(registry)) {
        printf("GPU solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }

    // Test various grid sizes including non-power-of-2
    size_t test_sizes[][2] = {
        {16, 16},
        {32, 32},
        {33, 33},  // Non-power-of-2
        {64, 32},  // Non-square
        {48, 48}
    };
    int num_tests = 5;

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    for (int t = 0; t < num_tests; t++) {
        size_t nx = test_sizes[t][0];
        size_t ny = test_sizes[t][1];

        test_result result = test_run_stability(SOLVER_TYPE_PROJECTION_JACOBI_GPU, nx, ny, &params, 5);

        printf("Grid %zux%zu: %s\n", nx, ny, result.passed ? "OK" : "FAILED");
        TEST_ASSERT_TRUE_MESSAGE(result.passed, "GPU should handle this grid size");
    }

    cfd_registry_destroy(registry);

    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU CONFIG DEFAULTS
//=============================================================================

void test_gpu_config_defaults(void) {
    printf("\n=== Test: GPU Config Defaults ===\n");

    gpu_config config = gpu_config_default();

    printf("Default config:\n");
    printf("  enable_gpu: %d\n", config.enable_gpu);
    printf("  block_size_x: %d\n", config.block_size_x);
    printf("  block_size_y: %d\n", config.block_size_y);
    printf("  min_grid_size: %zu\n", config.min_grid_size);
    printf("  poisson_max_iter: %d\n", config.poisson_max_iter);
    printf("  poisson_tolerance: %.2e\n", config.poisson_tolerance);

    // Check reasonable defaults
    TEST_ASSERT_TRUE(config.block_size_x > 0 && config.block_size_x <= 1024);
    TEST_ASSERT_TRUE(config.block_size_y > 0 && config.block_size_y <= 1024);
    TEST_ASSERT_TRUE(config.poisson_max_iter > 0);
    TEST_ASSERT_TRUE(config.poisson_tolerance > 0.0);

    printf("PASSED\n");
}

//=============================================================================
// TEST: GPU DIVERGENCE-FREE CONSTRAINT
//=============================================================================

void test_gpu_divergence_free(void) {
    printf("\n=== Test: GPU Divergence-Free Constraint ===\n");

    if (!gpu_is_available()) {
        printf("No GPU available, skipping divergence test\n");
        TEST_PASS();
        return;
    }

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    if (!gpu_solver_available(registry)) {
        printf("GPU solver not available, skipping\n");
        cfd_registry_destroy(registry);
        TEST_PASS();
        return;
    }
    cfd_registry_destroy(registry);

    solver_params params = solver_params_default();
    params.dt = 0.001;
    params.mu = 0.01;
    params.max_iter = 1;

    test_result result = test_run_divergence_free(SOLVER_TYPE_PROJECTION_JACOBI_GPU, 32, 32, &params, 10, 0.1);

    printf("Divergence norm after projection: %.6e\n", result.error_l2);

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
    printf("  GPU Projection Jacobi Solver Validation Tests\n");
    printf("================================================\n");

    RUN_TEST(test_gpu_availability);
    RUN_TEST(test_gpu_config_defaults);
    RUN_TEST(test_gpu_solver_creates);
    RUN_TEST(test_gpu_cpu_consistency);
    RUN_TEST(test_gpu_stability);
    RUN_TEST(test_gpu_energy_decay);
    RUN_TEST(test_gpu_divergence_free);
    RUN_TEST(test_gpu_various_grid_sizes);

    printf("\n================================================\n");

    return UNITY_END();
}
