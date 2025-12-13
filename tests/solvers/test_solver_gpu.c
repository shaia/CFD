#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/solver_gpu.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void setUp(void) {
    // Setup for each test
    ensure_directory_exists("../../artifacts");
    ensure_directory_exists("../../artifacts/output");
}

void tearDown(void) {
    // Cleanup after each test
}

// Test GPU configuration defaults
void test_gpu_config_defaults(void) {
    gpu_config config = gpu_config_default();

    int expected_enable = gpu_is_available() ? 1 : 0;
    TEST_ASSERT_EQUAL_INT(expected_enable, config.enable_gpu);
    TEST_ASSERT_EQUAL_INT(10000, config.min_grid_size);
    TEST_ASSERT_EQUAL_INT(10, config.min_steps);
    TEST_ASSERT_TRUE(config.block_size_x > 0);
    TEST_ASSERT_TRUE(config.block_size_y > 0);
}

// Test gpu_should_use logic
void test_gpu_should_use(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_should_use - No GPU available\n");
        return;
    }

    gpu_config config = gpu_config_default();
    config.min_grid_size = 100;
    config.min_steps = 10;

    // Should use GPU: optimized grid size and steps
    TEST_ASSERT_TRUE(gpu_should_use(&config, 1000, 1000, 20));

    // Should NOT use: grid too small
    TEST_ASSERT_FALSE(gpu_should_use(&config, 10, 10, 20));

    // Should NOT use: too few steps
    TEST_ASSERT_FALSE(gpu_should_use(&config, 1000, 1000, 5));

    // Should NOT use: disabled in config
    config.enable_gpu = 0;
    TEST_ASSERT_FALSE(gpu_should_use(&config, 1000, 1000, 20));
}

// Test basic initialization and step
void test_gpu_solver_execution(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_execution - No GPU available\n");
        return;
    }

    size_t nx = 100, ny = 50;  // grid size 5000 points (less than default min_grid_size of 10000)
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    flow_field* field = flow_field_create(nx, ny);

    TEST_ASSERT_NOT_NULL(grid);
    TEST_ASSERT_NOT_NULL(field);

    initialize_flow_field(field, grid);

    solver_params params = solver_params_default();
    params.max_iter = 5;
    params.dt = 0.001;

    gpu_config config = gpu_config_default();
    // Force usage even if defaults would say no (though 100*50 < 10000, so we adjust config)
    config.min_grid_size = 100;

    // Test Projection GPU Solver
    // Note: We are testing the public API wrapper which dispatches to GPU
    // But since we want to specifically test GPU code, we can use the direct call if exposed,
    // or rely on the registry. Let's use the registry via `solver_create`.

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* solver = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);

    // If we are on a machine with GPU but the solver creation failed (maybe build issue?), fail
    // test But if registry fallback works or returns null, handle it. Actually, solver_create
    // returns NULL if not found. However, if we built with CUDA support, it should be there.

    if (solver == NULL) {
        // Might happen if CUDA implementation wasn't linked or registered.
        // For now, let's assume it should be there.
        cfd_registry_destroy(registry);
        TEST_FAIL_MESSAGE("Could not create GPU solver. Check if SOLVER_TYPE_PROJECTION_JACOBI_GPU "
                          "is registered.");
    }

    solver_init(solver, grid, &params);
    solver_stats stats = solver_stats_default();

    // Run a few steps
    for (int i = 0; i < 3; i++) {
        int result = solver_step(solver, field, grid, &params, &stats);
        TEST_ASSERT_EQUAL(0, result);
    }

    solver_destroy(solver);
    cfd_registry_destroy(registry);

    // Verify results are finite (basic stability check)
    TEST_ASSERT_TRUE(isfinite(field->u[0]));
    TEST_ASSERT_TRUE(isfinite(field->p[0]));

    flow_field_destroy(field);
    grid_destroy(grid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_gpu_config_defaults);
    RUN_TEST(test_gpu_should_use);
    RUN_TEST(test_gpu_solver_execution);
    return UNITY_END();
}
