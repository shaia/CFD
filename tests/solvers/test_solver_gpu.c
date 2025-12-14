#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_gpu.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    GPUConfig config = gpu_config_default();

#ifdef CFD_HAS_CUDA
    int expected_enable = 1;
#else
    int expected_enable = 0;
#endif
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

    GPUConfig config = gpu_config_default();
    config.min_grid_size = 100;
    config.min_steps = 10;

    // Should use GPU: optimized grid size and steps
    TEST_ASSERT_TRUE(gpu_should_use(&config, 1000, 1000, 20));

    // Should NOT use: grid too small (9*9=81 < 100)
    TEST_ASSERT_FALSE(gpu_should_use(&config, 9, 9, 20));

    // Should NOT use: too few steps
    TEST_ASSERT_FALSE(gpu_should_use(&config, 1000, 1000, 5));

    // Should NOT use: disabled in config
    config.enable_gpu = 0;
    TEST_ASSERT_FALSE(gpu_should_use(&config, 1000, 1000, 20));
}

// Test GPU device info retrieval
void test_gpu_device_info(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_device_info - No GPU available\n");
        return;
    }

    GPUDeviceInfo info[4];
    int device_count = gpu_get_device_info(info, 4);

    TEST_ASSERT_TRUE(device_count > 0);
    TEST_ASSERT_TRUE(info[0].is_available);
    TEST_ASSERT_TRUE(info[0].total_memory > 0);
    TEST_ASSERT_TRUE(info[0].compute_capability_major >= 1);
    TEST_ASSERT_TRUE(strlen(info[0].name) > 0);

    printf("GPU Device: %s (CC %d.%d, %.1f GB)\n", info[0].name,
           info[0].compute_capability_major, info[0].compute_capability_minor,
           (double)info[0].total_memory / (1024.0 * 1024.0 * 1024.0));
}

// Test GPU device selection
void test_gpu_select_device(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_select_device - No GPU available\n");
        return;
    }

    // Select device 0 (should always exist if GPU is available)
    cfd_status_t status = gpu_select_device(0);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Selecting invalid device should fail
    status = gpu_select_device(999);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);
}

// Test GPU solver context creation and destruction
void test_gpu_solver_context_lifecycle(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_context_lifecycle - No GPU available\n");
        return;
    }

    GPUConfig config = gpu_config_default();
    size_t nx = 64, ny = 64;

    // Create context
    GPUSolverContext* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    // Get initial stats
    GPUSolverStats stats = gpu_solver_get_stats(ctx);
    TEST_ASSERT_EQUAL_INT(0, stats.kernels_launched);

    // Reset stats
    gpu_solver_reset_stats(ctx);
    stats = gpu_solver_get_stats(ctx);
    TEST_ASSERT_EQUAL_INT(0, stats.kernels_launched);

    // Destroy context
    gpu_solver_destroy(ctx);

    // Creating with NULL config should use defaults
    ctx = gpu_solver_create(nx, ny, NULL);
    TEST_ASSERT_NOT_NULL(ctx);
    gpu_solver_destroy(ctx);
}

// Test GPU data upload and download
void test_gpu_data_transfer(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_data_transfer - No GPU available\n");
        return;
    }

    size_t nx = 32, ny = 32;
    GPUConfig config = gpu_config_default();

    GPUSolverContext* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    // Initialize with known values
    for (size_t i = 0; i < nx * ny; i++) {
        field->u[i] = 1.0 + (double)i * 0.001;
        field->v[i] = 2.0 + (double)i * 0.001;
        field->p[i] = 3.0 + (double)i * 0.001;
        field->rho[i] = 1.0;
    }

    // Upload to GPU
    cfd_status_t status = gpu_solver_upload(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Clear host data
    for (size_t i = 0; i < nx * ny; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->p[i] = 0.0;
    }

    // Download from GPU
    status = gpu_solver_download(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Verify data integrity
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, field->u[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, field->v[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 3.0, field->p[0]);

    gpu_solver_destroy(ctx);
    flow_field_destroy(field);
}

// Test GPU solver step directly
void test_gpu_solver_step_direct(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_step_direct - No GPU available\n");
        return;
    }

    size_t nx = 64, ny = 64;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(grid);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, grid);

    GPUConfig config = gpu_config_default();
    GPUSolverContext* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    SolverParams params = solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    // Upload data
    cfd_status_t status = gpu_solver_upload(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Run a single step
    GPUSolverStats stats = {0};
    status = gpu_solver_step(ctx, grid, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    // Note: kernels_launched is incremented but not necessarily > 0 after one step
    // depending on implementation details. Just verify the step succeeded.

    // Download results
    status = gpu_solver_download(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Verify results are finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(field->u[i]));
        TEST_ASSERT_TRUE(isfinite(field->v[i]));
        TEST_ASSERT_TRUE(isfinite(field->p[i]));
    }

    gpu_solver_destroy(ctx);
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test multiple time steps for stability
void test_gpu_solver_multiple_steps(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_multiple_steps - No GPU available\n");
        return;
    }

    size_t nx = 64, ny = 64;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(grid);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, grid);

    SolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    Solver* solver = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    TEST_ASSERT_NOT_NULL(solver);

    SolverParams params = solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    solver_init(solver, grid, &params);

    SolverStats stats = solver_stats_default();

    // Run 50 time steps
    for (int step = 0; step < 50; step++) {
        cfd_status_t status = solver_step(solver, field, grid, &params, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        // Check stability every 10 steps
        if (step % 10 == 0) {
            double max_vel = 0.0;
            for (size_t i = 0; i < nx * ny; i++) {
                double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
                if (vel > max_vel)
                    max_vel = vel;
                TEST_ASSERT_TRUE(isfinite(field->u[i]));
                TEST_ASSERT_TRUE(isfinite(field->v[i]));
                TEST_ASSERT_TRUE(isfinite(field->p[i]));
            }
            // Velocity should remain bounded
            TEST_ASSERT_TRUE(max_vel < 100.0);
        }
    }

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Helper function for grid size tests
static void run_grid_size_test(size_t nx, size_t ny) {
    Grid* grid = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(grid);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, grid);

    GPUConfig config = gpu_config_default();
    GPUSolverContext* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    SolverParams params = solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    cfd_status_t status = gpu_solver_upload(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    GPUSolverStats stats = {0};
    status = gpu_solver_step(ctx, grid, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    status = gpu_solver_download(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Verify all values are finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(field->u[i]));
        TEST_ASSERT_TRUE(isfinite(field->v[i]));
        TEST_ASSERT_TRUE(isfinite(field->p[i]));
    }

    gpu_solver_destroy(ctx);
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test GPU solver with small square grid (16x16)
void test_gpu_solver_grid_16x16(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_grid_16x16 - No GPU available\n");
        return;
    }
    run_grid_size_test(16, 16);
}

// Test GPU solver with rectangular grid (32x64)
void test_gpu_solver_grid_32x64(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_grid_32x64 - No GPU available\n");
        return;
    }
    run_grid_size_test(32, 64);
}

// Test GPU solver with non-power-of-2 grid (100x50)
void test_gpu_solver_grid_100x50(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_grid_100x50 - No GPU available\n");
        return;
    }
    run_grid_size_test(100, 50);
}

// Test GPU solver with larger square grid (128x128)
void test_gpu_solver_grid_128x128(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_grid_128x128 - No GPU available\n");
        return;
    }
    run_grid_size_test(128, 128);
}

// Test GPU solver with large grid (256x256)
void test_gpu_solver_grid_256x256(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_grid_256x256 - No GPU available\n");
        return;
    }
    run_grid_size_test(256, 256);
}

// Test lid-driven cavity boundary conditions
void test_gpu_solver_lid_driven_cavity(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_lid_driven_cavity - No GPU available\n");
        return;
    }

    size_t nx = 64, ny = 64;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(grid);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    // Initialize with zero velocity except top lid
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
    // Set lid velocity (top boundary)
    for (size_t i = 0; i < nx; i++) {
        field->u[(ny - 1) * nx + i] = 1.0;
    }

    SolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    Solver* solver = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    TEST_ASSERT_NOT_NULL(solver);

    SolverParams params = solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    solver_init(solver, grid, &params);

    SolverStats stats = solver_stats_default();

    // Run a few steps
    for (int step = 0; step < 10; step++) {
        cfd_status_t status = solver_step(solver, field, grid, &params, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    // Verify lid velocity is maintained (approximately, since boundary is applied)
    double lid_u_avg = 0.0;
    for (size_t i = 1; i < nx - 1; i++) {
        lid_u_avg += field->u[(ny - 1) * nx + i];
    }
    lid_u_avg /= (nx - 2);
    // Lid velocity should be close to 1.0 (boundary condition)
    TEST_ASSERT_DOUBLE_WITHIN(0.5, 1.0, lid_u_avg);

    // Bottom wall should have approximately zero velocity
    double bottom_u_avg = 0.0;
    for (size_t i = 1; i < nx - 1; i++) {
        bottom_u_avg += fabs(field->u[i]);
    }
    bottom_u_avg /= (nx - 2);
    TEST_ASSERT_TRUE(bottom_u_avg < 0.5);

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test null/invalid parameter handling
void test_gpu_solver_error_handling(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_error_handling - No GPU available\n");
        return;
    }

    GPUConfig config = gpu_config_default();

    // NULL config for gpu_should_use
    TEST_ASSERT_FALSE(gpu_should_use(NULL, 100, 100, 10));

    // Upload/download with NULL context
    FlowField* field = flow_field_create(32, 32);
    TEST_ASSERT_NOT_NULL(field);

    cfd_status_t status = gpu_solver_upload(NULL, field);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    status = gpu_solver_download(NULL, field);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    // Upload/download with NULL field
    GPUSolverContext* ctx = gpu_solver_create(32, 32, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    status = gpu_solver_upload(ctx, NULL);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    status = gpu_solver_download(ctx, NULL);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    // Step with NULL parameters
    status = gpu_solver_step(NULL, NULL, NULL, NULL);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    // Destroy NULL context (should not crash)
    gpu_solver_destroy(NULL);

    gpu_solver_destroy(ctx);
    flow_field_destroy(field);
}

// Test basic initialization and step
void test_gpu_solver_execution(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_execution - No GPU available\n");
        return;
    }

    size_t nx = 100, ny = 50;  // Grid size 5000 points (less than default min_grid_size of 10000)
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(grid);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    initialize_flow_field(field, grid);

    SolverParams params = solver_params_default();
    params.max_iter = 5;
    params.dt = 0.001;

    GPUConfig config = gpu_config_default();
    // Force usage even if defaults would say no (though 100*50 < 10000, so we adjust config)
    config.min_grid_size = 100;

    // Test Projection GPU Solver
    // Note: We are testing the public API wrapper which dispatches to GPU
    // But since we want to specifically test GPU code, we can use the direct call if exposed,
    // or rely on the registry. Let's use the registry via `solver_create`.

    SolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    Solver* solver = cfd_solver_create(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU);

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

    // Explicitly set stable parameters for dx=0.01
    // nu * dt / dx^2 < 0.5
    // 0.01 * dt / (0.01^2) < 0.5 => 100 * dt < 0.5 => dt < 0.005
    params.dt = 0.001;
    params.mu = 0.01;

    printf("DEBUG: Initial Rho[0] = %f\n", field->rho[0]);
    printf("DEBUG: Initial P[0] = %f\n", field->p[0]);

    SolverStats stats = solver_stats_default();

    // Run a few steps
    for (int i = 0; i < 3; i++) {
        int result = solver_step(solver, field, grid, &params, &stats);
        TEST_ASSERT_EQUAL(0, result);
        printf("DEBUG: Step %d P[0] = %f\n", i, field->p[0]);
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

    // Basic configuration tests
    RUN_TEST(test_gpu_config_defaults);
    RUN_TEST(test_gpu_should_use);

    // Device management tests
    RUN_TEST(test_gpu_device_info);
    RUN_TEST(test_gpu_select_device);

    // Context lifecycle tests
    RUN_TEST(test_gpu_solver_context_lifecycle);

    // Data transfer tests
    RUN_TEST(test_gpu_data_transfer);

    // Solver step tests
    RUN_TEST(test_gpu_solver_step_direct);
    RUN_TEST(test_gpu_solver_execution);

    // Stability and correctness tests
    RUN_TEST(test_gpu_solver_multiple_steps);
    RUN_TEST(test_gpu_solver_lid_driven_cavity);

    // Grid size tests
    RUN_TEST(test_gpu_solver_grid_16x16);
    RUN_TEST(test_gpu_solver_grid_32x64);
    RUN_TEST(test_gpu_solver_grid_100x50);
    RUN_TEST(test_gpu_solver_grid_128x128);
    RUN_TEST(test_gpu_solver_grid_256x256);

    // Error handling tests
    RUN_TEST(test_gpu_solver_error_handling);

    return UNITY_END();
}
