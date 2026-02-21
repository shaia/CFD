#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {
    cfd_init();
    ensure_directory_exists("../../artifacts");
    ensure_directory_exists("../../artifacts/output");
}

void tearDown(void) {
    cfd_finalize();
}

// Test GPU configuration defaults
void test_gpu_config_defaults(void) {
    gpu_config_t config = gpu_config_default();

    // config.enable_gpu defaults to 1 if GPU is available at runtime
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

    gpu_config_t config = gpu_config_default();
    config.min_grid_size = 100;
    config.min_steps = 10;

    // Should use GPU: large grid size and enough steps
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

    gpu_device_info_t info[4];
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

    gpu_config_t config = gpu_config_default();
    size_t nx = 64, ny = 64;

    // Create context
    gpu_solver_context_t* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    // Get initial stats
    gpu_solver_stats_t stats = gpu_solver_get_stats(ctx);
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
    gpu_config_t config = gpu_config_default();

    gpu_solver_context_t* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    flow_field* field = flow_field_create(nx, ny);
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

    grid* g = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, g);

    gpu_config_t config = gpu_config_default();
    gpu_solver_context_t* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    // Upload data
    cfd_status_t status = gpu_solver_upload(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Run a single step
    gpu_solver_stats_t stats = {0};
    status = gpu_solver_step(ctx, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

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
    grid_destroy(g);
}

// Test multiple time steps for stability
void test_gpu_solver_multiple_steps(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_multiple_steps - No GPU available\n");
        return;
    }

    size_t nx = 64, ny = 64;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    grid* g = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, g);

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* s = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (s == NULL) {
        printf("Skipping test_gpu_solver_multiple_steps - GPU solver not available via registry\n");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    solver_init(s, g, &params);

    ns_solver_stats_t stats = ns_solver_stats_default();

    // Run 50 time steps
    for (int step = 0; step < 50; step++) {
        cfd_status_t status = solver_step(s, field, g, &params, &stats);
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
            TEST_ASSERT_TRUE(max_vel < 100.0);
        }
    }

    solver_destroy(s);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

// Helper function for grid size tests
static void run_grid_size_test(size_t nx, size_t ny) {
    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, g);

    gpu_config_t config = gpu_config_default();
    gpu_solver_context_t* ctx = gpu_solver_create(nx, ny, &config);
    TEST_ASSERT_NOT_NULL(ctx);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    cfd_status_t status = gpu_solver_upload(ctx, field);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    gpu_solver_stats_t stats = {0};
    status = gpu_solver_step(ctx, g, &params, &stats);
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
    grid_destroy(g);
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

    grid* g = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    // Initialize with zero velocity except top lid
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = IDX_2D(i, j, nx);
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
    // Set lid velocity (top boundary)
    for (size_t i = 0; i < nx; i++) {
        field->u[IDX_2D(i, ny-1, nx)] = 1.0;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* s = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (s == NULL) {
        printf("Skipping test_gpu_solver_lid_driven_cavity - GPU solver not available via registry\n");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.0001;
    params.mu = 0.01;

    solver_init(s, g, &params);

    ns_solver_stats_t stats = ns_solver_stats_default();

    // Run a few steps
    for (int step = 0; step < 10; step++) {
        cfd_status_t status = solver_step(s, field, g, &params, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    // Verify lid velocity is maintained (approximately)
    double lid_u_avg = 0.0;
    for (size_t i = 1; i < nx - 1; i++) {
        lid_u_avg += field->u[IDX_2D(i, ny-1, nx)];
    }
    lid_u_avg /= (nx - 2);
    TEST_ASSERT_DOUBLE_WITHIN(0.5, 1.0, lid_u_avg);

    // Bottom wall should have approximately zero velocity
    double bottom_u_avg = 0.0;
    for (size_t i = 1; i < nx - 1; i++) {
        bottom_u_avg += fabs(field->u[i]);
    }
    bottom_u_avg /= (nx - 2);
    TEST_ASSERT_TRUE(bottom_u_avg < 0.5);

    solver_destroy(s);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

// Test null/invalid parameter handling
void test_gpu_solver_error_handling(void) {
    if (!gpu_is_available()) {
        printf("Skipping test_gpu_solver_error_handling - No GPU available\n");
        return;
    }

    gpu_config_t config = gpu_config_default();

    // NULL config for gpu_should_use
    TEST_ASSERT_FALSE(gpu_should_use(NULL, 100, 100, 10));

    // Upload/download with NULL context
    flow_field* field = flow_field_create(32, 32);
    TEST_ASSERT_NOT_NULL(field);

    cfd_status_t status = gpu_solver_upload(NULL, field);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    status = gpu_solver_download(NULL, field);
    TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, status);

    // Upload/download with NULL field
    gpu_solver_context_t* ctx = gpu_solver_create(32, 32, &config);
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

    size_t nx = 100, ny = 50;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    grid* g = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    initialize_flow_field(field, g);

    ns_solver_params_t params = ns_solver_params_default();
    params.max_iter = 5;
    params.dt = 0.001;
    params.mu = 0.01;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* s = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);

    if (s == NULL) {
        printf("Skipping test_gpu_solver_execution - GPU solver not available via registry\n");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }

    solver_init(s, g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    // Run a few steps
    for (int i = 0; i < 3; i++) {
        cfd_status_t result = solver_step(s, field, g, &params, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, result);
    }

    solver_destroy(s);
    cfd_registry_destroy(registry);

    // Verify results are finite (basic stability check)
    TEST_ASSERT_TRUE(isfinite(field->u[0]));
    TEST_ASSERT_TRUE(isfinite(field->p[0]));

    flow_field_destroy(field);
    grid_destroy(g);
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

    // NSSolver step tests
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
