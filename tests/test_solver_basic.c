#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "unity.h"
#include "solver.h"
#include "grid.h"
#include "utils.h"

void setUp(void) {
    // Setup for each test - ensure output directory exists
    ensure_directory_exists("../../artifacts");
    ensure_directory_exists("../../artifacts/output");
}

void tearDown(void) {
    // Cleanup after each test
}

// Test that basic solver runs without crashing
void test_basic_solver_runs(void) {
    // Small grid for fast test
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    FlowField* field = flow_field_create(nx, ny);

    TEST_ASSERT_NOT_NULL(grid);
    TEST_ASSERT_NOT_NULL(field);

    // Initialize with simple values
    initialize_flow_field(field, grid);

    // Set up solver parameters for quick test
    SolverParams params = solver_params_default();
    params.max_iter = 2;  // Very few iterations for speed
    params.tolerance = 1e-1;  // Very relaxed tolerance

    // Test that solver runs without crashing
    solve_navier_stokes(field, grid, &params);

    // Verify solver completed successfully by checking field values are finite
    TEST_ASSERT_TRUE(isfinite(field->u[0]));
    TEST_ASSERT_TRUE(isfinite(field->v[0]));
    TEST_ASSERT_TRUE(isfinite(field->p[0]));
    TEST_ASSERT_TRUE(field->rho[0] > 0.0);

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that optimized solver runs without crashing
void test_optimized_solver_runs(void) {
    // Small grid for fast test
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    FlowField* field = flow_field_create(nx, ny);

    TEST_ASSERT_NOT_NULL(grid);
    TEST_ASSERT_NOT_NULL(field);

    // Initialize with simple values
    initialize_flow_field(field, grid);

    // Store initial density value to verify field structure remains intact
    double initial_rho = field->rho[0];

    // Set up solver parameters for quick test
    SolverParams params = solver_params_default();
    params.max_iter = 2;  // Very few iterations for speed
    params.tolerance = 1e-1;  // Very relaxed tolerance

    // Test that optimized solver runs without crashing and produces valid results
    solve_navier_stokes_optimized(field, grid, &params);

    // Verify solver completed successfully by checking field values are finite and valid
    TEST_ASSERT_TRUE(isfinite(field->u[0]));
    TEST_ASSERT_TRUE(isfinite(field->v[0]));
    TEST_ASSERT_TRUE(isfinite(field->p[0]));
    TEST_ASSERT_TRUE(field->rho[0] > 0.0);  // Density should remain positive

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that solver initialization works
void test_solver_initialization(void) {
    // Test basic flow field creation and initialization
    size_t nx = 3, ny = 3;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    FlowField* field = flow_field_create(nx, ny);

    TEST_ASSERT_NOT_NULL(grid);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_EQUAL(nx, field->nx);
    TEST_ASSERT_EQUAL(ny, field->ny);

    // Check that arrays are allocated
    TEST_ASSERT_NOT_NULL(field->u);
    TEST_ASSERT_NOT_NULL(field->v);
    TEST_ASSERT_NOT_NULL(field->p);
    TEST_ASSERT_NOT_NULL(field->rho);
    TEST_ASSERT_NOT_NULL(field->T);

    // Initialize field
    initialize_flow_field(field, grid);

    // Check that initialization completed (values should be set)
    TEST_ASSERT_TRUE(field->rho[0] > 0.0);  // Density should be positive
    TEST_ASSERT_TRUE(field->T[0] > 0.0);    // Temperature should be positive

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test main function
int main(void) {
    UNITY_BEGIN();

    printf("Testing basic solver functionality...\n");

    RUN_TEST(test_solver_initialization);
    RUN_TEST(test_basic_solver_runs);
    RUN_TEST(test_optimized_solver_runs);

    return UNITY_END();
}