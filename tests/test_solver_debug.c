#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "unity.h"
#include "solver_interface.h"
#include "grid.h"
#include "utils.h"

void setUp(void) {
    // Setup for each test
    ensure_directory_exists("../../artifacts");
    ensure_directory_exists("../../artifacts/output");
}

void tearDown(void) {
    // Cleanup after each test
}

// Helper function to print field values for debugging
void print_field_debug(FlowField* field, const char* name) {
    printf("\n=== %s ===\n", name);
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            printf("(%zu,%zu): u=%.3f v=%.3f p=%.3f rho=%.3f T=%.3f\n",
                   i, j, field->u[idx], field->v[idx], field->p[idx],
                   field->rho[idx], field->T[idx]);
        }
    }
}

// Check if any field contains NaN or infinity
void check_field_validity(FlowField* field, const char* stage) {
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < field->nx * field->ny; i++) {
        if (isnan(field->u[i]) || isnan(field->v[i]) || isnan(field->p[i]) ||
            isnan(field->rho[i]) || isnan(field->T[i])) {
            nan_count++;
        }
        if (isinf(field->u[i]) || isinf(field->v[i]) || isinf(field->p[i]) ||
            isinf(field->rho[i]) || isinf(field->T[i])) {
            inf_count++;
        }
    }
    printf("Stage: %s - NaN count: %d, Inf count: %d\n", stage, nan_count, inf_count);
}

// Test solver step by step to identify where NaN appears
void test_solver_step_by_step_debug(void) {
    // Use very simple 3x3 grid
    size_t nx = 3, ny = 3;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    FlowField* field = flow_field_create(nx, ny);

    TEST_ASSERT_NOT_NULL(grid);
    TEST_ASSERT_NOT_NULL(field);

    printf("=== DEBUGGING SOLVER STEP BY STEP ===\n");

    // Initialize field
    initialize_flow_field(field, grid);
    check_field_validity(field, "After initialization");
    print_field_debug(field, "Initial state");

    // Set up solver parameters - make them very conservative
    SolverParams params = solver_params_default();
    params.max_iter = 1;  // Only 1 iteration
    params.dt = 0.0001;   // Very small timestep
    params.tolerance = 1e-1;
    params.mu = 0.001;    // Small viscosity
    params.pressure_coupling = 0.01;  // Small pressure coupling

    printf("Solver parameters: dt=%.6f, mu=%.6f, pressure_coupling=%.6f\n",
           params.dt, params.mu, params.pressure_coupling);

    // Run solver using modern interface
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);
    check_field_validity(field, "After 1 solver iteration");
    print_field_debug(field, "After solver");

    // Check for NaN values
    int has_nan = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (isnan(field->p[i])) {
            has_nan = 1;
            break;
        }
    }

    printf("Has NaN in pressure field: %s\n", has_nan ? "YES" : "NO");

    // This test should not have NaN values with conservative parameters
    TEST_ASSERT_FALSE(has_nan);

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test main function
int main(void) {
    UNITY_BEGIN();

    printf("=== SOLVER DEBUG TEST ===\n");

    RUN_TEST(test_solver_step_by_step_debug);

    return UNITY_END();
}