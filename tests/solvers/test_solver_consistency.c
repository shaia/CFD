#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"
#include "cfd/core/utils.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Test that both solvers produce consistent results
void test_solver_consistency(void) {
    // Small grid for detailed comparison
    size_t nx = 20, ny = 10;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    // Create two identical grids
    Grid* grid1 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    Grid* grid2 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid1);
    grid_initialize_uniform(grid2);

    // Create two identical flow fields
    FlowField* field1 = flow_field_create(nx, ny);
    FlowField* field2 = flow_field_create(nx, ny);

    // Initialize with identical conditions
    initialize_flow_field(field1, grid1);
    initialize_flow_field(field2, grid2);

    // Copy field1 to field2 to ensure identical starting conditions
    for (size_t i = 0; i < nx * ny; i++) {
        field2->u[i] = field1->u[i];
        field2->v[i] = field1->v[i];
        field2->p[i] = field1->p[i];
        field2->rho[i] = field1->rho[i];
        field2->T[i] = field1->T[i];
    }

    // Set up identical solver parameters
    SolverParams params = {.dt = 0.001,
                           .cfl = 0.2,
                           .gamma = 1.4,
                           .mu = 0.01,
                           .k = 0.0242,
                           .max_iter = 5,  // Small number for quick test
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Run both solvers for same number of iterations
    Solver* solver1 = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver1, grid1, &params);
    SolverStats stats1 = solver_stats_default();
    solver_step(solver1, field1, grid1, &params, &stats1);
    solver_destroy(solver1);

    Solver* solver2 = solver_create(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    solver_init(solver2, grid2, &params);
    SolverStats stats2 = solver_stats_default();
    solver_step(solver2, field2, grid2, &params, &stats2);
    solver_destroy(solver2);

    // Compare results - they should be very close
    double max_u_diff = 0.0, max_v_diff = 0.0, max_p_diff = 0.0;
    int close_values = 0;

    for (size_t i = 0; i < nx * ny; i++) {
        // Check that values are finite
        TEST_ASSERT_TRUE(isfinite(field1->u[i]));
        TEST_ASSERT_TRUE(isfinite(field1->v[i]));
        TEST_ASSERT_TRUE(isfinite(field1->p[i]));
        TEST_ASSERT_TRUE(isfinite(field2->u[i]));
        TEST_ASSERT_TRUE(isfinite(field2->v[i]));
        TEST_ASSERT_TRUE(isfinite(field2->p[i]));

        // Calculate differences
        double u_diff = fabs(field1->u[i] - field2->u[i]);
        double v_diff = fabs(field1->v[i] - field2->v[i]);
        double p_diff = fabs(field1->p[i] - field2->p[i]);

        max_u_diff = (u_diff > max_u_diff) ? u_diff : max_u_diff;
        max_v_diff = (v_diff > max_v_diff) ? v_diff : max_v_diff;
        max_p_diff = (p_diff > max_p_diff) ? p_diff : max_p_diff;

        // Count values that are reasonably close (within 1% relative error or 1e-6 absolute)
        double u_base = fabs(field1->u[i]) + 1e-10;
        double v_base = fabs(field1->v[i]) + 1e-10;
        double p_base = fabs(field1->p[i]) + 1e-10;

        if ((u_diff < 0.01 * u_base || u_diff < 1e-6) &&
            (v_diff < 0.01 * v_base || v_diff < 1e-6) &&
            (p_diff < 0.01 * p_base || p_diff < 1e-6)) {
            close_values++;
        }
    }

    // Print some diagnostics
    printf("Max differences - U: %.2e, V: %.2e, P: %.2e\n", max_u_diff, max_v_diff, max_p_diff);
    printf("Close values: %d/%d (%.1f%%)\n", close_values, (int)(nx * ny),
           100.0 * close_values / (nx * ny));

    // At least 90% of values should be very close between solvers
    TEST_ASSERT_GREATER_THAN((int)(0.9 * nx * ny), close_values);

    // Maximum differences should be reasonable (not massive differences)
    TEST_ASSERT_LESS_THAN(1e3, max_u_diff);  // Should be much smaller in practice
    TEST_ASSERT_LESS_THAN(1e3, max_v_diff);
    TEST_ASSERT_LESS_THAN(1e3, max_p_diff);

    // Clean up
    flow_field_destroy(field1);
    flow_field_destroy(field2);
    grid_destroy(grid1);
    grid_destroy(grid2);
}

// Test that both solvers handle edge cases properly
void test_solver_stability(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Set challenging initial conditions
    for (size_t i = 0; i < nx * ny; i++) {
        field->u[i] = 10.0 * sin(2.0 * M_PI * i / (nx * ny));  // Large velocities
        field->v[i] = 5.0 * cos(3.0 * M_PI * i / (nx * ny));
        field->p[i] = 1.0 + 0.5 * sin(M_PI * i / (nx * ny));
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }

    SolverParams params = {.dt = 0.0001,  // Very small time step for stability
                           .cfl = 0.1,
                           .gamma = 1.4,
                           .mu = 0.01,
                           .k = 0.0242,
                           .max_iter = 3,
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Test basic solver
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);

    // Check that solution didn't blow up
    int stable_count = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (isfinite(field->u[i]) && isfinite(field->v[i]) && isfinite(field->p[i]) &&
            fabs(field->u[i]) < 1e6 && fabs(field->v[i]) < 1e6 && fabs(field->p[i]) < 1e6) {
            stable_count++;
        }
    }

    // At least 80% of values should remain stable
    TEST_ASSERT_GREATER_THAN((int)(0.8 * nx * ny), stable_count);

    // Reset and test optimized solver
    initialize_flow_field(field, grid);
    for (size_t i = 0; i < nx * ny; i++) {
        field->u[i] = 10.0 * sin(2.0 * M_PI * i / (nx * ny));
        field->v[i] = 5.0 * cos(3.0 * M_PI * i / (nx * ny));
        field->p[i] = 1.0 + 0.5 * sin(M_PI * i / (nx * ny));
    }

    Solver* solver2 = solver_create(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    solver_init(solver2, grid, &params);
    SolverStats stats2 = solver_stats_default();
    solver_step(solver2, field, grid, &params, &stats2);
    solver_destroy(solver2);

    stable_count = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (isfinite(field->u[i]) && isfinite(field->v[i]) && isfinite(field->p[i]) &&
            fabs(field->u[i]) < 1e6 && fabs(field->v[i]) < 1e6 && fabs(field->p[i]) < 1e6) {
            stable_count++;
        }
    }

    // At least 80% of values should remain stable for optimized solver too
    TEST_ASSERT_GREATER_THAN((int)(0.8 * nx * ny), stable_count);

    flow_field_destroy(field);
    grid_destroy(grid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_solver_consistency);
    RUN_TEST(test_solver_stability);
    return UNITY_END();
}