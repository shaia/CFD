#include "simulation_api.h"
#include "unity.h"
#include <math.h>

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Test basic simulation initialization
void test_simulation_initialization(void) {
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    TEST_ASSERT_NOT_NULL(sim_data);
    TEST_ASSERT_NOT_NULL(sim_data->field);
    TEST_ASSERT_NOT_NULL(sim_data->grid);

    // Test grid dimensions
    TEST_ASSERT_EQUAL_UINT(nx, sim_data->grid->nx);
    TEST_ASSERT_EQUAL_UINT(ny, sim_data->grid->ny);

    // Test grid bounds
    TEST_ASSERT_EQUAL_DOUBLE(xmin, sim_data->grid->xmin);
    TEST_ASSERT_EQUAL_DOUBLE(xmax, sim_data->grid->xmax);
    TEST_ASSERT_EQUAL_DOUBLE(ymin, sim_data->grid->ymin);
    TEST_ASSERT_EQUAL_DOUBLE(ymax, sim_data->grid->ymax);

    // Test field arrays are allocated
    TEST_ASSERT_NOT_NULL(sim_data->field->u);
    TEST_ASSERT_NOT_NULL(sim_data->field->v);
    TEST_ASSERT_NOT_NULL(sim_data->field->p);
    TEST_ASSERT_NOT_NULL(sim_data->field->rho);
    TEST_ASSERT_NOT_NULL(sim_data->field->T);

    free_simulation(sim_data);
}

// Test velocity magnitude calculation
void test_velocity_magnitude_calculation(void) {
    size_t nx = 3, ny = 3;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Set some test velocity values
    sim_data->field->u[0] = 3.0;  // u component
    sim_data->field->v[0] = 4.0;  // v component
    // Expected magnitude: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0

    sim_data->field->u[1] = 1.0;
    sim_data->field->v[1] = 1.0;
    // Expected magnitude: sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414

    double* velocity_magnitude = calculate_velocity_magnitude(sim_data->field, nx, ny);

    TEST_ASSERT_NOT_NULL(velocity_magnitude);
    TEST_ASSERT_EQUAL_DOUBLE(5.0, velocity_magnitude[0]);
    TEST_ASSERT_DOUBLE_WITHIN(0.001, 1.414, velocity_magnitude[1]);

    free(velocity_magnitude);
    free_simulation(sim_data);
}

// Test velocity magnitude squared calculation (performance version)
void test_velocity_magnitude_squared_calculation(void) {
    size_t nx = 2, ny = 2;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Set test velocity values
    sim_data->field->u[0] = 3.0;
    sim_data->field->v[0] = 4.0;
    // Expected magnitude squared: 3^2 + 4^2 = 9 + 16 = 25

    sim_data->field->u[1] = 2.0;
    sim_data->field->v[1] = 2.0;
    // Expected magnitude squared: 2^2 + 2^2 = 4 + 4 = 8

    double* velocity_magnitude_sq = calculate_velocity_magnitude_squared(sim_data->field, nx, ny);

    TEST_ASSERT_NOT_NULL(velocity_magnitude_sq);
    TEST_ASSERT_EQUAL_DOUBLE(25.0, velocity_magnitude_sq[0]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, velocity_magnitude_sq[1]);

    free(velocity_magnitude_sq);
    free_simulation(sim_data);
}

// Test simulation step execution (without VTK output)
void test_simulation_step_execution(void) {
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Store initial values to verify the solver runs
    double initial_u = sim_data->field->u[12];  // middle point
    double initial_v = sim_data->field->v[12];
    double initial_p = sim_data->field->p[12];

    // Run one simulation step
    run_simulation_step(sim_data);

    // Verify that the simulation ran (values may have changed or stayed the same)
    // The important thing is that no crashes occurred and memory is still valid
    TEST_ASSERT_NOT_NULL(sim_data->field->u);
    TEST_ASSERT_NOT_NULL(sim_data->field->v);
    TEST_ASSERT_NOT_NULL(sim_data->field->p);

    // Test that field values are finite (not NaN or infinite)
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(sim_data->field->u[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->v[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->p[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->rho[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->T[i]));
    }

    free_simulation(sim_data);
}

// Test multiple simulation steps
void test_multiple_simulation_steps(void) {
    size_t nx = 4, ny = 4;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Run multiple simulation steps
    for (int step = 0; step < 5; step++) {
        run_simulation_step(sim_data);

        // Verify stability after each step
        for (size_t i = 0; i < nx * ny; i++) {
            TEST_ASSERT_TRUE(isfinite(sim_data->field->u[i]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->v[i]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->p[i]));
        }
    }

    free_simulation(sim_data);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_simulation_initialization);
    RUN_TEST(test_velocity_magnitude_calculation);
    RUN_TEST(test_velocity_magnitude_squared_calculation);
    RUN_TEST(test_simulation_step_execution);
    RUN_TEST(test_multiple_simulation_steps);

    return UNITY_END();
}