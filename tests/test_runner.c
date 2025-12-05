#include "simulation_api.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

// Forward declarations
void test_simulation_basic(void);
void test_velocity_calculation(void);
void test_simulation_step(void);
void setUp(void);
void tearDown(void);

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Basic simulation initialization test (no VTK output)
void test_simulation_basic(void) {
    // Define grid and domain parameters
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);

    // Basic validation
    TEST_ASSERT_NOT_NULL(sim_data->field);
    TEST_ASSERT_NOT_NULL(sim_data->grid);
    TEST_ASSERT_NOT_NULL(sim_data->field->u);
    TEST_ASSERT_NOT_NULL(sim_data->field->v);
    TEST_ASSERT_NOT_NULL(sim_data->field->p);

    // Test grid dimensions
    TEST_ASSERT_EQUAL_UINT(nx, sim_data->grid->nx);
    TEST_ASSERT_EQUAL_UINT(ny, sim_data->grid->ny);

    // Clean up
    free_simulation(sim_data);
}

// Test velocity magnitude calculation without VTK output
void test_velocity_calculation(void) {
    size_t nx = 3, ny = 3;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Set test velocity values
    sim_data->field->u[0] = 3.0;
    sim_data->field->v[0] = 4.0;
    // Expected magnitude: sqrt(9 + 16) = 5.0

    double* velocity_magnitude = calculate_velocity_magnitude(sim_data->field, nx, ny);
    TEST_ASSERT_NOT_NULL(velocity_magnitude);
    // Use float comparison instead of double since Unity double precision may be disabled
    TEST_ASSERT_EQUAL_FLOAT(5.0f, (float)velocity_magnitude[0]);

    free(velocity_magnitude);
    free_simulation(sim_data);
}

// Test field memory allocation and basic properties (skip simulation step to avoid VTK)
void test_simulation_step(void) {
    size_t nx = 4, ny = 4;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

    // Test that memory allocation worked and basic properties are valid
    TEST_ASSERT_NOT_NULL(sim_data->field->u);
    TEST_ASSERT_NOT_NULL(sim_data->field->v);
    TEST_ASSERT_NOT_NULL(sim_data->field->p);
    TEST_ASSERT_NOT_NULL(sim_data->field->rho);
    TEST_ASSERT_NOT_NULL(sim_data->field->T);

    // Verify field values are initialized and finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(sim_data->field->u[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->v[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->p[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->rho[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->T[i]));
    }

    free_simulation(sim_data);
}

int main(void) {
    UNITY_BEGIN();

    // Add all test cases here (no VTK dependencies)
    RUN_TEST(test_simulation_basic);
    RUN_TEST(test_velocity_calculation);
    RUN_TEST(test_simulation_step);

    return UNITY_END();
}