#include "unity.h"
#include "simulation_api.h"

// Forward declarations
void test_simulation_basic(void);
void setUp(void);
void tearDown(void);

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

void test_simulation_basic(void) {
    // Define grid and domain parameters
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);

    // Run a small simulation
    for (int iter = 0; iter < 5; iter++) {
        run_simulation_step(sim_data);
    }

    // Basic validation
    TEST_ASSERT_NOT_NULL(sim_data->field);
    TEST_ASSERT_NOT_NULL(sim_data->grid);

    // Clean up
    free_simulation(sim_data);
}

int main(void) {
    UNITY_BEGIN();

    // Add all test cases here
    RUN_TEST(test_simulation_basic);

    return UNITY_END();
}