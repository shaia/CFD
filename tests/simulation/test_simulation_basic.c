#include "cfd/api/simulation_api.h"
#include "unity.h"
#include <math.h>
#include <stdio.h>

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

void test_simulation_initialization(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);
    TEST_ASSERT_NOT_NULL(sim_data->field);
    TEST_ASSERT_NOT_NULL(sim_data->grid);

    // Check initial simulation structure
    TEST_ASSERT_EQUAL_UINT(nx, sim_data->field->nx);
    TEST_ASSERT_EQUAL_UINT(ny, sim_data->field->ny);
    TEST_ASSERT_NOT_NULL(sim_data->field->u);
    TEST_ASSERT_NOT_NULL(sim_data->field->v);
    TEST_ASSERT_NOT_NULL(sim_data->field->p);
    TEST_ASSERT_NOT_NULL(sim_data->field->rho);
    TEST_ASSERT_NOT_NULL(sim_data->field->T);

    free_simulation(sim_data);
}

void test_simulation_parameters(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);

    // Check grid parameters (using float comparison since Unity double precision is disabled)
    TEST_ASSERT_EQUAL_FLOAT((float)xmin, (float)sim_data->grid->xmin);
    TEST_ASSERT_EQUAL_FLOAT((float)xmax, (float)sim_data->grid->xmax);
    TEST_ASSERT_EQUAL_FLOAT((float)ymin, (float)sim_data->grid->ymin);
    TEST_ASSERT_EQUAL_FLOAT((float)ymax, (float)sim_data->grid->ymax);

    free_simulation(sim_data);
}

void test_simulation_step_execution(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);

    // Initial check - ensure all fields finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(sim_data->field->u[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->v[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->p[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->rho[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->T[i]));
    }

    run_simulation_step(sim_data);

    // Verify stability - ensure all fields remain finite
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
    RUN_TEST(test_simulation_initialization);
    RUN_TEST(test_simulation_parameters);
    RUN_TEST(test_simulation_step_execution);
    return UNITY_END();
}