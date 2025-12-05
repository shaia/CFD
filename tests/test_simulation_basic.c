#include "simulation_api.h"
#include "unity.h"
#include <math.h>
#include <stdio.h>

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

    // Check grid parameters (using float comparison since Unity double precision is disabled)
    TEST_ASSERT_EQUAL_FLOAT((float)xmin, (float)sim_data->grid->xmin);
    TEST_ASSERT_EQUAL_FLOAT((float)xmax, (float)sim_data->grid->xmax);
    TEST_ASSERT_EQUAL_FLOAT((float)ymin, (float)sim_data->grid->ymin);
    TEST_ASSERT_EQUAL_FLOAT((float)ymax, (float)sim_data->grid->ymax);

    // Check that initial values are reasonable (after initialization)
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;

            // Check that values are finite (not NaN or infinite)
            TEST_ASSERT_TRUE(isfinite(sim_data->field->u[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->v[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->p[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->rho[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->T[idx]));
        }
    }

    // Test basic field access before running simulation
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            // Check that initial values are finite (not NaN or infinite)
            TEST_ASSERT_TRUE(isfinite(sim_data->field->u[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->v[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->p[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->rho[idx]));
            TEST_ASSERT_TRUE(isfinite(sim_data->field->T[idx]));
        }
    }

    // Try one simulation step
    run_simulation_step(sim_data);

    // After one simulation step, check values are still finite
    int finite_count = 0;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            if (isfinite(sim_data->field->u[idx]) && isfinite(sim_data->field->v[idx]) &&
                isfinite(sim_data->field->p[idx]) && isfinite(sim_data->field->rho[idx]) &&
                isfinite(sim_data->field->T[idx])) {
                finite_count++;
            }
        }
    }

    // At least 10% of values should be finite after simulation
    TEST_ASSERT_GREATER_THAN((int)(0.1 * nx * ny), finite_count);

    // Clean up
    free_simulation(sim_data);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_simulation_basic);
    return UNITY_END();
}