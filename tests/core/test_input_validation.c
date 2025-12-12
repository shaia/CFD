#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/logging.h"
#include "unity.h"
#include <string.h>

void setUp(void) {
    cfd_clear_error();
}

void tearDown(void) {}

void test_grid_creation_zero_width(void) {
    // Zero width
    Grid* grid = grid_create(0, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_zero_height(void) {
    // Zero height
    Grid* grid = grid_create(10, 0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_invalid_xmin_xmax(void) {
    // Invalid bounds xmin > xmax
    Grid* grid = grid_create(10, 10, 1.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_invalid_ymin_ymax(void) {
    // Invalid bounds ymin > ymax
    Grid* grid = grid_create(10, 10, 0.0, 1.0, 1.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_equal_xmin_xmax(void) {
    // Invalid bounds xmin == xmax
    Grid* grid = grid_create(10, 10, 1.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_equal_ymin_ymax(void) {
    // Invalid bounds ymin == ymax
    Grid* grid = grid_create(10, 10, 0.0, 1.0, 1.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_zero_width(void) {
    // Zero dimensions
    SimulationData* sim = init_simulation(0, 100, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_zero_height(void) {
    SimulationData* sim = init_simulation(100, 0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_invalid_bounds(void) {
    // Invalid bounds
    SimulationData* sim = init_simulation(100, 100, 1.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_equal_bounds(void) {
    // Equal bounds
    SimulationData* sim = init_simulation(100, 100, 1.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

// Dummy factory for testing
Solver* dummy_factory(void) {
    return NULL;
}

void test_registry_register_null_registry(void) {
    int res = cfd_registry_register(NULL, "test", dummy_factory);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_registry_register_null_factory(void) {
    SolverRegistry* registry = cfd_registry_create();
    int res = cfd_registry_register(registry, "test", NULL);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
    cfd_registry_destroy(registry);
}

void test_registry_register_empty_name(void) {
    SolverRegistry* registry = cfd_registry_create();
    int res = cfd_registry_register(registry, "", dummy_factory);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
    cfd_registry_destroy(registry);
}

void test_registry_register_limit_exceeded(void) {
    SolverRegistry* registry = cfd_registry_create();

    // Fill the registry
    char name_buf[32];
    int res;
    for (int i = 0; i < 32; i++) {
        snprintf(name_buf, sizeof(name_buf), "solver_%d", i);
        res = cfd_registry_register(registry, name_buf, dummy_factory);
        TEST_ASSERT_EQUAL(0, res);
    }

    // Attempt to register one more
    res = cfd_registry_register(registry, "overflow", dummy_factory);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_LIMIT_EXCEEDED, cfd_get_last_status());

    cfd_registry_destroy(registry);
}

void test_null_pointer_handling(void) {
    // API should be robust against NULL pointers
    simulation_set_solver(NULL, NULL);
    // Should not crash

    int res = simulation_set_solver_by_name(NULL, "explicit_euler");
    TEST_ASSERT_EQUAL(-1, res);

    simulation_register_output(NULL, OUTPUT_VELOCITY, 1, "test");
    // Should verify error is set if we implemented it, but mostly just check no crash

    simulation_write_outputs(NULL, 1);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_grid_creation_zero_width);
    RUN_TEST(test_grid_creation_zero_height);
    RUN_TEST(test_grid_creation_invalid_xmin_xmax);
    RUN_TEST(test_grid_creation_invalid_ymin_ymax);
    RUN_TEST(test_grid_creation_equal_xmin_xmax);
    RUN_TEST(test_grid_creation_equal_ymin_ymax);
    RUN_TEST(test_simulation_init_zero_width);
    RUN_TEST(test_simulation_init_zero_height);
    RUN_TEST(test_simulation_init_invalid_bounds);
    RUN_TEST(test_simulation_init_equal_bounds);
    RUN_TEST(test_registry_register_null_registry);
    RUN_TEST(test_registry_register_null_factory);
    RUN_TEST(test_registry_register_empty_name);
    RUN_TEST(test_registry_register_limit_exceeded);
    RUN_TEST(test_null_pointer_handling);
    return UNITY_END();
}
