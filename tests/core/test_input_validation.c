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

void test_grid_creation_invalid_bounds(void) {
    // 1. Invalid bounds xmin > xmax
    Grid* grid = grid_create(10, 10, 1.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 2. Invalid bounds ymin > ymax
    cfd_clear_error();
    grid = grid_create(10, 10, 0.0, 1.0, 1.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 3. Invalid bounds xmin == xmax (Equality check)
    cfd_clear_error();
    grid = grid_create(10, 10, 1.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 4. Invalid bounds ymin == ymax (Equality check)
    cfd_clear_error();
    grid = grid_create(10, 10, 0.0, 1.0, 1.0, 1.0);
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
    // 1. Invalid bounds (min > max)
    SimulationData* sim = init_simulation(100, 100, 1.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 2. Equal bounds (min == max)
    cfd_clear_error();
    sim = init_simulation(100, 100, 1.0, 1.0, 0.0, 1.0);
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

    // 1. Test with completely NULL arguments
    cfd_clear_error();
    simulation_set_solver(NULL, NULL);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    cfd_clear_error();
    int res = simulation_set_solver_by_name(NULL, "explicit_euler");
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    cfd_clear_error();
    simulation_register_output(NULL, OUTPUT_VELOCITY, 1, "test");
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    cfd_clear_error();
    simulation_write_outputs(NULL, 1);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 2. Test with partial NULL arguments where applicable
    SimulationData* sim = init_simulation(10, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(sim);

    // simulation_set_solver: valid sim, NULL solver
    cfd_clear_error();
    simulation_set_solver(sim, NULL);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // simulation_set_solver: NULL sim, valid (or dummy) solver
    // (We rely on logic that check happens before access, so passing a non-NULL dummy pointer is
    // "safe" for the check) However, to be cleaner, we can use the solver from the sim just for the
    // pointer value, or NULL. We already tested NULL/NULL. Let's strictly test NULL/Valid if we
    // could extract a solver. But simulation_set_solver(NULL, allocated_solver) is valid to test.
    // For now, NULL/NULL covers the "sim is NULL" branch in implementation usually: "if (!sim ||
    // !solver)"

    // simulation_set_solver_by_name: valid sim, NULL name
    cfd_clear_error();
    res = simulation_set_solver_by_name(sim, NULL);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // Clean up
    free_simulation(sim);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_grid_creation_zero_width);
    RUN_TEST(test_grid_creation_zero_height);
    RUN_TEST(test_grid_creation_invalid_bounds);
    RUN_TEST(test_simulation_init_zero_width);
    RUN_TEST(test_simulation_init_zero_height);
    RUN_TEST(test_simulation_init_invalid_bounds);
    RUN_TEST(test_registry_register_null_registry);
    RUN_TEST(test_registry_register_null_factory);
    RUN_TEST(test_registry_register_empty_name);
    RUN_TEST(test_registry_register_limit_exceeded);
    RUN_TEST(test_null_pointer_handling);
    return UNITY_END();
}
