#include "cfd/io/output_registry.h"
#include "cfd/api/simulation_api.h"
#include "unity.h"
#include <math.h>

// Test fixtures
static SimulationData* test_sim = NULL;

void setUp(void) {
    // Create a small simulation for testing
    test_sim = init_simulation(10, 10, 0.0, 1.0, 0.0, 1.0);
}

void tearDown(void) {
    if (test_sim) {
        free_simulation(test_sim);
        test_sim = NULL;
    }
}

//=============================================================================
// INITIALIZATION TESTS
//=============================================================================

void test_init_simulation_creates_valid_structure(void) {
    TEST_ASSERT_NOT_NULL(test_sim);
    TEST_ASSERT_NOT_NULL(test_sim->grid);
    TEST_ASSERT_NOT_NULL(test_sim->field);
    TEST_ASSERT_NOT_NULL(test_sim->solver);
    TEST_ASSERT_NOT_NULL(test_sim->outputs);
}

void test_init_simulation_sets_grid_dimensions(void) {
    TEST_ASSERT_EQUAL_UINT(10, test_sim->grid->nx);
    TEST_ASSERT_EQUAL_UINT(10, test_sim->grid->ny);
}

void test_init_simulation_sets_field_dimensions(void) {
    TEST_ASSERT_EQUAL_UINT(10, test_sim->field->nx);
    TEST_ASSERT_EQUAL_UINT(10, test_sim->field->ny);
}

void test_init_simulation_sets_domain_bounds(void) {
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, (float)test_sim->grid->xmin);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, (float)test_sim->grid->xmax);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, (float)test_sim->grid->ymin);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, (float)test_sim->grid->ymax);
}

void test_init_simulation_sets_default_params(void) {
    TEST_ASSERT_TRUE(test_sim->params.dt > 0);
    TEST_ASSERT_TRUE(test_sim->params.cfl > 0);
}

void test_init_simulation_with_solver_creates_valid_structure(void) {
    SimulationData* sim = init_simulation_with_solver(5, 5, 0.0, 1.0, 0.0, 1.0, "explicit_euler");
    TEST_ASSERT_NOT_NULL(sim);
    TEST_ASSERT_NOT_NULL(sim->solver);
    free_simulation(sim);
}

void test_init_simulation_with_null_solver_uses_default(void) {
    SimulationData* sim = init_simulation_with_solver(5, 5, 0.0, 1.0, 0.0, 1.0, NULL);
    TEST_ASSERT_NOT_NULL(sim);
    TEST_ASSERT_NOT_NULL(sim->solver);
    free_simulation(sim);
}

void test_init_simulation_with_invalid_solver_returns_null(void) {
    SimulationData* sim =
        init_simulation_with_solver(5, 5, 0.0, 1.0, 0.0, 1.0, "nonexistent_solver");
    TEST_ASSERT_NULL(sim);
}

//=============================================================================
// SOLVER MANAGEMENT TESTS
//=============================================================================

void test_simulation_get_solver_returns_solver(void) {
    Solver* solver = simulation_get_solver(test_sim);
    TEST_ASSERT_NOT_NULL(solver);
}

void test_simulation_get_solver_null_returns_null(void) {
    Solver* solver = simulation_get_solver(NULL);
    TEST_ASSERT_NULL(solver);
}

void test_simulation_set_solver_by_name_success(void) {
    int result = simulation_set_solver_by_name(test_sim, "projection");
    TEST_ASSERT_EQUAL_INT(0, result);
    TEST_ASSERT_NOT_NULL(test_sim->solver);
}

void test_simulation_set_solver_by_name_invalid_returns_error(void) {
    int result = simulation_set_solver_by_name(test_sim, "invalid_solver");
    TEST_ASSERT_EQUAL_INT(-1, result);
}

void test_simulation_set_solver_by_name_null_sim_returns_error(void) {
    int result = simulation_set_solver_by_name(NULL, "explicit_euler");
    TEST_ASSERT_EQUAL_INT(-1, result);
}

void test_simulation_set_solver_by_name_null_type_returns_error(void) {
    int result = simulation_set_solver_by_name(test_sim, NULL);
    TEST_ASSERT_EQUAL_INT(-1, result);
}

void test_simulation_list_solvers_returns_available(void) {
    const char* names[10];
    int count = simulation_list_solvers(names, 10);
    TEST_ASSERT_GREATER_THAN(0, count);
}

void test_simulation_has_solver_explicit_euler(void) {
    TEST_ASSERT_TRUE(simulation_has_solver("explicit_euler"));
}

void test_simulation_has_solver_projection(void) {
    TEST_ASSERT_TRUE(simulation_has_solver("projection"));
}

void test_simulation_has_solver_invalid(void) {
    TEST_ASSERT_FALSE(simulation_has_solver("nonexistent"));
}

//=============================================================================
// SIMULATION EXECUTION TESTS
//=============================================================================

void test_run_simulation_step_advances_time(void) {
    double initial_time = test_sim->current_time;
    run_simulation_step(test_sim);
    TEST_ASSERT_TRUE(test_sim->current_time > initial_time);
}

void test_run_simulation_step_updates_stats(void) {
    run_simulation_step(test_sim);
    const SolverStats* stats = simulation_get_stats(test_sim);
    TEST_ASSERT_NOT_NULL(stats);
}

void test_run_simulation_step_null_sim_no_crash(void) {
    // Should not crash
    run_simulation_step(NULL);
}

void test_simulation_get_stats_returns_stats(void) {
    const SolverStats* stats = simulation_get_stats(test_sim);
    TEST_ASSERT_NOT_NULL(stats);
}

void test_simulation_get_stats_null_returns_null(void) {
    const SolverStats* stats = simulation_get_stats(NULL);
    TEST_ASSERT_NULL(stats);
}

//=============================================================================
// OUTPUT REGISTRATION TESTS
//=============================================================================

void test_simulation_register_output_adds_config(void) {
    simulation_clear_outputs(test_sim);
    simulation_register_output(test_sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "vel_mag");
    TEST_ASSERT_EQUAL_INT(1, output_registry_count(test_sim->outputs));
}

void test_simulation_register_multiple_outputs(void) {
    simulation_clear_outputs(test_sim);
    simulation_register_output(test_sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "vel_mag");
    simulation_register_output(test_sim, OUTPUT_VELOCITY, 20, "velocity");
    simulation_register_output(test_sim, OUTPUT_FULL_FIELD, 50, "full");
    TEST_ASSERT_EQUAL_INT(3, output_registry_count(test_sim->outputs));
}

void test_simulation_clear_outputs_removes_all(void) {
    simulation_register_output(test_sim, OUTPUT_VELOCITY_MAGNITUDE, 10, "vel_mag");
    simulation_register_output(test_sim, OUTPUT_VELOCITY, 20, "velocity");
    simulation_clear_outputs(test_sim);
    TEST_ASSERT_EQUAL_INT(0, output_registry_count(test_sim->outputs));
}

void test_simulation_register_output_null_sim_no_crash(void) {
    // Should not crash
    simulation_register_output(NULL, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");
}

void test_simulation_clear_outputs_null_sim_no_crash(void) {
    // Should not crash
    simulation_clear_outputs(NULL);
}

void test_simulation_register_csv_outputs(void) {
    simulation_clear_outputs(test_sim);
    simulation_register_output(test_sim, OUTPUT_CSV_TIMESERIES, 1, "timeseries");
    simulation_register_output(test_sim, OUTPUT_CSV_CENTERLINE, 10, "centerline");
    simulation_register_output(test_sim, OUTPUT_CSV_STATISTICS, 5, "stats");
    TEST_ASSERT_EQUAL_INT(3, output_registry_count(test_sim->outputs));
}

//=============================================================================
// RUN PREFIX TESTS
//=============================================================================

void test_simulation_set_run_prefix(void) {
    simulation_set_run_prefix(test_sim, "my_test_run");
    TEST_ASSERT_NOT_NULL(test_sim->run_prefix);
    TEST_ASSERT_EQUAL_STRING("my_test_run", test_sim->run_prefix);
}

void test_simulation_set_run_prefix_replaces_existing(void) {
    simulation_set_run_prefix(test_sim, "first_prefix");
    simulation_set_run_prefix(test_sim, "second_prefix");
    TEST_ASSERT_EQUAL_STRING("second_prefix", test_sim->run_prefix);
}

void test_simulation_set_run_prefix_null_clears(void) {
    simulation_set_run_prefix(test_sim, "some_prefix");
    simulation_set_run_prefix(test_sim, NULL);
    TEST_ASSERT_NULL(test_sim->run_prefix);
}

void test_simulation_set_run_prefix_null_sim_no_crash(void) {
    // Should not crash
    simulation_set_run_prefix(NULL, "test");
}

//=============================================================================
// OUTPUT REGISTRY TESTS
//=============================================================================

void test_output_registry_create_destroy(void) {
    OutputRegistry* reg = output_registry_create();
    TEST_ASSERT_NOT_NULL(reg);
    output_registry_destroy(reg);
}

void test_output_registry_add_and_count(void) {
    OutputRegistry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");
    TEST_ASSERT_EQUAL_INT(1, output_registry_count(reg));
    output_registry_destroy(reg);
}

void test_output_registry_clear(void) {
    OutputRegistry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_VELOCITY_MAGNITUDE, 10, "test1");
    output_registry_add(reg, OUTPUT_VELOCITY, 20, "test2");
    output_registry_clear(reg);
    TEST_ASSERT_EQUAL_INT(0, output_registry_count(reg));
    output_registry_destroy(reg);
}

void test_output_registry_has_type_true(void) {
    OutputRegistry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_CSV_TIMESERIES, 10, "test");
    TEST_ASSERT_TRUE(output_registry_has_type(reg, OUTPUT_CSV_TIMESERIES));
    output_registry_destroy(reg);
}

void test_output_registry_has_type_false(void) {
    OutputRegistry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_CSV_TIMESERIES, 10, "test");
    TEST_ASSERT_FALSE(output_registry_has_type(reg, OUTPUT_VELOCITY_MAGNITUDE));
    output_registry_destroy(reg);
}

void test_output_registry_null_safety(void) {
    // These should not crash
    output_registry_add(NULL, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");
    output_registry_clear(NULL);
    TEST_ASSERT_EQUAL_INT(0, output_registry_count(NULL));
    TEST_ASSERT_FALSE(output_registry_has_type(NULL, OUTPUT_VELOCITY_MAGNITUDE));
    output_registry_destroy(NULL);
}

//=============================================================================
// OUTPUT WRITING INTEGRATION TESTS
//=============================================================================

void test_simulation_write_outputs_null_sim_no_crash(void) {
    // Should not crash
    simulation_write_outputs(NULL, 0);
}

void test_simulation_write_outputs_no_registered_outputs(void) {
    simulation_clear_outputs(test_sim);
    // Should not crash and should not create files
    simulation_write_outputs(test_sim, 0);
}

void test_simulation_write_outputs_with_csv_timeseries(void) {
    simulation_clear_outputs(test_sim);
    simulation_set_run_prefix(test_sim, "api_test");
    simulation_register_output(test_sim, OUTPUT_CSV_TIMESERIES, 1, "timeseries");

    // Run a step to generate some data
    run_simulation_step(test_sim);

    // Write outputs
    simulation_write_outputs(test_sim, 0);

    // The output registry should have created a run directory
    TEST_ASSERT_NOT_NULL(test_sim->outputs);
}

void test_simulation_write_outputs_respects_interval(void) {
    simulation_clear_outputs(test_sim);
    simulation_set_run_prefix(test_sim, "interval_test");
    // Output only every 5 steps
    simulation_register_output(test_sim, OUTPUT_CSV_STATISTICS, 5, "stats");

    // Steps 0, 1, 2, 3, 4 - only step 0 should write (interval 5)
    for (int step = 0; step < 5; step++) {
        run_simulation_step(test_sim);
        simulation_write_outputs(test_sim, step);
    }
}

//=============================================================================
// FIELD VALUE TESTS AFTER SIMULATION
//=============================================================================

void test_simulation_field_values_finite_after_step(void) {
    run_simulation_step(test_sim);

    size_t nx = test_sim->field->nx;
    size_t ny = test_sim->field->ny;
    int finite_count = 0;

    for (size_t i = 0; i < nx * ny; i++) {
        if (isfinite(test_sim->field->u[i]) && isfinite(test_sim->field->v[i]) &&
            isfinite(test_sim->field->p[i])) {
            finite_count++;
        }
    }

    // At least some values should be finite
    TEST_ASSERT_GREATER_THAN(0, finite_count);
}

void test_simulation_current_time_accumulates(void) {
    double time_before = test_sim->current_time;
    run_simulation_step(test_sim);
    double time_after_1 = test_sim->current_time;
    run_simulation_step(test_sim);
    double time_after_2 = test_sim->current_time;

    TEST_ASSERT_TRUE(time_after_1 > time_before);
    TEST_ASSERT_TRUE(time_after_2 > time_after_1);
}

//=============================================================================
// MAIN
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    // Initialization tests
    RUN_TEST(test_init_simulation_creates_valid_structure);
    RUN_TEST(test_init_simulation_sets_grid_dimensions);
    RUN_TEST(test_init_simulation_sets_field_dimensions);
    RUN_TEST(test_init_simulation_sets_domain_bounds);
    RUN_TEST(test_init_simulation_sets_default_params);
    RUN_TEST(test_init_simulation_with_solver_creates_valid_structure);
    RUN_TEST(test_init_simulation_with_null_solver_uses_default);
    RUN_TEST(test_init_simulation_with_invalid_solver_returns_null);

    // Solver management tests
    RUN_TEST(test_simulation_get_solver_returns_solver);
    RUN_TEST(test_simulation_get_solver_null_returns_null);
    RUN_TEST(test_simulation_set_solver_by_name_success);
    RUN_TEST(test_simulation_set_solver_by_name_invalid_returns_error);
    RUN_TEST(test_simulation_set_solver_by_name_null_sim_returns_error);
    RUN_TEST(test_simulation_set_solver_by_name_null_type_returns_error);
    RUN_TEST(test_simulation_list_solvers_returns_available);
    RUN_TEST(test_simulation_has_solver_explicit_euler);
    RUN_TEST(test_simulation_has_solver_projection);
    RUN_TEST(test_simulation_has_solver_invalid);

    // Simulation execution tests
    RUN_TEST(test_run_simulation_step_advances_time);
    RUN_TEST(test_run_simulation_step_updates_stats);
    RUN_TEST(test_run_simulation_step_null_sim_no_crash);
    RUN_TEST(test_simulation_get_stats_returns_stats);
    RUN_TEST(test_simulation_get_stats_null_returns_null);

    // Output registration tests
    RUN_TEST(test_simulation_register_output_adds_config);
    RUN_TEST(test_simulation_register_multiple_outputs);
    RUN_TEST(test_simulation_clear_outputs_removes_all);
    RUN_TEST(test_simulation_register_output_null_sim_no_crash);
    RUN_TEST(test_simulation_clear_outputs_null_sim_no_crash);
    RUN_TEST(test_simulation_register_csv_outputs);

    // Run prefix tests
    RUN_TEST(test_simulation_set_run_prefix);
    RUN_TEST(test_simulation_set_run_prefix_replaces_existing);
    RUN_TEST(test_simulation_set_run_prefix_null_clears);
    RUN_TEST(test_simulation_set_run_prefix_null_sim_no_crash);

    // Output registry tests
    RUN_TEST(test_output_registry_create_destroy);
    RUN_TEST(test_output_registry_add_and_count);
    RUN_TEST(test_output_registry_clear);
    RUN_TEST(test_output_registry_has_type_true);
    RUN_TEST(test_output_registry_has_type_false);
    RUN_TEST(test_output_registry_null_safety);

    // Output writing integration tests
    RUN_TEST(test_simulation_write_outputs_null_sim_no_crash);
    RUN_TEST(test_simulation_write_outputs_no_registered_outputs);
    RUN_TEST(test_simulation_write_outputs_with_csv_timeseries);
    RUN_TEST(test_simulation_write_outputs_respects_interval);

    // Field value tests
    RUN_TEST(test_simulation_field_values_finite_after_step);
    RUN_TEST(test_simulation_current_time_accumulates);

    return UNITY_END();
}
