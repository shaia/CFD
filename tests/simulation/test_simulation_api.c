#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/grid.h"
#include "cfd/io/output_registry.h"
#include "unity.h"
#include <math.h>
#include <string.h>


// Test fixtures
static simulation_data* test_sim = NULL;

void setUp(void) {
    // Create a small simulation for testing
    // This implicitly initializes the library if strictly needed,
    // but we can rely on lazy init too.
    test_sim = init_simulation(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
}

void tearDown(void) {
    if (test_sim) {
        free_simulation(test_sim);
        test_sim = NULL;
    }
    // Optional: could finalize here, but we want to test persistence usually
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

void test_init_simulation_performs_lazy_initialization(void) {
    // 1. Ensure clean state
    if (test_sim) {
        free_simulation(test_sim);
        test_sim = NULL;
    }
    cfd_finalize();
    TEST_ASSERT_FALSE(cfd_is_initialized());

    // 2. Call init_simulation
    test_sim = init_simulation(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);

    // 3. Verify it succeeded and initialized the library
    TEST_ASSERT_NOT_NULL(test_sim);
    TEST_ASSERT_TRUE(cfd_is_initialized());
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
    simulation_data* sim = init_simulation_with_solver(5, 5, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, "explicit_euler");
    TEST_ASSERT_NOT_NULL(sim);
    TEST_ASSERT_NOT_NULL(sim->solver);
    free_simulation(sim);
}

void test_init_simulation_with_null_solver_uses_default(void) {
    simulation_data* sim = init_simulation_with_solver(5, 5, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, NULL);
    TEST_ASSERT_NOT_NULL(sim);
    TEST_ASSERT_NOT_NULL(sim->solver);
    free_simulation(sim);
}

void test_init_simulation_with_invalid_solver_returns_null(void) {
    simulation_data* sim =
        init_simulation_with_solver(5, 5, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, "nonexistent_solver");
    TEST_ASSERT_NULL(sim);
}

void test_init_simulation_with_failing_solver_init_returns_null(void) {
    /* A failing solver init must yield NULL and propagate its status. Which
     * failure comes first depends on the build: explicit_euler_optimized
     * rejects the 2-wide grid with INVALID where SIMD is available, and
     * reports UNSUPPORTED before looking at the grid where it is not. */
    simulation_data* sim = init_simulation_with_solver(2, 5, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    TEST_ASSERT_NULL(sim);
    if (cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
    } else {
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED, cfd_get_last_status());
    }
}

//=============================================================================
// SOLVER MANAGEMENT TESTS
//=============================================================================

void test_simulation_get_solver_returns_solver(void) {
    ns_solver_t* solver = simulation_get_solver(test_sim);
    TEST_ASSERT_NOT_NULL(solver);
}

void test_simulation_get_solver_null_returns_null(void) {
    ns_solver_t* solver = simulation_get_solver(NULL);
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

void test_simulation_list_solvers_names_are_valid_strings(void) {
    // This test verifies the fix for use-after-free bug where
    // solver names were invalidated after function return
    const char* names[10];
    int count = simulation_list_solvers(names, 10);
    TEST_ASSERT_GREATER_THAN(0, count);

    // Verify each name is a valid, non-empty string
    for (int i = 0; i < count; i++) {
        TEST_ASSERT_NOT_NULL(names[i]);
        TEST_ASSERT_GREATER_THAN(0, (int)strlen(names[i]));
        // Names should be reasonable length (not garbage)
        TEST_ASSERT_LESS_THAN(64, (int)strlen(names[i]));
    }
}

void test_simulation_list_solvers_names_contain_known_solvers(void) {
    // This test verifies that the returned names include known solver types
    const char* names[10];
    int count = simulation_list_solvers(names, 10);
    TEST_ASSERT_GREATER_THAN(0, count);

    // Look for explicit_euler (should always be present)
    int found_explicit_euler = 0;
    int found_projection = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], "explicit_euler") == 0) {
            found_explicit_euler = 1;
        }
        if (strcmp(names[i], "projection") == 0) {
            found_projection = 1;
        }
    }
    TEST_ASSERT_TRUE(found_explicit_euler);
    TEST_ASSERT_TRUE(found_projection);
}

void test_simulation_list_solvers_names_usable_for_init(void) {
    // This test verifies that returned names can actually be used
    // to create simulations (would fail with dangling pointers)
    const char* names[10];
    int count = simulation_list_solvers(names, 10);
    TEST_ASSERT_GREATER_THAN(0, count);

    // Try to create a simulation with the first solver name
    simulation_data* sim = init_simulation_with_solver(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, names[0]);
    TEST_ASSERT_NOT_NULL(sim);
    free_simulation(sim);
}

void test_simulation_list_solvers_null_names_returns_count(void) {
    // Should be able to query count without providing names array
    int count = simulation_list_solvers(NULL, 0);
    TEST_ASSERT_GREATER_THAN(0, count);
}

void test_simulation_list_solvers_partial_fill(void) {
    // Request fewer names than available
    const char* names[2];
    int count = simulation_list_solvers(names, 2);
    // Should return total count, but only fill 2 names
    TEST_ASSERT_GREATER_OR_EQUAL(2, count);
    TEST_ASSERT_NOT_NULL(names[0]);
    TEST_ASSERT_NOT_NULL(names[1]);
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
    cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_TRUE(test_sim->current_time > initial_time);
}

/* The step is the caller's params.dt. run_simulation_step() and
 * run_simulation_solve() used to overwrite it with 0.005 on every call, which
 * made the documented `sim->params.dt = ...` a no-op and broke the diffusive
 * stability limit on fine grids. Projection, because the explicit Euler solvers
 * clamp their own step. */
void test_run_simulation_step_uses_params_dt(void) {
    simulation_data* sim =
        init_simulation_with_solver(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, "projection");
    TEST_ASSERT_NOT_NULL(sim);

    sim->params.dt = 3e-4;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_simulation_step(sim));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4, sim->last_stats.dt_used);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4, sim->current_time);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4, sim->params.dt);

    /* A change between steps takes effect on the next one */
    sim->params.dt = 7e-4;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_simulation_step(sim));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 7e-4, sim->last_stats.dt_used);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1e-3, sim->current_time);

    free_simulation(sim);
}

void test_run_simulation_solve_uses_params_dt(void) {
    simulation_data* sim =
        init_simulation_with_solver(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, "projection");
    TEST_ASSERT_NOT_NULL(sim);

    sim->params.dt = 3e-4;
    sim->params.max_iter = 3;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_simulation_solve(sim));
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4, sim->last_stats.dt_used);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4 * sim->last_stats.iterations, sim->current_time);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3e-4, sim->params.dt);

    free_simulation(sim);
}

/* A step of zero, a negative step or a NaN step is refused before it reaches
 * a solver: zero left current_time frozen, a negative dt integrated backwards
 * and NaN poisoned every field without an error. */
static const double BAD_DTS[] = {0.0, -1e-3, NAN, INFINITY};
#define NUM_BAD_DTS (sizeof(BAD_DTS) / sizeof(BAD_DTS[0]))

void test_run_simulation_rejects_bad_dt(void) {
    simulation_data* sim =
        init_simulation_with_solver(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, "projection");
    TEST_ASSERT_NOT_NULL(sim);

    for (size_t k = 0; k < NUM_BAD_DTS; k++) {
        sim->params.dt = BAD_DTS[k];
        sim->params.max_iter = 3;
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, run_simulation_step(sim));
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, run_simulation_solve(sim));
        TEST_ASSERT_EQUAL_DOUBLE(0.0, sim->current_time);
    }
    free_simulation(sim);
}

/* Every registered solver, since solver_step() and solver_solve() are the
 * dispatchers each of them is reached through. */
void test_solver_step_and_solve_reject_bad_dt(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);
    grid* g = grid_create(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(9, 9, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    const char* names[32];
    int count = cfd_registry_list(registry, names, 32);
    int checked = 0;
    for (int s = 0; s < count; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, names[s]);
        if (!slv) {
            continue;
        }
        ns_solver_params_t params = ns_solver_params_default();
        if (solver_init(slv, g, &params) != CFD_SUCCESS) {
            solver_destroy(slv); /* backend not available here */
            continue;
        }
        for (size_t k = 0; k < NUM_BAD_DTS; k++) {
            params.dt = BAD_DTS[k];
            ns_solver_stats_t stats = ns_solver_stats_default();
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID,
                                          solver_step(slv, field, g, &params, &stats), names[s]);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID,
                                          solver_solve(slv, field, g, &params, &stats), names[s]);
        }
        solver_destroy(slv);
        checked++;
    }
    TEST_ASSERT_TRUE(checked > 0);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

/* The exported GPU entry points reach the kernels without solver_step(). Without
 * CUDA the stubs refuse everything, so only "not a success" can be asserted. */
void test_exported_gpu_entry_points_reject_bad_dt(void) {
    grid* g = grid_create(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(9, 9, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);
    gpu_config_t config = gpu_config_default();
    int have_gpu = gpu_is_available();

    for (size_t k = 0; k < NUM_BAD_DTS; k++) {
        ns_solver_params_t params = ns_solver_params_default();
        params.dt = BAD_DTS[k];
        cfd_status_t results[] = {
            solve_navier_stokes_gpu(field, g, &params, &config),
            solve_projection_method_gpu(field, g, &params, &config),
            solve_rk2_method_gpu(field, g, &params, &config),
            solve_rk4_method_gpu(field, g, &params, &config),
        };
        for (size_t r = 0; r < sizeof(results) / sizeof(results[0]); r++) {
            if (have_gpu) {
                TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, results[r]);
            } else {
                TEST_ASSERT_NOT_EQUAL(CFD_SUCCESS, results[r]);
            }
        }
    }
    flow_field_destroy(field);
    grid_destroy(g);
}

void test_run_simulation_step_updates_stats(void) {
    cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    const ns_solver_stats_t* stats = simulation_get_stats(test_sim);
    TEST_ASSERT_NOT_NULL(stats);
}

void test_run_simulation_step_null_sim_no_crash(void) {
    // Should return error for NULL input
    cfd_status_t status = run_simulation_step(NULL);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
}

void test_simulation_get_stats_returns_stats(void) {
    const ns_solver_stats_t* stats = simulation_get_stats(test_sim);
    TEST_ASSERT_NOT_NULL(stats);
}

void test_simulation_get_stats_null_returns_null(void) {
    const ns_solver_stats_t* stats = simulation_get_stats(NULL);
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
    output_registry* reg = output_registry_create();
    TEST_ASSERT_NOT_NULL(reg);
    output_registry_destroy(reg);
}

void test_output_registry_add_and_count(void) {
    output_registry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_VELOCITY_MAGNITUDE, 10, "test");
    TEST_ASSERT_EQUAL_INT(1, output_registry_count(reg));
    output_registry_destroy(reg);
}

void test_output_registry_clear(void) {
    output_registry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_VELOCITY_MAGNITUDE, 10, "test1");
    output_registry_add(reg, OUTPUT_VELOCITY, 20, "test2");
    output_registry_clear(reg);
    TEST_ASSERT_EQUAL_INT(0, output_registry_count(reg));
    output_registry_destroy(reg);
}

void test_output_registry_has_type_true(void) {
    output_registry* reg = output_registry_create();
    output_registry_add(reg, OUTPUT_CSV_TIMESERIES, 10, "test");
    TEST_ASSERT_TRUE(output_registry_has_type(reg, OUTPUT_CSV_TIMESERIES));
    output_registry_destroy(reg);
}

void test_output_registry_has_type_false(void) {
    output_registry* reg = output_registry_create();
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
    cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

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
        cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
        simulation_write_outputs(test_sim, step);
    }
}

//=============================================================================
// FIELD VALUE TESTS AFTER SIMULATION
//=============================================================================

void test_simulation_field_values_finite_after_step(void) {
    cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

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
    cfd_status_t status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    double time_after_1 = test_sim->current_time;
    status = run_simulation_step(test_sim);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
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
    RUN_TEST(test_init_simulation_performs_lazy_initialization);
    RUN_TEST(test_init_simulation_sets_grid_dimensions);
    RUN_TEST(test_init_simulation_sets_field_dimensions);
    RUN_TEST(test_init_simulation_sets_domain_bounds);
    RUN_TEST(test_init_simulation_sets_default_params);
    RUN_TEST(test_init_simulation_with_solver_creates_valid_structure);
    RUN_TEST(test_init_simulation_with_null_solver_uses_default);
    RUN_TEST(test_init_simulation_with_invalid_solver_returns_null);
    RUN_TEST(test_init_simulation_with_failing_solver_init_returns_null);

    // NSSolver management tests
    RUN_TEST(test_simulation_get_solver_returns_solver);
    RUN_TEST(test_simulation_get_solver_null_returns_null);
    RUN_TEST(test_simulation_set_solver_by_name_success);
    RUN_TEST(test_simulation_set_solver_by_name_invalid_returns_error);
    RUN_TEST(test_simulation_set_solver_by_name_null_sim_returns_error);
    RUN_TEST(test_simulation_set_solver_by_name_null_type_returns_error);
    RUN_TEST(test_simulation_list_solvers_returns_available);
    RUN_TEST(test_simulation_list_solvers_names_are_valid_strings);
    RUN_TEST(test_simulation_list_solvers_names_contain_known_solvers);
    RUN_TEST(test_simulation_list_solvers_names_usable_for_init);
    RUN_TEST(test_simulation_list_solvers_null_names_returns_count);
    RUN_TEST(test_simulation_list_solvers_partial_fill);
    RUN_TEST(test_simulation_has_solver_explicit_euler);
    RUN_TEST(test_simulation_has_solver_projection);
    RUN_TEST(test_simulation_has_solver_invalid);

    // Simulation execution tests
    RUN_TEST(test_run_simulation_step_advances_time);
    RUN_TEST(test_run_simulation_step_uses_params_dt);
    RUN_TEST(test_run_simulation_solve_uses_params_dt);
    RUN_TEST(test_run_simulation_rejects_bad_dt);
    RUN_TEST(test_solver_step_and_solve_reject_bad_dt);
    RUN_TEST(test_exported_gpu_entry_points_reject_bad_dt);
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
