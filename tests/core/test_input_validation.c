#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "unity.h"

void setUp(void) {
    cfd_clear_error();
}

void tearDown(void) {}

void test_grid_creation_zero_width(void) {
    // Zero width
    grid* grid = grid_create(0, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_zero_height(void) {
    // Zero height
    grid* grid = grid_create(10, 0, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_grid_creation_invalid_bounds(void) {
    // 1. Invalid bounds xmin > xmax
    grid* grid = grid_create(10, 10, 1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 2. Invalid bounds ymin > ymax
    cfd_clear_error();
    grid = grid_create(10, 10, 1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 3. Invalid bounds xmin == xmax (Equality check)
    cfd_clear_error();
    grid = grid_create(10, 10, 1, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 4. Invalid bounds ymin == ymax (Equality check)
    cfd_clear_error();
    grid = grid_create(10, 10, 1, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(grid);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_zero_width(void) {
    // Zero dimensions
    simulation_data* sim = init_simulation(0, 100, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_zero_height(void) {
    simulation_data* sim = init_simulation(100, 0, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_simulation_init_invalid_bounds(void) {
    // 1. Invalid bounds (min > max)
    simulation_data* sim = init_simulation(100, 100, 1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());

    // 2. Equal bounds (min == max)
    cfd_clear_error();
    sim = init_simulation(100, 100, 1, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NULL(sim);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

// Dummy factory for testing
ns_solver_t* dummy_factory(void) {
    return NULL;
}

void test_registry_register_null_registry(void) {
    int res = cfd_registry_register(NULL, "test", dummy_factory);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_registry_register_null_factory(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    int res = cfd_registry_register(registry, "test", NULL);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
    cfd_registry_destroy(registry);
}

void test_registry_register_empty_name(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    int res = cfd_registry_register(registry, "", dummy_factory);
    TEST_ASSERT_EQUAL(-1, res);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
    cfd_registry_destroy(registry);
}

void test_registry_register_limit_exceeded(void) {
    ns_solver_registry_t* registry = cfd_registry_create();

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
    simulation_data* sim = init_simulation(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
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

/* ============================================================================
 * Helper: create a minimal initialized projection solver for NULL-arg tests
 * ============================================================================ */

typedef struct {
    ns_solver_registry_t* registry;
    ns_solver_t*          solver;
    grid*                 g;
    flow_field*           field;
    ns_solver_params_t    params;
} solver_test_ctx_t;

static int solver_test_ctx_init(solver_test_ctx_t* ctx) {
    ctx->registry = cfd_registry_create();
    if (!ctx->registry) return 0;
    cfd_registry_register_defaults(ctx->registry);

    ctx->g = grid_create(8, 8, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    if (!ctx->g) { cfd_registry_destroy(ctx->registry); return 0; }
    grid_initialize_uniform(ctx->g);

    ctx->field = flow_field_create(8, 8, 1);
    if (!ctx->field) {
        grid_destroy(ctx->g);
        cfd_registry_destroy(ctx->registry);
        return 0;
    }

    ctx->solver = cfd_solver_create(ctx->registry, NS_SOLVER_TYPE_PROJECTION);
    if (!ctx->solver) {
        flow_field_destroy(ctx->field);
        grid_destroy(ctx->g);
        cfd_registry_destroy(ctx->registry);
        return 0;
    }

    ctx->params = ns_solver_params_default();
    cfd_status_t st = solver_init(ctx->solver, ctx->g, &ctx->params);
    if (st != CFD_SUCCESS) {
        solver_destroy(ctx->solver);
        flow_field_destroy(ctx->field);
        grid_destroy(ctx->g);
        cfd_registry_destroy(ctx->registry);
        return 0;
    }

    return 1;
}

static void solver_test_ctx_destroy(solver_test_ctx_t* ctx) {
    solver_destroy(ctx->solver);
    flow_field_destroy(ctx->field);
    grid_destroy(ctx->g);
    cfd_registry_destroy(ctx->registry);
}

/* ============================================================================
 * cfd_solver_create — NULL registry / unknown name
 * ============================================================================ */

void test_solver_create_null_registry(void) {
    ns_solver_t* s = cfd_solver_create(NULL, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NULL(s);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_get_last_status());
}

void test_solver_create_unknown_name(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* s = cfd_solver_create(registry, "no_such_solver_xyz_999");
    TEST_ASSERT_NULL(s);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_NOT_FOUND, cfd_get_last_status());

    cfd_registry_destroy(registry);
}

/* ============================================================================
 * solver_destroy / solver_init / solver_step NULL guards
 * ============================================================================ */

void test_solver_destroy_null(void) {
    /* solver_registry.c:347 — guard: if (!solver) return */
    solver_destroy(NULL);
    TEST_PASS();
}

void test_solver_init_null_solver(void) {
    /* solver_registry.c:359 — guard: if (!solver) return CFD_ERROR_INVALID */
    cfd_status_t st = solver_init(NULL, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

void test_solver_step_null_solver(void) {
    /* solver_registry.c:372 — guard: if (!solver || ...) return CFD_ERROR_INVALID */
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t st = solver_step(NULL, NULL, NULL, NULL, &stats);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

void test_solver_step_null_field(void) {
    solver_test_ctx_t ctx;
    TEST_ASSERT_TRUE_MESSAGE(solver_test_ctx_init(&ctx), "Failed to init solver ctx");
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t st = solver_step(ctx.solver, NULL, ctx.g, &ctx.params, &stats);
    solver_test_ctx_destroy(&ctx);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

void test_solver_step_null_grid(void) {
    solver_test_ctx_t ctx;
    TEST_ASSERT_TRUE_MESSAGE(solver_test_ctx_init(&ctx), "Failed to init solver ctx");
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t st = solver_step(ctx.solver, ctx.field, NULL, &ctx.params, &stats);
    solver_test_ctx_destroy(&ctx);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

void test_solver_step_null_params(void) {
    solver_test_ctx_t ctx;
    TEST_ASSERT_TRUE_MESSAGE(solver_test_ctx_init(&ctx), "Failed to init solver ctx");
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t st = solver_step(ctx.solver, ctx.field, ctx.g, NULL, &stats);
    solver_test_ctx_destroy(&ctx);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

/* ============================================================================
 * solver_solve / solver_apply_boundary / solver_compute_dt NULL guards
 * ============================================================================ */

void test_solver_solve_null_solver(void) {
    /* solver_registry.c:395 — guard: if (!solver || ...) return CFD_ERROR_INVALID */
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t st = solver_solve(NULL, NULL, NULL, NULL, &stats);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, st);
}

void test_solver_apply_boundary_null(void) {
    /* solver_registry.c:416 — guard: if (!solver || !field || !grid) return */
    solver_apply_boundary(NULL, NULL, NULL);
    TEST_PASS();
}

void test_solver_compute_dt_null(void) {
    /* solver_registry.c:430 — guard: if (!solver || ...) return 0.0 */
    double dt = solver_compute_dt(NULL, NULL, NULL, NULL);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, dt);
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

    // cfd_solver_create — NULL registry / unknown name
    RUN_TEST(test_solver_create_null_registry);
    RUN_TEST(test_solver_create_unknown_name);

    // solver lifecycle NULL guards
    RUN_TEST(test_solver_destroy_null);
    RUN_TEST(test_solver_init_null_solver);
    RUN_TEST(test_solver_step_null_solver);
    RUN_TEST(test_solver_step_null_field);
    RUN_TEST(test_solver_step_null_grid);
    RUN_TEST(test_solver_step_null_params);

    // solver_solve / solver_apply_boundary / solver_compute_dt NULL guards
    RUN_TEST(test_solver_solve_null_solver);
    RUN_TEST(test_solver_apply_boundary_null);
    RUN_TEST(test_solver_compute_dt_null);

    return UNITY_END();
}
