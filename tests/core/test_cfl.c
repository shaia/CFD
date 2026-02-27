/**
 * @file test_cfl.c
 * @brief Unit tests for compute_time_step CFL condition
 *
 * Tests formula correctness, scaling relationships, clamping behavior,
 * sound speed effects, grid spacing effects, and edge cases.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"
#include <math.h>

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* Helper: fill all points of a 2D flow field with uniform values */
static void fill_uniform(flow_field* f, double u, double v,
                         double p, double rho) {
    size_t n = f->nx * f->ny;
    for (size_t i = 0; i < n; i++) {
        f->u[i] = u;
        f->v[i] = v;
        f->p[i] = p;
        f->rho[i] = rho;
    }
}

/* ============================================================================
 * Group 1: Formula Scaling
 * ============================================================================ */

void test_cfl_dt_scales_with_cfl_number(void) {
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 1.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();

    params.cfl = 0.1;
    compute_time_step(f, g, &params);
    double dt_low = params.dt;

    params.cfl = 0.2;
    compute_time_step(f, g, &params);
    double dt_high = params.dt;

    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, dt_high / dt_low);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_dt_scales_with_grid_spacing(void) {
    /* Same number of points, double the domain -> double dmin -> double dt */
    flow_field* f1 = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f1);
    fill_uniform(f1, 1.0, 0.0, 1.0, 1.0);

    flow_field* f2 = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f2);
    fill_uniform(f2, 1.0, 0.0, 1.0, 1.0);

    grid* g1 = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g1);
    grid_initialize_uniform(g1);

    grid* g2 = grid_create(51, 51, 1, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g2);
    grid_initialize_uniform(g2);

    ns_solver_params_t params = ns_solver_params_default();
    params.cfl = 0.1;

    compute_time_step(f1, g1, &params);
    double dt_fine = params.dt;

    compute_time_step(f2, g2, &params);
    double dt_coarse = params.dt;

    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, dt_coarse / dt_fine);

    flow_field_destroy(f2);
    grid_destroy(g2);
    flow_field_destroy(f1);
    grid_destroy(g1);
}

void test_cfl_dt_scales_inversely_with_velocity(void) {
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();
    double c = sqrt(params.gamma * 1.0 / 1.0); /* sound speed */

    fill_uniform(f, 1.0, 0.0, 1.0, 1.0);
    compute_time_step(f, g, &params);
    double dt_slow = params.dt;

    fill_uniform(f, 5.0, 0.0, 1.0, 1.0);
    compute_time_step(f, g, &params);
    double dt_fast = params.dt;

    /* dt_slow / dt_fast = max_speed_fast / max_speed_slow */
    double expected_ratio = (5.0 + c) / (1.0 + c);
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, expected_ratio, dt_slow / dt_fast);

    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Group 2: Velocity Effects
 * ============================================================================ */

void test_cfl_exact_value_zero_velocity(void) {
    grid* g = grid_create(21, 21, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(21, 21, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* dmin = 1/20 = 0.05, max_speed = sqrt(gamma) = sqrt(1.4) */
    double expected = params.cfl * (1.0 / 20.0) / sqrt(1.4);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_mixed_uv_velocity(void) {
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 3.0, 4.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* vel_mag = sqrt(9+16) = 5.0, max_speed = 5.0 + sqrt(1.4) */
    double expected = params.cfl * (1.0 / 50.0) / (5.0 + sqrt(1.4));
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_single_high_velocity_point_dominates(void) {
    grid* g = grid_create(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(10, 10, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    /* Set one point to high velocity */
    f->u[55] = 50.0;

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* max_speed from that single point: 50 + sqrt(1.4) */
    double expected = params.cfl * (1.0 / 9.0) / (50.0 + sqrt(1.4));
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Group 3: Sound Speed Effects
 * ============================================================================ */

void test_cfl_higher_pressure_reduces_dt(void) {
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();

    /* Low pressure: sound_speed = sqrt(gamma * 1 / 1) = sqrt(1.4) */
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);
    compute_time_step(f, g, &params);
    double dt_lowp = params.dt;

    /* 4x pressure: sound_speed = sqrt(gamma * 4 / 1) = 2*sqrt(1.4) */
    fill_uniform(f, 0.0, 0.0, 4.0, 1.0);
    compute_time_step(f, g, &params);
    double dt_highp = params.dt;

    /* Ratio should be 2.0 (sound speed doubled -> dt halved) */
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, dt_lowp / dt_highp);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_higher_density_increases_dt(void) {
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();

    /* Low density: sound_speed = sqrt(gamma * 1 / 1) = sqrt(1.4) */
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);
    compute_time_step(f, g, &params);
    double dt_light = params.dt;

    /* 4x density: sound_speed = sqrt(gamma * 1 / 4) = sqrt(1.4)/2 */
    fill_uniform(f, 0.0, 0.0, 1.0, 4.0);
    compute_time_step(f, g, &params);
    double dt_dense = params.dt;

    /* Ratio should be 2.0 (sound speed halved -> dt doubled) */
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, dt_dense / dt_light);

    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Group 4: Grid Spacing Effects
 * ============================================================================ */

void test_cfl_anisotropic_grid_uses_min_spacing(void) {
    /* dx = 2/20 = 0.1, dy = 1/20 = 0.05 -> dmin = 0.05 (same as isotropic) */
    grid* g_aniso = grid_create(21, 21, 1, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g_aniso);
    grid_initialize_uniform(g_aniso);

    grid* g_iso = grid_create(21, 21, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g_iso);
    grid_initialize_uniform(g_iso);

    flow_field* f1 = flow_field_create(21, 21, 1);
    TEST_ASSERT_NOT_NULL(f1);
    fill_uniform(f1, 1.0, 0.0, 1.0, 1.0);

    flow_field* f2 = flow_field_create(21, 21, 1);
    TEST_ASSERT_NOT_NULL(f2);
    fill_uniform(f2, 1.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();

    compute_time_step(f1, g_aniso, &params);
    double dt_aniso = params.dt;

    compute_time_step(f2, g_iso, &params);
    double dt_iso = params.dt;

    /* Both have dmin = 0.05, so dt should be identical */
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, dt_iso, dt_aniso);

    flow_field_destroy(f2);
    grid_destroy(g_iso);
    flow_field_destroy(f1);
    grid_destroy(g_aniso);
}

void test_cfl_stretched_grid_uses_min_spacing(void) {
    grid* g_uniform = grid_create(21, 21, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g_uniform);
    grid_initialize_uniform(g_uniform);

    grid* g_stretched = grid_create(21, 21, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g_stretched);
    grid_initialize_stretched(g_stretched, 2.0);

    flow_field* f1 = flow_field_create(21, 21, 1);
    TEST_ASSERT_NOT_NULL(f1);
    fill_uniform(f1, 1.0, 0.0, 1.0, 1.0);

    flow_field* f2 = flow_field_create(21, 21, 1);
    TEST_ASSERT_NOT_NULL(f2);
    fill_uniform(f2, 1.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();

    compute_time_step(f1, g_uniform, &params);
    double dt_uniform = params.dt;

    compute_time_step(f2, g_stretched, &params);
    double dt_stretched = params.dt;

    /* Stretched grid has smaller cells near boundaries -> smaller dt */
    TEST_ASSERT_TRUE_MESSAGE(dt_stretched < dt_uniform,
        "Stretched grid should produce smaller dt due to finer boundary cells");

    flow_field_destroy(f2);
    grid_destroy(g_stretched);
    flow_field_destroy(f1);
    grid_destroy(g_uniform);
}

/* ============================================================================
 * Group 5: Clamping Behavior
 * ============================================================================ */

void test_cfl_dt_clamped_at_max_limit(void) {
    /* Large grid spacing + zero velocity -> dt_cfl >> 0.01 */
    grid* g = grid_create(3, 3, 1, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(3, 3, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* dt_cfl = 0.2 * 5.0 / sqrt(1.4) = 0.845 >> 0.01 (DT_MAX_LIMIT) */
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.01, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_dt_clamped_at_min_limit(void) {
    /* Tiny grid + fast velocity -> dt_cfl << 1e-6 */
    grid* g = grid_create(10, 10, 1, 0.0, 0.0001, 0.0, 0.0001, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(10, 10, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 100.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* dt_cfl = 0.2 * 1.11e-5 / 101.18 = 2.2e-8 << 1e-6 (DT_MIN_LIMIT) */
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1e-6, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Group 6: Edge Cases
 * ============================================================================ */

void test_cfl_near_zero_speed_fallback(void) {
    /* Near-zero pressure -> sound speed < SPEED_EPSILON -> fallback max_speed=1.0 */
    grid* g = grid_create(51, 51, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(51, 51, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1e-25, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* Fallback: max_speed = 1.0, dt = 0.2 * 0.02 / 1.0 = 0.004 */
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.004, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_nonuniform_velocity_field_uses_max(void) {
    grid* g = grid_create(20, 20, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(20, 20, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    /* Set last row to high velocity */
    for (size_t i = 0; i < 20; i++) {
        f->u[19 * 20 + i] = 10.0;
    }

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* max_speed from the fast row: 10.0 + sqrt(1.4) */
    double expected = params.cfl * (1.0 / 19.0) / (10.0 + sqrt(1.4));
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Formula scaling */
    RUN_TEST(test_cfl_dt_scales_with_cfl_number);
    RUN_TEST(test_cfl_dt_scales_with_grid_spacing);
    RUN_TEST(test_cfl_dt_scales_inversely_with_velocity);

    /* Velocity effects */
    RUN_TEST(test_cfl_exact_value_zero_velocity);
    RUN_TEST(test_cfl_mixed_uv_velocity);
    RUN_TEST(test_cfl_single_high_velocity_point_dominates);

    /* Sound speed */
    RUN_TEST(test_cfl_higher_pressure_reduces_dt);
    RUN_TEST(test_cfl_higher_density_increases_dt);

    /* Grid spacing */
    RUN_TEST(test_cfl_anisotropic_grid_uses_min_spacing);
    RUN_TEST(test_cfl_stretched_grid_uses_min_spacing);

    /* Clamping */
    RUN_TEST(test_cfl_dt_clamped_at_max_limit);
    RUN_TEST(test_cfl_dt_clamped_at_min_limit);

    /* Edge cases */
    RUN_TEST(test_cfl_near_zero_speed_fallback);
    RUN_TEST(test_cfl_nonuniform_velocity_field_uses_max);

    return UNITY_END();
}
