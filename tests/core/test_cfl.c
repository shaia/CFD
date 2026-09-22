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


/* Regression: the convective CFL scan must cover every k-plane.
 *
 * ns_dt_convective() indexed with IDX_2D and looped only j and i, so on a 3D
 * grid it saw the k = 0 plane alone -- while explicitly handling w for 3D. A
 * field whose fastest flow sits above that plane got a dt sized from stagnant
 * fluid and blew past the CFL limit. Nothing caught it because every existing
 * CFL case is 2D, or puts its fast point in plane zero.
 */
void test_cfl_convective_scan_covers_all_k_planes(void) {
    const size_t nx = 10, ny = 10, nz = 5;
    grid* g = grid_create(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(nx, ny, nz);
    TEST_ASSERT_NOT_NULL(f);
    size_t total = nx * ny * nz;
    for (size_t i = 0; i < total; i++) {
        f->u[i] = 0.0;
        f->v[i] = 0.0;
        f->w[i] = 0.0;
        f->p[i] = 1.0;
        f->rho[i] = 1.0;
    }

    /* One fast cell, deliberately at k = 3 and NOT in the k = 0 plane. */
    size_t idx = 3 * nx * ny + 5 * nx + 5;
    f->u[idx] = 50.0;

    ns_solver_params_t params = ns_solver_params_default();
    compute_time_step(f, g, &params);

    /* dx = dy = 1/9 and dz = 1/4, so min spacing is 1/9. */
    double expected = params.cfl * (1.0 / 9.0) / (50.0 + sqrt(1.4));
    TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(
        1e-12, expected, params.dt,
        "convective CFL ignored a k > 0 plane; dt sized from stagnant fluid");

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
    /* Isolate the acoustic/advective branch: with the default mu the viscous
     * limit (dt ~ h^2) binds on this grid and masks the sound-speed scaling. */
    params.mu = 0.0;

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
    /* Isolate the acoustic/advective branch: with the default mu the viscous
     * limit (dt ~ h^2) binds on this grid and masks the sound-speed scaling. */
    params.mu = 0.0;

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
    /* Isolate the acoustic/advective branch: with the default mu the viscous
     * limit (dt ~ h^2) binds on this grid and masks the sound-speed scaling. */
    params.mu = 0.0;
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
 * Group 7: Viscous Diffusion Limit
 *
 * dt < cfl * h^2 / (2 * nu * ndim), nu = mu / rho.
 *
 * This limit scales as h^2 where the convective limit scales as h, so it
 * becomes the binding constraint as grids refine. It applies to the molecular
 * viscosity, not only to a turbulence model's eddy viscosity.
 * ============================================================================ */

void test_cfl_laminar_viscous_limit_binds_on_fine_grid(void) {
    /* The ROADMAP scenario: 129x129 unit cavity at Re=100 (nu = 1e-2). The
     * viscous limit is several times tighter than the CFL limit here, so a dt
     * chosen from CFL alone is unstable. */
    grid* g = grid_create(129, 129, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(129, 129, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-2;
    params.cfl = 0.5;

    compute_time_step(f, g, &params);

    const double h = 1.0 / 128.0;
    const double nu = params.mu / 1.0;
    const double dt_visc = params.cfl * h * h / (2.0 * nu * 2);
    const double dt_cfl = params.cfl * h / sqrt(1.4); /* zero velocity: sound speed only */

    TEST_ASSERT_DOUBLE_WITHIN(1e-12, dt_visc, params.dt);
    TEST_ASSERT_TRUE_MESSAGE(dt_visc < dt_cfl,
                             "viscous limit must bind before CFL on this grid");

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_viscous_limit_scales_as_h_squared(void) {
    /* Halving h quarters the viscous dt, where the CFL branch would only halve
     * it. This distinguishes the two branches unambiguously. */
    double dt[2];
    const size_t n[2] = {33, 65};

    for (int k = 0; k < 2; k++) {
        grid* g = grid_create(n[k], n[k], 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
        TEST_ASSERT_NOT_NULL(g);
        grid_initialize_uniform(g);

        flow_field* f = flow_field_create(n[k], n[k], 1);
        TEST_ASSERT_NOT_NULL(f);
        fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

        ns_solver_params_t params = ns_solver_params_default();
        params.mu = 0.5;  /* large enough that viscosity binds on both grids */
        params.cfl = 0.5;
        compute_time_step(f, g, &params);
        dt[k] = params.dt;

        flow_field_destroy(f);
        grid_destroy(g);
    }

    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 4.0, dt[0] / dt[1]);
}

void test_cfl_viscous_limit_scales_inversely_with_mu(void) {
    grid* g = grid_create(65, 65, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(65, 65, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    params.cfl = 0.5;

    params.mu = 0.25;
    compute_time_step(f, g, &params);
    double dt_low_mu = params.dt;

    params.mu = 0.5;
    compute_time_step(f, g, &params);
    double dt_high_mu = params.dt;

    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 2.0, dt_low_mu / dt_high_mu);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_zero_viscosity_disables_viscous_limit(void) {
    grid* g = grid_create(129, 129, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(129, 129, 1);
    TEST_ASSERT_NOT_NULL(f);
    fill_uniform(f, 0.0, 0.0, 1.0, 1.0);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 0.0;
    params.cfl = 0.5;

    compute_time_step(f, g, &params);

    const double h = 1.0 / 128.0;
    const double expected = params.cfl * h / sqrt(1.4);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, expected, params.dt);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_viscous_limit_uses_ndim_3_in_3d(void) {
    /* ndim goes 2 -> 3, so the 3D limit is 2/3 of the 2D one. */
    double dt[2];
    const size_t nz[2] = {1, 33};

    for (int k = 0; k < 2; k++) {
        grid* g = grid_create(33, 33, nz[k], 0.0, 1.0, 0.0, 1.0, 0.0,
                              (nz[k] > 1) ? 1.0 : 0.0);
        TEST_ASSERT_NOT_NULL(g);
        grid_initialize_uniform(g);

        flow_field* f = flow_field_create(33, 33, nz[k]);
        TEST_ASSERT_NOT_NULL(f);
        size_t total = f->nx * f->ny * f->nz;
        for (size_t i = 0; i < total; i++) {
            f->u[i] = 0.0;
            f->v[i] = 0.0;
            f->p[i] = 1.0;
            f->rho[i] = 1.0;
        }
        if (f->w) {
            for (size_t i = 0; i < total; i++) {
                f->w[i] = 0.0;
            }
        }

        ns_solver_params_t params = ns_solver_params_default();
        params.mu = 0.5;
        params.cfl = 0.5;
        compute_time_step(f, g, &params);
        dt[k] = params.dt;

        flow_field_destroy(f);
        grid_destroy(g);
    }

    TEST_ASSERT_DOUBLE_WITHIN(1e-9, 1.5, dt[0] / dt[1]);
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
    RUN_TEST(test_cfl_convective_scan_covers_all_k_planes);

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

    /* Viscous diffusion limit */
    RUN_TEST(test_cfl_laminar_viscous_limit_binds_on_fine_grid);
    RUN_TEST(test_cfl_viscous_limit_scales_as_h_squared);
    RUN_TEST(test_cfl_viscous_limit_scales_inversely_with_mu);
    RUN_TEST(test_cfl_zero_viscosity_disables_viscous_limit);
    RUN_TEST(test_cfl_viscous_limit_uses_ndim_3_in_3d);

    return UNITY_END();
}
