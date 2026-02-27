/**
 * @file test_flow_field_3d.c
 * @brief Unit tests for 3D flow_field and derived_fields extensions
 *
 * Verifies that flow_field_create works correctly for nz=1 and nz>1,
 * and that derived_fields correctly includes w in velocity magnitude.
 */

#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * flow_field_create tests
 * ============================================================================ */

void test_flow_field_create_3d_nz1(void) {
    flow_field* f = flow_field_create(10, 10, 1);
    TEST_ASSERT_NOT_NULL(f);

    TEST_ASSERT_EQUAL(10, f->nx);
    TEST_ASSERT_EQUAL(10, f->ny);
    TEST_ASSERT_EQUAL(1, f->nz);
    TEST_ASSERT_NOT_NULL(f->u);
    TEST_ASSERT_NOT_NULL(f->v);
    TEST_ASSERT_NOT_NULL(f->w);
    TEST_ASSERT_NOT_NULL(f->p);
    TEST_ASSERT_NOT_NULL(f->rho);
    TEST_ASSERT_NOT_NULL(f->T);

    // w should be all zeros
    for (size_t i = 0; i < 10 * 10; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, f->w[i]);
    }

    flow_field_destroy(f);
}

void test_flow_field_create_wrapper_sets_nz1(void) {
    flow_field* f = flow_field_create(8, 6, 1);
    TEST_ASSERT_NOT_NULL(f);

    TEST_ASSERT_EQUAL(8, f->nx);
    TEST_ASSERT_EQUAL(6, f->ny);
    TEST_ASSERT_EQUAL(1, f->nz);
    TEST_ASSERT_NOT_NULL(f->w);

    flow_field_destroy(f);
}

void test_flow_field_create_3d_allocates_correct_size(void) {
    flow_field* f = flow_field_create(10, 8, 6);
    TEST_ASSERT_NOT_NULL(f);

    TEST_ASSERT_EQUAL(10, f->nx);
    TEST_ASSERT_EQUAL(8, f->ny);
    TEST_ASSERT_EQUAL(6, f->nz);

    // All arrays should be allocated (write to last element to verify size)
    size_t last = 10 * 8 * 6 - 1;
    f->u[last] = 1.0;
    f->v[last] = 2.0;
    f->w[last] = 3.0;
    f->p[last] = 4.0;
    f->rho[last] = 5.0;
    f->T[last] = 6.0;

    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1.0, f->u[last]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 2.0, f->v[last]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 3.0, f->w[last]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 4.0, f->p[last]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 5.0, f->rho[last]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 6.0, f->T[last]);

    flow_field_destroy(f);
}

void test_flow_field_create_3d_zero_dims_fails(void) {
    TEST_ASSERT_NULL(flow_field_create(0, 10, 10));
    TEST_ASSERT_NULL(flow_field_create(10, 0, 10));
    TEST_ASSERT_NULL(flow_field_create(10, 10, 0));
}

/* ============================================================================
 * derived_fields 3D tests
 * ============================================================================ */

void test_derived_fields_create_3d_stores_nz(void) {
    derived_fields* d = derived_fields_create(10, 8, 6);
    TEST_ASSERT_NOT_NULL(d);

    TEST_ASSERT_EQUAL(10, d->nx);
    TEST_ASSERT_EQUAL(8, d->ny);
    TEST_ASSERT_EQUAL(6, d->nz);

    derived_fields_destroy(d);
}

void test_derived_fields_create_wrapper_nz1(void) {
    derived_fields* d = derived_fields_create(10, 8, 1);
    TEST_ASSERT_NOT_NULL(d);

    TEST_ASSERT_EQUAL(1, d->nz);

    derived_fields_destroy(d);
}

void test_velocity_magnitude_2d_unchanged(void) {
    // With w=0 (2D case), velocity magnitude should be sqrt(u^2 + v^2)
    flow_field* f = flow_field_create(4, 4, 1);
    TEST_ASSERT_NOT_NULL(f);

    // Set u=3, v=4 everywhere => |v| = 5
    for (size_t i = 0; i < 16; i++) {
        f->u[i] = 3.0;
        f->v[i] = 4.0;
        // w is already 0 from calloc
    }

    derived_fields* d = derived_fields_create(4, 4, 1);
    TEST_ASSERT_NOT_NULL(d);

    derived_fields_compute_velocity_magnitude(d, f);
    TEST_ASSERT_NOT_NULL(d->velocity_magnitude);

    for (size_t i = 0; i < 16; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 5.0, d->velocity_magnitude[i]);
    }

    derived_fields_destroy(d);
    flow_field_destroy(f);
}

void test_velocity_magnitude_3d_includes_w(void) {
    // u=1, v=2, w=2 => |v| = sqrt(1+4+4) = 3
    flow_field* f = flow_field_create(4, 4, 2);
    TEST_ASSERT_NOT_NULL(f);

    size_t n = 4 * 4 * 2;
    for (size_t i = 0; i < n; i++) {
        f->u[i] = 1.0;
        f->v[i] = 2.0;
        f->w[i] = 2.0;
    }

    derived_fields* d = derived_fields_create(4, 4, 2);
    TEST_ASSERT_NOT_NULL(d);

    derived_fields_compute_velocity_magnitude(d, f);
    TEST_ASSERT_NOT_NULL(d->velocity_magnitude);

    for (size_t i = 0; i < n; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 3.0, d->velocity_magnitude[i]);
    }

    derived_fields_destroy(d);
    flow_field_destroy(f);
}

void test_statistics_3d_includes_w(void) {
    flow_field* f = flow_field_create(4, 4, 2);
    TEST_ASSERT_NOT_NULL(f);

    size_t n = 4 * 4 * 2;
    for (size_t i = 0; i < n; i++) {
        f->u[i] = 1.0;
        f->v[i] = 2.0;
        f->w[i] = 3.0;
        f->p[i] = 4.0;
        f->rho[i] = 5.0;
        f->T[i] = 6.0;
    }

    derived_fields* d = derived_fields_create(4, 4, 2);
    TEST_ASSERT_NOT_NULL(d);

    derived_fields_compute_statistics(d, f);
    TEST_ASSERT_EQUAL(1, d->stats_computed);

    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, d->u_stats.avg_val);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, d->v_stats.avg_val);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 3.0, d->w_stats.avg_val);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 4.0, d->p_stats.avg_val);

    derived_fields_destroy(d);
    flow_field_destroy(f);
}

/* ============================================================================
 * compute_time_step 3D CFL tests
 * ============================================================================ */

void test_cfl_2d_ignores_w(void) {
    // In 2D (nz=1), w should not affect CFL even if non-zero
    grid* g = grid_create(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(10, 10, 1);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();

    // Set uniform velocity u=1, v=0, w=0 and valid thermodynamic state
    for (size_t i = 0; i < 100; i++) {
        f->u[i] = 1.0;
        f->v[i] = 0.0;
        f->w[i] = 0.0;
        f->p[i] = 1.0;
        f->rho[i] = 1.0;
    }

    compute_time_step(f, g, &params);
    double dt_no_w = params.dt;

    // Now set large w - should NOT change dt since nz=1
    for (size_t i = 0; i < 100; i++) {
        f->w[i] = 100.0;
    }

    compute_time_step(f, g, &params);
    double dt_with_w = params.dt;

    TEST_ASSERT_DOUBLE_WITHIN(1e-15, dt_no_w, dt_with_w);

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_3d_includes_w(void) {
    // In 3D (nz>1), w must contribute to max_speed
    grid* g = grid_create(10, 10, 10, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(10, 10, 10);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();
    size_t n = (size_t)10 * 10 * 10;

    // Set uniform u=1, v=0, w=0 with valid thermodynamic state
    for (size_t i = 0; i < n; i++) {
        f->u[i] = 1.0;
        f->v[i] = 0.0;
        f->w[i] = 0.0;
        f->p[i] = 1.0;
        f->rho[i] = 1.0;
    }

    compute_time_step(f, g, &params);
    double dt_no_w = params.dt;

    // Now set large w - should reduce dt since nz>1
    for (size_t i = 0; i < n; i++) {
        f->w[i] = 10.0;
    }

    compute_time_step(f, g, &params);
    double dt_with_w = params.dt;

    // dt must be smaller when w is large (higher max_speed => smaller CFL dt)
    TEST_ASSERT_TRUE_MESSAGE(dt_with_w < dt_no_w,
        "CFL dt should decrease when w velocity is large in 3D");

    flow_field_destroy(f);
    grid_destroy(g);
}

void test_cfl_3d_dz_limits_dt(void) {
    // When dz is the smallest spacing, it should limit the time step
    // Use fine z-spacing with coarse x/y
    grid* g = grid_create(5, 5, 20, 0.0, 1.0, 0.0, 1.0, 0.0, 0.1);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* f = flow_field_create(5, 5, 20);
    TEST_ASSERT_NOT_NULL(f);

    ns_solver_params_t params = ns_solver_params_default();
    size_t n = 5 * 5 * 20;

    for (size_t i = 0; i < n; i++) {
        f->u[i] = 1.0;
        f->v[i] = 0.0;
        f->w[i] = 0.0;
        f->p[i] = 1.0;
        f->rho[i] = 1.0;
    }

    compute_time_step(f, g, &params);
    double dt_fine_z = params.dt;

    // Compare with isotropic grid (same nx/ny/nz but equal domain extent)
    grid* g2 = grid_create(5, 5, 20, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g2);
    grid_initialize_uniform(g2);

    flow_field* f2 = flow_field_create(5, 5, 20);
    TEST_ASSERT_NOT_NULL(f2);

    for (size_t i = 0; i < n; i++) {
        f2->u[i] = 1.0;
        f2->v[i] = 0.0;
        f2->w[i] = 0.0;
        f2->p[i] = 1.0;
        f2->rho[i] = 1.0;
    }

    compute_time_step(f2, g2, &params);
    double dt_iso = params.dt;

    // Fine z-spacing (dz = 0.1/19 ~ 0.005) vs isotropic (dz = 1/19 ~ 0.053)
    // dt_fine_z should be smaller
    TEST_ASSERT_TRUE_MESSAGE(dt_fine_z < dt_iso,
        "CFL dt should be smaller when z-spacing is finer");

    flow_field_destroy(f2);
    grid_destroy(g2);
    flow_field_destroy(f);
    grid_destroy(g);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    // flow_field_create
    RUN_TEST(test_flow_field_create_3d_nz1);
    RUN_TEST(test_flow_field_create_wrapper_sets_nz1);
    RUN_TEST(test_flow_field_create_3d_allocates_correct_size);
    RUN_TEST(test_flow_field_create_3d_zero_dims_fails);

    // derived_fields 3D
    RUN_TEST(test_derived_fields_create_3d_stores_nz);
    RUN_TEST(test_derived_fields_create_wrapper_nz1);
    RUN_TEST(test_velocity_magnitude_2d_unchanged);
    RUN_TEST(test_velocity_magnitude_3d_includes_w);
    RUN_TEST(test_statistics_3d_includes_w);

    // compute_time_step 3D CFL
    RUN_TEST(test_cfl_2d_ignores_w);
    RUN_TEST(test_cfl_3d_includes_w);
    RUN_TEST(test_cfl_3d_dz_limits_dt);

    return UNITY_END();
}
