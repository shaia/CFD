/**
 * @file test_flow_field_3d.c
 * @brief Unit tests for 3D flow_field and derived_fields extensions
 *
 * Verifies that flow_field_create_3d works correctly for nz=1 and nz>1,
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
 * flow_field_create_3d tests
 * ============================================================================ */

void test_flow_field_create_3d_nz1(void) {
    flow_field* f = flow_field_create_3d(10, 10, 1);
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
    flow_field* f = flow_field_create(8, 6);
    TEST_ASSERT_NOT_NULL(f);

    TEST_ASSERT_EQUAL(8, f->nx);
    TEST_ASSERT_EQUAL(6, f->ny);
    TEST_ASSERT_EQUAL(1, f->nz);
    TEST_ASSERT_NOT_NULL(f->w);

    flow_field_destroy(f);
}

void test_flow_field_create_3d_allocates_correct_size(void) {
    flow_field* f = flow_field_create_3d(10, 8, 6);
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
    TEST_ASSERT_NULL(flow_field_create_3d(0, 10, 10));
    TEST_ASSERT_NULL(flow_field_create_3d(10, 0, 10));
    TEST_ASSERT_NULL(flow_field_create_3d(10, 10, 0));
}

/* ============================================================================
 * derived_fields 3D tests
 * ============================================================================ */

void test_derived_fields_create_3d_stores_nz(void) {
    derived_fields* d = derived_fields_create_3d(10, 8, 6);
    TEST_ASSERT_NOT_NULL(d);

    TEST_ASSERT_EQUAL(10, d->nx);
    TEST_ASSERT_EQUAL(8, d->ny);
    TEST_ASSERT_EQUAL(6, d->nz);

    derived_fields_destroy(d);
}

void test_derived_fields_create_wrapper_nz1(void) {
    derived_fields* d = derived_fields_create(10, 8);
    TEST_ASSERT_NOT_NULL(d);

    TEST_ASSERT_EQUAL(1, d->nz);

    derived_fields_destroy(d);
}

void test_velocity_magnitude_2d_unchanged(void) {
    // With w=0 (2D case), velocity magnitude should be sqrt(u^2 + v^2)
    flow_field* f = flow_field_create(4, 4);
    TEST_ASSERT_NOT_NULL(f);

    // Set u=3, v=4 everywhere => |v| = 5
    for (size_t i = 0; i < 16; i++) {
        f->u[i] = 3.0;
        f->v[i] = 4.0;
        // w is already 0 from calloc
    }

    derived_fields* d = derived_fields_create(4, 4);
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
    flow_field* f = flow_field_create_3d(4, 4, 2);
    TEST_ASSERT_NOT_NULL(f);

    size_t n = 4 * 4 * 2;
    for (size_t i = 0; i < n; i++) {
        f->u[i] = 1.0;
        f->v[i] = 2.0;
        f->w[i] = 2.0;
    }

    derived_fields* d = derived_fields_create_3d(4, 4, 2);
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
    flow_field* f = flow_field_create_3d(4, 4, 2);
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

    derived_fields* d = derived_fields_create_3d(4, 4, 2);
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
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    // flow_field_create_3d
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

    return UNITY_END();
}
