/**
 * @file test_grid_3d.c
 * @brief Unit tests for 3D grid extension
 *
 * Verifies that grid_create_3d works correctly for both nz=1 (2D)
 * and nz>1 (3D) cases, including precomputed constants and z-initialization.
 */

#include "cfd/core/grid.h"
#include "unity.h"
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * grid_create_3d with nz=1 (2D compatibility)
 * ============================================================================ */

void test_grid_create_3d_nz1_matches_2d(void) {
    grid* g2d = grid_create(10, 10, 0.0, 1.0, 0.0, 1.0);
    grid* g3d = grid_create_3d(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g2d);
    TEST_ASSERT_NOT_NULL(g3d);

    TEST_ASSERT_EQUAL(g2d->nx, g3d->nx);
    TEST_ASSERT_EQUAL(g2d->ny, g3d->ny);
    TEST_ASSERT_EQUAL(1, g3d->nz);
    TEST_ASSERT_NULL(g3d->z);
    TEST_ASSERT_NULL(g3d->dz);
    TEST_ASSERT_EQUAL(0, g3d->stride_z);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, g3d->inv_dz2);
    TEST_ASSERT_EQUAL(0, g3d->k_start);
    TEST_ASSERT_EQUAL(1, g3d->k_end);

    grid_destroy(g2d);
    grid_destroy(g3d);
}

void test_grid_create_wrapper_sets_nz1(void) {
    grid* g = grid_create(8, 8, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    TEST_ASSERT_EQUAL(1, g->nz);
    TEST_ASSERT_NULL(g->z);
    TEST_ASSERT_NULL(g->dz);
    TEST_ASSERT_EQUAL(0, g->stride_z);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, g->inv_dz2);
    TEST_ASSERT_EQUAL(0, g->k_start);
    TEST_ASSERT_EQUAL(1, g->k_end);

    grid_destroy(g);
}

/* ============================================================================
 * grid_create_3d with nz>1 (true 3D)
 * ============================================================================ */

void test_grid_create_3d_allocates_z_arrays(void) {
    grid* g = grid_create_3d(10, 10, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0);
    TEST_ASSERT_NOT_NULL(g);

    TEST_ASSERT_EQUAL(8, g->nz);
    TEST_ASSERT_NOT_NULL(g->z);
    TEST_ASSERT_NOT_NULL(g->dz);

    grid_destroy(g);
}

void test_grid_create_3d_precomputed_constants(void) {
    grid* g = grid_create_3d(10, 12, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0);
    TEST_ASSERT_NOT_NULL(g);

    TEST_ASSERT_EQUAL(10 * 12, g->stride_z);
    TEST_ASSERT_EQUAL(1, g->k_start);
    TEST_ASSERT_EQUAL(7, g->k_end);  // nz - 1 = 8 - 1 = 7
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 2.0, g->zmax);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, g->zmin);

    grid_destroy(g);
}

void test_grid_create_3d_uniform_z_coordinates(void) {
    grid* g = grid_create_3d(5, 5, 11, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_uniform(g);

    // z should span [0, 2] with 11 points => dz = 0.2
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->z[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, g->z[10]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->z[5]);

    double expected_dz = 0.2;
    for (size_t k = 0; k < g->nz - 1; k++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected_dz, g->dz[k]);
    }

    // inv_dz2 should be 1/(0.2^2) = 25.0
    TEST_ASSERT_DOUBLE_WITHIN(1e-8, 25.0, g->inv_dz2);

    grid_destroy(g);
}

void test_grid_create_3d_nz1_uniform_skips_z(void) {
    grid* g = grid_create_3d(5, 5, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_uniform(g);

    // z arrays should still be NULL, inv_dz2 should still be 0
    TEST_ASSERT_NULL(g->z);
    TEST_ASSERT_NULL(g->dz);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, g->inv_dz2);

    // x/y should be initialized normally
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->x[4]);

    grid_destroy(g);
}

void test_grid_create_3d_stretched_z(void) {
    grid* g = grid_create_3d(5, 5, 21, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_stretched(g, 2.0);

    // z should span full domain
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->z[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->z[20]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->z[10]);

    // z should be monotonically increasing
    for (size_t k = 1; k < g->nz; k++) {
        TEST_ASSERT_TRUE(g->z[k] > g->z[k - 1]);
    }

    // dz near boundary should be smaller than at center
    TEST_ASSERT_TRUE(g->dz[0] < g->dz[10]);

    // inv_dz2 should be positive
    TEST_ASSERT_TRUE(g->inv_dz2 > 0.0);

    grid_destroy(g);
}

/* ============================================================================
 * Error handling
 * ============================================================================ */

void test_grid_create_3d_zero_nz_fails(void) {
    grid* g = grid_create_3d(10, 10, 0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(g);
}

void test_grid_create_3d_invalid_z_bounds_fails(void) {
    // nz > 1 but zmax <= zmin
    grid* g = grid_create_3d(10, 10, 5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0);
    TEST_ASSERT_NULL(g);

    g = grid_create_3d(10, 10, 5, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0);
    TEST_ASSERT_NULL(g);
}

void test_grid_create_3d_nz1_zero_zbounds_ok(void) {
    // nz=1 with zmin=zmax=0 should succeed (2D mode)
    grid* g = grid_create_3d(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_destroy(g);
}

void test_grid_destroy_handles_null_z(void) {
    // nz=1 grid has z=NULL, dz=NULL — destroy should not crash
    grid* g = grid_create(5, 5, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_destroy(g);  // Should not crash
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    // 2D compatibility
    RUN_TEST(test_grid_create_3d_nz1_matches_2d);
    RUN_TEST(test_grid_create_wrapper_sets_nz1);

    // 3D creation
    RUN_TEST(test_grid_create_3d_allocates_z_arrays);
    RUN_TEST(test_grid_create_3d_precomputed_constants);
    RUN_TEST(test_grid_create_3d_uniform_z_coordinates);
    RUN_TEST(test_grid_create_3d_nz1_uniform_skips_z);
    RUN_TEST(test_grid_create_3d_stretched_z);

    // Error handling
    RUN_TEST(test_grid_create_3d_zero_nz_fails);
    RUN_TEST(test_grid_create_3d_invalid_z_bounds_fails);
    RUN_TEST(test_grid_create_3d_nz1_zero_zbounds_ok);
    RUN_TEST(test_grid_destroy_handles_null_z);

    return UNITY_END();
}
