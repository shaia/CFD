/**
 * @file test_grid.c
 * @brief Unit tests for grid creation and initialization functions
 *
 * Tests uniform and stretched grid generation, verifying:
 * - Grid spans correct domain [xmin, xmax] x [ymin, ymax]
 * - Stretched grids cluster points near boundaries
 * - Edge cases (beta=0, small grids)
 */

#include "cfd/core/grid.h"
#include "unity.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

/* ============================================================================
 * Uniform Grid Tests
 * ============================================================================ */

void test_uniform_grid_spans_full_domain(void) {
    grid* g = grid_create(11, 11, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_uniform(g);

    // First point should be at xmin, ymin
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->y[0]);

    // Last point should be at xmax, ymax
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->x[g->nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->y[g->ny - 1]);

    grid_destroy(g);
}

void test_uniform_grid_equal_spacing(void) {
    grid* g = grid_create(5, 5, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_uniform(g);

    // Expected spacing: (1.0 - 0.0) / (5 - 1) = 0.25
    double expected_dx = 0.25;
    double expected_dy = 0.25;

    for (size_t i = 0; i < g->nx - 1; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected_dx, g->dx[i]);
    }
    for (size_t j = 0; j < g->ny - 1; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, expected_dy, g->dy[j]);
    }

    grid_destroy(g);
}

void test_uniform_grid_non_unit_domain(void) {
    grid* g = grid_create(5, 5, -2.0, 3.0, 1.0, 6.0);
    TEST_ASSERT_NOT_NULL(g);

    grid_initialize_uniform(g);

    // First point should be at xmin, ymin
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, -2.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->y[0]);

    // Last point should be at xmax, ymax
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 3.0, g->x[g->nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 6.0, g->y[g->ny - 1]);

    // Expected spacing: 5.0 / 4 = 1.25
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.25, g->dx[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.25, g->dy[0]);

    grid_destroy(g);
}

/* ============================================================================
 * Stretched Grid Tests
 * ============================================================================ */

void test_stretched_grid_spans_full_domain(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;  // Moderate stretching
    grid_initialize_stretched(g, beta);

    // First point should be exactly at xmin, ymin
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->y[0]);

    // Last point should be exactly at xmax, ymax
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->x[g->nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->y[g->ny - 1]);

    // Middle point should be at domain center
    size_t mid = g->nx / 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->x[mid]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->y[mid]);

    grid_destroy(g);
}

void test_stretched_grid_clusters_near_boundaries(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;
    grid_initialize_stretched(g, beta);

    // For stretched grid, spacing should be smaller near boundaries
    // than in the center
    double dx_near_boundary = g->dx[0];  // First cell
    double dx_at_center = g->dx[g->nx / 2];  // Center cell

    // Near-boundary spacing should be smaller than center spacing
    TEST_ASSERT_TRUE(dx_near_boundary < dx_at_center);

    // Same for the other boundary
    double dx_near_other_boundary = g->dx[g->nx - 2];  // Last cell
    TEST_ASSERT_TRUE(dx_near_other_boundary < dx_at_center);

    // Both boundary spacings should be approximately equal (symmetric)
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, dx_near_boundary, dx_near_other_boundary);

    grid_destroy(g);
}

void test_stretched_grid_higher_beta_more_clustering(void) {
    grid* g1 = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    grid* g2 = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g1);
    TEST_ASSERT_NOT_NULL(g2);

    double beta_low = 1.0;
    double beta_high = 3.0;

    grid_initialize_stretched(g1, beta_low);
    grid_initialize_stretched(g2, beta_high);

    // Higher beta should have smaller spacing near boundary
    TEST_ASSERT_TRUE(g2->dx[0] < g1->dx[0]);

    // Higher beta should have larger spacing at center
    TEST_ASSERT_TRUE(g2->dx[g2->nx / 2] > g1->dx[g1->nx / 2]);

    grid_destroy(g1);
    grid_destroy(g2);
}

void test_stretched_grid_beta_zero_equals_uniform(void) {
    grid* g_stretched = grid_create(11, 11, 0.0, 1.0, 0.0, 1.0);
    grid* g_uniform = grid_create(11, 11, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g_stretched);
    TEST_ASSERT_NOT_NULL(g_uniform);

    grid_initialize_stretched(g_stretched, 0.0);  // beta = 0
    grid_initialize_uniform(g_uniform);

    // Should be identical to uniform grid
    for (size_t i = 0; i < g_stretched->nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, g_uniform->x[i], g_stretched->x[i]);
    }
    for (size_t j = 0; j < g_stretched->ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, g_uniform->y[j], g_stretched->y[j]);
    }

    grid_destroy(g_stretched);
    grid_destroy(g_uniform);
}

void test_stretched_grid_non_unit_domain(void) {
    grid* g = grid_create(21, 21, -1.0, 2.0, 0.5, 1.5);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;
    grid_initialize_stretched(g, beta);

    // First point should be at xmin, ymin
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, -1.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->y[0]);

    // Last point should be at xmax, ymax
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 2.0, g->x[g->nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.5, g->y[g->ny - 1]);

    // Middle point should be at domain center
    size_t mid = g->nx / 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->x[mid]);   // (-1 + 2) / 2 = 0.5
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->y[mid]);   // (0.5 + 1.5) / 2 = 1.0

    grid_destroy(g);
}

void test_stretched_grid_monotonically_increasing(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.5;
    grid_initialize_stretched(g, beta);

    // x coordinates should be strictly increasing
    for (size_t i = 1; i < g->nx; i++) {
        TEST_ASSERT_TRUE(g->x[i] > g->x[i - 1]);
    }

    // y coordinates should be strictly increasing
    for (size_t j = 1; j < g->ny; j++) {
        TEST_ASSERT_TRUE(g->y[j] > g->y[j - 1]);
    }

    // All dx values should be positive
    for (size_t i = 0; i < g->nx - 1; i++) {
        TEST_ASSERT_TRUE(g->dx[i] > 0.0);
    }

    // All dy values should be positive
    for (size_t j = 0; j < g->ny - 1; j++) {
        TEST_ASSERT_TRUE(g->dy[j] > 0.0);
    }

    grid_destroy(g);
}

void test_stretched_grid_y_direction_clustering(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;
    grid_initialize_stretched(g, beta);

    // Y-direction should also cluster near boundaries
    double dy_near_boundary = g->dy[0];
    double dy_at_center = g->dy[g->ny / 2];
    double dy_near_other_boundary = g->dy[g->ny - 2];

    // Near-boundary spacing should be smaller than center spacing
    TEST_ASSERT_TRUE(dy_near_boundary < dy_at_center);
    TEST_ASSERT_TRUE(dy_near_other_boundary < dy_at_center);

    // Both boundary spacings should be approximately equal (symmetric)
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, dy_near_boundary, dy_near_other_boundary);

    grid_destroy(g);
}

void test_stretched_grid_minimum_size(void) {
    // Test with minimum usable grid size (3x3)
    grid* g = grid_create(3, 3, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;
    grid_initialize_stretched(g, beta);

    // Should still span full domain
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->x[2]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->x[1]);  // Middle point at center

    // dx values should be positive
    TEST_ASSERT_TRUE(g->dx[0] > 0.0);
    TEST_ASSERT_TRUE(g->dx[1] > 0.0);

    grid_destroy(g);
}

void test_stretched_grid_negative_beta(void) {
    grid* g_pos = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    grid* g_neg = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g_pos);
    TEST_ASSERT_NOT_NULL(g_neg);

    // Due to tanh(-x) = -tanh(x), negative beta should produce same result
    grid_initialize_stretched(g_pos, 2.0);
    grid_initialize_stretched(g_neg, -2.0);

    for (size_t i = 0; i < g_pos->nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, g_pos->x[i], g_neg->x[i]);
    }
    for (size_t j = 0; j < g_pos->ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, g_pos->y[j], g_neg->y[j]);
    }

    grid_destroy(g_pos);
    grid_destroy(g_neg);
}

void test_stretched_grid_large_beta(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    // Very large beta should still work (extreme clustering)
    double beta = 10.0;
    grid_initialize_stretched(g, beta);

    // Should still span full domain
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, g->x[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, g->x[g->nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.5, g->x[g->nx / 2]);

    // Should be monotonically increasing
    for (size_t i = 1; i < g->nx; i++) {
        TEST_ASSERT_TRUE(g->x[i] > g->x[i - 1]);
    }

    // Boundary cells should be very small compared to center
    double ratio = g->dx[g->nx / 2] / g->dx[0];
    TEST_ASSERT_TRUE(ratio > 5.0);  // Large beta should give significant ratio

    grid_destroy(g);
}

void test_stretched_grid_dx_consistency(void) {
    grid* g = grid_create(21, 21, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);

    double beta = 2.0;
    grid_initialize_stretched(g, beta);

    // Verify dx[i] = x[i+1] - x[i] exactly
    for (size_t i = 0; i < g->nx - 1; i++) {
        double expected_dx = g->x[i + 1] - g->x[i];
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, expected_dx, g->dx[i]);
    }

    // Verify dy[j] = y[j+1] - y[j] exactly
    for (size_t j = 0; j < g->ny - 1; j++) {
        double expected_dy = g->y[j + 1] - g->y[j];
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, expected_dy, g->dy[j]);
    }

    grid_destroy(g);
}

/* ============================================================================
 * Grid Creation Error Handling Tests
 * ============================================================================ */

void test_grid_create_zero_dimensions_fails(void) {
    grid* g = grid_create(0, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(g);

    g = grid_create(10, 0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NULL(g);
}

void test_grid_create_invalid_bounds_fails(void) {
    // xmax <= xmin
    grid* g = grid_create(10, 10, 1.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(g);

    // ymax <= ymin
    g = grid_create(10, 10, 0.0, 1.0, 1.0, 0.0);
    TEST_ASSERT_NULL(g);

    // Equal bounds
    g = grid_create(10, 10, 0.0, 0.0, 0.0, 1.0);
    TEST_ASSERT_NULL(g);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    // Uniform grid tests
    RUN_TEST(test_uniform_grid_spans_full_domain);
    RUN_TEST(test_uniform_grid_equal_spacing);
    RUN_TEST(test_uniform_grid_non_unit_domain);

    // Stretched grid tests
    RUN_TEST(test_stretched_grid_spans_full_domain);
    RUN_TEST(test_stretched_grid_clusters_near_boundaries);
    RUN_TEST(test_stretched_grid_higher_beta_more_clustering);
    RUN_TEST(test_stretched_grid_beta_zero_equals_uniform);
    RUN_TEST(test_stretched_grid_non_unit_domain);
    RUN_TEST(test_stretched_grid_monotonically_increasing);
    RUN_TEST(test_stretched_grid_y_direction_clustering);
    RUN_TEST(test_stretched_grid_minimum_size);
    RUN_TEST(test_stretched_grid_negative_beta);
    RUN_TEST(test_stretched_grid_large_beta);
    RUN_TEST(test_stretched_grid_dx_consistency);

    // Error handling tests
    RUN_TEST(test_grid_create_zero_dimensions_fails);
    RUN_TEST(test_grid_create_invalid_bounds_fails);

    return UNITY_END();
}
