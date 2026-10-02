/**
 * @file test_steady_state.c
 * @brief Unit tests for the shared steady-state helper (steady_state.h)
 *
 * The validation runs only take the finite path, so a regression in the
 * non-finite guard would leave them green. These cases pin it directly.
 */

#include "steady_state.h"
#include "unity.h"

#include <math.h>

void setUp(void) {}
void tearDown(void) {}

void test_finite_returns_largest_change(void) {
    const double prev[] = {1.0, 2.0, 3.0, 4.0};
    const double cur[] = {1.5, 1.0, 3.25, 4.0};
    TEST_ASSERT_EQUAL_DOUBLE(1.0, steady_max_change(cur, prev, 4));
}

void test_no_change_is_zero(void) {
    const double v[] = {0.5, -2.0, 7.0};
    TEST_ASSERT_EQUAL_DOUBLE(0.0, steady_max_change(v, v, 3));
}

/* Each non-finite value is placed after a larger finite change, the case a
 * plain "keep the largest" comparison would skip. */
void test_non_finite_after_larger_change_returns_infinity(void) {
    const double bad[] = {NAN, INFINITY, -INFINITY};
    for (size_t b = 0; b < sizeof(bad) / sizeof(bad[0]); b++) {
        const double prev[] = {0.0, 0.0, 0.0};
        const double cur[] = {5.0, bad[b], 0.1};
        double change = steady_max_change(cur, prev, 3);
        TEST_ASSERT_TRUE(isinf(change) && change > 0.0);
    }
}

/* A non-finite previous value counts too: inf - inf is NaN. */
void test_non_finite_previous_value_returns_infinity(void) {
    const double prev[] = {1.0, INFINITY};
    const double cur[] = {1.0, INFINITY};
    double change = steady_max_change(cur, prev, 2);
    TEST_ASSERT_TRUE(isinf(change) && change > 0.0);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_finite_returns_largest_change);
    RUN_TEST(test_no_change_is_zero);
    RUN_TEST(test_non_finite_after_larger_change_returns_infinity);
    RUN_TEST(test_non_finite_previous_value_returns_infinity);
    return UNITY_END();
}
