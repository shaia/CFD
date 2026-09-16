/**
 * @file test_upwind_stencils.c
 * @brief Unit tests for the first-order upwind stencils in cfd/math/stencils.h
 *
 * The solvers' upwind convection terms call these stencils, so these tests pin
 * down the behavior every backend must reproduce:
 *   - Direction selection: vel >= 0 uses the backward difference, vel < 0 the
 *     forward difference (vel == 0 is backward; NaN is forward)
 *   - Exactness for linear functions on either side
 *   - O(h) convergence on a smooth function, against O(h^2) for the central
 *     stencil on the same grids
 */

#include "unity.h"
#include "cfd/math/stencils.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

#define CONVERGENCE_RATE_TOL 0.3  /* Allowed deviation from the expected order */
#define EXACT_TOL            1e-12

#define NUM_SIZES 3
static const size_t SIZES[NUM_SIZES] = {17, 33, 65};

/* Signature shared by the x/y/z upwind stencils */
typedef double (*upwind_stencil_fn)(double f_p, double f_c, double f_m, double h, double vel);

/* Central stencil adapted to the upwind signature, for the order contrast */
static double central_deriv(double f_p, double f_c, double f_m, double h, double vel) {
    (void)f_c;
    (void)vel;
    return stencil_first_deriv_x(f_p, f_m, h);
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static double compute_convergence_rate(double e_coarse, double e_fine,
                                       double h_coarse, double h_fine) {
    if (e_fine < 1e-15 || e_coarse < 1e-15) return 0.0;
    return log(e_coarse / e_fine) / log(h_coarse / h_fine);
}

/**
 * L2 error of a stencil's derivative of sin(s) over the interior of a uniform
 * grid on [0, 2*pi] with n points, for an advecting velocity vel.
 */
static double stencil_l2_error(upwind_stencil_fn stencil, size_t n, double vel, double* h_out) {
    double h = 2.0 * M_PI / (double)(n - 1);
    double sum_sq = 0.0;
    for (size_t i = 1; i < n - 1; i++) {
        double s = (double)i * h;
        double numerical = stencil(sin(s + h), sin(s), sin(s - h), h, vel);
        double err = numerical - cos(s);
        sum_sq += err * err;
    }
    *h_out = h;
    return sqrt(sum_sq / (double)(n - 2));
}

/**
 * Assert the stencil converges at the expected order on SIZES for velocity vel.
 * Returns the error on the finest grid.
 */
static double assert_convergence_order(upwind_stencil_fn stencil, double vel,
                                       double expected_order, const char* label) {
    double errors[NUM_SIZES];
    double spacings[NUM_SIZES];

    for (int s = 0; s < NUM_SIZES; s++) {
        errors[s] = stencil_l2_error(stencil, SIZES[s], vel, &spacings[s]);
        printf("      %s n=%zu vel=%+.0f: L2 error = %.6e\n", label, SIZES[s], vel, errors[s]);
    }

    for (int s = 1; s < NUM_SIZES; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      %s rate %zu->%zu: %.2f (expected ~%.1f)\n",
               label, SIZES[s - 1], SIZES[s], rate, expected_order);
        TEST_ASSERT_TRUE_MESSAGE(fabs(rate - expected_order) < CONVERGENCE_RATE_TOL,
                                 "Convergence rate deviates from the expected order");
    }

    return errors[NUM_SIZES - 1];
}

/* ============================================================================
 * UNITY SETUP
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * DIRECTION SELECTION
 * ============================================================================ */

void test_upwind_direction_selection(void) {
    /* f_m = 2, f_c = 3, f_p = 5: backward difference 1, forward difference 2 */
    const double f_p = 5.0, f_c = 3.0, f_m = 2.0;

    TEST_ASSERT_EQUAL_DOUBLE(1.0, stencil_upwind_diff(f_p, f_c, f_m, 1.0));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, stencil_upwind_diff(f_p, f_c, f_m, -1.0));

    /* Zero velocity (either sign) uses the backward difference */
    TEST_ASSERT_EQUAL_DOUBLE(1.0, stencil_upwind_diff(f_p, f_c, f_m, 0.0));
    TEST_ASSERT_EQUAL_DOUBLE(1.0, stencil_upwind_diff(f_p, f_c, f_m, -0.0));

    /* A NaN velocity fails the >= test and takes the forward difference */
    TEST_ASSERT_EQUAL_DOUBLE(2.0, stencil_upwind_diff(f_p, f_c, f_m, NAN));

    /* The axis stencils divide the undivided difference by the spacing */
    const double h = 0.5;
    TEST_ASSERT_EQUAL_DOUBLE(2.0, stencil_upwind_deriv_x(f_p, f_c, f_m, h, 1.0));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, stencil_upwind_deriv_x(f_p, f_c, f_m, h, -1.0));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, stencil_upwind_deriv_y(f_p, f_c, f_m, h, 1.0));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, stencil_upwind_deriv_y(f_p, f_c, f_m, h, -1.0));
    TEST_ASSERT_EQUAL_DOUBLE(2.0, stencil_upwind_deriv_z(f_p, f_c, f_m, h, 1.0));
    TEST_ASSERT_EQUAL_DOUBLE(4.0, stencil_upwind_deriv_z(f_p, f_c, f_m, h, -1.0));
}

/* ============================================================================
 * EXACTNESS FOR LINEAR FUNCTIONS
 * ============================================================================ */

void test_upwind_exact_for_linear_function(void) {
    /* f(s) = 3.5*s - 1.25 has slope 3.5; both one-sided differences are exact */
    const double slope = 3.5;
    const double h = 0.1;
    upwind_stencil_fn stencils[] = {stencil_upwind_deriv_x, stencil_upwind_deriv_y,
                                    stencil_upwind_deriv_z};
    const double velocities[] = {2.0, -2.0, 0.0};

    for (size_t a = 0; a < sizeof(stencils) / sizeof(stencils[0]); a++) {
        for (size_t v = 0; v < sizeof(velocities) / sizeof(velocities[0]); v++) {
            for (int i = 0; i < 10; i++) {
                double s = -0.3 + 0.17 * i;
                double deriv = stencils[a](slope * (s + h) - 1.25, slope * s - 1.25,
                                           slope * (s - h) - 1.25, h, velocities[v]);
                TEST_ASSERT_DOUBLE_WITHIN(EXACT_TOL, slope, deriv);
            }
        }
    }
}

/* ============================================================================
 * CONVERGENCE ORDER
 * ============================================================================ */

void test_upwind_deriv_x_first_order(void) {
    printf("\n    Testing upwind df/dx convergence...\n");
    assert_convergence_order(stencil_upwind_deriv_x, 1.0, 1.0, "upwind_x");
    assert_convergence_order(stencil_upwind_deriv_x, -1.0, 1.0, "upwind_x");
}

void test_upwind_deriv_y_first_order(void) {
    printf("\n    Testing upwind df/dy convergence...\n");
    assert_convergence_order(stencil_upwind_deriv_y, 1.0, 1.0, "upwind_y");
    assert_convergence_order(stencil_upwind_deriv_y, -1.0, 1.0, "upwind_y");
}

void test_upwind_deriv_z_first_order(void) {
    printf("\n    Testing upwind df/dz convergence...\n");
    assert_convergence_order(stencil_upwind_deriv_z, 1.0, 1.0, "upwind_z");
    assert_convergence_order(stencil_upwind_deriv_z, -1.0, 1.0, "upwind_z");
}

void test_central_more_accurate_than_upwind(void) {
    printf("\n    Contrasting central O(h^2) with upwind O(h)...\n");
    double central_fine = assert_convergence_order(central_deriv, 1.0, 2.0, "central");
    double upwind_fine = assert_convergence_order(stencil_upwind_deriv_x, 1.0, 1.0, "upwind");
    TEST_ASSERT_TRUE_MESSAGE(central_fine < upwind_fine,
                             "Central stencil should be more accurate on the finest grid");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("UPWIND STENCIL TESTS\n");
    printf("========================================\n");

    RUN_TEST(test_upwind_direction_selection);
    RUN_TEST(test_upwind_exact_for_linear_function);
    RUN_TEST(test_upwind_deriv_x_first_order);
    RUN_TEST(test_upwind_deriv_y_first_order);
    RUN_TEST(test_upwind_deriv_z_first_order);
    RUN_TEST(test_central_more_accurate_than_upwind);

    return UNITY_END();
}
