/**
 * @file test_taylor_green_3d.c
 * @brief 3D Taylor-Green vortex validation tests
 *
 * Tests the numerical solver against the 3D Taylor-Green vortex analytical solution
 * on a triply-periodic domain [0, 2pi]^3.
 *
 * Tests verify:
 *   - Velocity decay rate matches exp(-3vt)
 *   - Kinetic energy decay matches exp(-6vt)
 *   - Divergence-free constraint is satisfied
 *   - w-velocity remains near zero
 */

#include "taylor_green_3d_reference.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * VELOCITY DECAY TESTS
 * ============================================================================ */

void test_3d_velocity_decay_rate(void) {
    printf("\n    Testing 3D velocity decay rate (exp(-3vt))...\n");

    tg3_result_t result = tg3_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG3_DEFAULT_N, TG3_DEFAULT_NU, TG3_DEFAULT_DT, TG3_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    tg3_print_result(&result, "Projection CPU");

    double decay_error = fabs(result.measured_velocity_decay - result.expected_velocity_decay)
                         / result.expected_velocity_decay;
    printf("      Velocity decay relative error: %.4f (tolerance: %.4f)\n",
           decay_error, TG3_VELOCITY_DECAY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(decay_error < TG3_VELOCITY_DECAY_TOL,
        "3D velocity decay rate does not match analytical solution");
}

/* ============================================================================
 * KINETIC ENERGY DECAY TESTS
 * ============================================================================ */

void test_3d_kinetic_energy_decay(void) {
    printf("\n    Testing 3D kinetic energy decay rate (exp(-6vt))...\n");

    tg3_result_t result = tg3_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG3_DEFAULT_N, TG3_DEFAULT_NU, TG3_DEFAULT_DT, TG3_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    double ke_decay_error = fabs(result.measured_ke_decay - result.expected_ke_decay)
                            / result.expected_ke_decay;
    printf("      KE decay relative error: %.4f (tolerance: %.4f)\n",
           ke_decay_error, TG3_KE_DECAY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(ke_decay_error < TG3_KE_DECAY_TOL,
        "3D kinetic energy decay rate does not match analytical solution");
}

/* ============================================================================
 * DIVERGENCE TESTS
 * ============================================================================ */

void test_3d_divergence_free(void) {
    printf("\n    Testing 3D divergence-free constraint...\n");

    tg3_result_t result = tg3_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG3_DEFAULT_N, TG3_DEFAULT_NU, TG3_DEFAULT_DT, TG3_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Max divergence: %.6e (tolerance: %.6e)\n",
           result.max_divergence, TG3_DIVERGENCE_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.max_divergence < TG3_DIVERGENCE_TOL,
        "3D divergence exceeds tolerance");
}

/* ============================================================================
 * L2 ERROR TEST
 * ============================================================================ */

void test_3d_l2_error(void) {
    printf("\n    Testing 3D L2 error against analytical solution...\n");

    tg3_result_t result = tg3_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG3_DEFAULT_N, TG3_DEFAULT_NU, TG3_DEFAULT_DT, TG3_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      L2 error u: %.6f (tolerance: %.4f)\n",
           result.l2_error_u, TG3_L2_ERROR_TOL);
    printf("      L2 error v: %.6f (tolerance: %.4f)\n",
           result.l2_error_v, TG3_L2_ERROR_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.l2_error_u < TG3_L2_ERROR_TOL,
        "3D u-velocity L2 error exceeds tolerance");
    TEST_ASSERT_TRUE_MESSAGE(result.l2_error_v < TG3_L2_ERROR_TOL,
        "3D v-velocity L2 error exceeds tolerance");
}

/* ============================================================================
 * W-VELOCITY TEST
 * ============================================================================ */

void test_3d_w_remains_zero(void) {
    printf("\n    Testing w-velocity remains near zero...\n");

    tg3_result_t result = tg3_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG3_DEFAULT_N, TG3_DEFAULT_NU, TG3_DEFAULT_DT, TG3_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    /* The analytical w=0, so any nonzero w is numerical error.
     * We allow a generous tolerance since the projection method
     * on a coarse 3D grid can produce small spurious w. */
    printf("      max|w| = %.2e\n", result.max_w);
    TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(
        0.1, 0.0, result.max_w,
        "w-velocity should remain near zero (analytical w=0)");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_3d_velocity_decay_rate);
    RUN_TEST(test_3d_kinetic_energy_decay);
    RUN_TEST(test_3d_divergence_free);
    RUN_TEST(test_3d_l2_error);
    RUN_TEST(test_3d_w_remains_zero);

    return UNITY_END();
}
