/**
 * @file test_taylor_green_vortex.c
 * @brief Taylor-Green vortex validation tests
 *
 * Tests the numerical solver against the Taylor-Green vortex analytical solution.
 * This is an exact solution to the incompressible Navier-Stokes equations,
 * making it ideal for validating solver accuracy.
 *
 * Tests verify:
 *   - Velocity decay rate matches exp(-2νt)
 *   - Kinetic energy decay matches exp(-4νt)
 *   - L2 error remains bounded
 *   - Divergence-free constraint is satisfied
 */

#include "taylor_green_reference.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * VELOCITY DECAY TESTS
 * ============================================================================ */

/**
 * Test that velocity decays at the correct rate: exp(-2νt)
 */
void test_velocity_decay_rate(void) {
    printf("\n    Testing velocity decay rate (exp(-2νt))...\n");

    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    tg_print_result(&result, "Projection CPU");

    /* Velocity should decay as exp(-2νt) */
    double decay_error = fabs(result.measured_velocity_decay - result.expected_velocity_decay)
                         / result.expected_velocity_decay;
    printf("      Velocity decay relative error: %.4f (tolerance: %.4f)\n",
           decay_error, TG_VELOCITY_DECAY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(decay_error < TG_VELOCITY_DECAY_TOL,
        "Velocity decay rate does not match analytical solution");
}

/**
 * Test velocity decay with different viscosities
 */
void test_velocity_decay_viscosity_dependence(void) {
    printf("\n    Testing velocity decay with different viscosities...\n");

    double viscosities[] = {0.005, 0.01, 0.02};
    int num_viscosities = sizeof(viscosities) / sizeof(viscosities[0]);

    for (int i = 0; i < num_viscosities; i++) {
        double nu = viscosities[i];
        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_PROJECTION,
            TG_DEFAULT_NX, TG_DEFAULT_NY,
            nu, TG_DEFAULT_DT, TG_DEFAULT_STEPS
        );

        TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

        double decay_error = fabs(result.measured_velocity_decay - result.expected_velocity_decay)
                             / result.expected_velocity_decay;
        printf("      nu=%.3f: measured=%.4f, expected=%.4f, error=%.4f\n",
               nu, result.measured_velocity_decay, result.expected_velocity_decay, decay_error);

        TEST_ASSERT_TRUE_MESSAGE(decay_error < TG_VELOCITY_DECAY_TOL,
            "Velocity decay rate incorrect for viscosity");
    }
}

/* ============================================================================
 * KINETIC ENERGY DECAY TESTS
 * ============================================================================ */

/**
 * Test that kinetic energy decays at the correct rate: exp(-4νt)
 */
void test_kinetic_energy_decay_rate(void) {
    printf("\n    Testing kinetic energy decay rate (exp(-4νt))...\n");

    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    /* Kinetic energy should decay as exp(-4νt) */
    double ke_decay_error = fabs(result.measured_ke_decay - result.expected_ke_decay)
                            / result.expected_ke_decay;
    printf("      Initial KE: %.6f, Final KE: %.6f\n", result.initial_ke, result.final_ke);
    printf("      Measured decay: %.6f, Expected: %.6f\n",
           result.measured_ke_decay, result.expected_ke_decay);
    printf("      KE decay relative error: %.4f (tolerance: %.4f)\n",
           ke_decay_error, TG_KE_DECAY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(ke_decay_error < TG_KE_DECAY_TOL,
        "Kinetic energy decay rate does not match analytical solution");
}

/* ============================================================================
 * L2 ERROR TESTS
 * ============================================================================ */

/**
 * Test that L2 error remains bounded
 */
void test_l2_error_bounded(void) {
    printf("\n    Testing L2 error remains bounded...\n");

    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      L2 error u: %.6f (tolerance: %.4f)\n", result.l2_error_u, TG_L2_ERROR_TOL);
    printf("      L2 error v: %.6f (tolerance: %.4f)\n", result.l2_error_v, TG_L2_ERROR_TOL);

    TEST_ASSERT_TRUE_MESSAGE(result.l2_error_u < TG_L2_ERROR_TOL,
        "L2 error in u-velocity exceeds tolerance");
    TEST_ASSERT_TRUE_MESSAGE(result.l2_error_v < TG_L2_ERROR_TOL,
        "L2 error in v-velocity exceeds tolerance");
}

/**
 * Test grid convergence - error should decrease with finer grids
 */
void test_grid_convergence(void) {
    printf("\n    Testing grid convergence...\n");

    size_t grid_sizes[] = {16, 32, 64};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    double prev_error = 1.0;

    for (int i = 0; i < num_sizes; i++) {
        size_t n = grid_sizes[i];
        /* Adjust dt for CFL stability with finer grids */
        double dt = TG_DEFAULT_DT * (32.0 / n);
        int steps = (int)(TG_DEFAULT_STEPS * n / 32.0);

        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_PROJECTION,
            n, n,
            TG_DEFAULT_NU, dt, steps
        );

        TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

        double total_error = sqrt(result.l2_error_u * result.l2_error_u +
                                 result.l2_error_v * result.l2_error_v);
        printf("      %zux%zu: L2 error = %.6f", n, n, total_error);

        if (i > 0) {
            double convergence_rate = log(prev_error / total_error) / log(2.0);
            printf(" (rate: %.2f)", convergence_rate);
            /* Error should decrease with refinement */
            TEST_ASSERT_TRUE_MESSAGE(total_error < prev_error * 1.1,
                "Error increased with grid refinement");
        }
        printf("\n");

        prev_error = total_error;
    }
}

/* ============================================================================
 * DIVERGENCE-FREE TESTS
 * ============================================================================ */

/**
 * Test that divergence remains small (incompressibility constraint)
 */
void test_divergence_free(void) {
    printf("\n    Testing divergence-free constraint...\n");

    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Max divergence: %.2e (tolerance: %.2e)\n",
           result.max_divergence, TG_DIVERGENCE_TOL);

    /* Projection method should enforce near-zero divergence */
    TEST_ASSERT_TRUE_MESSAGE(result.max_divergence < TG_DIVERGENCE_TOL,
        "Divergence exceeds tolerance - incompressibility not satisfied");
}

/* ============================================================================
 * BACKEND COMPARISON TESTS
 * ============================================================================ */

/**
 * Test that different solver backends produce consistent results
 */
void test_backend_consistency(void) {
    printf("\n    Testing backend consistency...\n");

    const char* solvers[] = {
        NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_PROJECTION_OMP
    };
    const char* solver_names[] = {"Projection CPU", "Projection OMP"};
    int num_solvers = sizeof(solvers) / sizeof(solvers[0]);

    double reference_velocity_decay = 0.0;
    double reference_ke_decay = 0.0;

    for (int i = 0; i < num_solvers; i++) {
        tg_result_t result = tg_run_simulation(
            solvers[i],
            TG_DEFAULT_NX, TG_DEFAULT_NY,
            TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS / 2  /* Fewer steps for faster testing */
        );

        if (!result.success) {
            printf("      %s: SKIPPED (solver not available)\n", solver_names[i]);
            continue;
        }

        printf("      %s: velocity_decay=%.6f, ke_decay=%.6f\n",
               solver_names[i], result.measured_velocity_decay, result.measured_ke_decay);

        if (i == 0) {
            reference_velocity_decay = result.measured_velocity_decay;
            reference_ke_decay = result.measured_ke_decay;
        } else {
            /* Results should be within 1% of reference */
            double velocity_diff = fabs(result.measured_velocity_decay - reference_velocity_decay)
                                   / reference_velocity_decay;
            double ke_diff = fabs(result.measured_ke_decay - reference_ke_decay)
                            / reference_ke_decay;

            TEST_ASSERT_TRUE_MESSAGE(velocity_diff < 0.01,
                "Backend velocity decay differs from reference by more than 1%");
            TEST_ASSERT_TRUE_MESSAGE(ke_diff < 0.01,
                "Backend KE decay differs from reference by more than 1%");
        }
    }
}

/* ============================================================================
 * STABILITY TESTS
 * ============================================================================ */

/**
 * Test long-time stability
 */
void test_long_time_stability(void) {
    printf("\n    Testing long-time stability...\n");

    /* Run for longer time with stable parameters */
    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        TG_DEFAULT_NU, TG_DEFAULT_DT, TG_DEFAULT_STEPS * 2
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      Final time: %.4f\n", result.final_time);
    printf("      Final KE: %.6f (initial: %.6f)\n", result.final_ke, result.initial_ke);
    printf("      L2 error u: %.6f\n", result.l2_error_u);

    /* Simulation should remain stable */
    TEST_ASSERT_TRUE_MESSAGE(result.final_ke > 0.0,
        "Kinetic energy became negative (numerical instability)");
    TEST_ASSERT_TRUE_MESSAGE(result.final_ke < result.initial_ke * 2.0,
        "Kinetic energy grew (energy not conserved/dissipated correctly)");
    TEST_ASSERT_TRUE_MESSAGE(isfinite(result.l2_error_u),
        "L2 error is not finite (numerical blowup)");
}

/**
 * Test with low viscosity (more challenging)
 */
void test_low_viscosity_stability(void) {
    printf("\n    Testing low viscosity stability...\n");

    /* Lower viscosity requires smaller time step for stability */
    double nu = 0.005;
    double dt = 0.0005;  /* Smaller dt for stability */
    int steps = 500;

    tg_result_t result = tg_run_simulation(
        NS_SOLVER_TYPE_PROJECTION,
        TG_DEFAULT_NX, TG_DEFAULT_NY,
        nu, dt, steps
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    printf("      nu=%.3f, dt=%.4f, steps=%d\n", nu, dt, steps);
    printf("      Velocity decay: measured=%.6f, expected=%.6f\n",
           result.measured_velocity_decay, result.expected_velocity_decay);

    double decay_error = fabs(result.measured_velocity_decay - result.expected_velocity_decay)
                         / result.expected_velocity_decay;
    /* Use slightly relaxed tolerance for low viscosity */
    TEST_ASSERT_TRUE_MESSAGE(decay_error < TG_VELOCITY_DECAY_TOL * 1.5,
        "Low viscosity velocity decay incorrect");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("TAYLOR-GREEN VORTEX VALIDATION\n");
    printf("========================================\n");
    printf("\nAnalytical solution: u = cos(x)sin(y)exp(-2νt)\n");
    printf("Domain: [0, 2π] × [0, 2π]\n");
    printf("Default viscosity: %.3f\n", TG_DEFAULT_NU);

    /* Velocity decay tests */
    RUN_TEST(test_velocity_decay_rate);
    RUN_TEST(test_velocity_decay_viscosity_dependence);

    /* Kinetic energy decay tests */
    RUN_TEST(test_kinetic_energy_decay_rate);

    /* L2 error tests */
    RUN_TEST(test_l2_error_bounded);
    RUN_TEST(test_grid_convergence);

    /* Divergence-free tests */
    RUN_TEST(test_divergence_free);

    /* Backend consistency tests */
    RUN_TEST(test_backend_consistency);

    /* Stability tests */
    RUN_TEST(test_long_time_stability);
    RUN_TEST(test_low_viscosity_stability);

    return UNITY_END();
}
