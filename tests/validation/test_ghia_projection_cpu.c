/**
 * @file test_ghia_projection_cpu.c
 * @brief Ghia validation for Projection CPU (scalar) solver
 *
 * Tests the CPU scalar implementation of the projection solver
 * against Ghia et al. (1982) reference data.
 *
 * The projection method is the primary solver for incompressible flows.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_projection_cpu_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Projection CPU");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_MEDIUM,
        "Must meet scientific Ghia tolerance");
}

void test_projection_cpu_ghia_target(void) {
    printf("\n    Testing against scientific Ghia target (RMS < 0.10)...\n");

    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

    /* Print detailed comparison */
    printf("      u_center:  %.4f (Ghia: -0.20581, diff: %.4f)\n",
           result.u_at_center, fabs(result.u_at_center - (-0.20581)));
    printf("      u_min:     %.4f (Ghia: -0.21090, diff: %.4f)\n",
           result.u_min, fabs(result.u_min - (-0.21090)));
    printf("      v_center:  %.4f (Ghia:  0.05454, diff: %.4f)\n",
           result.v_at_center, fabs(result.v_at_center - 0.05454));

    if (result.rms_u_error <= GHIA_TOLERANCE_MEDIUM) {
        printf("\n      [SUCCESS] Solver meets scientific target!\n");
    }

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_MEDIUM,
        "Must meet scientific Ghia target (RMS < 0.10)");
}

void test_projection_cpu_grid_convergence(void) {
    printf("\n    Grid convergence study...\n");

    size_t sizes[] = {17, 25, 33};
    double errors[3];

    for (int i = 0; i < 3; i++) {
        size_t n = sizes[i];
        /* Scale dt inversely with grid size for CFL stability.
         * Scale iterations with grid size - finer grids need more
         * iterations for the Poisson solver to converge. */
        double dt = FINE_DT * (33.0 / n);
        int steps = (int)(MEDIUM_STEPS * n / 17.0);

        ghia_result_t result = run_ghia_validation(
            NS_SOLVER_TYPE_PROJECTION,
            n, n,
            100.0, 1.0,
            steps, dt
        );

        TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);

        errors[i] = result.rms_u_error;

        printf("      %zux%zu: RMS_u=%.4f", n, n, errors[i]);
        if (errors[i] > GHIA_TOLERANCE_MEDIUM) {
            printf(" [ABOVE TARGET]");
        }
        printf("\n");

        /* Error must strictly decrease with grid refinement */
        if (i > 0) {
            TEST_ASSERT_TRUE_MESSAGE(errors[i] < errors[i - 1],
                "Error must decrease with grid refinement (strict monotonicity)");
        }
    }

    /* Report convergence rates (informational — first-order BCs limit to ~O(h^1.5)) */
    double h[] = {1.0 / 17, 1.0 / 25, 1.0 / 33};
    for (int i = 0; i < 2; i++) {
        double rate = log(errors[i] / errors[i + 1]) / log(h[i] / h[i + 1]);
        printf("      Rate (%zu->%zu): %.2f\n", sizes[i], sizes[i + 1], rate);
    }
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PROJECTION CPU - GHIA VALIDATION\n");
    printf("========================================\n");
    printf("\nScientific target: RMS < %.2f\n", GHIA_TOLERANCE_MEDIUM);
    printf("Current baseline:  RMS < %.2f\n", GHIA_TOLERANCE_CURRENT);

    RUN_TEST(test_projection_cpu_ghia_re100);
    RUN_TEST(test_projection_cpu_ghia_target);
    RUN_TEST(test_projection_cpu_grid_convergence);

    return UNITY_END();
}
