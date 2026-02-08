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
    double prev_rms = 1.0;

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

        printf("      %zux%zu: RMS_u=%.4f", n, n, result.rms_u_error);
        if (result.rms_u_error > GHIA_TOLERANCE_MEDIUM) {
            printf(" [ABOVE TARGET]");
        }
        printf("\n");

        /* Error should decrease or stay similar with refinement */
        TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error <= prev_rms + 0.02,
            "Error increased significantly with grid refinement");
        prev_rms = result.rms_u_error;
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
