/**
 * @file test_ghia_euler_cpu.c
 * @brief Ghia validation for Explicit Euler CPU (scalar) solver
 *
 * Tests the CPU scalar implementation of the explicit Euler solver
 * against Ghia et al. (1982) reference data.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_euler_cpu_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        33, 33,
        100.0, 1.0,
        EULER_FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Explicit Euler CPU");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

void test_euler_cpu_stability(void) {
    printf("\n    Testing stability with 17x17 grid...\n");

    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        17, 17,
        100.0, 1.0,
        EULER_MEDIUM_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, "Simulation must complete without blowing up");
    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity > 0.1, "Flow must develop");
    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity < 10.0, "No velocity blow-up");
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("EXPLICIT EULER CPU - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_euler_cpu_ghia_re100);
    RUN_TEST(test_euler_cpu_stability);

    return UNITY_END();
}
