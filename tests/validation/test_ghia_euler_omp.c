/**
 * @file test_ghia_euler_omp.c
 * @brief Ghia validation for Explicit Euler OpenMP solver
 *
 * Tests the OpenMP-parallelized implementation of the explicit Euler solver
 * against Ghia et al. (1982) reference data.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_euler_omp_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Explicit Euler OMP");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

void test_euler_omp_matches_cpu(void) {
    printf("\n    Verifying OpenMP matches CPU scalar...\n");

    ghia_result_t cpu = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    ghia_result_t omp = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU must succeed");
    TEST_ASSERT_TRUE_MESSAGE(omp.success, "OMP must succeed");

    double diff = fabs(cpu.u_at_center - omp.u_at_center);
    printf("      CPU u_center: %.6f\n", cpu.u_at_center);
    printf("      OMP u_center: %.6f\n", omp.u_at_center);
    printf("      Difference:   %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < 0.001,
        "OpenMP must produce identical results to CPU");
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("EXPLICIT EULER OMP - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_euler_omp_ghia_re100);
    RUN_TEST(test_euler_omp_matches_cpu);

    return UNITY_END();
}
