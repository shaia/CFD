/**
 * @file test_ghia_projection_avx2.c
 * @brief Ghia validation for Projection AVX2/SIMD solver
 *
 * Tests the AVX2-optimized implementation of the projection solver
 * against Ghia et al. (1982) reference data.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_projection_avx2_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Projection AVX2");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

void test_projection_avx2_matches_cpu(void) {
    printf("\n    Verifying AVX2 matches CPU scalar...\n");

    ghia_result_t cpu = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    ghia_result_t avx2 = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU must succeed");
    TEST_ASSERT_TRUE_MESSAGE(avx2.success, "AVX2 must succeed");

    double diff = fabs(cpu.u_at_center - avx2.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      AVX2 u_center: %.6f\n", avx2.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < 0.001,
        "AVX2 must produce identical results to CPU");
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PROJECTION AVX2 - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_projection_avx2_ghia_re100);
    RUN_TEST(test_projection_avx2_matches_cpu);

    return UNITY_END();
}
