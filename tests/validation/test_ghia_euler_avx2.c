/**
 * @file test_ghia_euler_avx2.c
 * @brief Ghia validation for Explicit Euler AVX2/SIMD solver
 *
 * Tests the AVX2-optimized implementation of the explicit Euler solver
 * against Ghia et al. (1982) reference data.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_euler_avx2_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Explicit Euler AVX2");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

/* Note: Cross-architecture consistency (AVX2 vs CPU) is now tested in
 * test_solver_architecture.c which runs in the dedicated cross-arch-validation
 * CI job with proper tolerance settings. */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("EXPLICIT EULER AVX2 - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_euler_avx2_ghia_re100);

    return UNITY_END();
}
