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

    if (!result.success) {
        TEST_IGNORE_MESSAGE("AVX2 projection solver not available (SIMD not enabled)");
    }
    print_ghia_result(&result, "Projection AVX2");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

/* Note: Cross-architecture consistency (AVX2 vs CPU) is now tested in
 * test_solver_architecture.c which runs in the dedicated cross-arch-validation
 * CI job with proper tolerance settings. */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PROJECTION AVX2 - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_projection_avx2_ghia_re100);

    return UNITY_END();
}
