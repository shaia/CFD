/**
 * @file test_ghia_projection_omp.c
 * @brief Ghia validation for Projection OpenMP solver
 *
 * Tests the OpenMP-parallelized implementation of the projection solver
 * against Ghia et al. (1982) reference data.
 *
 * KNOWN ISSUE: The OMP projection solver currently produces different results
 * than the CPU/AVX2 implementations. This indicates a parallelization bug
 * that needs investigation.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

void test_projection_omp_ghia_re100(void) {
    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_OMP,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    /* Skip test if OMP solver is not available (not compiled with OpenMP) */
    if (!result.success && strstr(result.error_msg, "not available") != NULL) {
        TEST_IGNORE_MESSAGE("OpenMP solver not available (OpenMP not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Projection OMP");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

/* Note: Cross-architecture consistency (OMP vs CPU) is now tested in
 * test_solver_architecture.c which runs in the dedicated cross-arch-validation
 * CI job with proper tolerance settings. */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PROJECTION OMP - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_projection_omp_ghia_re100);

    return UNITY_END();
}
