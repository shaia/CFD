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
        EULER_FULL_STEPS, FINE_DT
    );

    /* Skip test if OMP solver is not available (not compiled with OpenMP) */
    if (result.solver_unavailable) {
        TEST_IGNORE_MESSAGE("OpenMP solver not available (OpenMP not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Explicit Euler OMP");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

/* Note: Cross-architecture consistency (OMP vs CPU) is now tested in
 * test_solver_architecture.c which runs in the dedicated cross-arch-validation
 * CI job with proper tolerance settings. */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("EXPLICIT EULER OMP - GHIA VALIDATION\n");
    printf("========================================\n");

    RUN_TEST(test_euler_omp_ghia_re100);

    return UNITY_END();
}
