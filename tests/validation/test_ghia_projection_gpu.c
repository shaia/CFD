/**
 * @file test_ghia_projection_gpu.c
 * @brief Ghia validation for Projection CUDA GPU solver
 *
 * Tests the CUDA GPU implementation of the projection solver
 * against Ghia et al. (1982) reference data.
 *
 * This test is OPTIONAL - it will skip gracefully if CUDA is not available.
 */

#include "cavity_validation_utils.h"

void setUp(void) {}
void tearDown(void) {}

static int gpu_available = -1;  /* -1 = not checked, 0 = no, 1 = yes */

static int check_gpu_available(void) {
    if (gpu_available < 0) {
        ns_solver_registry_t* registry = cfd_registry_create();
        cfd_registry_register_defaults(registry);

        ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
        gpu_available = (solver != NULL) ? 1 : 0;

        if (solver) solver_destroy(solver);
        cfd_registry_destroy(registry);
    }
    return gpu_available;
}

void test_gpu_availability(void) {
    printf("\n    Checking GPU solver availability...\n");

    if (check_gpu_available()) {
        printf("      GPU solver: AVAILABLE\n");
        TEST_PASS();
    } else {
        printf("      GPU solver: NOT AVAILABLE\n");
        printf("      (CUDA not installed or no compatible GPU detected)\n");
        TEST_PASS_MESSAGE("GPU optional, skipping");
    }
}

void test_projection_gpu_ghia_re100(void) {
    if (!check_gpu_available()) {
        TEST_IGNORE_MESSAGE("GPU not available, skipping Ghia validation");
        return;
    }

    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        33, 33,
        100.0, 1.0,
        FULL_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, result.error_msg);
    print_ghia_result(&result, "Projection GPU");

    TEST_ASSERT_TRUE_MESSAGE(result.rms_u_error < GHIA_TOLERANCE_CURRENT,
        "Must meet current baseline tolerance");
}

void test_projection_gpu_matches_cpu(void) {
    if (!check_gpu_available()) {
        TEST_IGNORE_MESSAGE("GPU not available, skipping consistency check");
        return;
    }

    printf("\n    Verifying GPU matches CPU scalar...\n");

    ghia_result_t cpu = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    ghia_result_t gpu = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        17, 17, 100.0, 1.0, 100, 0.005
    );

    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU must succeed");
    TEST_ASSERT_TRUE_MESSAGE(gpu.success, "GPU must succeed");

    double diff = fabs(cpu.u_at_center - gpu.u_at_center);
    printf("      CPU u_center: %.6f\n", cpu.u_at_center);
    printf("      GPU u_center: %.6f\n", gpu.u_at_center);
    printf("      Difference:   %.6f\n", diff);

    /* GPU may have slightly different floating point due to parallelism,
     * so we use a slightly looser tolerance of 0.01 (1%) */
    TEST_ASSERT_TRUE_MESSAGE(diff < 0.01,
        "GPU must produce similar results to CPU (within 1%)");
}

void test_projection_gpu_stability(void) {
    if (!check_gpu_available()) {
        TEST_IGNORE_MESSAGE("GPU not available, skipping stability test");
        return;
    }

    printf("\n    Testing GPU solver stability...\n");

    ghia_result_t result = run_ghia_validation(
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        25, 25,
        100.0, 1.0,
        MEDIUM_STEPS, FINE_DT
    );

    TEST_ASSERT_TRUE_MESSAGE(result.success, "GPU simulation must complete");
    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity > 0.1, "Flow must develop");
    TEST_ASSERT_TRUE_MESSAGE(result.max_velocity < 10.0, "No velocity blow-up");

    printf("      max_velocity: %.4f\n", result.max_velocity);
    printf("      u_center:     %.4f\n", result.u_at_center);
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PROJECTION GPU - GHIA VALIDATION\n");
    printf("========================================\n");
    printf("\nNOTE: GPU tests are OPTIONAL\n");
    printf("      Tests will skip if CUDA is not available\n");

    RUN_TEST(test_gpu_availability);
    RUN_TEST(test_projection_gpu_ghia_re100);
    RUN_TEST(test_projection_gpu_matches_cpu);
    RUN_TEST(test_projection_gpu_stability);

    return UNITY_END();
}
