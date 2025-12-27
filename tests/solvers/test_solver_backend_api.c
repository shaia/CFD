/**
 * Test Suite for Solver Backend API
 *
 * Tests the backend availability and solver creation API:
 * - cfd_backend_is_available()
 * - cfd_backend_get_name()
 * - cfd_registry_list_by_backend()
 * - cfd_solver_create_checked()
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ns_solver_registry_t* registry = NULL;

void setUp(void) {
    cfd_init();
    cfd_clear_error();
    registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
}

void tearDown(void) {
    if (registry) {
        cfd_registry_destroy(registry);
        registry = NULL;
    }
    cfd_clear_error();
    cfd_finalize();
}

//=============================================================================
// cfd_backend_is_available() Tests
//=============================================================================

void test_backend_scalar_always_available(void) {
    // Scalar backend should always be available
    TEST_ASSERT_TRUE(cfd_backend_is_available(NS_SOLVER_BACKEND_SCALAR));
}

void test_backend_simd_matches_cpu_features(void) {
    // SIMD availability should match cpu_features detection
    int expected = cfd_has_simd() ? 1 : 0;
    int actual = cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD);
    TEST_ASSERT_EQUAL_INT(expected, actual);
}

void test_backend_cuda_matches_gpu_available(void) {
    // CUDA availability should match gpu_is_available()
    int expected = gpu_is_available() ? 1 : 0;
    int actual = cfd_backend_is_available(NS_SOLVER_BACKEND_CUDA);
    TEST_ASSERT_EQUAL_INT(expected, actual);
}

void test_backend_invalid_returns_false(void) {
    // Invalid backend type should return 0
    TEST_ASSERT_FALSE(cfd_backend_is_available((ns_solver_backend_t)999));
}

//=============================================================================
// cfd_backend_get_name() Tests
//=============================================================================

void test_backend_name_scalar(void) {
    const char* name = cfd_backend_get_name(NS_SOLVER_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL_STRING("scalar", name);
}

void test_backend_name_simd(void) {
    const char* name = cfd_backend_get_name(NS_SOLVER_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL_STRING("simd", name);
}

void test_backend_name_omp(void) {
    const char* name = cfd_backend_get_name(NS_SOLVER_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL_STRING("openmp", name);
}

void test_backend_name_cuda(void) {
    const char* name = cfd_backend_get_name(NS_SOLVER_BACKEND_CUDA);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL_STRING("cuda", name);
}

void test_backend_name_invalid(void) {
    const char* name = cfd_backend_get_name((ns_solver_backend_t)999);
    TEST_ASSERT_NOT_NULL(name);
    TEST_ASSERT_EQUAL_STRING("unknown", name);
}

//=============================================================================
// cfd_registry_list_by_backend() Tests
//=============================================================================

void test_list_by_backend_scalar(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SCALAR, names, 16);

    // Should have scalar solvers: explicit_euler, projection
    TEST_ASSERT_TRUE(count >= 2);

    // Verify names contain expected solvers
    int found_euler = 0, found_projection = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], NS_SOLVER_TYPE_EXPLICIT_EULER) == 0)
            found_euler = 1;
        if (strcmp(names[i], NS_SOLVER_TYPE_PROJECTION) == 0)
            found_projection = 1;
    }
    TEST_ASSERT_TRUE(found_euler);
    TEST_ASSERT_TRUE(found_projection);
}

void test_list_by_backend_simd(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SIMD, names, 16);

    // Should have SIMD solvers: explicit_euler_optimized, projection_optimized
    TEST_ASSERT_TRUE(count >= 2);

    int found_euler_opt = 0, found_projection_opt = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(names[i], NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED) == 0)
            found_euler_opt = 1;
        if (strcmp(names[i], NS_SOLVER_TYPE_PROJECTION_OPTIMIZED) == 0)
            found_projection_opt = 1;
    }
    TEST_ASSERT_TRUE(found_euler_opt);
    TEST_ASSERT_TRUE(found_projection_opt);
}

void test_list_by_backend_omp(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_OMP, names, 16);

    // OMP solvers are only registered if CFD_ENABLE_OPENMP is defined at compile time
    if (cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        // OpenMP is available - should have OMP solvers registered
        TEST_ASSERT_TRUE(count >= 2);

        int found_euler_omp = 0, found_projection_omp = 0;
        for (int i = 0; i < count; i++) {
            if (strcmp(names[i], NS_SOLVER_TYPE_EXPLICIT_EULER_OMP) == 0)
                found_euler_omp = 1;
            if (strcmp(names[i], NS_SOLVER_TYPE_PROJECTION_OMP) == 0)
                found_projection_omp = 1;
        }
        TEST_ASSERT_TRUE(found_euler_omp);
        TEST_ASSERT_TRUE(found_projection_omp);
    } else {
        // OpenMP not available - no OMP solvers registered
        printf("OpenMP not available - OMP solvers not registered (count=%d)\n", count);
        TEST_ASSERT_EQUAL_INT(0, count);
    }
}

void test_list_by_backend_cuda(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_CUDA, names, 16);

    // GPU solvers are only registered if CFD_HAS_CUDA is defined at compile time
    // AND gpu_is_available() returns true at runtime (factory checks this).
    // If CUDA not compiled in, count will be 0.
    // If CUDA compiled in but no GPU available, factory returns NULL so count is 0.
    if (gpu_is_available()) {
        // GPU is available - should have CUDA solvers registered and creatable
        TEST_ASSERT_TRUE(count >= 2);

        int found_euler_gpu = 0, found_projection_gpu = 0;
        for (int i = 0; i < count; i++) {
            if (strcmp(names[i], NS_SOLVER_TYPE_EXPLICIT_EULER_GPU) == 0)
                found_euler_gpu = 1;
            if (strcmp(names[i], NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU) == 0)
                found_projection_gpu = 1;
        }
        TEST_ASSERT_TRUE(found_euler_gpu);
        TEST_ASSERT_TRUE(found_projection_gpu);
    } else {
        // GPU not available at runtime - solvers either not registered or factory returns NULL
        // Either way, count should be 0
        printf("GPU not available - CUDA solvers not creatable (count=%d)\n", count);
        TEST_ASSERT_EQUAL_INT(0, count);
    }
}

void test_list_by_backend_null_registry(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(NULL, NS_SOLVER_BACKEND_SCALAR, names, 16);
    TEST_ASSERT_EQUAL_INT(0, count);
}

void test_list_by_backend_null_names(void) {
    // When names is NULL, function still counts solvers for the backend
    // but doesn't try to store them. This is valid behavior.
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SCALAR, NULL, 16);
    // Should have scalar solvers available (explicit_euler, projection)
    TEST_ASSERT_TRUE(count >= 2);
}

void test_list_by_backend_zero_max(void) {
    const char* names[16];
    int count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SCALAR, names, 0);
    TEST_ASSERT_EQUAL_INT(0, count);
}

//=============================================================================
// cfd_solver_create_checked() Tests
//=============================================================================

void test_create_checked_scalar_succeeds(void) {
    // Scalar solvers should always be creatable
    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SCALAR, solver->backend);
    solver_destroy(solver);

    solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SCALAR, solver->backend);
    solver_destroy(solver);
}

void test_create_checked_simd_conditional(void) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);

    if (cfd_has_simd()) {
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SIMD, solver->backend);
        solver_destroy(solver);
    } else {
        TEST_ASSERT_NULL(solver);
        TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, cfd_get_last_status());
        printf("SIMD not available - error correctly returned\n");
    }
}

void test_create_checked_omp_conditional(void) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);

    if (cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_OMP, solver->backend);
        solver_destroy(solver);
    } else {
        // OpenMP not available - solver creation should fail
        // Error could be:
        // - CFD_ERROR_INVALID: if OpenMP not compiled in (solver type not registered)
        // - CFD_ERROR_UNSUPPORTED: if OpenMP compiled in but not available at runtime
        TEST_ASSERT_NULL(solver);
        cfd_status_t status = cfd_get_last_status();
        TEST_ASSERT_TRUE(status == CFD_ERROR_UNSUPPORTED || status == CFD_ERROR_INVALID);
        printf("OpenMP not available - error correctly returned (status=%d)\n", status);
    }
}

void test_create_checked_cuda_conditional(void) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);

    if (gpu_is_available()) {
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_CUDA, solver->backend);
        solver_destroy(solver);
    } else {
        // GPU not available - solver creation should fail
        // Error could be:
        // - CFD_ERROR_INVALID: if CUDA not compiled in (solver type not registered)
        // - CFD_ERROR_UNSUPPORTED: if CUDA compiled in but no GPU at runtime
        TEST_ASSERT_NULL(solver);
        cfd_status_t status = cfd_get_last_status();
        TEST_ASSERT_TRUE(status == CFD_ERROR_UNSUPPORTED || status == CFD_ERROR_INVALID);
        printf("CUDA not available - error correctly returned (status=%d)\n", status);
    }
}

void test_create_checked_invalid_type(void) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, "nonexistent_solver");
    TEST_ASSERT_NULL(solver);
}

void test_create_checked_null_registry(void) {
    ns_solver_t* solver = cfd_solver_create_checked(NULL, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NULL(solver);
}

void test_create_checked_null_type(void) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, NULL);
    TEST_ASSERT_NULL(solver);
}

//=============================================================================
// Solver Backend Field Tests
//=============================================================================

void test_solver_backend_field_set_correctly(void) {
    // Test that each solver type has the correct backend field set
    ns_solver_t* solver;

    // Scalar solvers
    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SCALAR, solver->backend);
    solver_destroy(solver);

    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SCALAR, solver->backend);
    solver_destroy(solver);

    // SIMD solvers
    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SIMD, solver->backend);
    solver_destroy(solver);

    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_SIMD, solver->backend);
    solver_destroy(solver);

    // OMP solvers - only test if OpenMP is available
    if (cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_OMP, solver->backend);
        solver_destroy(solver);

        solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OMP);
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_OMP, solver->backend);
        solver_destroy(solver);
    } else {
        printf("OpenMP not available - skipping OMP solver backend field tests\n");
    }
}

void test_gpu_solver_backend_field(void) {
    // GPU solvers - only test if GPU is available
    // Note: cfd_solver_create returns NULL if GPU not available due to factory check
    ns_solver_t* solver;

    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU);
    if (solver != NULL) {
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_CUDA, solver->backend);
        solver_destroy(solver);
    } else {
        printf("GPU solver creation returned NULL (expected when GPU not available)\n");
    }

    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (solver != NULL) {
        TEST_ASSERT_EQUAL_INT(NS_SOLVER_BACKEND_CUDA, solver->backend);
        solver_destroy(solver);
    } else {
        printf("GPU solver creation returned NULL (expected when GPU not available)\n");
    }
}

//=============================================================================
// Error Message Tests
//=============================================================================

void test_error_message_on_unavailable_backend(void) {
    // Only test if we know a backend is unavailable
    if (!gpu_is_available()) {
        cfd_clear_error();
        ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
        TEST_ASSERT_NULL(solver);

        const char* error = cfd_get_last_error();
        TEST_ASSERT_NOT_NULL(error);
        // Error message should mention the backend or availability
        TEST_ASSERT_TRUE(strlen(error) > 0);
        printf("Error message: %s\n", error);
    } else {
        printf("Skipping error message test - GPU is available\n");
    }
}

//=============================================================================
// Main
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    // cfd_backend_is_available() tests
    RUN_TEST(test_backend_scalar_always_available);
    RUN_TEST(test_backend_simd_matches_cpu_features);
    RUN_TEST(test_backend_cuda_matches_gpu_available);
    RUN_TEST(test_backend_invalid_returns_false);

    // cfd_backend_get_name() tests
    RUN_TEST(test_backend_name_scalar);
    RUN_TEST(test_backend_name_simd);
    RUN_TEST(test_backend_name_omp);
    RUN_TEST(test_backend_name_cuda);
    RUN_TEST(test_backend_name_invalid);

    // cfd_registry_list_by_backend() tests
    RUN_TEST(test_list_by_backend_scalar);
    RUN_TEST(test_list_by_backend_simd);
    RUN_TEST(test_list_by_backend_omp);
    RUN_TEST(test_list_by_backend_cuda);
    RUN_TEST(test_list_by_backend_null_registry);
    RUN_TEST(test_list_by_backend_null_names);
    RUN_TEST(test_list_by_backend_zero_max);

    // cfd_solver_create_checked() tests
    RUN_TEST(test_create_checked_scalar_succeeds);
    RUN_TEST(test_create_checked_simd_conditional);
    RUN_TEST(test_create_checked_omp_conditional);
    RUN_TEST(test_create_checked_cuda_conditional);
    RUN_TEST(test_create_checked_invalid_type);
    RUN_TEST(test_create_checked_null_registry);
    RUN_TEST(test_create_checked_null_type);

    // Solver backend field tests
    RUN_TEST(test_solver_backend_field_set_correctly);
    RUN_TEST(test_gpu_solver_backend_field);

    // Error message tests
    RUN_TEST(test_error_message_on_unavailable_backend);

    return UNITY_END();
}
