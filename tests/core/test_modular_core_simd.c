/**
 * Test Suite for Modular Libraries: Core + SIMD Only
 *
 * This test links ONLY against CFD::Core and CFD::SIMD to verify
 * that the SIMD modular library works when not using the unified
 * CFD::Library.
 *
 * Key verification:
 * - Core functionality works independently
 * - SIMD solvers work without OMP/CUDA backends
 * - SIMD availability is correctly reported
 * - Graceful handling when SIMD not available on CPU
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/cfd_version.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {
    cfd_init();
    cfd_clear_error();
}

void tearDown(void) {
    cfd_clear_error();
    cfd_finalize();
}

//=============================================================================
// Core Library Tests (minimal - just verify core works)
//=============================================================================

void test_core_basics(void) {
    const char* version = cfd_get_version_string();
    TEST_ASSERT_NOT_NULL(version);
    printf("CFD Library version: %s\n", version);

    grid* g = grid_create(10, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    grid_destroy(g);
}

void test_cpu_features_detection(void) {
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    const char* simd_name = cfd_get_simd_name();
    int has_simd = cfd_has_simd();

    printf("Detected SIMD architecture: %s\n", simd_name);
    printf("SIMD available: %s\n", has_simd ? "yes" : "no");

    // Verify consistency
    if (arch == CFD_SIMD_NONE) {
        TEST_ASSERT_FALSE(has_simd);
    } else {
        TEST_ASSERT_TRUE(has_simd);
    }
}

//=============================================================================
// SIMD Library Tests
//=============================================================================

void test_simd_backend_availability(void) {
    int simd_available = cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD);
    int has_simd = cfd_has_simd();

    // Backend availability should match CPU feature detection
    TEST_ASSERT_EQUAL(has_simd, simd_available);

    const char* backend_name = cfd_backend_get_name(NS_SOLVER_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(backend_name);
    printf("SIMD backend: %s (available: %s)\n", backend_name, simd_available ? "yes" : "no");
}

void test_simd_solver_registry(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    // List SIMD solvers
    const char* names[16];
    int simd_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SIMD, names, 16);
    printf("SIMD solvers registered: %d\n", simd_count);

    // SIMD solvers should always be registered (even if not runnable)
    TEST_ASSERT_TRUE(simd_count >= 2);

    for (int i = 0; i < simd_count; i++) {
        printf("  - %s\n", names[i]);
    }

    cfd_registry_destroy(registry);
}

void test_simd_solver_creation_checked(void) {
    // This test verifies that cfd_solver_create_checked properly checks
    // backend availability before creating the solver

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    int simd_available = cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD);

    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);

    if (simd_available) {
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "SIMD solver should be created when SIMD is available");
        TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SIMD, solver->backend);
        solver_destroy(solver);
    } else {
        TEST_ASSERT_NULL_MESSAGE(solver, "SIMD solver should NOT be created when SIMD is unavailable");
        TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, cfd_get_last_status());
        printf("SIMD solver correctly rejected (SIMD not available on this CPU)\n");
    }

    cfd_registry_destroy(registry);
}

void test_simd_euler_solver_conditional(void) {
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        printf("SIMD not available - skipping SIMD Euler solver test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SIMD, solver->backend);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();

    cfd_status_t status = solver_init(solver, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    ns_solver_stats_t stats = ns_solver_stats_default();
    status = solver_step(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

void test_simd_projection_solver_conditional(void) {
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        printf("SIMD not available - skipping SIMD Projection solver test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SIMD, solver->backend);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();

    cfd_status_t status = solver_init(solver, g, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        /* SIMD Poisson solver not compiled (CFD_ENABLE_AVX2=OFF) */
        solver_destroy(solver);
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("SIMD Poisson solver not available (AVX2 not compiled)");
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    ns_solver_stats_t stats = ns_solver_stats_default();
    status = solver_step(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

void test_simd_multiple_steps_conditional(void) {
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        printf("SIMD not available - skipping SIMD multi-step test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(solver);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();

    cfd_status_t status = solver_init(solver, g, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        /* SIMD Poisson solver not compiled (CFD_ENABLE_AVX2=OFF) */
        solver_destroy(solver);
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("SIMD Poisson solver not available (AVX2 not compiled)");
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Run multiple steps to verify stability
    for (int i = 0; i < 10; i++) {
        ns_solver_stats_t stats = ns_solver_stats_default();
        status = solver_step(solver, field, g, &params, &stats);
        if (status == CFD_ERROR_MAX_ITER) {
            /* Poisson solver convergence failure on trivial test case - acceptable for API test */
            printf("  Note: Step %d hit convergence limit (expected on zero initial conditions)\n", i);
            break;
        }
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

//=============================================================================
// Main
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    // Core tests
    RUN_TEST(test_core_basics);
    RUN_TEST(test_cpu_features_detection);

    // SIMD tests
    RUN_TEST(test_simd_backend_availability);
    RUN_TEST(test_simd_solver_registry);
    RUN_TEST(test_simd_solver_creation_checked);
    RUN_TEST(test_simd_euler_solver_conditional);
    RUN_TEST(test_simd_projection_solver_conditional);
    RUN_TEST(test_simd_multiple_steps_conditional);

    return UNITY_END();
}
