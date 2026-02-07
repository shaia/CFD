/**
 * Test Suite for Modular Backend Libraries
 *
 * Tests that the modular library split works correctly:
 * - CFD::Core provides basic functionality
 * - CFD::Scalar provides scalar CPU solvers
 * - CFD::SIMD provides SIMD-optimized solvers
 * - CFD::OMP provides OpenMP parallelized solvers
 * - CFD::CUDA provides GPU solvers (when available)
 * - CFD::Library provides unified access to all backends
 *
 * This test links against CFD::Library (unified) and verifies
 * that all backend functionality is accessible.
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/cfd_version.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/io/vtk_output.h"
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
// Core Library Tests (cfd_core)
//=============================================================================

void test_core_version_available(void) {
    // Version macros should be available from core
    int major = CFD_VERSION_MAJOR;
    int minor = CFD_VERSION_MINOR;
    int patch = CFD_VERSION_PATCH;

    TEST_ASSERT_TRUE(major >= 0);
    TEST_ASSERT_TRUE(minor >= 0);
    TEST_ASSERT_TRUE(patch >= 0);

    const char* version = cfd_get_version_string();
    TEST_ASSERT_NOT_NULL(version);
    TEST_ASSERT_TRUE(strlen(version) > 0);
    printf("CFD Library version: %s\n", version);
}

void test_core_init_finalize(void) {
    // Init/finalize should work (already called in setUp/tearDown)
    // Just verify we can call status functions
    cfd_status_t status = cfd_get_last_status();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
}

void test_core_grid_creation(void) {
    // Grid creation is part of core
    grid* g = grid_create(10, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    TEST_ASSERT_EQUAL(10, g->nx);
    TEST_ASSERT_EQUAL(10, g->ny);
    grid_destroy(g);
}

void test_core_memory_allocation(void) {
    // Memory utilities are part of core
    double* data = cfd_malloc(100 * sizeof(double));
    TEST_ASSERT_NOT_NULL(data);

    // Initialize and verify
    for (int i = 0; i < 100; i++) {
        data[i] = (double)i;
    }
    TEST_ASSERT_EQUAL_DOUBLE(50.0, data[50]);

    cfd_free(data);
}

void test_core_cpu_features(void) {
    // CPU feature detection is part of core
    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    TEST_ASSERT_TRUE(arch == CFD_SIMD_NONE || arch == CFD_SIMD_AVX2 || arch == CFD_SIMD_NEON);

    const char* simd_name = cfd_get_simd_name();
    TEST_ASSERT_NOT_NULL(simd_name);
    printf("Detected SIMD: %s\n", simd_name);
}

void test_core_error_handling(void) {
    // Error handling is part of core
    cfd_clear_error();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_get_last_status());

    cfd_set_error(CFD_ERROR_INVALID, "Test error message");
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, cfd_get_last_status());

    const char* error = cfd_get_last_error();
    TEST_ASSERT_NOT_NULL(error);
    TEST_ASSERT_TRUE(strstr(error, "Test error message") != NULL);

    cfd_clear_error();
}

//=============================================================================
// Scalar Library Tests (cfd_scalar)
//=============================================================================

void test_scalar_solver_creation(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    // Scalar solvers should always be available
    ns_solver_t* euler = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(euler);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SCALAR, euler->backend);
    solver_destroy(euler);

    ns_solver_t* proj = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(proj);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SCALAR, proj->backend);
    solver_destroy(proj);

    cfd_registry_destroy(registry);
}

void test_scalar_flow_field(void) {
    // Flow field creation uses scalar backend
    flow_field* field = flow_field_create(10, 10);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(field->u);
    TEST_ASSERT_NOT_NULL(field->v);
    TEST_ASSERT_NOT_NULL(field->p);
    flow_field_destroy(field);
}

void test_scalar_solver_step(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(solver);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();

    // Initialize solver
    cfd_status_t status = solver_init(solver, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Take a single step
    ns_solver_stats_t stats = ns_solver_stats_default();
    status = solver_step(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

//=============================================================================
// SIMD Library Tests (cfd_simd)
//=============================================================================

void test_simd_solver_creation(void) {
    // SIMD solvers may not be available on all CPUs
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        printf("SIMD not available - skipping SIMD solver creation test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    // SIMD solvers should be registered
    ns_solver_t* euler = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(euler);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SIMD, euler->backend);
    solver_destroy(euler);

    ns_solver_t* proj = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);
    TEST_ASSERT_NOT_NULL(proj);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SIMD, proj->backend);
    solver_destroy(proj);

    cfd_registry_destroy(registry);
}

void test_simd_backend_availability(void) {
    // SIMD availability should match CPU features
    int simd_available = cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD);
    int has_simd = cfd_has_simd();

    TEST_ASSERT_EQUAL(has_simd, simd_available);
    printf("SIMD backend available: %s\n", simd_available ? "yes" : "no");
}

void test_simd_solver_step_conditional(void) {
    if (!cfd_has_simd()) {
        printf("SIMD not available on this CPU - skipping step test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED);
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

    ns_solver_stats_t stats = ns_solver_stats_default();
    status = solver_step(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

//=============================================================================
// OpenMP Library Tests (cfd_omp)
//=============================================================================

void test_omp_solver_creation(void) {
    // OMP solvers are only registered if OpenMP is available
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        printf("OpenMP not available - skipping OMP solver creation test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    // OMP solvers should be registered
    ns_solver_t* euler = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP);
    TEST_ASSERT_NOT_NULL(euler);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_OMP, euler->backend);
    solver_destroy(euler);

    ns_solver_t* proj = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OMP);
    TEST_ASSERT_NOT_NULL(proj);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_OMP, proj->backend);
    solver_destroy(proj);

    cfd_registry_destroy(registry);
}

void test_omp_backend_availability(void) {
    int omp_available = cfd_backend_is_available(NS_SOLVER_BACKEND_OMP);
    printf("OpenMP backend available: %s\n", omp_available ? "yes" : "no");

    // OMP should be available if compiled with OpenMP
#ifdef CFD_ENABLE_OPENMP
    TEST_ASSERT_TRUE(omp_available);
#endif
}

void test_omp_solver_step_conditional(void) {
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        printf("OpenMP not available - skipping step test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_OMP);
    TEST_ASSERT_NOT_NULL(solver);

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

//=============================================================================
// CUDA Library Tests (cfd_cuda)
//=============================================================================

void test_cuda_backend_availability(void) {
    int cuda_available = cfd_backend_is_available(NS_SOLVER_BACKEND_CUDA);
    int gpu_available = gpu_is_available();

    TEST_ASSERT_EQUAL(gpu_available, cuda_available);
    printf("CUDA backend available: %s\n", cuda_available ? "yes" : "no");
}

void test_cuda_solver_creation_conditional(void) {
    if (!gpu_is_available()) {
        printf("GPU not available - skipping CUDA solver creation test\n");
        TEST_PASS();
        return;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create_checked(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_CUDA, solver->backend);
    solver_destroy(solver);

    cfd_registry_destroy(registry);
}

void test_cuda_gpu_config(void) {
    // GPU config functions should always be available (stub or real)
    gpu_config_t config = gpu_config_default();

    TEST_ASSERT_TRUE(config.min_grid_size > 0);
    TEST_ASSERT_TRUE(config.block_size_x > 0);
    TEST_ASSERT_TRUE(config.block_size_y > 0);
}

//=============================================================================
// Unified Library Tests (cfd_library)
//=============================================================================

void test_unified_all_backends_accessible(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    const char* names[32];
    int total = cfd_registry_list(registry, names, 32);

    printf("Total solvers in unified library: %d\n", total);
    TEST_ASSERT_TRUE(total >= 4);  // At least scalar + simd solvers

    // List all available solvers
    for (int i = 0; i < total; i++) {
        printf("  - %s\n", names[i]);
    }

    cfd_registry_destroy(registry);
}

void test_unified_backend_counts(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    const char* names[16];

    int scalar_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SCALAR, names, 16);
    int simd_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SIMD, names, 16);
    int omp_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_OMP, names, 16);
    int cuda_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_CUDA, names, 16);

    printf("Backend solver counts:\n");
    printf("  Scalar: %d\n", scalar_count);
    printf("  SIMD:   %d\n", simd_count);
    printf("  OMP:    %d\n", omp_count);
    printf("  CUDA:   %d\n", cuda_count);

    // Scalar and SIMD should always have solvers
    TEST_ASSERT_TRUE(scalar_count >= 2);
    TEST_ASSERT_TRUE(simd_count >= 2);

    // OMP depends on compile-time flag
    if (cfd_backend_is_available(NS_SOLVER_BACKEND_OMP)) {
        TEST_ASSERT_TRUE(omp_count >= 2);
    }

    // CUDA depends on runtime GPU availability
    if (gpu_is_available()) {
        TEST_ASSERT_TRUE(cuda_count >= 2);
    }

    cfd_registry_destroy(registry);
}

void test_unified_solver_switching(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();
    ns_solver_stats_t stats;

    int simd_available = cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD);

    // Test switching between different backends
    // Only test SIMD solvers if SIMD is available
    const char* solver_types[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        NS_SOLVER_TYPE_PROJECTION,
    };
    int num_solvers = 2;

    const char* simd_solver_types[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    };

    // Test scalar solvers (always available)
    for (int i = 0; i < num_solvers; i++) {
        ns_solver_t* solver = cfd_solver_create(registry, solver_types[i]);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, solver_types[i]);

        cfd_status_t status = solver_init(solver, g, &params);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        stats = ns_solver_stats_default();
        status = solver_step(solver, field, g, &params, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        solver_destroy(solver);
    }

    // Test SIMD solvers only if available
    if (simd_available) {
        for (int i = 0; i < 2; i++) {
            ns_solver_t* solver = cfd_solver_create(registry, simd_solver_types[i]);
            TEST_ASSERT_NOT_NULL_MESSAGE(solver, simd_solver_types[i]);

            cfd_status_t status = solver_init(solver, g, &params);
            if (status == CFD_ERROR_UNSUPPORTED) {
                /* SIMD Poisson solver not compiled - skip this solver */
                printf("  %s: init returned UNSUPPORTED, skipping\n", simd_solver_types[i]);
                solver_destroy(solver);
                continue;
            }
            TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

            stats = ns_solver_stats_default();
            status = solver_step(solver, field, g, &params, &stats);
            TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

            solver_destroy(solver);
        }
    } else {
        printf("SIMD not available - skipping SIMD solver switching tests\n");
    }

    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

//=============================================================================
// Main
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    // Core library tests
    RUN_TEST(test_core_version_available);
    RUN_TEST(test_core_init_finalize);
    RUN_TEST(test_core_grid_creation);
    RUN_TEST(test_core_memory_allocation);
    RUN_TEST(test_core_cpu_features);
    RUN_TEST(test_core_error_handling);

    // Scalar library tests
    RUN_TEST(test_scalar_solver_creation);
    RUN_TEST(test_scalar_flow_field);
    RUN_TEST(test_scalar_solver_step);

    // SIMD library tests
    RUN_TEST(test_simd_solver_creation);
    RUN_TEST(test_simd_backend_availability);
    RUN_TEST(test_simd_solver_step_conditional);

    // OpenMP library tests
    RUN_TEST(test_omp_solver_creation);
    RUN_TEST(test_omp_backend_availability);
    RUN_TEST(test_omp_solver_step_conditional);

    // CUDA library tests
    RUN_TEST(test_cuda_backend_availability);
    RUN_TEST(test_cuda_solver_creation_conditional);
    RUN_TEST(test_cuda_gpu_config);

    // Unified library tests
    RUN_TEST(test_unified_all_backends_accessible);
    RUN_TEST(test_unified_backend_counts);
    RUN_TEST(test_unified_solver_switching);

    return UNITY_END();
}
