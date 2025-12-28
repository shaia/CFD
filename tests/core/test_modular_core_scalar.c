/**
 * Test Suite for Modular Libraries: Core + Scalar Only
 *
 * This test links ONLY against CFD::Core and CFD::Scalar to verify
 * that the modular library architecture actually works when not using
 * the unified CFD::Library.
 *
 * Key verification:
 * - Core functionality (grid, memory, I/O) works independently
 * - Scalar solvers work without SIMD/OMP/CUDA backends
 * - No unresolved symbols from missing backends
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/cfd_version.h"
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
// Core Library Tests
//=============================================================================

void test_core_version(void) {
    const char* version = cfd_get_version_string();
    TEST_ASSERT_NOT_NULL(version);
    TEST_ASSERT_TRUE(strlen(version) > 0);
    printf("CFD Library version: %s\n", version);
}

void test_core_grid_creation(void) {
    grid* g = grid_create(10, 10, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    TEST_ASSERT_EQUAL(10, g->nx);
    TEST_ASSERT_EQUAL(10, g->ny);
    grid_destroy(g);
}

void test_core_memory(void) {
    double* data = cfd_malloc(100 * sizeof(double));
    TEST_ASSERT_NOT_NULL(data);
    for (int i = 0; i < 100; i++) {
        data[i] = (double)i;
    }
    TEST_ASSERT_EQUAL_DOUBLE(50.0, data[50]);
    cfd_free(data);
}

void test_core_error_handling(void) {
    cfd_clear_error();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_get_last_status());

    cfd_set_error(CFD_ERROR_INVALID, "Test error");
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, cfd_get_last_status());

    cfd_clear_error();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_get_last_status());
}

//=============================================================================
// Scalar Library Tests
//=============================================================================

void test_scalar_flow_field(void) {
    flow_field* field = flow_field_create(10, 10);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(field->u);
    TEST_ASSERT_NOT_NULL(field->v);
    TEST_ASSERT_NOT_NULL(field->p);
    flow_field_destroy(field);
}

void test_scalar_solver_registry(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);

    // Register only scalar solvers (no SIMD/OMP/CUDA)
    // In modular build, only scalar factories should be available
    cfd_registry_register_defaults(registry);

    // Scalar solvers should be available
    const char* names[16];
    int scalar_count = cfd_registry_list_by_backend(registry, NS_SOLVER_BACKEND_SCALAR, names, 16);
    printf("Scalar solvers available: %d\n", scalar_count);
    TEST_ASSERT_TRUE(scalar_count >= 2);  // At least euler and projection

    for (int i = 0; i < scalar_count; i++) {
        printf("  - %s\n", names[i]);
    }

    cfd_registry_destroy(registry);
}

void test_scalar_euler_solver(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SCALAR, solver->backend);

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

void test_scalar_projection_solver(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(NS_SOLVER_BACKEND_SCALAR, solver->backend);

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

void test_scalar_multiple_steps(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(solver);

    grid* g = grid_create(16, 16, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(16, 16);
    ns_solver_params_t params = ns_solver_params_default();

    cfd_status_t status = solver_init(solver, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    // Run multiple steps to verify stability
    for (int i = 0; i < 10; i++) {
        ns_solver_stats_t stats = ns_solver_stats_default();
        status = solver_step(solver, field, g, &params, &stats);
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

    // Core library tests
    RUN_TEST(test_core_version);
    RUN_TEST(test_core_grid_creation);
    RUN_TEST(test_core_memory);
    RUN_TEST(test_core_error_handling);

    // Scalar library tests
    RUN_TEST(test_scalar_flow_field);
    RUN_TEST(test_scalar_solver_registry);
    RUN_TEST(test_scalar_euler_solver);
    RUN_TEST(test_scalar_projection_solver);
    RUN_TEST(test_scalar_multiple_steps);

    return UNITY_END();
}
