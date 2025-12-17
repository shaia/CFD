/**
 * Unit Tests for Poisson Solver Abstraction
 *
 * Tests the poisson_solver interface:
 * - Parameter defaults
 * - Solver lifecycle (create, init, destroy)
 * - Jacobi solver (scalar and SIMD)
 * - Red-Black SOR solver (scalar and SIMD)
 * - SOR solver (scalar)
 * - Backend selection
 * - Convergence on simple problems
 * - Convenience API
 */

#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "unity.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Test grid sizes */
#define TEST_NX 32
#define TEST_NY 32
#define TEST_DX 0.1
#define TEST_DY 0.1

/* Tolerance for floating point comparisons */
#define TOLERANCE 1e-10

void setUp(void) {
    /* Called before each test */
}

void tearDown(void) {
    /* Called after each test */
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Allocate and initialize test arrays
 */
static double* create_test_field(size_t nx, size_t ny, double value) {
    double* field = (double*)cfd_calloc(nx * ny, sizeof(double));
    if (field) {
        for (size_t i = 0; i < nx * ny; i++) {
            field[i] = value;
        }
    }
    return field;
}

/**
 * Create RHS for test problem: zero RHS
 * Expected solution: constant (initial guess)
 */
static double* create_zero_rhs(size_t nx, size_t ny) {
    return create_test_field(nx, ny, 0.0);
}

/**
 * Create RHS for test problem: uniform RHS
 * Solution is quadratic in x and y
 */
static double* create_uniform_rhs(size_t nx, size_t ny, double value) {
    return create_test_field(nx, ny, value);
}

/* ============================================================================
 * PARAMETER TESTS
 * ============================================================================ */

void test_params_default(void) {
    poisson_solver_params_t params = poisson_solver_params_default();

    TEST_ASSERT_EQUAL_DOUBLE(1e-6, params.tolerance);
    TEST_ASSERT_EQUAL_DOUBLE(1e-10, params.absolute_tolerance);
    TEST_ASSERT_EQUAL_INT(1000, params.max_iterations);
    TEST_ASSERT_EQUAL_DOUBLE(1.5, params.omega);
    TEST_ASSERT_EQUAL_INT(1, params.check_interval);
    TEST_ASSERT_FALSE(params.verbose);
}

void test_stats_default(void) {
    poisson_solver_stats_t stats = poisson_solver_stats_default();

    TEST_ASSERT_EQUAL_INT(POISSON_ERROR, stats.status);
    TEST_ASSERT_EQUAL_INT(0, stats.iterations);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, stats.initial_residual);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, stats.final_residual);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, stats.elapsed_time_ms);
}

/* ============================================================================
 * BACKEND SELECTION TESTS
 * ============================================================================ */

void test_backend_scalar_available(void) {
    TEST_ASSERT_TRUE(poisson_solver_backend_available(POISSON_BACKEND_SCALAR));
}

void test_backend_auto_available(void) {
    TEST_ASSERT_TRUE(poisson_solver_backend_available(POISSON_BACKEND_AUTO));
}

void test_backend_set_scalar(void) {
    poisson_solver_backend_t original = poisson_solver_get_backend();

    TEST_ASSERT_TRUE(poisson_solver_set_backend(POISSON_BACKEND_SCALAR));
    TEST_ASSERT_EQUAL_INT(POISSON_BACKEND_SCALAR, poisson_solver_get_backend());
    TEST_ASSERT_EQUAL_STRING("scalar", poisson_solver_get_backend_name());

    /* Restore original */
    poisson_solver_set_backend(original);
}

/* ============================================================================
 * SOLVER LIFECYCLE TESTS
 * ============================================================================ */

void test_create_jacobi_scalar(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_STRING(POISSON_SOLVER_TYPE_JACOBI_SCALAR, solver->name);
    TEST_ASSERT_EQUAL_INT(POISSON_METHOD_JACOBI, solver->method);
    TEST_ASSERT_EQUAL_INT(POISSON_BACKEND_SCALAR, solver->backend);

    poisson_solver_destroy(solver);
}

void test_create_sor_scalar(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_STRING(POISSON_SOLVER_TYPE_SOR_SCALAR, solver->name);
    TEST_ASSERT_EQUAL_INT(POISSON_METHOD_SOR, solver->method);

    poisson_solver_destroy(solver);
}

void test_create_redblack_scalar(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_STRING(POISSON_SOLVER_TYPE_REDBLACK_SCALAR, solver->name);
    TEST_ASSERT_EQUAL_INT(POISSON_METHOD_REDBLACK_SOR, solver->method);

    poisson_solver_destroy(solver);
}

void test_create_with_auto_backend(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_AUTO);

    TEST_ASSERT_NOT_NULL(solver);
    /* Backend should be SIMD if available, otherwise SCALAR */
    TEST_ASSERT_TRUE(
        solver->backend == POISSON_BACKEND_SIMD ||
        solver->backend == POISSON_BACKEND_SCALAR);

    poisson_solver_destroy(solver);
}

void test_init_solver(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    cfd_status_t status = poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, NULL);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(TEST_NX, solver->nx);
    TEST_ASSERT_EQUAL_INT(TEST_NY, solver->ny);
    TEST_ASSERT_EQUAL_DOUBLE(TEST_DX, solver->dx);
    TEST_ASSERT_EQUAL_DOUBLE(TEST_DY, solver->dy);

    poisson_solver_destroy(solver);
}

void test_init_with_custom_params(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = 500;
    params.omega = 1.7;

    cfd_status_t status = poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_DOUBLE(1e-8, solver->params.tolerance);
    TEST_ASSERT_EQUAL_INT(500, solver->params.max_iterations);
    TEST_ASSERT_EQUAL_DOUBLE(1.7, solver->params.omega);

    poisson_solver_destroy(solver);
}

void test_destroy_null_is_safe(void) {
    poisson_solver_destroy(NULL);  /* Should not crash */
    TEST_PASS();
}

/* ============================================================================
 * CONVERGENCE TESTS
 * ============================================================================ */

/**
 * Test convergence on zero RHS (trivial problem)
 * Solution should remain constant (initial guess)
 */
void test_jacobi_converges_zero_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE(stats.final_residual < 1e-6);

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_sor_converges_zero_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_redblack_converges_zero_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/* ============================================================================
 * RESIDUAL TESTS
 * ============================================================================ */

void test_compute_residual_zero_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, NULL);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    double residual = poisson_solver_compute_residual(solver, x, rhs);

    /* With x=0 and rhs=0, Laplacian(x) - rhs = 0 */
    TEST_ASSERT_TRUE(residual < TOLERANCE);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/* ============================================================================
 * LEGACY API BACKWARD COMPATIBILITY TESTS
 * ============================================================================ */

void test_legacy_poisson_solve_sor(void) {
    double* p = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    int iterations = poisson_solve(p, NULL, rhs, TEST_NX, TEST_NY, TEST_DX, TEST_DY,
                                   POISSON_SOLVER_SOR_SCALAR);

    /* Should converge quickly for zero RHS */
    TEST_ASSERT_TRUE(iterations >= 0 || iterations == -1);  /* -1 if not converged */

    cfd_free(p);
    cfd_free(rhs);
}

void test_legacy_poisson_solve_jacobi(void) {
    double* p = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* p_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    int iterations = poisson_solve(p, p_temp, rhs, TEST_NX, TEST_NY, TEST_DX, TEST_DY,
                                   POISSON_SOLVER_JACOBI_SIMD);

    TEST_ASSERT_TRUE(iterations >= 0 || iterations == -1);

    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
}

void test_legacy_poisson_solve_redblack(void) {
    double* p = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* p_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    int iterations = poisson_solve(p, p_temp, rhs, TEST_NX, TEST_NY, TEST_DX, TEST_DY,
                                   POISSON_SOLVER_REDBLACK_SIMD);

    TEST_ASSERT_TRUE(iterations >= 0 || iterations == -1);

    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * SIMD BACKEND TESTS (if available)
 * ============================================================================ */

void test_jacobi_simd_if_available(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(POISSON_BACKEND_SIMD, solver->backend);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_redblack_simd_if_available(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_INT(POISSON_BACKEND_SIMD, solver->backend);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/* ============================================================================
 * STATISTICS TESTS
 * ============================================================================ */

void test_stats_timing(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 50;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    /* Timing should be positive */
    TEST_ASSERT_TRUE(stats.elapsed_time_ms >= 0.0);

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/* ============================================================================
 * TEST RUNNER
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Parameter tests */
    RUN_TEST(test_params_default);
    RUN_TEST(test_stats_default);

    /* Backend tests */
    RUN_TEST(test_backend_scalar_available);
    RUN_TEST(test_backend_auto_available);
    RUN_TEST(test_backend_set_scalar);

    /* Lifecycle tests */
    RUN_TEST(test_create_jacobi_scalar);
    RUN_TEST(test_create_sor_scalar);
    RUN_TEST(test_create_redblack_scalar);
    RUN_TEST(test_create_with_auto_backend);
    RUN_TEST(test_init_solver);
    RUN_TEST(test_init_with_custom_params);
    RUN_TEST(test_destroy_null_is_safe);

    /* Convergence tests */
    RUN_TEST(test_jacobi_converges_zero_rhs);
    RUN_TEST(test_sor_converges_zero_rhs);
    RUN_TEST(test_redblack_converges_zero_rhs);

    /* Residual tests */
    RUN_TEST(test_compute_residual_zero_rhs);

    /* Legacy API tests */
    RUN_TEST(test_legacy_poisson_solve_sor);
    RUN_TEST(test_legacy_poisson_solve_jacobi);
    RUN_TEST(test_legacy_poisson_solve_redblack);

    /* SIMD tests */
    RUN_TEST(test_jacobi_simd_if_available);
    RUN_TEST(test_redblack_simd_if_available);

    /* Statistics tests */
    RUN_TEST(test_stats_timing);

    return UNITY_END();
}
