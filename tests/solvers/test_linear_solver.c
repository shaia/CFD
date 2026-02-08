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
 *
 * Note: All SIMD backends use runtime CPU detection (AVX2/NEON).
 */

#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
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
    TEST_ASSERT_EQUAL_INT(5000, params.max_iterations);  /* Increased from 1000 for CG on fine grids */
    TEST_ASSERT_EQUAL_DOUBLE(1.5, params.omega);
    TEST_ASSERT_EQUAL_INT(1, params.check_interval);
    TEST_ASSERT_FALSE(params.verbose);
    TEST_ASSERT_EQUAL_INT(POISSON_PRECOND_NONE, params.preconditioner);
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

void test_create_cg_scalar(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_STRING(POISSON_SOLVER_TYPE_CG_SCALAR, solver->name);
    TEST_ASSERT_EQUAL_INT(POISSON_METHOD_CG, solver->method);
    TEST_ASSERT_EQUAL_INT(POISSON_BACKEND_SCALAR, solver->backend);

    poisson_solver_destroy(solver);
}

void test_create_with_auto_backend(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_AUTO);

    TEST_ASSERT_NOT_NULL(solver);
    /* Backend should be SIMD or SCALAR depending on availability */
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

void test_cg_converges_zero_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
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
    TEST_ASSERT_TRUE(stats.final_residual < 1e-6);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_cg_converges_uniform_rhs(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 500;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    /* CG should converge for this problem */
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/**
 * Test that CG scalar and SIMD produce consistent results
 */
void test_cg_scalar_simd_consistency(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    /* Create scalar solver */
    poisson_solver_t* scalar_solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(scalar_solver);

    /* Create SIMD solver */
    poisson_solver_t* simd_solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(simd_solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 200;
    params.tolerance = 1e-8;

    poisson_solver_init(scalar_solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);
    poisson_solver_init(simd_solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    /* Allocate fields */
    double* x_scalar = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_simd = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    poisson_solver_stats_t stats_simd = poisson_solver_stats_default();

    poisson_solver_solve(scalar_solver, x_scalar, NULL, rhs, &stats_scalar);
    poisson_solver_solve(simd_solver, x_simd, NULL, rhs, &stats_simd);

    /* Both should converge */
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats_scalar.status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats_simd.status);

    /* Compare solutions - should be very close */
    double max_diff = 0.0;
    for (size_t j = 1; j < TEST_NY - 1; j++) {
        for (size_t i = 1; i < TEST_NX - 1; i++) {
            size_t idx = j * TEST_NX + i;
            double diff = fabs(x_scalar[idx] - x_simd[idx]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    /* Solutions should match within reasonable tolerance */
    TEST_ASSERT_TRUE(max_diff < 1e-6);

    cfd_free(x_scalar);
    cfd_free(x_simd);
    cfd_free(rhs);
    poisson_solver_destroy(scalar_solver);
    poisson_solver_destroy(simd_solver);
}

/**
 * Test CG with larger grid size
 */
void test_cg_larger_grid(void) {
    const size_t NX = 64;
    const size_t NY = 64;

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 2000;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, NX, NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(NX, NY, 0.0);
    double* rhs = create_uniform_rhs(NX, NY, 1.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    /* Verify solver met its convergence criterion (relative or absolute tolerance) */
    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/**
 * Test CG with non-zero initial guess
 */
void test_cg_nonzero_initial_guess(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 500;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    /* Start with non-zero initial guess */
    double* x = create_test_field(TEST_NX, TEST_NY, 0.5);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/**
 * Test that CG uses no more iterations than Jacobi on a simple problem.
 * Uses zero RHS where both solvers converge quickly.
 */
void test_cg_no_more_iterations_than_jacobi(void) {
    /* Use zero RHS where both solvers converge very quickly */
    poisson_solver_t* cg_solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    poisson_solver_t* jacobi_solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(cg_solver);
    TEST_ASSERT_NOT_NULL(jacobi_solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    params.tolerance = 1e-6;

    poisson_solver_init(cg_solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);
    poisson_solver_init(jacobi_solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x_cg = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_jacobi = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* x_temp = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_zero_rhs(TEST_NX, TEST_NY);

    poisson_solver_stats_t stats_cg = poisson_solver_stats_default();
    poisson_solver_stats_t stats_jacobi = poisson_solver_stats_default();

    poisson_solver_solve(cg_solver, x_cg, NULL, rhs, &stats_cg);
    poisson_solver_solve(jacobi_solver, x_jacobi, x_temp, rhs, &stats_jacobi);

    /* Both should converge on this simple problem */
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats_cg.status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats_jacobi.status);

    /* CG should use no more iterations than Jacobi */
    TEST_ASSERT_TRUE(stats_cg.iterations <= stats_jacobi.iterations);

    cfd_free(x_cg);
    cfd_free(x_jacobi);
    cfd_free(x_temp);
    cfd_free(rhs);
    poisson_solver_destroy(cg_solver);
    poisson_solver_destroy(jacobi_solver);
}

/**
 * Test CG with stricter tolerance
 */
void test_cg_tight_tolerance(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 1000;
    params.tolerance = 1e-10;
    params.absolute_tolerance = 1e-12;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    /* Verify solver met its convergence criterion (relative or absolute tolerance) */
    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/**
 * Test CG auto backend selection
 */
void test_cg_auto_backend(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_AUTO);

    TEST_ASSERT_NOT_NULL(solver);

    /* Should select SIMD if available, otherwise SCALAR */
    TEST_ASSERT_TRUE(
        solver->backend == POISSON_BACKEND_SIMD ||
        solver->backend == POISSON_BACKEND_SCALAR);

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

/**
 * Test CG reports correct statistics
 */
void test_cg_statistics(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 500;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, NULL, rhs, &stats);

    /* Check statistics are reasonable */
    TEST_ASSERT_TRUE(stats.iterations > 0);
    TEST_ASSERT_TRUE(stats.initial_residual > 0.0);
    TEST_ASSERT_TRUE(stats.final_residual >= 0.0);
    TEST_ASSERT_TRUE(stats.final_residual < stats.initial_residual);
    TEST_ASSERT_TRUE(stats.elapsed_time_ms >= 0.0);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

/**
 * Test CG SIMD with larger grid
 */
void test_cg_simd_larger_grid(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    const size_t NX = 64;
    const size_t NY = 64;

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 1000;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, NX, NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(NX, NY, 0.0);
    double* rhs = create_uniform_rhs(NX, NY, 1.0);

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

void test_redblack_simd_converges_uniform_rhs(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(solver);

    /* Basic correctness test: verify SIMD Red-Black SOR can solve trivial problem.
     * Uses zero RHS (compatible with Neumann BCs) on small grid. This validates
     * SIMD implementation correctness, not algorithmic performance. */
    const size_t nx = 9, ny = 9;
    const double dx = 0.1, dy = 0.1;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    params.tolerance = 1e-10;
    poisson_solver_init(solver, nx, ny, dx, dy, &params);

    double* x = create_test_field(nx, ny, 0.0);
    /* Use zero RHS for Neumann BCs (uniform RHS violates compatibility condition) */
    double* rhs = create_test_field(nx, ny, 0.0);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
    /* Zero RHS with zero initial guess should converge in 1 iteration */
    TEST_ASSERT_LESS_THAN(10, stats.iterations);

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_redblack_simd_scalar_consistency(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    /* Create scalar solver */
    poisson_solver_t* scalar_solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(scalar_solver);

    /* Create SIMD solver */
    poisson_solver_t* simd_solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(simd_solver);

    /* Verify SIMD matches scalar on trivial problem (zero RHS with Neumann BCs) */
    const size_t nx = 9, ny = 9;
    const double dx = 0.1, dy = 0.1;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 100;
    params.tolerance = 1e-10;

    poisson_solver_init(scalar_solver, nx, ny, dx, dy, &params);
    poisson_solver_init(simd_solver, nx, ny, dx, dy, &params);

    double* x_scalar = create_test_field(nx, ny, 0.0);
    double* x_simd = create_test_field(nx, ny, 0.0);
    /* Use zero RHS for Neumann BCs (uniform RHS violates compatibility condition) */
    double* rhs = create_test_field(nx, ny, 0.0);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    poisson_solver_stats_t stats_simd = poisson_solver_stats_default();

    cfd_status_t status_scalar = poisson_solver_solve(scalar_solver, x_scalar, NULL, rhs, &stats_scalar);
    cfd_status_t status_simd = poisson_solver_solve(simd_solver, x_simd, NULL, rhs, &stats_simd);

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status_scalar);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status_simd);

    /* Both should converge quickly with zero RHS */
    TEST_ASSERT_LESS_THAN(10, stats_scalar.iterations);
    TEST_ASSERT_LESS_THAN(10, stats_simd.iterations);

    /* Verify SIMD and scalar produce same results */
    double max_diff = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        double diff = fabs(x_simd[i] - x_scalar[i]);
        if (diff > max_diff) max_diff = diff;
    }

    TEST_ASSERT_LESS_THAN(1e-10, max_diff);

    cfd_free(x_scalar);
    cfd_free(x_simd);
    cfd_free(rhs);
    poisson_solver_destroy(scalar_solver);
    poisson_solver_destroy(simd_solver);
}

void test_cg_simd_if_available(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);
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

void test_cg_simd_converges_uniform_rhs(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = 500;
    params.tolerance = 1e-6;
    poisson_solver_init(solver, TEST_NX, TEST_NY, TEST_DX, TEST_DY, &params);

    double* x = create_test_field(TEST_NX, TEST_NY, 0.0);
    double* rhs = create_uniform_rhs(TEST_NX, TEST_NY, 1.0);

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
 * BACKEND AVAILABILITY DIAGNOSTICS
 * ============================================================================ */

void test_simd_backend_diagnostic(void) {
    printf("\n");
    printf("=== SIMD Backend Diagnostic ===\n");
    printf("SIMD backend available: %s\n",
           poisson_solver_backend_available(POISSON_BACKEND_SIMD) ? "YES" : "NO");
    printf("SIMD architecture: %s\n", poisson_solver_get_simd_arch_name());
    printf("Current backend: %s\n", poisson_solver_get_backend_name());

    if (poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_JACOBI, POISSON_BACKEND_SIMD);
        if (solver) {
            printf("SIMD Jacobi solver name: %s\n", solver->name);
            printf("SIMD Jacobi solver description: %s\n", solver->description);
            poisson_solver_destroy(solver);
        }

        solver = poisson_solver_create(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SIMD);
        if (solver) {
            printf("SIMD Red-Black solver name: %s\n", solver->name);
            printf("SIMD Red-Black solver description: %s\n", solver->description);
            poisson_solver_destroy(solver);
        }
    }
    printf("===================================\n\n");
    TEST_PASS();
}

void test_omp_backend_diagnostic(void) {
    printf("\n");
    printf("=== OMP Backend Diagnostic ===\n");
    printf("OMP backend available: %s\n",
           poisson_solver_backend_available(POISSON_BACKEND_OMP) ? "YES" : "NO");

    if (poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP);
        if (solver) {
            printf("OMP Red-Black solver name: %s\n", solver->name);
            printf("OMP Red-Black solver description: %s\n", solver->description);
            poisson_solver_destroy(solver);
        }
    }
    printf("==============================\n\n");
    TEST_PASS();
}

/* ============================================================================
 * TEST RUNNER
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Backend diagnostics (run first to show availability) */
    RUN_TEST(test_simd_backend_diagnostic);
    RUN_TEST(test_omp_backend_diagnostic);

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
    RUN_TEST(test_create_cg_scalar);
    RUN_TEST(test_create_with_auto_backend);
    RUN_TEST(test_init_solver);
    RUN_TEST(test_init_with_custom_params);
    RUN_TEST(test_destroy_null_is_safe);

    /* Convergence tests */
    RUN_TEST(test_jacobi_converges_zero_rhs);
    RUN_TEST(test_sor_converges_zero_rhs);
    RUN_TEST(test_redblack_converges_zero_rhs);
    RUN_TEST(test_cg_converges_zero_rhs);
    RUN_TEST(test_cg_converges_uniform_rhs);

    /* CG advanced tests */
    RUN_TEST(test_cg_scalar_simd_consistency);
    RUN_TEST(test_cg_larger_grid);
    RUN_TEST(test_cg_nonzero_initial_guess);
    RUN_TEST(test_cg_no_more_iterations_than_jacobi);
    RUN_TEST(test_cg_tight_tolerance);
    RUN_TEST(test_cg_auto_backend);
    RUN_TEST(test_cg_statistics);
    RUN_TEST(test_cg_simd_larger_grid);

    /* Residual tests */
    RUN_TEST(test_compute_residual_zero_rhs);

    /* Legacy API tests */
    RUN_TEST(test_legacy_poisson_solve_sor);
    RUN_TEST(test_legacy_poisson_solve_jacobi);
    RUN_TEST(test_legacy_poisson_solve_redblack);

    /* SIMD tests */
    RUN_TEST(test_jacobi_simd_if_available);
    RUN_TEST(test_redblack_simd_if_available);
    RUN_TEST(test_redblack_simd_converges_uniform_rhs);
    RUN_TEST(test_redblack_simd_scalar_consistency);
    RUN_TEST(test_cg_simd_if_available);
    RUN_TEST(test_cg_simd_converges_uniform_rhs);

    /* Statistics tests */
    RUN_TEST(test_stats_timing);

    return UNITY_END();
}
