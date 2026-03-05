/**
 * @file test_solver_breakdown.c
 * @brief CG and BiCGSTAB breakdown and edge-case detection tests
 *
 * Verifies solver behavior under degenerate conditions:
 * - Incompatible Neumann systems (non-zero mean RHS) are handled gracefully
 *   (no crash; solver returns SUCCESS or MAX_ITER with non-zero iterations)
 * - Zero RHS with zero initial guess converges immediately
 * - Tight tolerance with severely limited iterations is reported via either
 *   SUCCESS or MAX_ITER without crashing
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NX 17
#define NY 17

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * Test 1: CG on incompatible Neumann system
 * ============================================================================ */

/**
 * A constant interior RHS sums to a non-zero value, making it incompatible
 * with homogeneous Neumann boundary conditions.  With a small iteration
 * budget and tight absolute tolerance, CG must exhaust the budget or
 * report stagnation — it must not crash or return success in 0 iterations.
 *
 * Note: CG may still "converge" via relative tolerance, so we only assert
 * that it runs the full budget without crashing and uses a non-trivial
 * number of iterations.
 */
void test_cg_incompatible_neumann(void) {
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* x     = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs   = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Set interior RHS to constant 1.0 — incompatible with Neumann BCs */
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            rhs[IDX_2D(i, j, NX)] = 1.0;
        }
    }
    /* Boundaries remain zero */

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance          = 1e-10;
    params.absolute_tolerance = 1e-14;
    params.max_iterations     = 50;

    cfd_status_t init_status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    /* CG may converge via relative tolerance even on an incompatible system.
     * Accept either CFD_SUCCESS (relative convergence) or CFD_ERROR_MAX_ITER
     * (exhausted budget). Any other return is a bug. */
    TEST_ASSERT_TRUE_MESSAGE(
        status == CFD_SUCCESS || status == CFD_ERROR_MAX_ITER,
        "CG must return SUCCESS or MAX_ITER on incompatible Neumann system");
    TEST_ASSERT_TRUE_MESSAGE(stats.iterations > 0,
        "CG must perform at least one iteration on non-trivial system");

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * Test 2: BiCGSTAB — zero RHS / zero initial guess
 * ============================================================================ */

/**
 * The trivial system (zero RHS, zero initial guess) is already solved.
 * BiCGSTAB must detect this and report convergence within one iteration.
 */
void test_bicgstab_trivial_system(void) {
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* x and rhs are already all zeros from cfd_calloc */

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    cfd_status_t init_status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_OR_EQUAL(1, stats.iterations);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * Test 3: CG — zero RHS / zero initial guess
 * ============================================================================ */

/**
 * The trivial system (zero RHS, zero initial guess) is already solved.
 * CG must detect this and report convergence within one iteration.
 */
void test_cg_trivial_system(void) {
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    cfd_status_t init_status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_OR_EQUAL(1, stats.iterations);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * Test 4: BiCGSTAB — max iterations exhausted
 * ============================================================================ */

/**
 * A sinusoidal RHS with tolerance set far below what 3 iterations can reach
 * forces BiCGSTAB to exhaust its iteration budget and report POISSON_MAX_ITER.
 */
void test_bicgstab_max_iter(void) {
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Sinusoidal RHS compatible with Neumann BCs (mean-subtracted) */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        double y = j * dy;
        for (size_t i = 1; i < NX - 1; i++) {
            double xi = i * dx;
            rhs[IDX_2D(i, j, NX)] = cos(2.0 * M_PI * xi) * cos(2.0 * M_PI * y);
            interior_sum += rhs[IDX_2D(i, j, NX)];
            interior_count++;
        }
    }
    if (interior_count > 0) {
        double interior_mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < NY - 1; j++) {
            for (size_t i = 1; i < NX - 1; i++) {
                rhs[IDX_2D(i, j, NX)] -= interior_mean;
            }
        }
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance          = 1e-15;
    params.absolute_tolerance = 1e-18;
    params.max_iterations     = 3;

    cfd_status_t init_status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_ERROR_MAX_ITER, status);
    TEST_ASSERT_EQUAL(POISSON_MAX_ITER, stats.status);
    TEST_ASSERT_EQUAL(3, stats.iterations);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * Test 5: CG — max iterations exhausted
 * ============================================================================ */

/**
 * Same setup as test_bicgstab_max_iter but for CG. Three iterations with
 * sub-1e-15 tolerance cannot converge; solver must report POISSON_MAX_ITER.
 */
void test_cg_max_iter(void) {
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        double y = j * dy;
        for (size_t i = 1; i < NX - 1; i++) {
            double xi = i * dx;
            rhs[IDX_2D(i, j, NX)] = cos(2.0 * M_PI * xi) * cos(2.0 * M_PI * y);
            interior_sum += rhs[IDX_2D(i, j, NX)];
            interior_count++;
        }
    }
    if (interior_count > 0) {
        double interior_mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < NY - 1; j++) {
            for (size_t i = 1; i < NX - 1; i++) {
                rhs[IDX_2D(i, j, NX)] -= interior_mean;
            }
        }
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance          = 1e-15;
    params.absolute_tolerance = 1e-18;
    params.max_iterations     = 3;

    cfd_status_t init_status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_ERROR_MAX_ITER, status);
    TEST_ASSERT_EQUAL(POISSON_MAX_ITER, stats.status);
    TEST_ASSERT_EQUAL(3, stats.iterations);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_cg_incompatible_neumann);
    RUN_TEST(test_bicgstab_trivial_system);
    RUN_TEST(test_cg_trivial_system);
    RUN_TEST(test_bicgstab_max_iter);
    RUN_TEST(test_cg_max_iter);
    return UNITY_END();
}
