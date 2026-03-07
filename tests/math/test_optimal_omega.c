/**
 * @file test_optimal_omega.c
 * @brief Tests for auto-computed optimal SOR omega
 *
 * Verifies that the optimal omega formula produces correct values and
 * that Red-Black SOR converges reliably on grids where it previously failed
 * with the hard-coded omega=1.5 default.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2*pi*x) * cos(2*pi*y) with discrete interior mean subtracted.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy)
{
    for (size_t j = 0; j < ny; j++) {
        double y = j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Subtract interior mean for Neumann compatibility */
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += rhs[IDX_2D(i, j, nx)];
            count++;
        }
    }
    double mean = sum / (double)count;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            rhs[IDX_2D(i, j, nx)] -= mean;
        }
    }
}

/**
 * Helper: solve Poisson with given method/backend on NxN grid, return stats.
 * Returns CFD_SUCCESS if converged, or error status.
 */
static cfd_status_t solve_poisson_test(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend,
    size_t nx, size_t ny,
    int max_iterations,
    poisson_solver_stats_t* out_stats)
{
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    size_t n = nx * ny;

    double* x = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    if (!x || !x_temp || !rhs) {
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return CFD_ERROR_NOMEM;
    }

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-6;
    params.max_iterations = max_iterations;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return status;
    }

    *out_stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, out_stats);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    return status;
}

/**
 * Test: RB-SOR scalar converges on 33x33 within 1000 iterations with auto-omega.
 * This was the configuration that previously failed with omega=1.5.
 */
void test_redblack_scalar_33x33_converges(void) {
    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
        33, 33, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    /* With optimal omega ~1.83, should converge in well under 1000 iterations */
    TEST_ASSERT_LESS_THAN(500, stats.iterations);
}

/**
 * Test: RB-SOR OMP converges on 33x33 within 1000 iterations with auto-omega.
 */
void test_redblack_omp_33x33_converges(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP);
    if (!solver) {
        TEST_IGNORE_MESSAGE("OMP Red-Black SOR solver not available");
        return;
    }
    poisson_solver_destroy(solver);

    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP,
        33, 33, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_THAN(500, stats.iterations);
}

/**
 * Test: RB-SOR scalar converges across multiple grid sizes.
 */
void test_redblack_scalar_multi_grid(void) {
    size_t grid_sizes[] = {17, 33, 65};
    for (int g = 0; g < 3; g++) {
        size_t n = grid_sizes[g];
        poisson_solver_stats_t stats;
        cfd_status_t status = solve_poisson_test(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
            n, n, 1000, &stats);

        char msg[128];
        snprintf(msg, sizeof(msg), "Failed to converge on %zux%zu grid", n, n);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, msg);
        TEST_ASSERT_EQUAL_MESSAGE(POISSON_CONVERGED, stats.status, msg);
    }
}

/**
 * Test: RB-SOR converges on non-square grid (33x65).
 */
void test_redblack_scalar_nonsquare(void) {
    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
        33, 65, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
}

/**
 * Test: Explicit omega override is respected (omega=1.2 should converge
 * slower than auto-optimal).
 */
void test_explicit_omega_override(void) {
    size_t nx = 33, ny = 33;
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    size_t n = nx * ny;

    double* x1 = (double*)cfd_calloc(n, sizeof(double));
    double* x2 = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x1);
    TEST_ASSERT_NOT_NULL(x2);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    /* Solve with auto-omega (default) */
    poisson_solver_t* solver_auto = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_auto);

    poisson_solver_params_t params_auto = poisson_solver_params_default();
    params_auto.tolerance = 1e-6;
    params_auto.max_iterations = 2000;
    cfd_status_t status = poisson_solver_init(solver_auto, nx, ny, 1, dx, dy, 0.0, &params_auto);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_auto = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_auto, x1, x_temp, rhs, &stats_auto);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve with explicit suboptimal omega=1.2 */
    poisson_solver_t* solver_explicit = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_explicit);

    poisson_solver_params_t params_explicit = poisson_solver_params_default();
    params_explicit.tolerance = 1e-6;
    params_explicit.max_iterations = 2000;
    params_explicit.omega = 1.2;  /* Explicit suboptimal value */
    status = poisson_solver_init(solver_explicit, nx, ny, 1, dx, dy, 0.0, &params_explicit);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_explicit = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_explicit, x2, x_temp, rhs, &stats_explicit);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Explicit omega=1.2 should need more iterations than auto-optimal */
    TEST_ASSERT_GREATER_THAN(stats_auto.iterations, stats_explicit.iterations);

    poisson_solver_destroy(solver_auto);
    poisson_solver_destroy(solver_explicit);
    cfd_free(x1);
    cfd_free(x2);
    cfd_free(x_temp);
    cfd_free(rhs);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_redblack_scalar_33x33_converges);
    RUN_TEST(test_redblack_omp_33x33_converges);
    RUN_TEST(test_redblack_scalar_multi_grid);
    RUN_TEST(test_redblack_scalar_nonsquare);
    RUN_TEST(test_explicit_omega_override);
    return UNITY_END();
}
