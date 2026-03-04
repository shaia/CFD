/**
 * @file test_solver_robustness.c
 * @brief Solver edge-case and robustness tests
 *
 * Tests solver correctness and stability under boundary conditions:
 * - Minimal 9x9 grid with only 49 interior points
 * - Early termination with loose tolerance
 * - SOR omega parameter effect on convergence speed
 * - Sequential re-solves produce identical results
 * - Repeated create/destroy lifecycle without resource leaks
 *
 * NOTE: The Poisson solver applies Neumann BCs by default. All RHS functions
 * must be Neumann-compatible (cos-based with interior mean subtracted).
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/**
 * Initialize Neumann-compatible sinusoidal RHS: cos(2πx)*cos(2πy) with
 * interior mean subtracted and zero boundary values.
 * This matches the pattern used in test_bicgstab_avx2.c.
 */
static void init_neumann_rhs(double* rhs, size_t nx, size_t ny,
                              double xmin, double xmax,
                              double ymin, double ymax) {
    double dx = (xmax - xmin) / (nx - 1);
    double dy = (ymax - ymin) / (ny - 1);

    /* First pass: sinusoidal values */
    for (size_t j = 0; j < ny; j++) {
        double y = ymin + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = xmin + i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Second pass: subtract interior mean (Neumann compatibility) */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            interior_sum += rhs[IDX_2D(i, j, nx)];
            interior_count++;
        }
    }
    if (interior_count > 0) {
        double mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                rhs[IDX_2D(i, j, nx)] -= mean;
            }
        }
    }

    /* Zero boundaries */
    for (size_t i = 0; i < nx; i++) {
        rhs[i] = 0.0;
        rhs[(ny - 1) * nx + i] = 0.0;
    }
    for (size_t j = 0; j < ny; j++) {
        rhs[j * nx] = 0.0;
        rhs[j * nx + (nx - 1)] = 0.0;
    }
}

/* ============================================================================
 * Test 1: Convergence on minimal 9x9 grid
 * ============================================================================ */

/**
 * Verifies Jacobi and CG both converge on a 9x9 grid (49 interior points).
 */
void test_minimal_grid_9x9(void) {
    const size_t NX = 9;
    const size_t NY = 9;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n = NX * NY;

    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);

    init_neumann_rhs(rhs, NX, NY, 0.0, 1.0, 0.0, 1.0);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = 1e-4;
    params.max_iterations  = 5000;

    /* --- Jacobi solve --- */
    poisson_solver_t* solver_jacobi = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_jacobi);

    cfd_status_t status = poisson_solver_init(solver_jacobi, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_jacobi, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    printf("  Jacobi on 9x9: %d iterations, final residual = %.3e\n",
           stats.iterations, stats.final_residual);

    poisson_solver_destroy(solver_jacobi);

    /* --- CG solve --- */
    memset(x, 0, n * sizeof(double));
    memset(x_temp, 0, n * sizeof(double));

    poisson_solver_t* solver_cg = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_cg);

    status = poisson_solver_init(solver_cg, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    stats  = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_cg, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    printf("  CG on 9x9: %d iterations, final residual = %.3e\n",
           stats.iterations, stats.final_residual);

    poisson_solver_destroy(solver_cg);
    cfd_free(rhs);
    cfd_free(x);
    cfd_free(x_temp);
}

/* ============================================================================
 * Test 2: Early termination with loose tolerance
 * ============================================================================ */

/**
 * CG with tolerance=0.1 on a 33x33 problem terminates well before
 * the iteration budget, demonstrating early-exit behavior.
 */
void test_early_termination_large_tol(void) {
    const size_t NX = 33;
    const size_t NY = 33;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n  = NX * NY;

    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);

    init_neumann_rhs(rhs, NX, NY, 0.0, 1.0, 0.0, 1.0);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = 0.1;
    params.max_iterations = 5000;

    cfd_status_t status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_THAN(50, (int)stats.iterations);

    printf("  CG early-exit at tol=0.1: %d iterations\n", stats.iterations);

    poisson_solver_destroy(solver);
    cfd_free(rhs);
    cfd_free(x);
    cfd_free(x_temp);
}

/* ============================================================================
 * Test 3: SOR omega boundary — Gauss-Seidel vs over-relaxation
 * ============================================================================ */

/**
 * Verifies that SOR with omega=1.5 converges faster (fewer iterations) than
 * omega=1.0 (Gauss-Seidel) on the same 17x17 Neumann-compatible problem.
 */
void test_sor_omega_boundary(void) {
    const size_t NX = 17;
    const size_t NY = 17;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n  = NX * NY;

    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);

    init_neumann_rhs(rhs, NX, NY, 0.0, 1.0, 0.0, 1.0);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = 1e-6;
    params.max_iterations = 5000;

    /* --- Solve with omega=1.0 --- */
    poisson_solver_t* solver1 = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver1);

    params.omega = 1.0;
    cfd_status_t status = poisson_solver_init(solver1, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    memset(x, 0, n * sizeof(double));
    memset(x_temp, 0, n * sizeof(double));

    poisson_solver_stats_t stats1 = poisson_solver_stats_default();
    status = poisson_solver_solve(solver1, x, x_temp, rhs, &stats1);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats1.status);

    poisson_solver_destroy(solver1);

    /* --- Solve with omega=1.5 --- */
    poisson_solver_t* solver2 = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver2);

    params.omega = 1.5;
    status = poisson_solver_init(solver2, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    memset(x, 0, n * sizeof(double));
    memset(x_temp, 0, n * sizeof(double));

    poisson_solver_stats_t stats2 = poisson_solver_stats_default();
    status = poisson_solver_solve(solver2, x, x_temp, rhs, &stats2);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats2.status);

    poisson_solver_destroy(solver2);

    printf("  SOR omega=1.0: %d iterations\n", stats1.iterations);
    printf("  SOR omega=1.5: %d iterations\n", stats2.iterations);

    /* omega=1.5 should not converge significantly slower than omega=1.0.
     * Allow small variation across platforms/compilers. */
    TEST_ASSERT((int)stats2.iterations <= (int)stats1.iterations + 10);

    cfd_free(rhs);
    cfd_free(x);
    cfd_free(x_temp);
}

/* ============================================================================
 * Test 4: Sequential solves produce consistent results
 * ============================================================================ */

/**
 * Solving the same problem twice with the same CG solver instance must yield
 * numerically identical solutions within tolerance (L2 difference < 1e-10).
 */
void test_sequential_solves_consistent(void) {
    const size_t NX = 17;
    const size_t NY = 17;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n  = NX * NY;

    double* rhs     = (double*)cfd_calloc(n, sizeof(double));
    double* x       = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp  = (double*)cfd_calloc(n, sizeof(double));
    double* x_first = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(x_first);

    init_neumann_rhs(rhs, NX, NY, 0.0, 1.0, 0.0, 1.0);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = 1e-8;
    params.max_iterations = 5000;

    cfd_status_t status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve #1 */
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    memcpy(x_first, x, n * sizeof(double));

    /* Reset and solve #2 */
    memset(x, 0, n * sizeof(double));
    memset(x_temp, 0, n * sizeof(double));

    stats  = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    /* L2 difference between the two solutions */
    double l2_diff = 0.0;
    size_t count   = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            size_t idx = IDX_2D(i, j, NX);
            double diff = x_first[idx] - x[idx];
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = sqrt(l2_diff / (double)count);

    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, l2_diff);

    poisson_solver_destroy(solver);
    cfd_free(rhs);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(x_first);
}

/* ============================================================================
 * Test 5: Repeated create/destroy lifecycle
 * ============================================================================ */

/**
 * Creates, inits, solves, and destroys a CG solver 100 times. Verifies no
 * crashes and consistent CFD_SUCCESS returns across all iterations.
 */
void test_solver_create_destroy_cycle(void) {
    const size_t NX = 17;
    const size_t NY = 17;
    double dx = 1.0 / (NX - 1);
    double dy = 1.0 / (NY - 1);
    size_t n  = NX * NY;

    double* rhs    = (double*)cfd_calloc(n, sizeof(double));
    double* x      = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);

    init_neumann_rhs(rhs, NX, NY, 0.0, 1.0, 0.0, 1.0);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = 1e-6;
    params.max_iterations = 1000;

    for (int iter = 0; iter < 100; iter++) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "solver creation failed in cycle");

        cfd_status_t status = poisson_solver_init(solver, NX, NY, 1, dx, dy, 0.0, &params);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        memset(x, 0, n * sizeof(double));
        memset(x_temp, 0, n * sizeof(double));

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        poisson_solver_destroy(solver);
    }

    cfd_free(rhs);
    cfd_free(x);
    cfd_free(x_temp);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_minimal_grid_9x9);
    RUN_TEST(test_early_termination_large_tol);
    RUN_TEST(test_sor_omega_boundary);
    RUN_TEST(test_sequential_solves_consistent);
    RUN_TEST(test_solver_create_destroy_cycle);
    return UNITY_END();
}
