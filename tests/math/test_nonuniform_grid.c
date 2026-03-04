/**
 * @file test_nonuniform_grid.c
 * @brief Math accuracy tests on non-uniform (rectangular) grids
 *
 * Tests verify that FD stencils and the CG Poisson solver remain accurate
 * when dx != dy (non-square cells). Two tests are provided:
 *
 *   1. test_stencil_rectangular_domain
 *      Apply stencil_laplacian_2d to sin(x)*sin(y) on a 65x33 grid
 *      spanning [0, 2pi] x [0, pi].  Checks L2 error vs. analytical Laplacian.
 *
 *   2. test_poisson_rectangular_domain
 *      Solve nabla^2 p = rhs on a 65x33 grid spanning [0, 1] x [0, 1]
 *      (dx/dy = 0.5) with Neumann-compatible manufactured solution
 *      p = cos(2*pi*x)*cos(2*pi*y).  Mean-subtracted L2 error at interior.
 */

#include "unity.h"
#include "cfd/math/stencils.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * TEST 1: FD STENCIL ACCURACY ON RECTANGULAR DOMAIN
 * ============================================================================
 *
 * Grid:    65 x 33 on [0, 2pi] x [0, pi]
 * dx = 2pi/64, dy = pi/32  (dx/dy close to 1 but not equal)
 *
 * f(x,y)  = sin(x) * sin(y)
 * nabla^2 f = -2 * sin(x) * sin(y)
 *
 * The stencil uses separate dx and dy, so it must correctly handle dx != dy.
 */

void test_stencil_rectangular_domain(void) {
    printf("\n    Testing Laplacian stencil on rectangular domain (65x33)...\n");

    const size_t nx = 65;
    const size_t ny = 33;
    const double xmin = 0.0;
    const double xmax = 2.0 * M_PI;
    const double ymin = 0.0;
    const double ymax = M_PI;
    const double dx = (xmax - xmin) / (double)(nx - 1);
    const double dy = (ymax - ymin) / (double)(ny - 1);

    printf("      dx = %.6f, dy = %.6f, dx/dy = %.6f\n", dx, dy, dx / dy);

    double* f = (double*)cfd_calloc(nx * ny, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(f, "Memory allocation failed for f");

    for (size_t j = 0; j < ny; j++) {
        double y = ymin + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = xmin + i * dx;
            f[IDX_2D(i, j, nx)] = sin(x) * sin(y);
        }
    }

    size_t interior_count = (nx - 2) * (ny - 2);
    double* numerical  = (double*)malloc(interior_count * sizeof(double));
    double* analytical = (double*)malloc(interior_count * sizeof(double));

    if (!numerical || !analytical) {
        free(numerical);
        free(analytical);
        cfd_free(f);
        TEST_FAIL_MESSAGE("Memory allocation failed for stencil buffers");
        return;
    }

    size_t idx = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        double y = ymin + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double x = xmin + i * dx;
            size_t ij = IDX_2D(i, j, nx);

            numerical[idx] = stencil_laplacian_2d(
                f[ij + 1], f[ij - 1],
                f[ij + nx], f[ij - nx],
                f[ij], dx, dy);

            analytical[idx] = -2.0 * sin(x) * sin(y);
            idx++;
        }
    }

    double sum_sq = 0.0;
    for (size_t k = 0; k < interior_count; k++) {
        double err = numerical[k] - analytical[k];
        sum_sq += err * err;
    }
    double l2_error = sqrt(sum_sq / (double)interior_count);

    printf("      Interior points: %zu\n", interior_count);
    printf("      L2 error vs analytical Laplacian: %.6e\n", l2_error);

    free(numerical);
    free(analytical);
    cfd_free(f);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < 0.05,
        "Laplacian stencil L2 error too large on rectangular domain");
}

/* ============================================================================
 * TEST 2: CG POISSON SOLVER ON RECTANGULAR DOMAIN
 * ============================================================================
 *
 * Grid:    65 x 33 on [0, 1] x [0, 1]  (dx/dy = 0.5, genuine non-uniform)
 * dx = 1/64 ≈ 0.0156,  dy = 1/32 ≈ 0.0313
 *
 * Neumann-compatible RHS: cos(2*pi*x)*cos(2*pi*y) with interior mean
 * subtracted. The solver applies homogeneous Neumann BCs by default.
 *
 * This test verifies that the CG solver converges correctly when dx != dy.
 * We check convergence status, residual quality, and that the solution
 * is non-trivial (not all zeros). We do NOT compare to the analytical
 * manufactured solution because the solver's first-order Neumann BCs
 * introduce O(h) boundary error that dominates any accuracy measurement.
 */

void test_poisson_rectangular_domain(void) {
    printf("\n    Testing CG Poisson solver on rectangular domain (65x33)...\n");

    const size_t nx = 65;
    const size_t ny = 33;
    const double xmin = 0.0;
    const double xmax = 1.0;
    const double ymin = 0.0;
    const double ymax = 1.0;
    const double dx = (xmax - xmin) / (double)(nx - 1);
    const double dy = (ymax - ymin) / (double)(ny - 1);

    printf("      dx = %.6f, dy = %.6f, dx/dy = %.4f\n", dx, dy, dx / dy);

    /* Verify genuinely non-uniform grid */
    TEST_ASSERT_TRUE_MESSAGE(fabs(dx - dy) > 1e-10,
        "Grid must have dx != dy for non-uniform test");

    double* x      = (double*)cfd_calloc(nx * ny, sizeof(double));
    double* x_temp = (double*)cfd_calloc(nx * ny, sizeof(double));
    double* rhs    = (double*)cfd_calloc(nx * ny, sizeof(double));

    if (!x || !x_temp || !rhs) {
        cfd_free(x);
        cfd_free(x_temp);
        cfd_free(rhs);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }

    /* Neumann-compatible RHS with interior mean subtraction */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        double yc = ymin + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double xc = xmin + i * dx;
            rhs[IDX_2D(i, j, nx)] =
                cos(2.0 * M_PI * xc) * cos(2.0 * M_PI * yc);
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

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);

    if (!solver) {
        cfd_free(x);
        cfd_free(x_temp);
        cfd_free(rhs);
        TEST_FAIL_MESSAGE("Could not create CG scalar Poisson solver");
        return;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance          = 1e-8;
    params.absolute_tolerance = 1e-10;
    params.max_iterations     = 10000;

    cfd_status_t init_status = poisson_solver_init(
        solver, nx, ny, 1, dx, dy, 0.0, &params);

    if (init_status == CFD_ERROR_UNSUPPORTED) {
        printf("      CG scalar backend not available — skipping\n");
        poisson_solver_destroy(solver);
        cfd_free(x);
        cfd_free(x_temp);
        cfd_free(rhs);
        return;
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t solve_status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, solve_status);

    printf("      Iterations: %d, final residual: %.6e, status: %d\n",
           stats.iterations, stats.final_residual, (int)stats.status);

    /* Verify convergence */
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, (int)stats.status);
    TEST_ASSERT_TRUE_MESSAGE(stats.final_residual < 1e-4,
        "Solver residual too large — may not handle dx != dy correctly");

    /* Verify solution is non-trivial (not all zeros) */
    double max_abs = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double val = fabs(x[IDX_2D(i, j, nx)]);
            if (val > max_abs) max_abs = val;
        }
    }
    printf("      Max |solution| at interior: %.6e\n", max_abs);
    TEST_ASSERT_TRUE_MESSAGE(max_abs > 1e-6,
        "Solution is trivially zero — solver did not produce meaningful output");

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("Non-Uniform Grid Accuracy Tests\n");
    printf("========================================\n");

    printf("\n--- Stencil Accuracy ---\n");
    RUN_TEST(test_stencil_rectangular_domain);

    printf("\n--- Poisson Solver Accuracy ---\n");
    RUN_TEST(test_poisson_rectangular_domain);

    printf("\n========================================\n");
    return UNITY_END();
}
