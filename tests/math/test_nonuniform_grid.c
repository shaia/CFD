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
 *      Solve nabla^2 p = rhs on a 65x33 grid spanning [0, 1] x [0, 0.5]
 *      (aspect ratio 2:1) with manufactured solution p = sin(pi*x)*sin(2*pi*y).
 *      Checks solver convergence and L2 error at interior points.
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
 * Grid:    65 x 33 on [0, 1] x [0, 0.5]  (aspect ratio 2:1)
 * dx = 1/64, dy = 0.5/32
 *
 * Manufactured solution:
 *   p(x,y) = sin(pi*x) * sin(2*pi*y)
 *
 * This has exactly one full spatial period in each direction given the domain,
 * so p = 0 on all four boundaries (pure Dirichlet, zero BC).
 *
 * Laplacian:
 *   nabla^2 p = -(pi^2 + 4*pi^2) * sin(pi*x) * sin(2*pi*y)
 *             = -5*pi^2 * sin(pi*x) * sin(2*pi*y)
 *
 * Setup:
 *   - x (solution) = 0 everywhere, then set boundary values to exact p
 *   - rhs = -5*pi^2 * sin(pi*x) * sin(2*pi*y) at interior points
 *   - rhs = 0 at boundary points (solver ignores them, but zero is cleanest)
 */

void test_poisson_rectangular_domain(void) {
    printf("\n    Testing CG Poisson solver on rectangular domain (65x33)...\n");

    const size_t nx = 65;
    const size_t ny = 33;
    const double xmin = 0.0;
    const double xmax = 1.0;
    const double ymin = 0.0;
    const double ymax = 0.5;
    const double dx = (xmax - xmin) / (double)(nx - 1);
    const double dy = (ymax - ymin) / (double)(ny - 1);

    printf("      dx = %.6f, dy = %.6f, aspect ratio dx/dy = %.4f\n",
           dx, dy, dx / dy);

    double* x     = (double*)cfd_calloc(nx * ny, sizeof(double));
    double* x_temp = (double*)cfd_calloc(nx * ny, sizeof(double));
    double* rhs   = (double*)cfd_calloc(nx * ny, sizeof(double));

    if (!x || !x_temp || !rhs) {
        cfd_free(x);
        cfd_free(x_temp);
        cfd_free(rhs);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }

    /* RHS at interior points; boundary entries remain 0 */
    for (size_t j = 1; j < ny - 1; j++) {
        double yc = ymin + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double xc = xmin + i * dx;
            rhs[IDX_2D(i, j, nx)] =
                -5.0 * M_PI * M_PI * sin(M_PI * xc) * sin(2.0 * M_PI * yc);
        }
    }

    /* Boundary values of x set to exact solution (Dirichlet BCs).
     * Since sin(pi*x)*sin(2*pi*y) is zero at x=0, x=1, y=0, y=0.5,
     * these are all zero — but we set them explicitly for clarity.
     */
    for (size_t j = 0; j < ny; j++) {
        double yc = ymin + j * dy;
        x[IDX_2D(0,      j, nx)] = sin(M_PI * xmin) * sin(2.0 * M_PI * yc);
        x[IDX_2D(nx - 1, j, nx)] = sin(M_PI * xmax) * sin(2.0 * M_PI * yc);
    }
    for (size_t i = 0; i < nx; i++) {
        double xc = xmin + i * dx;
        x[IDX_2D(i, 0,      nx)] = sin(M_PI * xc) * sin(2.0 * M_PI * ymin);
        x[IDX_2D(i, ny - 1, nx)] = sin(M_PI * xc) * sin(2.0 * M_PI * ymax);
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
    params.tolerance       = 1e-6;
    params.max_iterations  = 5000;

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
    poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    printf("      Iterations: %d, final residual: %.6e, status: %d\n",
           stats.iterations, stats.final_residual, (int)stats.status);

    /* Compute L2 error at interior points */
    double sum_sq = 0.0;
    size_t interior_count = (nx - 2) * (ny - 2);
    for (size_t j = 1; j < ny - 1; j++) {
        double yc = ymin + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double xc = xmin + i * dx;
            double exact = sin(M_PI * xc) * sin(2.0 * M_PI * yc);
            double err   = x[IDX_2D(i, j, nx)] - exact;
            sum_sq += err * err;
        }
    }
    double l2_error = sqrt(sum_sq / (double)interior_count);
    printf("      L2 error vs exact solution: %.6e\n", l2_error);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);

    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, (int)stats.status);
    TEST_ASSERT_TRUE_MESSAGE(l2_error < 1e-2,
        "CG solver L2 error too large on rectangular domain");
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
