/**
 * @file test_finite_differences.c
 * @brief Unit tests for finite difference stencil accuracy
 *
 * These tests verify O(h^2) accuracy for the standard central difference
 * stencils defined in cfd/math/stencils.h. This validates the mathematical
 * correctness of the stencil formulas.
 *
 * NOTE: These are standalone mathematical validation tests. The production
 * solver code (solver_explicit_euler.c, solver_projection.c, etc.) currently
 * implements stencils inline and does NOT use cfd/math/stencils.h. For tests
 * that exercise actual production code paths, see the Taylor-Green vortex
 * validation tests which run the full solver.
 *
 * Tests cover:
 *   - First derivative (central difference)
 *   - Second derivative (central difference)
 *   - 2D Laplacian (5-point stencil)
 *   - Divergence operator
 *   - Gradient operator
 *
 * Test approach: Use smooth analytical functions (e.g., sin(kx)*sin(ky)),
 * compute numerical derivatives using the stencil functions, compare to
 * analytical derivatives, verify O(h^2) error scaling.
 */

#include "unity.h"
#include "cfd/math/stencils.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Helper macro to check two allocations and free both if either fails */
#define CHECK_ALLOC2(ptr1, ptr2) \
    do { \
        if ((ptr1) == NULL || (ptr2) == NULL) { \
            free(ptr1); \
            free(ptr2); \
            TEST_FAIL_MESSAGE("Memory allocation failed"); \
            return; \
        } \
    } while (0)

/* Helper macro to check four allocations and free all if any fails */
#define CHECK_ALLOC4(ptr1, ptr2, ptr3, ptr4) \
    do { \
        if ((ptr1) == NULL || (ptr2) == NULL || (ptr3) == NULL || (ptr4) == NULL) { \
            free(ptr1); \
            free(ptr2); \
            free(ptr3); \
            free(ptr4); \
            TEST_FAIL_MESSAGE("Memory allocation failed"); \
            return; \
        } \
    } while (0)

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

/* Wavenumber for test functions */
#define TEST_K 1.0

/* Tolerances */
#define CONVERGENCE_RATE_TOL 0.3  /* Allow 0.3 deviation from expected order */
#define ABSOLUTE_ERROR_TOL 0.05   /* Max error at finest grid */

/* ============================================================================
 * ANALYTICAL FUNCTIONS AND DERIVATIVES
 * ============================================================================ */

/**
 * Scalar test function: f(x,y) = sin(kx) * sin(ky)
 */
static inline double test_f(double x, double y, double k) {
    return sin(k * x) * sin(k * y);
}

/**
 * Analytical first derivative: df/dx = k * cos(kx) * sin(ky)
 */
static inline double test_df_dx(double x, double y, double k) {
    return k * cos(k * x) * sin(k * y);
}

/**
 * Analytical first derivative: df/dy = k * sin(kx) * cos(ky)
 */
static inline double test_df_dy(double x, double y, double k) {
    return k * sin(k * x) * cos(k * y);
}

/**
 * Analytical second derivative: d2f/dx2 = -k^2 * sin(kx) * sin(ky)
 */
static inline double test_d2f_dx2(double x, double y, double k) {
    return -k * k * sin(k * x) * sin(k * y);
}

/**
 * Analytical second derivative: d2f/dy2 = -k^2 * sin(kx) * sin(ky)
 */
static inline double test_d2f_dy2(double x, double y, double k) {
    return -k * k * sin(k * x) * sin(k * y);
}

/**
 * Analytical Laplacian: nabla^2 f = d2f/dx2 + d2f/dy2 = -2k^2 * sin(kx) * sin(ky)
 */
static inline double test_laplacian(double x, double y, double k) {
    return -2.0 * k * k * sin(k * x) * sin(k * y);
}

/**
 * Vector field u-component: u(x,y) = sin(kx) * cos(ky)
 */
static inline double test_u(double x, double y, double k) {
    return sin(k * x) * cos(k * y);
}

/**
 * Vector field v-component: v(x,y) = -cos(kx) * sin(ky)
 * Note: This is chosen so that div(u,v) = 0 (divergence-free)
 */
static inline double test_v(double x, double y, double k) {
    return -cos(k * x) * sin(k * y);
}

/**
 * Analytical divergence: div(u,v) = du/dx + dv/dy
 * For our test field: du/dx = k*cos(kx)*cos(ky), dv/dy = -k*cos(kx)*cos(ky)
 * So div = 0 (divergence-free field)
 */
static inline double test_divergence(double x, double y, double k) {
    (void)x; (void)y; (void)k;  /* Suppress unused parameter warnings */
    return 0.0;  /* Divergence-free by construction */
}

/**
 * Non-zero divergence test field: u = sin(2x)*sin(y), v = sin(x)*sin(2y)
 * du/dx = 2*cos(2x)*sin(y)
 * dv/dy = 2*sin(x)*cos(2y)
 * div = 2*cos(2x)*sin(y) + 2*sin(x)*cos(2y)
 */
static inline double test_u_nonzero_div(double x, double y) {
    return sin(2.0 * x) * sin(y);
}

static inline double test_v_nonzero_div(double x, double y) {
    return sin(x) * sin(2.0 * y);
}

static inline double test_divergence_nonzero(double x, double y) {
    return 2.0 * cos(2.0 * x) * sin(y) + 2.0 * sin(x) * cos(2.0 * y);
}

/* ============================================================================
 * STENCIL FUNCTION ALIASES
 * ============================================================================
 * Use the shared stencil implementations from cfd/math/stencils.h
 * These aliases maintain backward compatibility with existing test code.
 */

#define fd_first_deriv_x  stencil_first_deriv_x
#define fd_first_deriv_y  stencil_first_deriv_y
#define fd_second_deriv_x stencil_second_deriv_x
#define fd_second_deriv_y stencil_second_deriv_y
#define fd_laplacian      stencil_laplacian_2d

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Compute L2 error norm over interior points
 */
static double compute_l2_error(const double* numerical, const double* analytical, size_t n) {
    if (n == 0) return 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = numerical[i] - analytical[i];
        sum_sq += err * err;
    }
    return sqrt(sum_sq / n);
}

/**
 * Compute max (L-infinity) error
 */
static double compute_max_error(const double* numerical, const double* analytical, size_t n) {
    if (n == 0) return 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs(numerical[i] - analytical[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/**
 * Compute convergence rate from two error values and grid spacings
 * rate = log(e_coarse/e_fine) / log(h_coarse/h_fine)
 *
 * @param e_coarse  Error on coarser grid
 * @param e_fine    Error on finer grid
 * @param h_coarse  Grid spacing on coarser grid
 * @param h_fine    Grid spacing on finer grid
 * @return Estimated convergence order
 */
static double compute_convergence_rate(double e_coarse, double e_fine,
                                       double h_coarse, double h_fine) {
    if (e_fine < 1e-15 || e_coarse < 1e-15) return 0.0;
    if (h_fine < 1e-15 || h_coarse < 1e-15) return 0.0;
    return log(e_coarse / e_fine) / log(h_coarse / h_fine);
}

/* ============================================================================
 * UNITY SETUP
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * FIRST DERIVATIVE TESTS
 * ============================================================================ */

/**
 * Test first derivative df/dx accuracy
 */
void test_first_derivative_x_accuracy(void) {
    printf("\n    Testing first derivative df/dx accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = dx;

        /* Allocate arrays for interior points */
        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                /* Get function values for stencil */
                double f_ip1 = test_f(x + dx, y, k);
                double f_im1 = test_f(x - dx, y, k);

                /* Compute numerical derivative */
                numerical[idx] = fd_first_deriv_x(f_ip1, f_im1, dx);

                /* Compute analytical derivative */
                analytical[idx] = test_df_dx(x, y, k);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "First derivative df/dx convergence rate below O(h^2)");
    }

    /* Verify absolute error at finest grid */
    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "First derivative df/dx error too large at finest grid");
}

/**
 * Test first derivative df/dy accuracy
 */
void test_first_derivative_y_accuracy(void) {
    printf("\n    Testing first derivative df/dy accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = dy;

        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                double f_jp1 = test_f(x, y + dy, k);
                double f_jm1 = test_f(x, y - dy, k);

                numerical[idx] = fd_first_deriv_y(f_jp1, f_jm1, dy);
                analytical[idx] = test_df_dy(x, y, k);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "First derivative df/dy convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "First derivative df/dy error too large at finest grid");
}

/* ============================================================================
 * SECOND DERIVATIVE TESTS
 * ============================================================================ */

/**
 * Test second derivative d2f/dx2 accuracy
 */
void test_second_derivative_x_accuracy(void) {
    printf("\n    Testing second derivative d2f/dx2 accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = dx;

        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                double f_ip1 = test_f(x + dx, y, k);
                double f_i = test_f(x, y, k);
                double f_im1 = test_f(x - dx, y, k);

                numerical[idx] = fd_second_deriv_x(f_ip1, f_i, f_im1, dx);
                analytical[idx] = test_d2f_dx2(x, y, k);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Second derivative d2f/dx2 convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Second derivative d2f/dx2 error too large at finest grid");
}

/**
 * Test second derivative d2f/dy2 accuracy
 */
void test_second_derivative_y_accuracy(void) {
    printf("\n    Testing second derivative d2f/dy2 accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = dy;

        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                double f_jp1 = test_f(x, y + dy, k);
                double f_j = test_f(x, y, k);
                double f_jm1 = test_f(x, y - dy, k);

                numerical[idx] = fd_second_deriv_y(f_jp1, f_j, f_jm1, dy);
                analytical[idx] = test_d2f_dy2(x, y, k);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Second derivative d2f/dy2 convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Second derivative d2f/dy2 error too large at finest grid");
}

/* ============================================================================
 * LAPLACIAN TESTS
 * ============================================================================ */

/**
 * Test 2D Laplacian (5-point stencil) accuracy
 */
void test_laplacian_5point_accuracy(void) {
    printf("\n    Testing 2D Laplacian (5-point stencil) accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = (dx > dy) ? dx : dy;  /* Use max(dx, dy) for 2D stencils */

        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                double f_ip1 = test_f(x + dx, y, k);
                double f_im1 = test_f(x - dx, y, k);
                double f_jp1 = test_f(x, y + dy, k);
                double f_jm1 = test_f(x, y - dy, k);
                double f_ij = test_f(x, y, k);

                numerical[idx] = fd_laplacian(f_ip1, f_im1, f_jp1, f_jm1, f_ij, dx, dy);
                analytical[idx] = test_laplacian(x, y, k);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Laplacian convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Laplacian error too large at finest grid");
}

/**
 * Test Laplacian with non-square grid spacing
 */
void test_laplacian_nonsquare_grid(void) {
    printf("\n    Testing Laplacian with non-square grid (dx != dy)...\n");

    double k = TEST_K;
    size_t nx = 65;
    size_t ny = 33;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = M_PI;  /* Different domain size */

    double dx = (xmax - xmin) / (nx - 1);
    double dy = (ymax - ymin) / (ny - 1);

    printf("      Grid: %zux%zu, dx=%.4f, dy=%.4f\n", nx, ny, dx, dy);

    size_t interior_count = (nx - 2) * (ny - 2);
    double* numerical = malloc(interior_count * sizeof(double));
    double* analytical = malloc(interior_count * sizeof(double));
    CHECK_ALLOC2(numerical, analytical);

    size_t idx = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        double y = ymin + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double x = xmin + i * dx;

            double f_ip1 = test_f(x + dx, y, k);
            double f_im1 = test_f(x - dx, y, k);
            double f_jp1 = test_f(x, y + dy, k);
            double f_jm1 = test_f(x, y - dy, k);
            double f_ij = test_f(x, y, k);

            numerical[idx] = fd_laplacian(f_ip1, f_im1, f_jp1, f_jm1, f_ij, dx, dy);
            analytical[idx] = test_laplacian(x, y, k);

            idx++;
        }
    }

    double l2_error = compute_l2_error(numerical, analytical, interior_count);
    double max_error = compute_max_error(numerical, analytical, interior_count);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    /* Allow slightly higher tolerance for non-square grid */
    TEST_ASSERT_TRUE_MESSAGE(l2_error < ABSOLUTE_ERROR_TOL * 2.0,
        "Laplacian error too large on non-square grid");

    free(numerical);
    free(analytical);
}

/* ============================================================================
 * DIVERGENCE TESTS
 * ============================================================================ */

/**
 * Test divergence operator with divergence-free field
 */
void test_divergence_free_field(void) {
    printf("\n    Testing divergence of divergence-free field...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t n = 65;
    double dx = (xmax - xmin) / (n - 1);
    double dy = (ymax - ymin) / (n - 1);

    double max_div = 0.0;
    double l2_sum = 0.0;
    size_t count = 0;

    for (size_t j = 1; j < n - 1; j++) {
        double y = ymin + j * dy;
        for (size_t i = 1; i < n - 1; i++) {
            double x = xmin + i * dx;

            /* Compute du/dx */
            double u_ip1 = test_u(x + dx, y, k);
            double u_im1 = test_u(x - dx, y, k);
            double du_dx = fd_first_deriv_x(u_ip1, u_im1, dx);

            /* Compute dv/dy */
            double v_jp1 = test_v(x, y + dy, k);
            double v_jm1 = test_v(x, y - dy, k);
            double dv_dy = fd_first_deriv_y(v_jp1, v_jm1, dy);

            /* Divergence = du/dx + dv/dy */
            double div = du_dx + dv_dy;
            double analytical = test_divergence(x, y, k);  /* Should be 0 */

            double err = fabs(div - analytical);
            if (err > max_div) max_div = err;
            l2_sum += err * err;
            count++;
        }
    }

    double l2_error = sqrt(l2_sum / count);
    printf("      Max divergence: %.6e\n", max_div);
    printf("      L2 divergence: %.6e\n", l2_error);

    /* Divergence-free field should have small divergence numerically.
     * Use O(h²) tolerance since central differences are 2nd order accurate.
     * The specific test field has error cancellation, so actual error is
     * typically much smaller, but we use a principled tolerance. */
    double h2_tolerance = dx * dx;
    printf("      Tolerance (h²): %.6e\n", h2_tolerance);
    TEST_ASSERT_TRUE_MESSAGE(max_div < h2_tolerance,
        "Divergence of divergence-free field exceeds O(h²) tolerance");
}

/**
 * Test divergence operator with non-zero divergence field
 */
void test_divergence_nonzero_accuracy(void) {
    printf("\n    Testing divergence accuracy with non-zero divergence field...\n");

    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings[s] = (dx > dy) ? dx : dy;

        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                /* Compute du/dx for u = sin(2x)*sin(y) */
                double u_ip1 = test_u_nonzero_div(x + dx, y);
                double u_im1 = test_u_nonzero_div(x - dx, y);
                double du_dx = fd_first_deriv_x(u_ip1, u_im1, dx);

                /* Compute dv/dy for v = sin(x)*sin(2y) */
                double v_jp1 = test_v_nonzero_div(x, y + dy);
                double v_jm1 = test_v_nonzero_div(x, y - dy);
                double dv_dy = fd_first_deriv_y(v_jp1, v_jm1, dy);

                numerical[idx] = du_dx + dv_dy;
                analytical[idx] = test_divergence_nonzero(x, y);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Divergence operator convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Divergence operator error too large at finest grid");
}

/* ============================================================================
 * GRADIENT TESTS
 * ============================================================================ */

/**
 * Test gradient operator accuracy (both components)
 */
void test_gradient_accuracy(void) {
    printf("\n    Testing gradient operator accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors_x[3], errors_y[3];
    double spacings_x[3], spacings_y[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        spacings_x[s] = dx;
        spacings_y[s] = dy;

        size_t interior_count = (n - 2) * (n - 2);
        double* num_grad_x = malloc(interior_count * sizeof(double));
        double* num_grad_y = malloc(interior_count * sizeof(double));
        double* ana_grad_x = malloc(interior_count * sizeof(double));
        double* ana_grad_y = malloc(interior_count * sizeof(double));
        CHECK_ALLOC4(num_grad_x, num_grad_y, ana_grad_x, ana_grad_y);

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                /* Gradient x-component: df/dx */
                double f_ip1 = test_f(x + dx, y, k);
                double f_im1 = test_f(x - dx, y, k);
                num_grad_x[idx] = fd_first_deriv_x(f_ip1, f_im1, dx);
                ana_grad_x[idx] = test_df_dx(x, y, k);

                /* Gradient y-component: df/dy */
                double f_jp1 = test_f(x, y + dy, k);
                double f_jm1 = test_f(x, y - dy, k);
                num_grad_y[idx] = fd_first_deriv_y(f_jp1, f_jm1, dy);
                ana_grad_y[idx] = test_df_dy(x, y, k);

                idx++;
            }
        }

        errors_x[s] = compute_l2_error(num_grad_x, ana_grad_x, interior_count);
        errors_y[s] = compute_l2_error(num_grad_y, ana_grad_y, interior_count);
        printf("      %zux%zu: L2 error (grad_x: %.6e, grad_y: %.6e)\n",
               n, n, errors_x[s], errors_y[s]);

        free(num_grad_x);
        free(num_grad_y);
        free(ana_grad_x);
        free(ana_grad_y);
    }

    /* Verify O(h^2) convergence for both components */
    for (int s = 1; s < num_sizes; s++) {
        double rate_x = compute_convergence_rate(errors_x[s - 1], errors_x[s],
                                                 spacings_x[s - 1], spacings_x[s]);
        double rate_y = compute_convergence_rate(errors_y[s - 1], errors_y[s],
                                                 spacings_y[s - 1], spacings_y[s]);
        printf("      Convergence rate %zu->%zu: grad_x=%.2f, grad_y=%.2f\n",
               sizes[s - 1], sizes[s], rate_x, rate_y);

        TEST_ASSERT_TRUE_MESSAGE(rate_x > 2.0 - CONVERGENCE_RATE_TOL,
            "Gradient x-component convergence rate below O(h^2)");
        TEST_ASSERT_TRUE_MESSAGE(rate_y > 2.0 - CONVERGENCE_RATE_TOL,
            "Gradient y-component convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors_x[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Gradient x-component error too large at finest grid");
    TEST_ASSERT_TRUE_MESSAGE(errors_y[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Gradient y-component error too large at finest grid");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("FINITE DIFFERENCE STENCIL TESTS\n");
    printf("========================================\n");
    printf("\nVerifying O(h^2) accuracy for central differences\n");
    printf("Test function: f(x,y) = sin(kx) * sin(ky), k=%.1f\n", TEST_K);

    /* First derivative tests */
    RUN_TEST(test_first_derivative_x_accuracy);
    RUN_TEST(test_first_derivative_y_accuracy);

    /* Second derivative tests */
    RUN_TEST(test_second_derivative_x_accuracy);
    RUN_TEST(test_second_derivative_y_accuracy);

    /* Laplacian tests */
    RUN_TEST(test_laplacian_5point_accuracy);
    RUN_TEST(test_laplacian_nonsquare_grid);

    /* Divergence tests */
    RUN_TEST(test_divergence_free_field);
    RUN_TEST(test_divergence_nonzero_accuracy);

    /* Gradient tests */
    RUN_TEST(test_gradient_accuracy);

    return UNITY_END();
}
