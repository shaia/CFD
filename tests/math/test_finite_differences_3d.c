/**
 * @file test_finite_differences_3d.c
 * @brief Unit tests for 3D finite difference stencil accuracy
 *
 * These tests verify O(h^2) accuracy for the 3D central difference stencils
 * defined in cfd/math/stencils.h. This validates the mathematical correctness
 * of the 3D stencil formulas including the z-direction derivatives, 3D
 * Laplacian, and 3D divergence operator.
 *
 * NOTE: These are standalone mathematical validation tests. They link against
 * unity only (no CFD::Library) since stencils.h is a header-only inline library.
 * For tests that exercise actual production code paths, see the 3D solver
 * validation tests.
 *
 * Tests cover:
 *   - First derivative in z (central difference)
 *   - Second derivative in z (central difference)
 *   - 3D Laplacian (7-point stencil)
 *   - 3D Divergence operator (non-zero divergence field)
 *   - 3D Divergence operator (divergence-free field)
 *
 * Test approach: Use smooth analytical functions (e.g., sin(kx)*sin(ky)*sin(kz)),
 * compute numerical derivatives using the stencil functions, compare to
 * analytical derivatives, verify O(h^2) error scaling.
 *
 * Grid sizes are kept small (9^3 and 17^3) to avoid excessive memory use and
 * long runtimes from the cubic growth of 3D allocations.
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

/* Flat 3D array indexing: k*nx*ny + j*nx + i */
#define IDX_3D(i, j, k, nx, ny) ((k) * (nx) * (ny) + (j) * (nx) + (i))

/* ============================================================================
 * ANALYTICAL FUNCTIONS AND DERIVATIVES
 * ============================================================================ */

/**
 * Scalar test function: f(x,y,z) = sin(kx) * sin(ky) * sin(kz)
 */
static inline double test_f3(double x, double y, double z, double k) {
    return sin(k * x) * sin(k * y) * sin(k * z);
}

/**
 * Analytical first derivative: df/dz = k * sin(kx) * sin(ky) * cos(kz)
 */
static inline double test_df_dz(double x, double y, double z, double k) {
    return k * sin(k * x) * sin(k * y) * cos(k * z);
}

/**
 * Analytical second derivative: d2f/dz2 = -k^2 * sin(kx) * sin(ky) * sin(kz)
 */
static inline double test_d2f_dz2(double x, double y, double z, double k) {
    return -k * k * sin(k * x) * sin(k * y) * sin(k * z);
}

/**
 * Analytical 3D Laplacian: nabla^2 f = -3k^2 * sin(kx) * sin(ky) * sin(kz)
 */
static inline double test_laplacian_3d(double x, double y, double z, double k) {
    return -3.0 * k * k * sin(k * x) * sin(k * y) * sin(k * z);
}

/**
 * Non-zero divergence test field components:
 *   u = sin(2x)*sin(y)*sin(z)
 *   v = sin(x)*sin(2y)*sin(z)
 *   w = sin(x)*sin(y)*sin(2z)
 */
static inline double test_u_nonzero_div(double x, double y, double z) {
    return sin(2.0 * x) * sin(y) * sin(z);
}

static inline double test_v_nonzero_div(double x, double y, double z) {
    return sin(x) * sin(2.0 * y) * sin(z);
}

static inline double test_w_nonzero_div(double x, double y, double z) {
    return sin(x) * sin(y) * sin(2.0 * z);
}

/**
 * Analytical divergence of the non-zero divergence field:
 *   du/dx = 2*cos(2x)*sin(y)*sin(z)
 *   dv/dy = 2*sin(x)*cos(2y)*sin(z)
 *   dw/dz = 2*sin(x)*sin(y)*cos(2z)
 *   div   = sum of the above
 */
static inline double test_divergence_nonzero_3d(double x, double y, double z) {
    return 2.0 * cos(2.0 * x) * sin(y) * sin(z)
         + 2.0 * sin(x) * cos(2.0 * y) * sin(z)
         + 2.0 * sin(x) * sin(y) * cos(2.0 * z);
}

/**
 * Divergence-free field components:
 *   u = sin(ky) * cos(kz)   -> du/dx = 0
 *   v = sin(kz) * cos(kx)   -> dv/dy = 0
 *   w = sin(kx) * cos(ky)   -> dw/dz = 0
 * Therefore div(u,v,w) = 0 analytically.
 */
static inline double test_u_div_free(double x, double y, double z, double k) {
    (void)x;
    return sin(k * y) * cos(k * z);
}

static inline double test_v_div_free(double x, double y, double z, double k) {
    (void)y;
    return sin(k * z) * cos(k * x);
}

static inline double test_w_div_free(double x, double y, double z, double k) {
    (void)z;
    return sin(k * x) * cos(k * y);
}

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
 * FIRST DERIVATIVE Z TESTS
 * ============================================================================ */

/**
 * Test first derivative df/dz accuracy
 */
void test_first_deriv_z_accuracy(void) {
    printf("\n    Testing first derivative df/dz accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;
    double zmin = 0.0, zmax = 2.0 * M_PI;

    size_t sizes[] = {9, 17};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[2];
    double spacings[2];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        double dz = (zmax - zmin) / (n - 1);
        spacings[s] = dz;

        size_t interior_count = (n - 2) * (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t k3 = 1; k3 < n - 1; k3++) {
            double z = zmin + k3 * dz;
            for (size_t j = 1; j < n - 1; j++) {
                double y = ymin + j * dy;
                for (size_t i = 1; i < n - 1; i++) {
                    double x = xmin + i * dx;

                    double f_kp1 = test_f3(x, y, z + dz, k);
                    double f_km1 = test_f3(x, y, z - dz, k);

                    numerical[idx] = stencil_first_deriv_z(f_kp1, f_km1, dz);
                    analytical[idx] = test_df_dz(x, y, z, k);

                    idx++;
                }
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zux%zu: L2 error = %.6e\n", n, n, n, errors[s]);

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
            "First derivative df/dz convergence rate below O(h^2)");
    }

    /* Verify absolute error at finest grid */
    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "First derivative df/dz error too large at finest grid");
}

/* ============================================================================
 * SECOND DERIVATIVE Z TESTS
 * ============================================================================ */

/**
 * Test second derivative d2f/dz2 accuracy
 */
void test_second_deriv_z_accuracy(void) {
    printf("\n    Testing second derivative d2f/dz2 accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;
    double zmin = 0.0, zmax = 2.0 * M_PI;

    size_t sizes[] = {9, 17};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[2];
    double spacings[2];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        double dz = (zmax - zmin) / (n - 1);
        spacings[s] = dz;

        size_t interior_count = (n - 2) * (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t k3 = 1; k3 < n - 1; k3++) {
            double z = zmin + k3 * dz;
            for (size_t j = 1; j < n - 1; j++) {
                double y = ymin + j * dy;
                for (size_t i = 1; i < n - 1; i++) {
                    double x = xmin + i * dx;

                    double f_kp1 = test_f3(x, y, z + dz, k);
                    double f_k   = test_f3(x, y, z,      k);
                    double f_km1 = test_f3(x, y, z - dz, k);

                    numerical[idx] = stencil_second_deriv_z(f_kp1, f_k, f_km1, dz);
                    analytical[idx] = test_d2f_dz2(x, y, z, k);

                    idx++;
                }
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zux%zu: L2 error = %.6e\n", n, n, n, errors[s]);

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
            "Second derivative d2f/dz2 convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Second derivative d2f/dz2 error too large at finest grid");
}

/* ============================================================================
 * 3D LAPLACIAN TESTS
 * ============================================================================ */

/**
 * Test 3D Laplacian (7-point stencil) accuracy
 */
void test_laplacian_3d_accuracy(void) {
    printf("\n    Testing 3D Laplacian (7-point stencil) accuracy...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;
    double zmin = 0.0, zmax = 2.0 * M_PI;

    size_t sizes[] = {9, 17};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[2];
    double spacings[2];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        double dz = (zmax - zmin) / (n - 1);
        spacings[s] = dx;  /* Uniform grid: dx == dy == dz */

        size_t interior_count = (n - 2) * (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t k3 = 1; k3 < n - 1; k3++) {
            double z = zmin + k3 * dz;
            for (size_t j = 1; j < n - 1; j++) {
                double y = ymin + j * dy;
                for (size_t i = 1; i < n - 1; i++) {
                    double x = xmin + i * dx;

                    double f_ip1 = test_f3(x + dx, y,      z,      k);
                    double f_im1 = test_f3(x - dx, y,      z,      k);
                    double f_jp1 = test_f3(x,      y + dy, z,      k);
                    double f_jm1 = test_f3(x,      y - dy, z,      k);
                    double f_kp1 = test_f3(x,      y,      z + dz, k);
                    double f_km1 = test_f3(x,      y,      z - dz, k);
                    double f_ijk = test_f3(x,      y,      z,      k);

                    numerical[idx] = stencil_laplacian_3d(
                        f_ip1, f_im1, f_jp1, f_jm1, f_kp1, f_km1, f_ijk,
                        dx, dy, dz);
                    analytical[idx] = test_laplacian_3d(x, y, z, k);

                    idx++;
                }
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zux%zu: L2 error = %.6e\n", n, n, n, errors[s]);

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
            "3D Laplacian convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "3D Laplacian error too large at finest grid");
}

/* ============================================================================
 * 3D DIVERGENCE TESTS
 * ============================================================================ */

/**
 * Test 3D divergence operator accuracy with non-zero divergence field
 */
void test_divergence_3d_accuracy(void) {
    printf("\n    Testing 3D divergence accuracy with non-zero divergence field...\n");

    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;
    double zmin = 0.0, zmax = 2.0 * M_PI;

    /* Use 9 and 17 for convergence rate, 33 for absolute error check.
     * The wavenumber-2 field has larger absolute error than k=1 tests;
     * a 33^3 grid brings the L2 error below ABSOLUTE_ERROR_TOL. */
    size_t sizes[] = {9, 17, 33};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (xmax - xmin) / (n - 1);
        double dy = (ymax - ymin) / (n - 1);
        double dz = (zmax - zmin) / (n - 1);
        spacings[s] = dx;

        size_t interior_count = (n - 2) * (n - 2) * (n - 2);
        double* numerical = malloc(interior_count * sizeof(double));
        double* analytical = malloc(interior_count * sizeof(double));
        CHECK_ALLOC2(numerical, analytical);

        size_t idx = 0;
        for (size_t k3 = 1; k3 < n - 1; k3++) {
            double z = zmin + k3 * dz;
            for (size_t j = 1; j < n - 1; j++) {
                double y = ymin + j * dy;
                for (size_t i = 1; i < n - 1; i++) {
                    double x = xmin + i * dx;

                    double u_ip1 = test_u_nonzero_div(x + dx, y,      z     );
                    double u_im1 = test_u_nonzero_div(x - dx, y,      z     );
                    double v_jp1 = test_v_nonzero_div(x,      y + dy, z     );
                    double v_jm1 = test_v_nonzero_div(x,      y - dy, z     );
                    double w_kp1 = test_w_nonzero_div(x,      y,      z + dz);
                    double w_km1 = test_w_nonzero_div(x,      y,      z - dz);

                    numerical[idx] = stencil_divergence_3d(
                        u_ip1, u_im1, v_jp1, v_jm1, w_kp1, w_km1,
                        dx, dy, dz);
                    analytical[idx] = test_divergence_nonzero_3d(x, y, z);

                    idx++;
                }
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        printf("      %zux%zux%zu: L2 error = %.6e\n", n, n, n, errors[s]);

        free(numerical);
        free(analytical);
    }

    /* Verify O(h^2) convergence using the first two grid levels */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "3D divergence operator convergence rate below O(h^2)");
    }

    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "3D divergence operator error too large at finest grid");
}

/**
 * Test 3D divergence-free field gives near-zero divergence
 *
 * Field: u = sin(ky)*cos(kz), v = sin(kz)*cos(kx), w = sin(kx)*cos(ky)
 * Analytically: du/dx = 0, dv/dy = 0, dw/dz = 0, so div = 0.
 * Numerically the central-difference approximation also gives exactly 0
 * because u has no x dependence, v no y dependence, and w no z dependence.
 * We verify max|div| < h^2 as a principled bound.
 */
void test_divergence_3d_free_field(void) {
    printf("\n    Testing 3D divergence of divergence-free field...\n");

    double k = TEST_K;
    double xmin = 0.0, xmax = 2.0 * M_PI;
    double ymin = 0.0, ymax = 2.0 * M_PI;
    double zmin = 0.0, zmax = 2.0 * M_PI;

    size_t n = 17;
    double dx = (xmax - xmin) / (n - 1);
    double dy = (ymax - ymin) / (n - 1);
    double dz = (zmax - zmin) / (n - 1);

    double* numerical = malloc((n - 2) * (n - 2) * (n - 2) * sizeof(double));
    double* analytical = malloc((n - 2) * (n - 2) * (n - 2) * sizeof(double));
    CHECK_ALLOC2(numerical, analytical);

    size_t interior_count = (n - 2) * (n - 2) * (n - 2);
    size_t idx = 0;
    for (size_t k3 = 1; k3 < n - 1; k3++) {
        double z = zmin + k3 * dz;
        for (size_t j = 1; j < n - 1; j++) {
            double y = ymin + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = xmin + i * dx;

                double u_ip1 = test_u_div_free(x + dx, y,      z,      k);
                double u_im1 = test_u_div_free(x - dx, y,      z,      k);
                double v_jp1 = test_v_div_free(x,      y + dy, z,      k);
                double v_jm1 = test_v_div_free(x,      y - dy, z,      k);
                double w_kp1 = test_w_div_free(x,      y,      z + dz, k);
                double w_km1 = test_w_div_free(x,      y,      z - dz, k);

                numerical[idx] = stencil_divergence_3d(
                    u_ip1, u_im1, v_jp1, v_jm1, w_kp1, w_km1,
                    dx, dy, dz);
                analytical[idx] = 0.0;  /* Divergence-free by construction */

                idx++;
            }
        }
    }

    double max_div = compute_max_error(numerical, analytical, interior_count);
    printf("      %zux%zux%zu: max|div| = %.6e\n", n, n, n, max_div);

    /* Divergence-free field should have small divergence numerically.
     * Use O(h^2) tolerance since central differences are 2nd order accurate. */
    double h2_tolerance = dx * dx;
    printf("      Tolerance (h^2): %.6e\n", h2_tolerance);
    TEST_ASSERT_TRUE_MESSAGE(max_div < h2_tolerance,
        "Divergence of divergence-free field exceeds O(h^2) tolerance");

    free(numerical);
    free(analytical);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("3D FINITE DIFFERENCE STENCIL TESTS\n");
    printf("========================================\n");
    printf("\nVerifying O(h^2) accuracy for 3D central differences\n");
    printf("Test function: f(x,y,z) = sin(kx)*sin(ky)*sin(kz), k=%.1f\n", TEST_K);
    printf("Grid sizes: 9^3 and 17^3 (kept small for 3D memory/time constraints)\n");

    /* z-direction derivative tests */
    RUN_TEST(test_first_deriv_z_accuracy);
    RUN_TEST(test_second_deriv_z_accuracy);

    /* 3D Laplacian test */
    RUN_TEST(test_laplacian_3d_accuracy);

    /* 3D divergence tests */
    RUN_TEST(test_divergence_3d_accuracy);
    RUN_TEST(test_divergence_3d_free_field);

    return UNITY_END();
}
