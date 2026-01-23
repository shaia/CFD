/**
 * @file test_laplacian_accuracy.c
 * @brief Laplacian operator accuracy validation tests
 *
 * These tests verify the accuracy of the discrete Laplacian operator
 * against manufactured solutions. This validates the core numerical
 * operator used by Poisson/CG solvers.
 *
 * Tests cover (ROADMAP 1.2.2):
 *   - Manufactured solution: p = sin(πx)sin(πy) → ∇²p = -2π²p
 *   - 2nd-order accuracy O(dx²) verification with grid refinement
 *   - Backend comparison: CPU scalar vs SIMD (CG implementations)
 *
 * Test approach:
 *   1. Direct stencil test: Apply stencil to known function, compare to analytical
 *   2. Indirect test via CG solver: Verify different backends produce same result
 *   3. Grid convergence: Verify O(h²) error scaling
 */

#include "unity.h"
#include "cfd/math/stencils.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

/* Domain: [0, 1] x [0, 1] */
#define DOMAIN_XMIN 0.0
#define DOMAIN_XMAX 1.0
#define DOMAIN_YMIN 0.0
#define DOMAIN_YMAX 1.0

/* Tolerances */
#define CONVERGENCE_RATE_TOL    0.3    /* Minimum expected rate is 2.0 - 0.3 = 1.7 */
#define ABSOLUTE_ERROR_TOL      1e-3   /* Max error at finest grid */
#define BACKEND_COMPARISON_TOL  1e-10  /* Backends should match to machine precision */

/* ============================================================================
 * MANUFACTURED SOLUTION
 * ============================================================================
 *
 * p(x,y) = sin(πx) * sin(πy)
 *
 * Derivatives:
 *   ∂p/∂x = π * cos(πx) * sin(πy)
 *   ∂p/∂y = π * sin(πx) * cos(πy)
 *   ∂²p/∂x² = -π² * sin(πx) * sin(πy)
 *   ∂²p/∂y² = -π² * sin(πx) * sin(πy)
 *
 * Laplacian:
 *   ∇²p = ∂²p/∂x² + ∂²p/∂y² = -2π² * sin(πx) * sin(πy) = -2π² * p
 */

static inline double manufactured_p(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

static inline double manufactured_laplacian(double x, double y) {
    return -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static double* create_field(size_t nx, size_t ny) {
    return (double*)cfd_calloc(nx * ny, sizeof(double));
}

/**
 * Initialize field with manufactured solution p = sin(πx)sin(πy)
 */
static void init_manufactured_solution(double* p, size_t nx, size_t ny,
                                        double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            p[j * nx + i] = manufactured_p(x, y);
        }
    }
}

/**
 * Compute L2 error norm between numerical and analytical arrays
 */
static double compute_l2_error(const double* numerical, const double* analytical,
                                size_t n) {
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
static double compute_max_error(const double* numerical, const double* analytical,
                                 size_t n) {
    if (n == 0) return 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs(numerical[i] - analytical[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/**
 * Compute convergence rate from two error values
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
 * TEST: DIRECT STENCIL ACCURACY
 * ============================================================================
 *
 * Apply the Laplacian stencil directly to the manufactured solution
 * and compare with analytical Laplacian.
 */

void test_laplacian_stencil_manufactured_solution(void) {
    printf("\n    Testing Laplacian stencil with manufactured solution p=sin(πx)sin(πy)...\n");

    size_t sizes[] = {17, 33, 65, 129};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[4];
    double spacings[4];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        spacings[s] = dx;

        /* Allocate field and initialize with manufactured solution */
        double* p = create_field(n, n);
        TEST_ASSERT_NOT_NULL(p);
        init_manufactured_solution(p, n, n, dx, dy);

        /* Compute numerical and analytical Laplacian at interior points */
        size_t interior_count = (n - 2) * (n - 2);
        double* numerical = (double*)malloc(interior_count * sizeof(double));
        double* analytical = (double*)malloc(interior_count * sizeof(double));

        if (!numerical || !analytical) {
            free(numerical);
            free(analytical);
            cfd_free(p);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        size_t idx = 0;
        for (size_t j = 1; j < n - 1; j++) {
            double y = DOMAIN_YMIN + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double x = DOMAIN_XMIN + i * dx;
                size_t ij = j * n + i;

                /* Get stencil values */
                double p_ip1 = p[ij + 1];
                double p_im1 = p[ij - 1];
                double p_jp1 = p[ij + n];
                double p_jm1 = p[ij - n];
                double p_ij = p[ij];

                /* Compute numerical Laplacian using stencil */
                numerical[idx] = stencil_laplacian_2d(p_ip1, p_im1, p_jp1, p_jm1, p_ij, dx, dy);

                /* Analytical Laplacian */
                analytical[idx] = manufactured_laplacian(x, y);

                idx++;
            }
        }

        errors[s] = compute_l2_error(numerical, analytical, interior_count);
        double max_error = compute_max_error(numerical, analytical, interior_count);
        printf("      %3zux%-3zu: L2 error = %.6e, Max error = %.6e\n",
               n, n, errors[s], max_error);

        free(numerical);
        free(analytical);
        cfd_free(p);
    }

    /* Verify O(h²) convergence */
    printf("\n    Convergence rates:\n");
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s - 1], errors[s],
                                               spacings[s - 1], spacings[s]);
        printf("      %3zu->%-3zu: rate = %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Laplacian convergence rate below O(h²)");
    }

    /* Verify absolute error at finest grid */
    TEST_ASSERT_TRUE_MESSAGE(errors[num_sizes - 1] < ABSOLUTE_ERROR_TOL,
        "Laplacian error too large at finest grid");
}

/* ============================================================================
 * TEST: CG SOLVER BACKEND COMPARISON
 * ============================================================================
 *
 * Verify that CPU scalar and SIMD CG implementations produce identical
 * results when solving the same Poisson problem.
 */

void test_cg_backend_comparison(void) {
    printf("\n    Testing CG solver backend comparison (scalar vs SIMD)...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    /* RHS: Laplacian of manufactured solution */
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(rhs);

    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[j * nx + i] = manufactured_laplacian(x, y);
        }
    }

    /* Storage for solutions from different backends */
    double* p_scalar = create_field(nx, ny);
    double* p_simd = create_field(nx, ny);
    double* p_temp = create_field(nx, ny);

    if (!p_scalar || !p_simd || !p_temp) {
        cfd_free(rhs);
        cfd_free(p_scalar);
        cfd_free(p_simd);
        cfd_free(p_temp);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }

    /* Apply Dirichlet BCs (analytical solution on boundary) */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        p_scalar[j * nx + 0] = manufactured_p(DOMAIN_XMIN, y);
        p_scalar[j * nx + (nx - 1)] = manufactured_p(DOMAIN_XMAX, y);
        p_simd[j * nx + 0] = manufactured_p(DOMAIN_XMIN, y);
        p_simd[j * nx + (nx - 1)] = manufactured_p(DOMAIN_XMAX, y);
    }
    for (size_t i = 0; i < nx; i++) {
        double x = DOMAIN_XMIN + i * dx;
        p_scalar[0 * nx + i] = manufactured_p(x, DOMAIN_YMIN);
        p_scalar[(ny - 1) * nx + i] = manufactured_p(x, DOMAIN_YMAX);
        p_simd[0 * nx + i] = manufactured_p(x, DOMAIN_YMIN);
        p_simd[(ny - 1) * nx + i] = manufactured_p(x, DOMAIN_YMAX);
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 5000;

    int scalar_available = 0;
    int simd_available = 0;

    /* Test scalar CG */
    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);

    if (solver_scalar) {
        cfd_status_t status = poisson_solver_init(solver_scalar, nx, ny, dx, dy, &params);
        if (status == CFD_SUCCESS) {
            poisson_solver_stats_t stats = poisson_solver_stats_default();
            poisson_solver_solve(solver_scalar, p_scalar, p_temp, rhs, &stats);
            printf("      Scalar CG: %d iterations, residual = %.6e\n",
                   stats.iterations, stats.final_residual);
            scalar_available = 1;
        }
        poisson_solver_destroy(solver_scalar);
    } else {
        printf("      Scalar CG: not available\n");
    }

    /* Test SIMD CG */
    poisson_solver_t* solver_simd = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD);

    if (solver_simd) {
        cfd_status_t status = poisson_solver_init(solver_simd, nx, ny, dx, dy, &params);
        if (status == CFD_SUCCESS) {
            poisson_solver_stats_t stats = poisson_solver_stats_default();
            poisson_solver_solve(solver_simd, p_simd, p_temp, rhs, &stats);
            printf("      SIMD CG:   %d iterations, residual = %.6e\n",
                   stats.iterations, stats.final_residual);
            simd_available = 1;
        }
        poisson_solver_destroy(solver_simd);
    } else {
        printf("      SIMD CG: not available (SIMD support not compiled)\n");
    }

    /* Compare solutions if both backends available */
    if (scalar_available && simd_available) {
        double max_diff = 0.0;
        for (size_t i = 0; i < nx * ny; i++) {
            double diff = fabs(p_scalar[i] - p_simd[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("      Backend difference: %.6e\n", max_diff);
        TEST_ASSERT_TRUE_MESSAGE(max_diff < BACKEND_COMPARISON_TOL,
            "CG scalar and SIMD backends produce different results");
    } else {
        printf("      Backend comparison skipped (need both scalar and SIMD)\n");
    }

    /* At least one backend must be available */
    TEST_ASSERT_TRUE_MESSAGE(scalar_available || simd_available,
        "At least one CG backend must be available");

    cfd_free(rhs);
    cfd_free(p_scalar);
    cfd_free(p_simd);
    cfd_free(p_temp);
}

/* ============================================================================
 * TEST: LAPLACIAN VIA RESIDUAL COMPUTATION
 * ============================================================================
 *
 * Verify the Laplacian computation used in solvers via the public
 * residual computation API. This tests that:
 *   residual = ||∇²p - rhs||
 *
 * If p is the exact solution of ∇²p = rhs, residual should be O(h²).
 */

void test_laplacian_via_residual(void) {
    printf("\n    Testing Laplacian via residual computation...\n");

    size_t sizes[] = {17, 33, 65, 129};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double residuals[4];
    double spacings[4];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        spacings[s] = dx;

        double* p = create_field(n, n);
        double* rhs = create_field(n, n);

        if (!p || !rhs) {
            cfd_free(p);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        /* Set p to manufactured solution, rhs to its Laplacian */
        for (size_t j = 0; j < n; j++) {
            double y = DOMAIN_YMIN + j * dy;
            for (size_t i = 0; i < n; i++) {
                double x = DOMAIN_XMIN + i * dx;
                p[j * n + i] = manufactured_p(x, y);
                rhs[j * n + i] = manufactured_laplacian(x, y);
            }
        }

        /* Create solver just to use compute_residual */
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);

        if (!solver) {
            cfd_free(p);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Could not create solver");
            return;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        /* Compute residual: ||∇²p - rhs||_∞
         * This should be O(h²) since we're comparing numerical Laplacian
         * to analytical Laplacian of the manufactured solution.
         */
        residuals[s] = poisson_solver_compute_residual(solver, p, rhs);
        printf("      %3zux%-3zu: residual = %.6e\n", n, n, residuals[s]);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
    }

    /* Verify O(h²) convergence of residual */
    printf("\n    Convergence rates:\n");
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(residuals[s - 1], residuals[s],
                                               spacings[s - 1], spacings[s]);
        printf("      %3zu->%-3zu: rate = %.2f (expected ~2.0)\n",
               sizes[s - 1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Residual convergence rate below O(h²)");
    }
}

/* ============================================================================
 * TEST: LAPLACIAN SYMMETRY
 * ============================================================================
 *
 * Verify that the Laplacian stencil is symmetric (self-adjoint).
 * For a symmetric stencil: <∇²u, v> = <u, ∇²v>
 *
 * IMPORTANT: For discrete Laplacian symmetry, both functions must have
 * homogeneous Dirichlet BCs (zero on all boundaries). We use:
 *   u = sin(πx)sin(πy)    - zero on all boundaries of [0,1]²
 *   v = sin(πx)sin(2πy)   - zero on all boundaries of [0,1]²
 */

void test_laplacian_symmetry(void) {
    printf("\n    Testing Laplacian stencil symmetry...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    /* Create two test functions */
    double* u = create_field(nx, ny);
    double* v = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Both functions are zero on all boundaries of [0,1]²:
     * u = sin(πx)sin(πy)           - smooth eigenfunction
     * v = x(1-x)y(1-y)             - polynomial, not an eigenfunction
     *
     * These are NOT orthogonal, so <u, v>, <∇²u, v>, and <u, ∇²v>
     * are all non-zero, making this a meaningful symmetry test.
     */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            u[j * nx + i] = sin(M_PI * x) * sin(M_PI * y);
            v[j * nx + i] = x * (1.0 - x) * y * (1.0 - y);
        }
    }

    /* Compute <∇²u, v> and <u, ∇²v> over interior points */
    double inner_lapu_v = 0.0;
    double inner_u_lapv = 0.0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t ij = j * nx + i;

            /* Laplacian of u at (i,j) */
            double lap_u = stencil_laplacian_2d(
                u[ij + 1], u[ij - 1],
                u[ij + nx], u[ij - nx],
                u[ij], dx, dy);

            /* Laplacian of v at (i,j) */
            double lap_v = stencil_laplacian_2d(
                v[ij + 1], v[ij - 1],
                v[ij + nx], v[ij - nx],
                v[ij], dx, dy);

            inner_lapu_v += lap_u * v[ij];
            inner_u_lapv += u[ij] * lap_v;
        }
    }

    /* Scale by cell area */
    inner_lapu_v *= dx * dy;
    inner_u_lapv *= dx * dy;

    printf("      <∇²u, v> = %.10f\n", inner_lapu_v);
    printf("      <u, ∇²v> = %.10f\n", inner_u_lapv);

    double abs_diff = fabs(inner_lapu_v - inner_u_lapv);
    double scale = fabs(inner_lapu_v) + fabs(inner_u_lapv);
    printf("      Absolute difference: %.6e\n", abs_diff);
    printf("      Relative difference: %.6e\n", abs_diff / (scale + 1e-15));

    /* Should be equal to machine precision (discrete symmetry) */
    /* The difference should be small relative to the magnitudes */
    TEST_ASSERT_TRUE_MESSAGE(abs_diff < 1e-10 * scale + 1e-12,
        "Laplacian stencil is not symmetric");

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("Laplacian Operator Validation Tests\n");
    printf("========================================\n");

    /* Direct stencil accuracy */
    printf("\n--- Direct Stencil Accuracy Tests ---\n");
    RUN_TEST(test_laplacian_stencil_manufactured_solution);
    RUN_TEST(test_laplacian_symmetry);

    /* Solver backend tests */
    printf("\n--- Solver Backend Tests ---\n");
    RUN_TEST(test_cg_backend_comparison);
    RUN_TEST(test_laplacian_via_residual);

    printf("\n========================================\n");
    return UNITY_END();
}
