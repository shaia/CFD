/**
 * @file test_residual_computation.c
 * @brief Tests for poisson_solver_compute_residual()
 *
 * Tests cover:
 *   - Exact discrete solution yields near-zero residual
 *   - Wrong solution (all zeros) yields non-trivial residual
 *   - Residual of analytical solution converges O(h²) with grid refinement
 *
 * The residual is defined as ||∇²_h x - rhs||_∞ over interior points.
 * When x is the exact discrete solution, ∇²_h x == rhs exactly, so the
 * residual is zero (up to floating-point rounding).  When x is the
 * continuous analytical solution, the mismatch between the discrete and
 * continuous Laplacian is the truncation error, which is O(h²).
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DOMAIN_XMIN 0.0
#define DOMAIN_XMAX 1.0
#define DOMAIN_YMIN 0.0
#define DOMAIN_YMAX 1.0

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * HELPERS
 * ============================================================================ */

static double* alloc_field(size_t n) {
    return (double*)cfd_calloc(n * n, sizeof(double));
}

/**
 * Create a CG scalar solver, initialised for an n×n grid on [0,1]².
 * Caller must destroy the returned solver.
 */
static poisson_solver_t* make_cg_solver(size_t n, double dx, double dy) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    if (!solver) return NULL;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = 1e-10;
    params.max_iterations  = 2000;
    params.preconditioner  = POISSON_PRECOND_NONE;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return NULL;
    }
    return solver;
}

/* ============================================================================
 * TEST 1: Residual of exact discrete solution is near zero
 * ============================================================================
 *
 * Set x = sin(πx)*sin(πy) (the manufactured solution).
 * Compute rhs as the discrete Laplacian of x:
 *   rhs[i,j] = (x[i+1,j] - 2x[i,j] + x[i-1,j])/dx²
 *            + (x[i,j+1] - 2x[i,j] + x[i,j-1])/dy²
 * for interior points; boundary nodes get rhs = 0.
 * Then poisson_solver_compute_residual(solver, x, rhs) must be ~0 because
 * the discrete equation ∇²_h x = rhs is satisfied exactly.
 */
void test_residual_exact_solution(void) {
    printf("\n    test_residual_exact_solution: discrete Laplacian residual ~0\n");

    const size_t n  = 17;
    const double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    const double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    double* x   = alloc_field(n);
    double* rhs = alloc_field(n);

    TEST_ASSERT_NOT_NULL_MESSAGE(x,   "Memory allocation failed for x");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

    /* Fill x with the manufactured solution */
    for (size_t j = 0; j < n; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < n; i++) {
            double xi = DOMAIN_XMIN + i * dx;
            x[IDX_2D(i, j, n)] = sin(M_PI * xi) * sin(M_PI * y);
        }
    }

    /* Compute rhs as discrete Laplacian of x at interior points */
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double lap_x = (x[IDX_2D(i + 1, j, n)] - 2.0 * x[IDX_2D(i, j, n)]
                            + x[IDX_2D(i - 1, j, n)]) / (dx * dx);
            double lap_y = (x[IDX_2D(i, j + 1, n)] - 2.0 * x[IDX_2D(i, j, n)]
                            + x[IDX_2D(i, j - 1, n)]) / (dy * dy);
            rhs[IDX_2D(i, j, n)] = lap_x + lap_y;
        }
    }
    /* Boundary nodes: rhs stays 0 (already zero-initialised) */

    poisson_solver_t* solver = make_cg_solver(n, dx, dy);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

    double residual = poisson_solver_compute_residual(solver, x, rhs);

    printf("      residual = %.3e  (should be < 1e-10)\n", residual);

    TEST_ASSERT_TRUE_MESSAGE(residual < 1e-10,
        "Residual of exact discrete solution must be near zero");

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST 2: Residual is non-zero for wrong solution
 * ============================================================================
 *
 * x = all zeros, rhs = sin(πx)*sin(πy).
 * The discrete Laplacian of x is identically zero, but rhs is non-zero, so
 * the residual = ||0 - rhs||_∞ > 0.
 */
void test_residual_wrong_solution(void) {
    printf("\n    test_residual_wrong_solution: wrong solution has non-zero residual\n");

    const size_t n  = 17;
    const double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    const double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    double* x   = alloc_field(n);  /* all zeros */
    double* rhs = alloc_field(n);

    TEST_ASSERT_NOT_NULL_MESSAGE(x,   "Memory allocation failed for x");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

    /* rhs = sin(πx)*sin(πy) on interior */
    for (size_t j = 1; j < n - 1; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 1; i < n - 1; i++) {
            double xi = DOMAIN_XMIN + i * dx;
            rhs[IDX_2D(i, j, n)] = sin(M_PI * xi) * sin(M_PI * y);
        }
    }

    poisson_solver_t* solver = make_cg_solver(n, dx, dy);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

    double residual = poisson_solver_compute_residual(solver, x, rhs);

    printf("      residual = %.3e  (should be > 0.01)\n", residual);

    TEST_ASSERT_TRUE_MESSAGE(residual > 0.01,
        "Residual of wrong (zero) solution must be clearly non-zero");

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST 3: Residual of analytical solution converges O(h²)
 * ============================================================================
 *
 * x    = sin(πx)*sin(πy)           (continuous analytical solution)
 * rhs  = 2π²*sin(πx)*sin(πy)      (analytical RHS from -∇²x)
 *
 * The solver computes ||∇²_h x - rhs||_∞.  Since ∇²_h x ≈ ∇²x + O(h²),
 * the residual measures truncation error and should decrease as O(h²) when
 * the grid is refined.
 *
 * Grid sizes tested: 17, 33, 65  (not 129 — keep fast).
 */
void test_residual_convergence_rate(void) {
    printf("\n    test_residual_convergence_rate: truncation error O(h²)\n");

    const size_t sizes[3] = {17, 33, 65};
    const int    num_sizes = 3;
    double       residuals[3];
    double       hs[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n  = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        hs[s]     = dx;

        double* x   = alloc_field(n);
        double* rhs = alloc_field(n);

        if (!x || !rhs) {
            cfd_free(x);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        /* x = sin(πx)*sin(πy) */
        for (size_t j = 0; j < n; j++) {
            double y = DOMAIN_YMIN + j * dy;
            for (size_t i = 0; i < n; i++) {
                double xi = DOMAIN_XMIN + i * dx;
                x[IDX_2D(i, j, n)] = sin(M_PI * xi) * sin(M_PI * y);
            }
        }

        /* rhs = -2π²*sin(πx)*sin(πy)  (∇²sin(πx)sin(πy) = -2π²sin(πx)sin(πy)) */
        for (size_t j = 1; j < n - 1; j++) {
            double y = DOMAIN_YMIN + j * dy;
            for (size_t i = 1; i < n - 1; i++) {
                double xi = DOMAIN_XMIN + i * dx;
                rhs[IDX_2D(i, j, n)] = -2.0 * M_PI * M_PI * sin(M_PI * xi) * sin(M_PI * y);
            }
        }

        poisson_solver_t* solver = make_cg_solver(n, dx, dy);
        if (!solver) {
            cfd_free(x);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Could not create CG solver");
            return;
        }

        residuals[s] = poisson_solver_compute_residual(solver, x, rhs);

        printf("      n=%2zu  h=%.4f  residual=%.4e\n", n, dx, residuals[s]);

        poisson_solver_destroy(solver);
        cfd_free(x);
        cfd_free(rhs);
    }

    /* Compute convergence rates between consecutive grid refinements */
    double rate_sum = 0.0;
    int    rate_count = 0;
    for (int s = 1; s < num_sizes; s++) {
        if (residuals[s] < 1e-15 || residuals[s - 1] < 1e-15) continue;
        double rate = log(residuals[s - 1] / residuals[s]) / log(hs[s - 1] / hs[s]);
        printf("      convergence rate (n=%zu → n=%zu): %.2f\n",
               sizes[s - 1], sizes[s], rate);
        rate_sum += rate;
        rate_count++;
    }

    TEST_ASSERT_TRUE_MESSAGE(rate_count > 0, "No convergence rate could be computed");

    double avg_rate = rate_sum / rate_count;
    printf("      average rate = %.2f  (should be > 1.7)\n", avg_rate);

    TEST_ASSERT_TRUE_MESSAGE(avg_rate > 1.7,
        "Truncation error must converge at least O(h^1.7) — expected O(h²)");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("Residual Computation Tests\n");
    printf("========================================\n");

    RUN_TEST(test_residual_exact_solution);
    RUN_TEST(test_residual_wrong_solution);
    RUN_TEST(test_residual_convergence_rate);

    printf("\n========================================\n");
    return UNITY_END();
}
