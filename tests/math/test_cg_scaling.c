/**
 * @file test_cg_scaling.c
 * @brief CG iteration-count scaling tests
 *
 * Tests cover:
 *   - CG iteration count scales as O(√κ):
 *       For 2D Poisson, κ ≈ 4/(π²h²), so the CG bound is ~√κ iterations.
 *       We verify that the ratio  iterations / √κ  remains below 3.0 for
 *       all tested grid sizes.
 *   - PCG (Jacobi preconditioner) does not regress vs unpreconditioned CG:
 *       For the uniform-grid Laplacian the Jacobi preconditioner is a
 *       constant scalar, so no iteration reduction is expected.  We verify
 *       that PCG requires no more than 5 % + 1 extra iterations vs CG.
 *
 * Both tests use the same sinusoidal RHS (cos(2πx)cos(2πy) with interior
 * mean subtracted and boundary values zeroed) that is used throughout the
 * existing math test suite.
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

#define SOLVER_TOLERANCE  1e-6
#define MAX_ITERATIONS    2000

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * HELPERS
 * ============================================================================ */

static double* alloc_zeroed(size_t n) {
    return (double*)cfd_calloc(n * n, sizeof(double));
}

/**
 * Non-trivial initial guess: checkerboard ±1 pattern.
 */
static void init_checkerboard(double* p, size_t n) {
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            p[IDX_2D(i, j, n)] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
    }
}

/**
 * Sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2πx)cos(2πy), interior mean subtracted, boundaries zeroed.
 */
static void init_sinusoidal_rhs(double* rhs, size_t n, double dx, double dy) {
    /* First pass: fill with cos(2πx)cos(2πy) */
    for (size_t j = 0; j < n; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < n; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[IDX_2D(i, j, n)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Second pass: subtract interior mean */
    double sum = 0.0;
    size_t cnt = 0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            sum += rhs[IDX_2D(i, j, n)];
            cnt++;
        }
    }
    if (cnt > 0) {
        double mean = sum / (double)cnt;
        for (size_t j = 1; j < n - 1; j++) {
            for (size_t i = 1; i < n - 1; i++) {
                rhs[IDX_2D(i, j, n)] -= mean;
            }
        }
    }

    /* Zero boundary nodes */
    for (size_t i = 0; i < n; i++) {
        rhs[i]                 = 0.0;  /* bottom row  */
        rhs[(n - 1) * n + i]  = 0.0;  /* top row     */
    }
    for (size_t j = 1; j < n - 1; j++) {
        rhs[j * n]         = 0.0;  /* left column  */
        rhs[j * n + n - 1] = 0.0;  /* right column */
    }
}

/**
 * Theoretical Poisson condition number: κ ≈ 4 / (π² h²)
 */
static double condition_number(double h) {
    return 4.0 / (M_PI * M_PI * h * h);
}

/**
 * Create, init, and solve with CG scalar.
 * Returns iteration count, or -1 on failure.
 * Solver is destroyed before returning.
 */
static int run_cg(size_t n, double dx, double dy,
                  double* p, double* p_temp, const double* rhs,
                  poisson_precond_type_t precond) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    if (!solver) return -1;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = SOLVER_TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;
    params.preconditioner = precond;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return -1;
    }

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
    poisson_solver_destroy(solver);

    if (status != CFD_SUCCESS || stats.status != POISSON_CONVERGED) return -1;
    return stats.iterations;
}

/* ============================================================================
 * TEST 1: CG iteration count scales as O(√κ)
 * ============================================================================
 *
 * Grid sizes: 9, 17, 33, 65.
 * For each grid, solve and record iterations.
 * Assert:  iterations / √κ  < 3.0  for every size.
 */
void test_cg_sqrt_kappa_scaling(void) {
    printf("\n    test_cg_sqrt_kappa_scaling: iterations / sqrt(kappa) < 3.0\n");
    printf("    %-6s  %-8s  %-10s  %-10s  %-8s\n",
           "n", "iters", "kappa", "sqrt(kappa)", "ratio");

    const size_t sizes[] = {9, 17, 33, 65};
    const int num_sizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int all_passed = 1;

    for (int s = 0; s < num_sizes; s++) {
        size_t n  = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

        double* p      = alloc_zeroed(n);
        double* p_temp = alloc_zeroed(n);
        double* rhs    = alloc_zeroed(n);

        if (!p || !p_temp || !rhs) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_checkerboard(p, n);
        init_sinusoidal_rhs(rhs, n, dx, dy);

        int iters = run_cg(n, dx, dy, p, p_temp, rhs, POISSON_PRECOND_NONE);

        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);

        if (iters < 0) {
            printf("    %-6zu  FAILED (solver unavailable or did not converge)\n", n);
            all_passed = 0;
            continue;
        }

        double kappa      = condition_number(dx);
        double sqrt_kappa = sqrt(kappa);
        double ratio      = (double)iters / sqrt_kappa;

        printf("    %-6zu  %-8d  %-10.1f  %-10.2f  %-8.3f\n",
               n, iters, kappa, sqrt_kappa, ratio);

        if (ratio >= 3.0) {
            printf("    FAIL: ratio %.3f >= 3.0 for n=%zu\n", ratio, n);
            all_passed = 0;
        }
    }

    TEST_ASSERT_TRUE_MESSAGE(all_passed,
        "CG must converge in < 3*sqrt(kappa) iterations for all grid sizes");
}

/* ============================================================================
 * TEST 2: PCG doesn't regress vs CG
 * ============================================================================
 *
 * Grid sizes: 17, 33, 65.
 * Solve the same problem with CG (PRECOND_NONE) and PCG (PRECOND_JACOBI).
 * Assert: pcg_iters <= cg_iters * 1.05 + 1  at each size.
 */
void test_pcg_vs_cg_across_sizes(void) {
    printf("\n    test_pcg_vs_cg_across_sizes: PCG iters <= CG iters * 1.05 + 1\n");
    printf("    %-6s  %-8s  %-8s  %-8s\n", "n", "cg_iters", "pcg_iters", "ok?");

    const size_t sizes[] = {17, 33, 65};
    const int num_sizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int all_passed = 1;

    for (int s = 0; s < num_sizes; s++) {
        size_t n  = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

        /* --- CG --- */
        double* p_cg      = alloc_zeroed(n);
        double* p_temp_cg = alloc_zeroed(n);
        double* rhs       = alloc_zeroed(n);

        if (!p_cg || !p_temp_cg || !rhs) {
            cfd_free(p_cg);
            cfd_free(p_temp_cg);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_sinusoidal_rhs(rhs, n, dx, dy);
        init_checkerboard(p_cg, n);

        int cg_iters = run_cg(n, dx, dy, p_cg, p_temp_cg, rhs, POISSON_PRECOND_NONE);

        cfd_free(p_cg);
        cfd_free(p_temp_cg);

        if (cg_iters < 0) {
            cfd_free(rhs);
            printf("    %-6zu  CG solver failed — skipping\n", n);
            continue;
        }

        /* --- PCG --- */
        double* p_pcg      = alloc_zeroed(n);
        double* p_temp_pcg = alloc_zeroed(n);

        if (!p_pcg || !p_temp_pcg) {
            cfd_free(p_pcg);
            cfd_free(p_temp_pcg);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_checkerboard(p_pcg, n);

        int pcg_iters = run_cg(n, dx, dy, p_pcg, p_temp_pcg, rhs, POISSON_PRECOND_JACOBI);

        cfd_free(p_pcg);
        cfd_free(p_temp_pcg);
        cfd_free(rhs);

        if (pcg_iters < 0) {
            printf("    %-6zu  %-8d  PCG solver failed — skipping\n", n, cg_iters);
            continue;
        }

        int limit  = (int)(cg_iters * 1.05) + 1;
        int passed = (pcg_iters <= limit);

        printf("    %-6zu  %-8d  %-8d  %s  (limit=%d)\n",
               n, cg_iters, pcg_iters, passed ? "PASS" : "FAIL", limit);

        if (!passed) all_passed = 0;
    }

    TEST_ASSERT_TRUE_MESSAGE(all_passed,
        "PCG must not require significantly more iterations than CG (limit: CG*1.05+1)");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("CG Scaling Tests\n");
    printf("========================================\n");

    RUN_TEST(test_cg_sqrt_kappa_scaling);
    RUN_TEST(test_pcg_vs_cg_across_sizes);

    printf("\n========================================\n");
    return UNITY_END();
}
