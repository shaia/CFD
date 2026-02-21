/**
 * @file test_pcg_convergence.c
 * @brief Preconditioned Conjugate Gradient convergence tests
 *
 * These tests verify that the Jacobi-preconditioned CG (PCG) solver:
 *   - Converges correctly to the same solution as standard CG
 *   - Requires the same or fewer iterations than unpreconditioned CG (depending on problem structure)
 *   - Works consistently across scalar and SIMD backends
 *   - Behaves identically to standard CG when preconditioner is disabled
 *
 * Jacobi preconditioner for -∇²:
 *   M = diag(A) = 2/dx² + 2/dy²
 *   M⁻¹ = 1 / (2/dx² + 2/dy²)
 *
 * For uniform-grid Laplacian with constant coefficients, the Jacobi diagonal
 * is constant (2/dx² + 2/dy² = 4/h²), so preconditioning does not reduce
 * iterations. Iteration reduction occurs with variable coefficients or
 * non-uniform grids where the diagonal varies spatially.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

#define DOMAIN_XMIN 0.0
#define DOMAIN_XMAX 1.0
#define DOMAIN_YMIN 0.0
#define DOMAIN_YMAX 1.0

#define TOLERANCE 1e-8
#define MAX_ITERATIONS 2000

/* For uniform-grid Laplacian, Jacobi preconditioner doesn't reduce iterations
 * because the diagonal is constant. We verify PCG doesn't perform worse. */
#define MAX_ITER_RATIO 1.05  /* Allow 5% tolerance for numerical differences */

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static double* create_field(size_t nx, size_t ny) {
    double* field = (double*)cfd_malloc(nx * ny * sizeof(double));
    if (field) {
        for (size_t i = 0; i < nx * ny; i++) {
            field[i] = 0.0;
        }
    }
    return field;
}

static void init_nontrivial_guess(double* p, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            p[IDX_2D(i, j, nx)] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
    }
}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2πx)cos(2πy) with discrete interior mean subtracted.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy) {
    /* First pass: initialize sinusoidal values */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Second pass: compute interior mean and subtract */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            interior_sum += rhs[IDX_2D(i, j, nx)];
            interior_count++;
        }
    }

    if (interior_count > 0) {
        double interior_mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                rhs[IDX_2D(i, j, nx)] -= interior_mean;
            }
        }
    }

    /* Set boundary RHS to 0 */
    for (size_t i = 0; i < nx; i++) {
        rhs[i] = 0.0;
        rhs[(ny - 1) * nx + i] = 0.0;
    }
    for (size_t j = 1; j < ny - 1; j++) {
        rhs[j * nx] = 0.0;
        rhs[j * nx + (nx - 1)] = 0.0;
    }
}

/**
 * Compute L2 norm of difference between two fields (interior only)
 */
static double compute_l2_difference(const double* a, const double* b,
                                     size_t nx, size_t ny) {
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double diff = a[idx] - b[idx];
            sum_sq += diff * diff;
            count++;
        }
    }
    return sqrt(sum_sq / count);
}

/* ============================================================================
 * TEST: PCG CONVERGES CORRECTLY
 * ============================================================================
 *
 * Verify that PCG converges to the same solution as standard CG.
 */

void test_pcg_converges_correctly(void) {
    printf("\n    Testing PCG converges to correct solution...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    double* p_cg = create_field(n, n);
    double* p_pcg = create_field(n, n);
    double* p_temp = create_field(n, n);
    double* rhs = create_field(n, n);

    TEST_ASSERT_NOT_NULL_MESSAGE(p_cg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_pcg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs(rhs, n, n, dx, dy);

    /* Solve with standard CG */
    init_nontrivial_guess(p_cg, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        params.preconditioner = POISSON_PRECOND_NONE;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p_cg, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

        printf("      CG:  %d iterations, final residual: %.2e\n",
               stats.iterations, stats.final_residual);

        poisson_solver_destroy(solver);
    }

    /* Solve with PCG */
    init_nontrivial_guess(p_pcg, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create PCG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        params.preconditioner = POISSON_PRECOND_JACOBI;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p_pcg, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

        printf("      PCG: %d iterations, final residual: %.2e\n",
               stats.iterations, stats.final_residual);

        poisson_solver_destroy(solver);
    }

    /* Compare solutions */
    double l2_diff = compute_l2_difference(p_cg, p_pcg, n, n);
    printf("      L2 difference between CG and PCG solutions: %.2e\n", l2_diff);

    /* Solutions should be nearly identical (within solver tolerance) */
    TEST_ASSERT_TRUE_MESSAGE(l2_diff < 1e-6,
        "PCG and CG should converge to the same solution");

    cfd_free(p_cg);
    cfd_free(p_pcg);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST: PCG DOES NOT INCREASE ITERATIONS
 * ============================================================================
 *
 * Verify that Jacobi-preconditioned CG doesn't perform worse than standard CG.
 *
 * Note: For the uniform-grid Laplacian with constant coefficients, the Jacobi
 * preconditioner (M = diag(A)) is a constant scalar (2/dx² + 2/dy²), which
 * doesn't change the condition number. Therefore, PCG and CG have essentially
 * the same convergence rate for this problem.
 *
 * For problems with variable coefficients or non-uniform grids, Jacobi
 * preconditioning would provide more benefit.
 */

void test_pcg_iteration_comparison(void) {
    printf("\n    Testing PCG doesn't perform worse than CG...\n");
    printf("      (For uniform-grid Laplacian, Jacobi is constant - no speedup expected)\n");

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

        double* p = create_field(n, n);
        double* p_temp = create_field(n, n);
        double* rhs = create_field(n, n);

        if (!p || !p_temp || !rhs) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        int cg_iters = 0, pcg_iters = 0;

        /* Standard CG */
        init_nontrivial_guess(p, n, n);
        {
            poisson_solver_t* solver = poisson_solver_create(
                POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
            if (!solver) {
                cfd_free(p);
                cfd_free(p_temp);
                cfd_free(rhs);
                printf("      %3zux%-3zu: CG solver not available\n", n, n);
                continue;
            }

            poisson_solver_params_t params = poisson_solver_params_default();
            params.tolerance = TOLERANCE;
            params.max_iterations = MAX_ITERATIONS;
            params.preconditioner = POISSON_PRECOND_NONE;

            cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
            TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

            poisson_solver_stats_t stats = poisson_solver_stats_default();
            status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
            TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
            cg_iters = stats.iterations;

            poisson_solver_destroy(solver);
        }

        /* Preconditioned CG */
        init_nontrivial_guess(p, n, n);
        {
            poisson_solver_t* solver = poisson_solver_create(
                POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
            TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create PCG solver");

            poisson_solver_params_t params = poisson_solver_params_default();
            params.tolerance = TOLERANCE;
            params.max_iterations = MAX_ITERATIONS;
            params.preconditioner = POISSON_PRECOND_JACOBI;

            cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
            TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

            poisson_solver_stats_t stats = poisson_solver_stats_default();
            status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
            TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
            TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
            pcg_iters = stats.iterations;

            poisson_solver_destroy(solver);
        }

        double ratio = (double)pcg_iters / (double)cg_iters;

        printf("      %3zux%-3zu: CG=%3d iters, PCG=%3d iters (ratio=%.2f)\n",
               n, n, cg_iters, pcg_iters, ratio);

        /* PCG should not require significantly more iterations than CG */
        TEST_ASSERT_TRUE_MESSAGE(ratio <= MAX_ITER_RATIO,
            "PCG should not perform significantly worse than CG");

        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
    }
}

/* ============================================================================
 * TEST: DISABLED PRECONDITIONER EQUALS STANDARD CG
 * ============================================================================
 *
 * Verify that setting preconditioner to NONE gives identical behavior
 * to not specifying a preconditioner.
 */

void test_disabled_precond_equals_cg(void) {
    printf("\n    Testing disabled preconditioner equals standard CG...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    double* p1 = create_field(n, n);
    double* p2 = create_field(n, n);
    double* p_temp = create_field(n, n);
    double* rhs = create_field(n, n);

    TEST_ASSERT_NOT_NULL_MESSAGE(p1, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p2, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs(rhs, n, n, dx, dy);

    int iters1 = 0, iters2 = 0;

    /* CG with default params (no explicit preconditioner setting) */
    init_nontrivial_guess(p1, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        /* Don't explicitly set preconditioner - use default (NONE) */

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p1, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
        iters1 = stats.iterations;

        poisson_solver_destroy(solver);
    }

    /* CG with explicit POISSON_PRECOND_NONE */
    init_nontrivial_guess(p2, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        params.preconditioner = POISSON_PRECOND_NONE;  /* Explicit NONE */

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p2, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
        iters2 = stats.iterations;

        poisson_solver_destroy(solver);
    }

    printf("      Default params: %d iterations\n", iters1);
    printf("      Explicit NONE:  %d iterations\n", iters2);

    /* Iteration counts should be identical */
    TEST_ASSERT_EQUAL_INT_MESSAGE(iters1, iters2,
        "Default params and explicit NONE should give same iteration count");

    /* Solutions should be identical */
    double l2_diff = compute_l2_difference(p1, p2, n, n);
    TEST_ASSERT_TRUE_MESSAGE(l2_diff < 1e-12,
        "Default params and explicit NONE should give identical solutions");

    cfd_free(p1);
    cfd_free(p2);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST: SIMD BACKEND CONSISTENCY
 * ============================================================================
 *
 * Verify that scalar and SIMD backends produce consistent PCG results.
 */

void test_simd_backend_consistency(void) {
    printf("\n    Testing SIMD backend consistency with scalar...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    double* p_scalar = create_field(n, n);
    double* p_simd = create_field(n, n);
    double* p_temp = create_field(n, n);
    double* rhs = create_field(n, n);

    TEST_ASSERT_NOT_NULL_MESSAGE(p_scalar, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_simd, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs(rhs, n, n, dx, dy);

    int scalar_iters = 0, simd_iters = 0;

    /* Scalar PCG */
    init_nontrivial_guess(p_scalar, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create scalar CG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        params.preconditioner = POISSON_PRECOND_JACOBI;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p_scalar, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
        scalar_iters = stats.iterations;

        printf("      Scalar PCG: %d iterations\n", scalar_iters);

        poisson_solver_destroy(solver);
    }

    /* SIMD PCG (if available) */
    init_nontrivial_guess(p_simd, n, n);
    {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SIMD);

        if (!solver) {
            printf("      SIMD PCG: not available (skipping)\n");
            cfd_free(p_scalar);
            cfd_free(p_simd);
            cfd_free(p_temp);
            cfd_free(rhs);
            return;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = TOLERANCE;
        params.max_iterations = MAX_ITERATIONS;
        params.preconditioner = POISSON_PRECOND_JACOBI;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p_simd, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);
        simd_iters = stats.iterations;

        printf("      SIMD PCG:   %d iterations\n", simd_iters);

        poisson_solver_destroy(solver);
    }

    /* Compare results */
    double l2_diff = compute_l2_difference(p_scalar, p_simd, n, n);
    printf("      L2 difference: %.2e\n", l2_diff);

    /* Iteration counts should be identical or very close */
    int iter_diff = abs(scalar_iters - simd_iters);
    TEST_ASSERT_TRUE_MESSAGE(iter_diff <= 1,
        "Scalar and SIMD should have same (or ±1) iteration count");

    /* Solutions should be nearly identical */
    TEST_ASSERT_TRUE_MESSAGE(l2_diff < 1e-10,
        "Scalar and SIMD should produce consistent solutions");

    cfd_free(p_scalar);
    cfd_free(p_simd);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("PCG Convergence Tests\n");
    printf("========================================\n");

    RUN_TEST(test_pcg_converges_correctly);
    RUN_TEST(test_pcg_iteration_comparison);
    RUN_TEST(test_disabled_precond_equals_cg);
    RUN_TEST(test_simd_backend_consistency);

    printf("\n========================================\n");
    return UNITY_END();
}
