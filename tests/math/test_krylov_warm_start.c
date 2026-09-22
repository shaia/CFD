/**
 * @file test_krylov_warm_start.c
 * @brief The Krylov solvers must be idempotent at their own solution.
 *
 * Re-solving from an already-converged field must return that field. It is the
 * weakest correctness property a linear solver has -- weaker than accuracy,
 * weaker than a convergence rate -- and it holds for an arbitrary initial guess
 * only if the operator used to form the initial residual is the operator the
 * iteration then inverts.
 *
 * The Krylov solvers here update interior points only, and their search
 * directions carry a permanent zero halo, so the operator they invert holds the
 * walls at zero. The initial residual, in contrast, is formed from x after
 * apply_bc has filled its halo with zero-gradient copies of the interior. For a
 * zero initial guess the two agree, the copies being copies of zeros, which is
 * why the existing tests pass. For any other guess they disagree at
 * wall-adjacent points by O(x/h^2), and the solve converges to a field that
 * solves neither system.
 *
 * That reaches real runs because the projection solver warm-starts each
 * pressure solve from the pressure of the previous step, so every solve after
 * the first took the inconsistent path.
 *
 * These tests state the property without reference to halos: solve, then solve
 * again from the answer, and require the answer back.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

#define GRID_N 33

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/**
 * Smooth interior RHS, shifted to exactly zero interior mean.
 *
 * The mean matters only for the Red-Black SOR control below: that solver honours
 * the zero-gradient walls on every sweep, so it inverts the true singular Neumann
 * operator and stalls on any component of the RHS that lies in its nullspace, the
 * constants. The Krylov solvers hold the walls at zero and are insensitive to it.
 */
static void build_rhs(double* rhs, size_t n, double h) {
    memset(rhs, 0, n * n * sizeof(double));
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double x = (double)i * h;
            double y = (double)j * h;
            double v = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
            rhs[IDX_2D(i, j, n)] = v;
            sum += v;
            count++;
        }
    }

    double mean = sum / (double)count;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            rhs[IDX_2D(i, j, n)] -= mean;
        }
    }
}

static double max_interior_diff(const double* a, const double* b, size_t n) {
    double worst = 0.0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double d = fabs(a[IDX_2D(i, j, n)] - b[IDX_2D(i, j, n)]);
            if (d > worst) {
                worst = d;
            }
        }
    }
    return worst;
}

static double max_interior_abs(const double* a, size_t n) {
    double worst = 0.0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double d = fabs(a[IDX_2D(i, j, n)]);
            if (d > worst) {
                worst = d;
            }
        }
    }
    return worst;
}

/**
 * Solve once from x, in place. Returns 0 on success and -1 when the backend is
 * not built or not available at runtime; any other failure is a test failure.
 */
static int solve_once(poisson_solver_method_t method,
                      poisson_solver_backend_t backend,
                      double* x, const double* rhs, size_t n, double h,
                      double tolerance,
                      poisson_solver_stats_t* stats_out) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        return -1;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = tolerance;
    params.max_iterations = 200000;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        poisson_solver_destroy(solver);
        return -1;
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    double* work = (double*)cfd_calloc(n * n, sizeof(double));
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, work, rhs, &stats);
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "solve did not converge");
    if (stats_out) {
        *stats_out = stats;
    }

    cfd_free(work);
    poisson_solver_destroy(solver);
    return 0;
}

/** Solve, then solve again from the answer; the second must return the first. */
static void assert_idempotent(poisson_solver_method_t method,
                              poisson_solver_backend_t backend,
                              double tolerance, const char* label) {
    const size_t n = GRID_N;
    const double h = 1.0 / (double)(n - 1);
    size_t total = n * n;

    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x_cold = (double*)cfd_calloc(total, sizeof(double));
    double* x_warm = (double*)cfd_calloc(total, sizeof(double));
    build_rhs(rhs, n, h);

    if (solve_once(method, backend, x_cold, rhs, n, h, tolerance, NULL) != 0) {
        cfd_free(rhs);
        cfd_free(x_cold);
        cfd_free(x_warm);
        TEST_IGNORE_MESSAGE("backend unavailable");
        return;
    }

    /* A non-trivial answer, or the comparison below would prove nothing. */
    double scale = max_interior_abs(x_cold, n);
    TEST_ASSERT_TRUE_MESSAGE(scale > 1e-3, "first solve produced a near-zero field");

    memcpy(x_warm, x_cold, total * sizeof(double));
    poisson_solver_stats_t warm;
    TEST_ASSERT_EQUAL(0, solve_once(method, backend, x_warm, rhs, n, h, tolerance, &warm));

    double drift = max_interior_diff(x_warm, x_cold, n);
    printf("%-18s max|x_warm - x_cold| = %.3e   |x| = %.3e   warm iters = %d\n",
           label, drift, scale, warm.iterations);

    /* The warm solve starts at the answer, so it must recognise it and move
     * nothing measurable. Two runs of the same solver to the same relative
     * tolerance can legitimately differ by about that tolerance, so allow a
     * decade of slack above it and no more. */
    TEST_ASSERT_TRUE_MESSAGE(drift < 10.0 * tolerance * scale,
        "re-solving from the converged field moved it: the initial residual and "
        "the iteration disagree about the operator");

    cfd_free(rhs);
    cfd_free(x_cold);
    cfd_free(x_warm);
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

/* The Krylov solvers reach 1e-12 on a 33x33 grid in tens of iterations. */
#define KRYLOV_TOL 1e-12

/* Red-Black SOR needs O(n^2) sweeps for the same residual, so it is run to a
 * tolerance it can actually reach in a test. */
#define SOR_TOL 1e-9

void test_cg_scalar_idempotent(void) {
    assert_idempotent(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, KRYLOV_TOL, "CG scalar");
}

void test_cg_simd_idempotent(void) {
    assert_idempotent(POISSON_METHOD_CG, POISSON_BACKEND_SIMD, KRYLOV_TOL, "CG SIMD");
}

void test_cg_omp_idempotent(void) {
    assert_idempotent(POISSON_METHOD_CG, POISSON_BACKEND_OMP, KRYLOV_TOL, "CG OMP");
}

void test_bicgstab_scalar_idempotent(void) {
    assert_idempotent(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR, KRYLOV_TOL,
                      "BiCGSTAB scalar");
}

void test_bicgstab_simd_idempotent(void) {
    assert_idempotent(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SIMD, KRYLOV_TOL,
                      "BiCGSTAB SIMD");
}

void test_bicgstab_omp_idempotent(void) {
    assert_idempotent(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_OMP, KRYLOV_TOL,
                      "BiCGSTAB OMP");
}

void test_gmres_scalar_idempotent(void) {
    assert_idempotent(POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR, KRYLOV_TOL,
                      "GMRES scalar");
}

void test_gmres_simd_idempotent(void) {
    assert_idempotent(POISSON_METHOD_GMRES, POISSON_BACKEND_SIMD, KRYLOV_TOL,
                      "GMRES SIMD");
}

void test_gmres_omp_idempotent(void) {
    assert_idempotent(POISSON_METHOD_GMRES, POISSON_BACKEND_OMP, KRYLOV_TOL,
                      "GMRES OMP");
}

/* The stationary solvers re-apply the BC hook on every sweep, so they never had
 * the mismatch. Pinned here as the control: it passed before this fix too. */
void test_redblack_sor_scalar_idempotent(void) {
    assert_idempotent(POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR, SOR_TOL,
                      "RB-SOR scalar");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_cg_scalar_idempotent);
    RUN_TEST(test_cg_simd_idempotent);
    RUN_TEST(test_cg_omp_idempotent);
    RUN_TEST(test_bicgstab_scalar_idempotent);
    RUN_TEST(test_bicgstab_simd_idempotent);
    RUN_TEST(test_bicgstab_omp_idempotent);
    RUN_TEST(test_gmres_scalar_idempotent);
    RUN_TEST(test_gmres_simd_idempotent);
    RUN_TEST(test_gmres_omp_idempotent);
    RUN_TEST(test_redblack_sor_scalar_idempotent);
    return UNITY_END();
}
