/**
 * @file test_helmholtz_shift.c
 * @brief Tests for poisson_solver_params_t.helmholtz_shift
 *
 * The solved equation is
 *
 *     nabla^2 x - sigma*x = rhs
 *
 * with sigma = 0 reproducing the pure Poisson path exactly. The shift exists
 * for implicit diffusion, where (I - nu*dt*nabla^2)u = b becomes
 * sigma = 1/(nu*dt) with rhs = -b/(nu*dt).
 *
 * WHAT CG ACTUALLY SOLVES, which shapes every test below. CG updates interior
 * points only, and its search directions carry a permanent zero halo, so the
 * iteration inverts the operator under homogeneous Dirichlet walls. The
 * default apply_bc hook overwrites the boundary with zero-gradient copies of
 * the interior before the loop starts, so an initial guess of zero asks for
 * zero walls. These tests therefore use manufactured solutions that vanish on
 * the boundary, which is the problem the discrete system actually poses.
 *
 * Two consequences differ from the unshifted pressure solve this module was
 * built for:
 *
 *  - The Neumann nullspace, and hence the zero-mean compatibility condition on
 *    the rhs, belongs to the multigrid and SOR paths, which honour the BC hook
 *    throughout. It never reaches CG.
 *  - sigma > 0 makes the operator strictly diagonally dominant, so CG converges
 *    faster than on the unshifted problem and, at large sigma, nearly stops
 *    caring how fine the grid is. That is the economic premise of using this
 *    for implicit diffusion, so it is asserted rather than assumed.
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

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/** Solve with the scalar CG backend at the given shift. */
static cfd_status_t solve_shifted(size_t nx, size_t ny, size_t nz,
                                  double dx, double dy, double dz,
                                  double sigma, double tolerance,
                                  const double* rhs, double* x,
                                  poisson_solver_stats_t* stats_out) {
    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                     POISSON_BACKEND_SCALAR);
    if (!solver) {
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.helmholtz_shift = sigma;
    params.tolerance = tolerance;
    params.max_iterations = 5000;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return status;
    }

    size_t n = nx * ny * nz;
    double* work = (double*)cfd_calloc(n, sizeof(double));
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, work, rhs, &stats);
    if (stats_out) {
        *stats_out = stats;
    }

    cfd_free(work);
    poisson_solver_destroy(solver);
    return CFD_SUCCESS;
}

/** Max |x - expected| over interior points. */
static double max_interior_error(const double* x, const double* expected,
                                 size_t nx, size_t ny, size_t nz) {
    size_t stride_z = (nz > 1) ? nx * ny : 0;
    size_t k0 = (nz > 1) ? 1 : 0;
    size_t k1 = (nz > 1) ? nz - 1 : 1;
    double worst = 0.0;

    for (size_t k = k0; k < k1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                double e = fabs(x[idx] - expected[idx]);
                if (e > worst || isnan(e)) {
                    worst = e;
                }
            }
        }
    }
    return worst;
}

/**
 * A manufactured solution vanishing on the boundary of the unit square.
 *
 * Two modes rather than one: a single discrete eigenvector is solved by CG in
 * exactly one iteration, which would let the exactness test pass without
 * exercising the Krylov recurrence at all.
 */
static void fill_manufactured_2d(double* u, size_t n, double h) {
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            double xx = (double)i * h;
            double yy = (double)j * h;
            u[IDX_2D(i, j, n)] =
                  sin(M_PI * xx) * sin(M_PI * yy)
                + 0.5 * sin(3.0 * M_PI * xx) * sin(2.0 * M_PI * yy);
        }
    }
}

/** rhs = nabla^2_h u - sigma*u, using the solver's own stencil. */
static void build_rhs_2d(const double* u, double* rhs, size_t n, double h,
                         double sigma) {
    memset(rhs, 0, n * n * sizeof(double));
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            size_t idx = IDX_2D(i, j, n);
            double lap = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) / (h * h)
                       + (u[idx + n] - 2.0 * u[idx] + u[idx - n]) / (h * h);
            rhs[idx] = lap - sigma * u[idx];
        }
    }
}

/* ============================================================================
 * EXACTNESS
 *
 * The rhs is built from the same stencil the solver uses, so the discrete
 * problem has an exact answer, checked absolutely rather than up to an
 * additive constant. Sweeping sigma over eight orders of magnitude catches a
 * shift that is dropped, doubled, or signed wrongly: at sigma = 1e8 the mass
 * term dominates and x -> -rhs/sigma, where a sign error is not subtle.
 * ============================================================================ */

void test_discrete_mms_is_exact(void) {
    const size_t n = 33;
    const double h = 1.0 / (double)(n - 1);
    const double sigmas[] = {0.0, 1.0, 1e3, 1e8};

    size_t total = n * n;
    double* u = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x = (double*)cfd_calloc(total, sizeof(double));

    fill_manufactured_2d(u, n, h);

    for (size_t s = 0; s < sizeof(sigmas) / sizeof(sigmas[0]); s++) {
        double sigma = sigmas[s];
        build_rhs_2d(u, rhs, n, h, sigma);
        memset(x, 0, total * sizeof(double));

        cfd_status_t status = solve_shifted(n, n, 1, h, h, 0.0, sigma, 1e-13,
                                            rhs, x, NULL);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

        double err = max_interior_error(x, u, n, n, 1);
        printf("discrete MMS sigma=%-8.0e max|x - u| = %.3e\n", sigma, err);
        TEST_ASSERT_TRUE_MESSAGE(err < 1e-8,
                                 "shifted solve missed the exact discrete solution");
    }

    cfd_free(u);
    cfd_free(rhs);
    cfd_free(x);
}

void test_shift_changes_the_answer(void) {
    /* Guards against the shift being accepted and then ignored: one rhs under
     * two different shifts must give materially different solutions. */
    const size_t n = 33;
    const double h = 1.0 / (double)(n - 1);
    size_t total = n * n;

    double* u = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x0 = (double*)cfd_calloc(total, sizeof(double));
    double* x1 = (double*)cfd_calloc(total, sizeof(double));

    fill_manufactured_2d(u, n, h);
    build_rhs_2d(u, rhs, n, h, 0.0);

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      solve_shifted(n, n, 1, h, h, 0.0, 0.0, 1e-12, rhs, x0, NULL));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      solve_shifted(n, n, 1, h, h, 0.0, 50.0, 1e-12, rhs, x1, NULL));

    double diff = max_interior_error(x0, x1, n, n, 1);
    printf("sigma 0 vs 50: max|x0 - x1| = %.3e\n", diff);
    TEST_ASSERT_TRUE_MESSAGE(diff > 1e-3, "the shift was accepted but not applied");

    cfd_free(u);
    cfd_free(rhs);
    cfd_free(x0);
    cfd_free(x1);
}

void test_discrete_mms_is_exact_3d(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);
    const double sigma = 25.0;
    size_t total = n * n * n;
    size_t plane = n * n;

    double* u = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x = (double*)cfd_calloc(total, sizeof(double));

    for (size_t k = 0; k < n; k++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < n; i++) {
                double xx = (double)i * h;
                double yy = (double)j * h;
                double zz = (double)k * h;
                u[k * plane + IDX_2D(i, j, n)] =
                    sin(M_PI * xx) * sin(M_PI * yy) * sin(M_PI * zz);
            }
        }
    }

    for (size_t k = 1; k < n - 1; k++) {
        for (size_t j = 1; j < n - 1; j++) {
            for (size_t i = 1; i < n - 1; i++) {
                size_t idx = k * plane + IDX_2D(i, j, n);
                double lap = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) / (h * h)
                           + (u[idx + n] - 2.0 * u[idx] + u[idx - n]) / (h * h)
                           + (u[idx + plane] - 2.0 * u[idx] + u[idx - plane]) / (h * h);
                rhs[idx] = lap - sigma * u[idx];
            }
        }
    }

    cfd_status_t status = solve_shifted(n, n, n, h, h, h, sigma, 1e-13, rhs, x, NULL);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    double err = max_interior_error(x, u, n, n, n);
    printf("3D discrete MMS: max|x - u| = %.3e\n", err);
    TEST_ASSERT_TRUE(err < 1e-8);

    cfd_free(u);
    cfd_free(rhs);
    cfd_free(x);
}

/* ============================================================================
 * BIT-IDENTITY AT SIGMA = 0
 *
 * The shift goes through a guarded axpy rather than being fused into the
 * kernel precisely so that sigma = 0 executes the original instruction stream.
 * memcmp, not TEST_ASSERT_EQUAL_DOUBLE: the latter has a tolerance and would
 * not notice +0.0 where -0.0 belongs.
 * ============================================================================ */

void test_zero_shift_is_bit_identical(void) {
    const size_t n = 33;
    const double h = 1.0 / (double)(n - 1);
    size_t total = n * n;

    double* u = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    fill_manufactured_2d(u, n, h);
    build_rhs_2d(u, rhs, n, h, 0.0);

    double* x_default = (double*)cfd_calloc(total, sizeof(double));
    double* x_zero = (double*)cfd_calloc(total, sizeof(double));
    double* x_negzero = (double*)cfd_calloc(total, sizeof(double));
    poisson_solver_stats_t s_zero, s_negzero;

    /* Baseline: params_default(), the field untouched by the caller. */
    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                     POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params));
    double* work = (double*)cfd_calloc(total, sizeof(double));
    poisson_solver_stats_t s_default = poisson_solver_stats_default();
    poisson_solver_solve(solver, x_default, work, rhs, &s_default);
    poisson_solver_destroy(solver);
    cfd_free(work);

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      solve_shifted(n, n, 1, h, h, 0.0, 0.0, 1e-10, rhs, x_zero, &s_zero));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      solve_shifted(n, n, 1, h, h, 0.0, -0.0, 1e-10, rhs, x_negzero, &s_negzero));

    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(x_default, x_zero, total * sizeof(double),
                                     "sigma = 0.0 changed the unshifted result");
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(x_default, x_negzero, total * sizeof(double),
                                     "sigma = -0.0 changed the unshifted result");
    TEST_ASSERT_EQUAL_INT(s_default.iterations, s_zero.iterations);
    TEST_ASSERT_EQUAL_INT(s_default.iterations, s_negzero.iterations);
    TEST_ASSERT_EQUAL_MEMORY(&s_default.final_residual, &s_zero.final_residual,
                             sizeof(double));

    cfd_free(u);
    cfd_free(rhs);
    cfd_free(x_default);
    cfd_free(x_zero);
    cfd_free(x_negzero);
}

/* ============================================================================
 * CONDITIONING
 *
 * lambda_min(A) = sigma exactly and lambda_max ~ sigma + 8/h^2, so
 * cond = 1 + 8/(sigma*h^2) and CG iterations go as its square root.
 * ============================================================================ */

static int iterations_at(size_t n, double sigma) {
    const double h = 1.0 / (double)(n - 1);
    size_t total = n * n;
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x = (double*)cfd_calloc(total, sizeof(double));

    /* A single discrete eigenmode is solved by CG in one step, which would
     * make this measure nothing. Mix several modes plus a constant. */
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double xx = (double)i * h;
            double yy = (double)j * h;
            rhs[IDX_2D(i, j, n)] = 1.0
                + sin(2.0 * M_PI * xx) * sin(2.0 * M_PI * yy)
                + 0.7 * sin(7.0 * M_PI * xx) * cos(3.0 * M_PI * yy)
                + 0.3 * cos(11.0 * M_PI * xx) * sin(5.0 * M_PI * yy);
        }
    }

    poisson_solver_stats_t stats;
    cfd_status_t status = solve_shifted(n, n, 1, h, h, 0.0, sigma, 1e-10, rhs, x, &stats);
    cfd_free(rhs);
    cfd_free(x);
    return (status == CFD_SUCCESS) ? stats.iterations : -1;
}

void test_larger_shift_converges_faster(void) {
    const double sigmas[] = {0.0, 1e1, 1e3, 1e6};
    int prev = -1;

    for (size_t s = 0; s < sizeof(sigmas) / sizeof(sigmas[0]); s++) {
        int iters = iterations_at(65, sigmas[s]);
        TEST_ASSERT_GREATER_THAN_INT(0, iters);
        printf("65x65 sigma=%-8.0e iterations=%d\n", sigmas[s], iters);
        if (prev >= 0) {
            TEST_ASSERT_TRUE_MESSAGE(iters <= prev,
                                     "iteration count should not grow with the shift");
        }
        prev = iters;
    }
    TEST_ASSERT_TRUE_MESSAGE(prev <= 10,
                             "a large shift should converge in a few iterations");
}

void test_large_shift_nearly_removes_grid_dependence(void) {
    /* Unshifted, the count grows like 1/h. At sigma = 1e6 the mass term
     * dominates and it barely moves. Asserted against the unshifted growth on
     * the same grids rather than a bare constant, so this measures the
     * property and not the machine. */
    int u33 = iterations_at(33, 0.0);
    int u129 = iterations_at(129, 0.0);
    int s33 = iterations_at(33, 1e6);
    int s129 = iterations_at(129, 1e6);

    printf("unshifted 33=%d 129=%d   shifted(1e6) 33=%d 129=%d\n",
           u33, u129, s33, s129);

    TEST_ASSERT_GREATER_THAN_INT(0, u33);
    TEST_ASSERT_GREATER_THAN_INT(0, s33);
    TEST_ASSERT_TRUE_MESSAGE(u129 > 2 * u33,
                             "unshifted CG should degrade with refinement");
    TEST_ASSERT_TRUE_MESSAGE(s129 <= s33 + 3,
                             "a large shift should be nearly grid-independent");
    TEST_ASSERT_TRUE(s129 * 4 < u129);
}

/* ============================================================================
 * VALIDATION AND THE CAPABILITY TABLE
 *
 * The rejection sweep is what makes central default-deny trustworthy: a
 * backend that has not been taught the shift must refuse it rather than
 * silently solve the unshifted equation.
 * ============================================================================ */

void test_invalid_shift_rejected(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);
    const double bad[] = {-1.0, -1e-12, NAN, INFINITY};

    for (size_t b = 0; b < sizeof(bad) / sizeof(bad[0]); b++) {
        poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                         POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL(solver);
        poisson_solver_params_t params = poisson_solver_params_default();
        params.helmholtz_shift = bad[b];
        cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID, status,
                                  "a negative or non-finite shift must be rejected");
        poisson_solver_destroy(solver);
    }
}

void test_unsupported_backends_reject_the_shift(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_JACOBI, POISSON_METHOD_GAUSS_SEIDEL, POISSON_METHOD_SOR,
        POISSON_METHOD_REDBLACK_SOR, POISSON_METHOD_CG, POISSON_METHOD_BICGSTAB,
        POISSON_METHOD_GMRES, POISSON_METHOD_MULTIGRID
    };
    const poisson_solver_backend_t backends[] = {
        POISSON_BACKEND_SCALAR, POISSON_BACKEND_SIMD,
        POISSON_BACKEND_OMP, POISSON_BACKEND_GPU
    };

    int checked = 0;
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        for (size_t b = 0; b < sizeof(backends) / sizeof(backends[0]); b++) {
            poisson_solver_t* solver = poisson_solver_create(methods[m], backends[b]);
            if (!solver) {
                continue;  /* backend not compiled in */
            }

            poisson_solver_params_t params = poisson_solver_params_default();
            params.helmholtz_shift = 1.0;
            cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);

            int supported = (methods[m] == POISSON_METHOD_CG &&
                             backends[b] == POISSON_BACKEND_SCALAR);
            if (supported) {
                TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status,
                                          "scalar CG must accept the shift");
            } else {
                /* Multigrid rejects a 17x17 grid as non-2^k+1 before reaching
                 * the shift check, and a backend can be absent at runtime;
                 * both are legitimate. Silently succeeding is not. */
                TEST_ASSERT_NOT_EQUAL_MESSAGE(CFD_SUCCESS, status,
                    "a backend without shift support must not silently accept it");
            }
            checked++;
            poisson_solver_destroy(solver);
        }
    }
    printf("rejection sweep covered %d (method, backend) pairs\n", checked);
    TEST_ASSERT_GREATER_THAN_INT(4, checked);
}

void test_mg_preconditioner_rejects_the_shift(void) {
    /* The inner V-cycle is built from default params, so it would precondition
     * with the unshifted operator. Refuse until multigrid carries the shift. */
    const size_t n = 33;
    const double h = 1.0 / (double)(n - 1);

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                     POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    poisson_solver_params_t params = poisson_solver_params_default();
    params.helmholtz_shift = 1.0;
    params.preconditioner = POISSON_PRECOND_MULTIGRID;
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params));
    poisson_solver_destroy(solver);
}

void test_jacobi_preconditioner_accepts_the_shift(void) {
    /* One diagonal term, folded into diag_inv at init. */
    const size_t n = 33;
    const double h = 1.0 / (double)(n - 1);
    const double sigma = 30.0;
    size_t total = n * n;

    double* u = (double*)cfd_calloc(total, sizeof(double));
    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x = (double*)cfd_calloc(total, sizeof(double));
    fill_manufactured_2d(u, n, h);
    build_rhs_2d(u, rhs, n, h, sigma);

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                     POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    poisson_solver_params_t params = poisson_solver_params_default();
    params.helmholtz_shift = sigma;
    params.preconditioner = POISSON_PRECOND_JACOBI;
    params.tolerance = 1e-13;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params));

    double* work = (double*)cfd_calloc(total, sizeof(double));
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, work, rhs, &stats);

    double err = max_interior_error(x, u, n, n, 1);
    printf("Jacobi-preconditioned shifted solve: max|x - u| = %.3e\n", err);
    TEST_ASSERT_TRUE(err < 1e-8);

    cfd_free(work);
    poisson_solver_destroy(solver);
    cfd_free(u);
    cfd_free(rhs);
    cfd_free(x);
}

/* ============================================================================
 * RUNNER
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_discrete_mms_is_exact);
    RUN_TEST(test_shift_changes_the_answer);
    RUN_TEST(test_discrete_mms_is_exact_3d);
    RUN_TEST(test_zero_shift_is_bit_identical);
    RUN_TEST(test_larger_shift_converges_faster);
    RUN_TEST(test_large_shift_nearly_removes_grid_dependence);

    RUN_TEST(test_invalid_shift_rejected);
    RUN_TEST(test_unsupported_backends_reject_the_shift);
    RUN_TEST(test_mg_preconditioner_rejects_the_shift);
    RUN_TEST(test_jacobi_preconditioner_accepts_the_shift);

    return UNITY_END();
}
