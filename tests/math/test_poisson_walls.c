/**
 * @file test_poisson_walls.c
 * @brief Per-face walls on the Poisson operator (poisson_solver_params_t.walls)
 *
 * The Krylov solvers update interior points only, so the operator they invert is
 * whatever their vectors' halos say it is. Per-face walls make that explicit: each
 * face is either zero-gradient or holds a prescribed value.
 *
 * The split that matters is between the ITERATE, which carries the prescribed
 * values, and the SEARCH DIRECTIONS, which carry only the homogeneous part -- a
 * zero halo on a Dirichlet face. Applying the prescribed value to a direction
 * would make the operator affine rather than linear, which the Krylov recurrences
 * do not describe. test_dirichlet_lift_is_linear is the assertion that catches it.
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

#define WALLS_N 17
#define RAMP_SLOPE 2.0
#define RAMP_OFFSET 1.0

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/**
 * Solve with the given walls. Returns 0 on success, -1 if the backend is not
 * available; any other failure is a test failure.
 */
static int solve_with_walls(poisson_solver_method_t method,
                            poisson_solver_backend_t backend,
                            const poisson_walls_t* walls,
                            const double* rhs, double* x,
                            size_t n, double h) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        return -1;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-13;
    params.max_iterations = 20000;
    params.walls = *walls;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        poisson_solver_destroy(solver);
        return -1;
    }
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "init rejected a supported configuration");

    double* work = (double*)cfd_calloc(n * n, sizeof(double));
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, work, rhs, &stats);
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "solve did not converge");

    cfd_free(work);
    poisson_solver_destroy(solver);
    return 0;
}

/** Walls for a streamwise ramp: x-faces prescribed, y-faces zero-gradient. */
static poisson_walls_t ramp_walls(double left, double right) {
    poisson_walls_t w = poisson_walls_default();
    w.left = POISSON_WALL_DIRICHLET;
    w.right = POISSON_WALL_DIRICHLET;
    w.values.left = left;
    w.values.right = right;
    return w;
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

/* ============================================================================
 * 1. A LINEAR FIELD IS REPRODUCED EXACTLY
 * ============================================================================ */

/**
 * p = a*x + b has zero Laplacian and zero y-derivative, so it satisfies the
 * discrete system exactly under prescribed x-faces and zero-gradient y-faces.
 * Any error here is the operator, not discretisation.
 */
static void assert_linear_field_exact(poisson_solver_method_t method,
                                      poisson_solver_backend_t backend,
                                      const char* label) {
    const size_t n = WALLS_N;
    const double h = 1.0 / (double)(n - 1);
    size_t total = n * n;

    double* rhs = (double*)cfd_calloc(total, sizeof(double));   /* Laplace: rhs = 0 */
    double* x = (double*)cfd_calloc(total, sizeof(double));
    double* expected = (double*)cfd_calloc(total, sizeof(double));

    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            expected[IDX_2D(i, j, n)] = (RAMP_SLOPE * (double)i * h) + RAMP_OFFSET;
        }
    }

    poisson_walls_t w = ramp_walls(RAMP_OFFSET, (RAMP_SLOPE * 1.0) + RAMP_OFFSET);
    if (solve_with_walls(method, backend, &w, rhs, x, n, h) != 0) {
        cfd_free(rhs); cfd_free(x); cfd_free(expected);
        TEST_IGNORE_MESSAGE("backend unavailable");
        return;
    }

    double err = max_interior_diff(x, expected, n);
    printf("%-18s max|x - (a*x + b)| = %.3e\n", label, err);
    TEST_ASSERT_TRUE_MESSAGE(err < 1e-10,
        "prescribed x-faces did not reproduce the linear field");

    cfd_free(rhs);
    cfd_free(x);
    cfd_free(expected);
}

void test_linear_field_exact_cg_scalar(void) {
    assert_linear_field_exact(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, "CG scalar");
}
void test_linear_field_exact_cg_simd(void) {
    assert_linear_field_exact(POISSON_METHOD_CG, POISSON_BACKEND_SIMD, "CG SIMD");
}
void test_linear_field_exact_cg_omp(void) {
    assert_linear_field_exact(POISSON_METHOD_CG, POISSON_BACKEND_OMP, "CG OMP");
}
void test_linear_field_exact_bicgstab(void) {
    assert_linear_field_exact(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR, "BiCGSTAB scalar");
}
void test_linear_field_exact_gmres(void) {
    assert_linear_field_exact(POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR, "GMRES scalar");
}

/* ============================================================================
 * 2. THE DIRICHLET LIFT IS LINEAR
 * ============================================================================ */

/**
 * The operator is linear, so raising one face by V must shift the solution by
 * exactly the ramp V*x/Lx and nothing else:
 *
 *     solve(rhs, left=0, right=V) == solve(rhs, left=0, right=0) + V*x/Lx
 *
 * That ramp is discretely harmonic and y-independent, so it satisfies the
 * homogeneous problem exactly and the identity holds to solver tolerance. It
 * fails the moment a search direction is handed the prescribed value instead of
 * zero, because the iteration then describes an affine map rather than A.
 */
void test_dirichlet_lift_is_linear(void) {
    const size_t n = WALLS_N;
    const double h = 1.0 / (double)(n - 1);
    const double V = 3.5;
    size_t total = n * n;

    double* rhs = (double*)cfd_calloc(total, sizeof(double));
    double* x_lifted = (double*)cfd_calloc(total, sizeof(double));
    double* x_homog = (double*)cfd_calloc(total, sizeof(double));

    /* A non-trivial source, so this is not merely the ramp on both sides. */
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double xx = (double)i * h;
            double yy = (double)j * h;
            rhs[IDX_2D(i, j, n)] = sin(M_PI * xx) * cos(2.0 * M_PI * yy);
        }
    }

    poisson_walls_t lifted = ramp_walls(0.0, V);
    poisson_walls_t homog = ramp_walls(0.0, 0.0);

    if (solve_with_walls(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                         &lifted, rhs, x_lifted, n, h) != 0) {
        cfd_free(rhs); cfd_free(x_lifted); cfd_free(x_homog);
        TEST_IGNORE_MESSAGE("backend unavailable");
        return;
    }
    TEST_ASSERT_EQUAL(0, solve_with_walls(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                                          &homog, rhs, x_homog, n, h));

    /* Add the lift analytically and compare. */
    double worst = 0.0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double ramp = V * ((double)i * h) / 1.0;
            double d = fabs(x_lifted[IDX_2D(i, j, n)]
                            - (x_homog[IDX_2D(i, j, n)] + ramp));
            if (d > worst) {
                worst = d;
            }
        }
    }

    printf("lift linearity      max|lifted - (homogeneous + ramp)| = %.3e\n", worst);
    TEST_ASSERT_TRUE_MESSAGE(worst < 1e-9,
        "the Dirichlet lift is not linear: a search direction saw the prescribed value");

    cfd_free(rhs);
    cfd_free(x_lifted);
    cfd_free(x_homog);
}

/* ============================================================================
 * 3. THE SINGULARITY PREDICATE
 * ============================================================================ */

void test_singularity_predicate(void) {
    poisson_walls_t w = poisson_walls_default();
    TEST_ASSERT_TRUE_MESSAGE(poisson_walls_are_singular(&w, 1),
                             "all zero-gradient walls are singular");
    TEST_ASSERT_TRUE(poisson_walls_are_singular(&w, 8));

    w.right = POISSON_WALL_DIRICHLET;
    TEST_ASSERT_FALSE_MESSAGE(poisson_walls_are_singular(&w, 1),
                              "one prescribed face removes the nullspace");

    /* The z-faces do not exist on a 2D grid, so a Dirichlet one there cannot
     * pin anything and the operator is still singular. */
    poisson_walls_t z = poisson_walls_default();
    z.front = POISSON_WALL_DIRICHLET;
    TEST_ASSERT_TRUE_MESSAGE(poisson_walls_are_singular(&z, 1),
                             "a z-face on a 2D grid must not count");
    TEST_ASSERT_FALSE_MESSAGE(poisson_walls_are_singular(&z, 8),
                              "the same z-face does count in 3D");

    /* NULL means the defaults. */
    TEST_ASSERT_TRUE(poisson_walls_are_singular(NULL, 1));

    poisson_walls_t u = poisson_walls_uniform(POISSON_WALL_DIRICHLET, 0.0);
    TEST_ASSERT_FALSE_MESSAGE(poisson_walls_are_singular(&u, 1),
                              "uniform Dirichlet walls are nonsingular");
    poisson_walls_t g = poisson_walls_uniform(POISSON_WALL_ZERO_GRADIENT, 0.0);
    TEST_ASSERT_TRUE_MESSAGE(poisson_walls_are_singular(&g, 1),
                             "uniform zero-gradient walls are the default operator");
}

/* ============================================================================
 * 4. REJECTIONS
 * ============================================================================ */

static void noop_hook(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

/** Methods whose sweeps apply whole-domain walls must refuse, not ignore. */
void test_unsupported_methods_reject_walls(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_JACOBI, POISSON_METHOD_GAUSS_SEIDEL, POISSON_METHOD_SOR,
        POISSON_METHOD_REDBLACK_SOR, POISSON_METHOD_MULTIGRID
    };

    int checked = 0;
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        poisson_solver_t* solver = poisson_solver_create(methods[m], POISSON_BACKEND_SCALAR);
        if (!solver) {
            continue;
        }
        poisson_solver_params_t params = poisson_solver_params_default();
        params.walls = ramp_walls(0.0, 1.0);
        cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
            "a method that cannot honour per-face walls must reject them");
        checked++;
        poisson_solver_destroy(solver);
    }
    /* Derived from the list, not a constant: every method here has a scalar
     * implementation in every build, so all of them must have been reached. A
     * hand-written number would start lying the moment the list changed, or the
     * moment an optional backend joined the loop. */
    TEST_ASSERT_EQUAL_INT_MESSAGE((int)(sizeof(methods) / sizeof(methods[0])), checked,
        "every listed method must have been exercised");
}

/** Walls and a hook both prescribe wall values; taking both is ambiguous. */
void test_walls_with_hook_rejected(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    solver->apply_bc = noop_hook;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.walls = ramp_walls(0.0, 1.0);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
                      poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params));
    poisson_solver_destroy(solver);
}

/** A hook on its own, and default walls on their own, both stay fine. */
void test_hook_alone_and_defaults_accepted(void) {
    const size_t n = 17;
    const double h = 1.0 / (double)(n - 1);

    poisson_solver_t* a = poisson_solver_create(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(a);
    a->apply_bc = noop_hook;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(a, n, n, 1, h, h, 0.0, NULL));
    poisson_solver_destroy(a);

    poisson_solver_t* b = poisson_solver_create(POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(b, n, n, 1, h, h, 0.0, NULL));
    poisson_solver_destroy(b);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_linear_field_exact_cg_scalar);
    RUN_TEST(test_linear_field_exact_cg_simd);
    RUN_TEST(test_linear_field_exact_cg_omp);
    RUN_TEST(test_linear_field_exact_bicgstab);
    RUN_TEST(test_linear_field_exact_gmres);
    RUN_TEST(test_dirichlet_lift_is_linear);
    RUN_TEST(test_singularity_predicate);
    RUN_TEST(test_unsupported_methods_reject_walls);
    RUN_TEST(test_walls_with_hook_rejected);
    RUN_TEST(test_hook_alone_and_defaults_accepted);
    return UNITY_END();
}
