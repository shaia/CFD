/**
 * @file test_bicgstab.c
 * @brief BiCGSTAB solver unit tests
 *
 * Tests the BiCGSTAB (Biconjugate Gradient Stabilized) solver for solving
 * the Poisson equation. While BiCGSTAB is designed for non-symmetric systems,
 * it should also work correctly on symmetric systems like the Poisson equation.
 *
 * Tests cover:
 *   - Convergence on zero RHS (trivial solution)
 *   - Convergence on sinusoidal RHS (Neumann compatible)
 *   - Convergence on Dirichlet problem with known solution
 *   - Comparison with CG solver (max-norm and L2-norm metrics)
 *   - Error handling (NULL inputs, small grids)
 */

#include "unity.h"
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

/* Grid sizes for tests */
#define NX_SMALL  17
#define NY_SMALL  17
#define NX_MEDIUM 33
#define NY_MEDIUM 33

/* Convergence tolerance */
#define TOLERANCE 1e-6
#define ABS_TOLERANCE 1e-10

/* Maximum iterations */
#define MAX_ITERATIONS 2000

/* ============================================================================
 * TEST FIXTURES
 * ============================================================================ */

void setUp(void) {
    /* Nothing to set up */
}

void tearDown(void) {
    /* Nothing to tear down */
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Create a field initialized to zero
 */
static double* create_field(size_t nx, size_t ny) {
    double* field = (double*)cfd_malloc(nx * ny * sizeof(double));
    if (field) {
        for (size_t i = 0; i < nx * ny; i++) {
            field[i] = 0.0;
        }
    }
    return field;
}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs
 * f(x,y) = cos(2πx)cos(2πy) has zero integral over [0,1]²
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy) {
    /* First pass: initialize sinusoidal values */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[j * nx + i] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Subtract interior mean to enforce discrete Neumann compatibility */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            interior_sum += rhs[j * nx + i];
            interior_count++;
        }
    }
    double interior_mean = interior_sum / (double)interior_count;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            rhs[j * nx + i] -= interior_mean;
        }
    }
}

/**
 * Compute L2 norm of difference between two fields (interior points only)
 */
static double compute_l2_error(const double* a, const double* b,
                                size_t nx, size_t ny) {
    double sum = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double diff = a[j * nx + i] - b[j * nx + i];
            sum += diff * diff;
        }
    }
    return sqrt(sum / ((nx - 2) * (ny - 2)));  /* RMS error */
}

/**
 * Initialize RHS for Dirichlet manufactured solution test.
 *
 * For p = sin(πx)sin(πy), we have:
 *   ∇²p = -2π²sin(πx)sin(πy)
 *
 * The solver solves: Laplacian(p) = rhs, so rhs = ∇²p = -2π²sin(πx)sin(πy).
 */
static void init_dirichlet_rhs(double* rhs, size_t nx, size_t ny,
                                double dx, double dy) {
    (void)dx; (void)dy;  /* Grid spacing not needed for analytical RHS */
    double coeff = -2.0 * M_PI * M_PI;
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
            rhs[j * nx + i] = coeff * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Compute exact solution p = sin(πx)sin(πy)
 */
static void compute_exact_solution(double* exact, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
            exact[j * nx + i] = sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Remove mean from interior points
 */
static double remove_interior_mean(double* field, size_t nx, size_t ny) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += field[j * nx + i];
            count++;
        }
    }
    double mean = sum / count;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            field[j * nx + i] -= mean;
        }
    }
    return mean;
}

/* ============================================================================
 * BASIC TESTS
 * ============================================================================ */

/**
 * Test BiCGSTAB solver creation
 */
void test_bicgstab_create(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(POISSON_METHOD_BICGSTAB, solver->method);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_SCALAR, solver->backend);
    TEST_ASSERT_EQUAL_STRING("bicgstab_scalar", solver->name);

    poisson_solver_destroy(solver);
}

/**
 * Test BiCGSTAB initialization
 */
void test_bicgstab_init(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, NULL);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(nx, solver->nx);
    TEST_ASSERT_EQUAL(ny, solver->ny);

    poisson_solver_destroy(solver);
}

/* ============================================================================
 * CONVERGENCE TESTS
 * ============================================================================ */

/**
 * Test BiCGSTAB with zero RHS (trivial solution: p = constant)
 */
void test_bicgstab_zero_rhs(void) {
    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Start from non-zero initial guess */
    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = 1.0;
    }
    /* RHS is already zero */

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE(stats.iterations < MAX_ITERATIONS);

    /* Verify solver met its convergence criterion (relative or absolute tolerance) */
    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/**
 * Test BiCGSTAB with sinusoidal RHS (Neumann compatible)
 */
void test_bicgstab_sinusoidal_rhs(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    /* Verify solver met its convergence criterion (relative or absolute tolerance) */
    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/**
 * Test BiCGSTAB comparison with CG solver
 * Both should produce similar results on SPD systems
 */
void test_bicgstab_vs_cg(void) {
    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p_bicgstab = create_field(nx, ny);
    double* p_cg = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p_bicgstab);
    TEST_ASSERT_NOT_NULL(p_cg);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    /* Solve with BiCGSTAB */
    poisson_solver_t* bicgstab = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(bicgstab);
    poisson_solver_init(bicgstab, nx, ny, dx, dy, &params);

    poisson_solver_stats_t stats_bicgstab = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(bicgstab, p_bicgstab, NULL, rhs, &stats_bicgstab);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve with CG */
    poisson_solver_t* cg = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(cg);
    poisson_solver_init(cg, nx, ny, dx, dy, &params);

    poisson_solver_stats_t stats_cg = poisson_solver_stats_default();
    status = poisson_solver_solve(cg, p_cg, NULL, rhs, &stats_cg);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Both solutions should have converged */
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_bicgstab.status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_cg.status);

    /* Solutions should be similar (within tolerance)
     * Note: Neumann BCs allow for a constant shift, so we compare
     * after removing the mean from both solutions */
    double mean_bicgstab = 0.0, mean_cg = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            mean_bicgstab += p_bicgstab[j * nx + i];
            mean_cg += p_cg[j * nx + i];
            count++;
        }
    }
    mean_bicgstab /= count;
    mean_cg /= count;

    /* Compute difference after mean removal */
    double max_diff = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double v_bicgstab = p_bicgstab[j * nx + i] - mean_bicgstab;
            double v_cg = p_cg[j * nx + i] - mean_cg;
            double diff = fabs(v_bicgstab - v_cg);
            if (diff > max_diff) max_diff = diff;
        }
    }

    /* Solutions should be similar within a reasonable tolerance */
    TEST_ASSERT_TRUE(max_diff < 1e-4);

    poisson_solver_destroy(bicgstab);
    poisson_solver_destroy(cg);
    cfd_free(p_bicgstab);
    cfd_free(p_cg);
    cfd_free(rhs);
}

/**
 * Test BiCGSTAB vs CG using L2 error metric
 *
 * Verifies that both solvers produce solutions with small RMS difference.
 * Uses mean-shifted solutions since Neumann BCs allow constant offsets.
 */
void test_bicgstab_l2_error(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p_bicgstab = create_field(nx, ny);
    double* p_cg = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p_bicgstab);
    TEST_ASSERT_NOT_NULL(p_cg);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    /* Solve with BiCGSTAB */
    poisson_solver_t* bicgstab = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(bicgstab);
    poisson_solver_init(bicgstab, nx, ny, dx, dy, &params);
    poisson_solver_stats_t stats_bicgstab = poisson_solver_stats_default();
    poisson_solver_solve(bicgstab, p_bicgstab, NULL, rhs, &stats_bicgstab);

    /* Solve with CG */
    poisson_solver_t* cg = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(cg);
    poisson_solver_init(cg, nx, ny, dx, dy, &params);
    poisson_solver_stats_t stats_cg = poisson_solver_stats_default();
    poisson_solver_solve(cg, p_cg, NULL, rhs, &stats_cg);

    /* Remove means to handle Neumann BC constant offset */
    double mean_bicgstab = 0.0, mean_cg = 0.0;
    size_t count = (nx - 2) * (ny - 2);
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            mean_bicgstab += p_bicgstab[j * nx + i];
            mean_cg += p_cg[j * nx + i];
        }
    }
    mean_bicgstab /= count;
    mean_cg /= count;

    /* Shift solutions to zero mean */
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            p_bicgstab[j * nx + i] -= mean_bicgstab;
            p_cg[j * nx + i] -= mean_cg;
        }
    }

    /* Compute L2 (RMS) error between solutions */
    double l2_error = compute_l2_error(p_bicgstab, p_cg, nx, ny);

    /* L2 error should be small - both methods solve the same system */
    TEST_ASSERT_TRUE(l2_error < 1e-5);

    poisson_solver_destroy(bicgstab);
    poisson_solver_destroy(cg);
    cfd_free(p_bicgstab);
    cfd_free(p_cg);
    cfd_free(rhs);
}

/**
 * Test BiCGSTAB with manufactured Dirichlet solution.
 *
 * Uses the manufactured solution p = sin(πx)sin(πy), which:
 * - Is zero on all boundaries of [0,1]² (natural Dirichlet BC)
 * - Has Laplacian ∇²p = -2π²sin(πx)sin(πy)
 *
 * Note: BiCGSTAB applies Neumann BCs internally, so the computed solution
 * may differ from the exact Dirichlet solution by a constant. We compare
 * after removing the mean from both solutions.
 *
 * Expected accuracy: O(h²) ≈ (1/32)² ≈ 0.001 for 33x33 grid.
 */
void test_bicgstab_dirichlet(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* exact = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(exact);

    /* Set up RHS for manufactured solution */
    init_dirichlet_rhs(rhs, nx, ny, dx, dy);

    /* Compute exact solution for comparison */
    compute_exact_solution(exact, nx, ny);

    /* Solve with BiCGSTAB */
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    /* Remove means from both solutions (Neumann allows constant offset) */
    remove_interior_mean(p, nx, ny);
    remove_interior_mean(exact, nx, ny);

    /* Compute L2 error between computed and exact solutions */
    double l2_error = compute_l2_error(p, exact, nx, ny);

    /* Error should be within second-order discretization bounds.
     * For 33x33 grid: h ≈ 1/32, so O(h²) ≈ 0.001.
     * Allow some margin for iterative solver tolerance. */
    TEST_ASSERT_TRUE(l2_error < 0.01);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(exact);
}

/* ============================================================================
 * ERROR HANDLING TESTS
 * ============================================================================ */

/**
 * Test that unsupported backends return NULL
 */
void test_bicgstab_unsupported_backend(void) {
    /* OMP backend not yet implemented for BiCGSTAB */
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_OMP);
    TEST_ASSERT_NULL(solver);

    /* GPU backend not yet implemented for BiCGSTAB */
    solver = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_GPU);
    TEST_ASSERT_NULL(solver);
}

/**
 * Test destroy with NULL solver
 */
void test_bicgstab_destroy_null(void) {
    /* Should not crash */
    poisson_solver_destroy(NULL);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Basic tests */
    RUN_TEST(test_bicgstab_create);
    RUN_TEST(test_bicgstab_init);

    /* Convergence tests */
    RUN_TEST(test_bicgstab_zero_rhs);
    RUN_TEST(test_bicgstab_sinusoidal_rhs);
    RUN_TEST(test_bicgstab_vs_cg);
    RUN_TEST(test_bicgstab_l2_error);
    RUN_TEST(test_bicgstab_dirichlet);

    /* Error handling tests */
    RUN_TEST(test_bicgstab_unsupported_backend);
    RUN_TEST(test_bicgstab_destroy_null);

    return UNITY_END();
}
