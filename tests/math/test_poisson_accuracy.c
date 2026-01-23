/**
 * @file test_poisson_accuracy.c
 * @brief Poisson solver accuracy tests with analytical solutions
 *
 * These tests verify the accuracy of Poisson solvers against known analytical
 * solutions. This validates that the iterative solvers converge to the correct
 * solution and achieve expected convergence rates.
 *
 * Tests cover:
 *   - Zero RHS test: solution should remain constant (with Neumann BCs)
 *   - Uniform RHS test: solution is quadratic
 *   - Sinusoidal RHS test: p = sin(πx)sin(πy) → ∇²p = -2π²p
 *   - Convergence rate verification for all Poisson variants
 *   - Residual convergence tracking
 *
 * Test approach: Use manufactured solutions where we know the analytical
 * relationship between the solution and RHS, solve numerically, and compare.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

/* Domain: [0, 1] × [0, 1] for simplicity */
#define DOMAIN_XMIN 0.0
#define DOMAIN_XMAX 1.0
#define DOMAIN_YMIN 0.0
#define DOMAIN_YMAX 1.0

/* Tolerances */
#define ZERO_RHS_TOL       1e-10  /* Solution should not change significantly */
#define UNIFORM_RHS_TOL    1e-4   /* Allow some numerical error */
#define SINUSOIDAL_RHS_TOL 1e-2   /* L2 error tolerance for sinusoidal */
#define CONVERGENCE_RATE_TOL 0.3  /* Tolerance for convergence order check */

/* Solver parameters */
#define MAX_ITERATIONS 5000
#define SOLVER_TOLERANCE 1e-8

/* ============================================================================
 * UNITY SETUP
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Allocate a 2D field initialized to zero
 */
static double* create_field(size_t nx, size_t ny) {
    return (double*)cfd_calloc(nx * ny, sizeof(double));
}

/**
 * Allocate a 2D field initialized to a constant value
 */
static double* create_field_constant(size_t nx, size_t ny, double value) {
    double* field = create_field(nx, ny);
    if (field) {
        for (size_t i = 0; i < nx * ny; i++) {
            field[i] = value;
        }
    }
    return field;
}

/**
 * Compute the L2 error between numerical and analytical solutions
 */
static double compute_l2_error(const double* numerical, const double* analytical,
                                size_t nx, size_t ny) {
    double sum_sq = 0.0;
    size_t count = 0;
    /* Compare interior points only */
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double err = numerical[idx] - analytical[idx];
            sum_sq += err * err;
            count++;
        }
    }
    if (count == 0) return 0.0;
    return sqrt(sum_sq / count);
}

/**
 * Compute max (L-infinity) error
 */
static double compute_max_error(const double* numerical, const double* analytical,
                                 size_t nx, size_t ny) {
    double max_err = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double err = fabs(numerical[idx] - analytical[idx]);
            if (err > max_err) max_err = err;
        }
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
 * ANALYTICAL SOLUTIONS
 * ============================================================================
 *
 * For Poisson equation: ∇²p = f
 * With Neumann BCs: ∂p/∂n = 0 on boundaries
 *
 * Solution is unique up to an additive constant.
 */

/**
 * Sinusoidal solution: p = sin(πx)sin(πy)
 * Then: ∇²p = -π²sin(πx)sin(πy) - π²sin(πx)sin(πy) = -2π²p
 * So RHS f = -2π² sin(πx)sin(πy)
 *
 * Note: This doesn't satisfy Neumann BCs exactly (∂p/∂x|_{x=0} = π·sin(πy) ≠ 0)
 * so we use Dirichlet-like BCs (enforce analytical values on boundary).
 */
static double sinusoidal_solution(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

static double sinusoidal_rhs(double x, double y) {
    return -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

/**
 * Initialize field with sinusoidal analytical solution
 */
static void init_sinusoidal_analytical(double* p, size_t nx, size_t ny,
                                        double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            p[j * nx + i] = sinusoidal_solution(x, y);
        }
    }
}

/**
 * Initialize RHS for sinusoidal problem
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[j * nx + i] = sinusoidal_rhs(x, y);
        }
    }
}

/**
 * Apply Dirichlet BCs using analytical solution (for sinusoidal test)
 */
static void apply_sinusoidal_bc(double* p, size_t nx, size_t ny,
                                 double dx, double dy) {
    /* Left and right boundaries */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        p[j * nx + 0] = sinusoidal_solution(DOMAIN_XMIN, y);
        p[j * nx + (nx-1)] = sinusoidal_solution(DOMAIN_XMAX, y);
    }
    /* Bottom and top boundaries */
    for (size_t i = 0; i < nx; i++) {
        double x = DOMAIN_XMIN + i * dx;
        p[0 * nx + i] = sinusoidal_solution(x, DOMAIN_YMIN);
        p[(ny-1) * nx + i] = sinusoidal_solution(x, DOMAIN_YMAX);
    }
}

/**
 * Quadratic solution for uniform RHS test: p = (x² + y²) / 4
 * Then: ∇²p = (2 + 2) / 4 = 1
 * So RHS f = 1 (uniform)
 *
 * This tests that the solver correctly handles constant source terms
 * and produces the expected quadratic profile.
 */
static double quadratic_solution(double x, double y) {
    return (x * x + y * y) / 4.0;
}

static double uniform_rhs_value(void) {
    return 1.0;  /* ∇²p = 1 for p = (x² + y²) / 4 */
}

/**
 * Initialize field with quadratic analytical solution
 */
static void init_quadratic_analytical(double* p, size_t nx, size_t ny,
                                       double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            p[j * nx + i] = quadratic_solution(x, y);
        }
    }
}

/**
 * Initialize uniform RHS
 */
static void init_uniform_rhs(double* rhs, size_t nx, size_t ny) {
    double val = uniform_rhs_value();
    for (size_t i = 0; i < nx * ny; i++) {
        rhs[i] = val;
    }
}

/**
 * Apply Dirichlet BCs using quadratic analytical solution
 */
static void apply_quadratic_bc(double* p, size_t nx, size_t ny,
                                double dx, double dy) {
    /* Left and right boundaries */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        p[j * nx + 0] = quadratic_solution(DOMAIN_XMIN, y);
        p[j * nx + (nx-1)] = quadratic_solution(DOMAIN_XMAX, y);
    }
    /* Bottom and top boundaries */
    for (size_t i = 0; i < nx; i++) {
        double x = DOMAIN_XMIN + i * dx;
        p[0 * nx + i] = quadratic_solution(x, DOMAIN_YMIN);
        p[(ny-1) * nx + i] = quadratic_solution(x, DOMAIN_YMAX);
    }
}

/* ============================================================================
 * ZERO RHS TESTS
 * ============================================================================
 *
 * If ∇²p = 0 with Neumann BCs (∂p/∂n = 0), any constant is a solution.
 * The solver should preserve the initial guess (or converge immediately).
 */

void test_zero_rhs_jacobi(void) {
    printf("\n    Testing zero RHS with Jacobi solver...\n");

    size_t nx = 32, ny = 32;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    /* Initial guess: constant = 1.0 */
    double initial_value = 1.0;
    double* p = create_field_constant(nx, ny, initial_value);
    double* p_temp = create_field(nx, ny);
    double* rhs = create_field(nx, ny);  /* Zero RHS */

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(p_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Create solver */
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create Jacobi solver");

    /* Configure */
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 100;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    /* Solve */
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    printf("      Iterations: %d, Final residual: %.2e\n",
           stats.iterations, stats.final_residual);

    /* Check solution is still approximately constant */
    double min_val = p[0], max_val = p[0];
    for (size_t i = 1; i < nx * ny; i++) {
        if (p[i] < min_val) min_val = p[i];
        if (p[i] > max_val) max_val = p[i];
    }
    double variation = max_val - min_val;
    printf("      Solution variation: %.2e (tolerance: %.2e)\n",
           variation, ZERO_RHS_TOL);

    TEST_ASSERT_TRUE_MESSAGE(variation < ZERO_RHS_TOL,
        "Solution should remain constant for zero RHS");

    /* Cleanup */
    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
}

void test_zero_rhs_sor(void) {
    printf("\n    Testing zero RHS with SOR solver...\n");

    size_t nx = 32, ny = 32;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double initial_value = 1.0;
    double* p = create_field_constant(nx, ny, initial_value);
    double* rhs = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create SOR solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 100;
    params.omega = 1.5;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    printf("      Iterations: %d, Final residual: %.2e\n",
           stats.iterations, stats.final_residual);

    double min_val = p[0], max_val = p[0];
    for (size_t i = 1; i < nx * ny; i++) {
        if (p[i] < min_val) min_val = p[i];
        if (p[i] > max_val) max_val = p[i];
    }
    double variation = max_val - min_val;
    printf("      Solution variation: %.2e\n", variation);

    TEST_ASSERT_TRUE_MESSAGE(variation < ZERO_RHS_TOL,
        "Solution should remain constant for zero RHS");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

void test_zero_rhs_redblack(void) {
    printf("\n    Testing zero RHS with Red-Black SOR solver...\n");

    size_t nx = 32, ny = 32;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double initial_value = 1.0;
    double* p = create_field_constant(nx, ny, initial_value);
    double* rhs = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create Red-Black SOR solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 100;
    params.omega = 1.5;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    printf("      Iterations: %d, Final residual: %.2e\n",
           stats.iterations, stats.final_residual);

    double min_val = p[0], max_val = p[0];
    for (size_t i = 1; i < nx * ny; i++) {
        if (p[i] < min_val) min_val = p[i];
        if (p[i] > max_val) max_val = p[i];
    }
    double variation = max_val - min_val;
    printf("      Solution variation: %.2e\n", variation);

    TEST_ASSERT_TRUE_MESSAGE(variation < ZERO_RHS_TOL,
        "Solution should remain constant for zero RHS");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/* ============================================================================
 * UNIFORM RHS TESTS (Quadratic Solution)
 * ============================================================================
 *
 * Manufactured solution: p = (x² + y²) / 4
 * RHS: f = 1 (uniform)
 *
 * This verifies that the solver correctly handles constant source terms
 * and produces the expected quadratic profile.
 *
 * Use Dirichlet BCs (enforce analytical solution on boundary).
 */

/**
 * Custom solver loop for quadratic solution with Dirichlet BCs
 */
static int solve_quadratic_with_dirichlet_bc(poisson_solver_t* solver,
                                              double* p, double* p_temp, const double* rhs,
                                              size_t nx, size_t ny, double dx, double dy,
                                              int max_iters, double tol) {
    double residual = 0.0;

    for (int iter = 0; iter < max_iters; iter++) {
        cfd_status_t status = poisson_solver_iterate(solver, p, p_temp, rhs, &residual);
        if (status != CFD_SUCCESS) return -1;

        apply_quadratic_bc(p, nx, ny, dx, dy);

        if (residual < tol) {
            return iter + 1;
        }
    }
    return max_iters;
}

void test_uniform_rhs_jacobi(void) {
    printf("\n    Testing uniform RHS (quadratic solution) with Jacobi solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* p_temp = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(p_temp);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_uniform_rhs(rhs, nx, ny);
    init_quadratic_analytical(analytical, nx, ny, dx, dy);
    apply_quadratic_bc(p, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 10000;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    int iters = solve_quadratic_with_dirichlet_bc(solver, p, p_temp, rhs,
                                                   nx, ny, dx, dy, 10000, 1e-10);
    printf("      Iterations: %d\n", iters);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < UNIFORM_RHS_TOL,
        "Uniform RHS L2 error exceeds tolerance");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
    cfd_free(analytical);
}

void test_uniform_rhs_sor(void) {
    printf("\n    Testing uniform RHS (quadratic solution) with SOR solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_uniform_rhs(rhs, nx, ny);
    init_quadratic_analytical(analytical, nx, ny, dx, dy);
    apply_quadratic_bc(p, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 10000;
    params.omega = 1.7;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    int iters = solve_quadratic_with_dirichlet_bc(solver, p, NULL, rhs,
                                                   nx, ny, dx, dy, 10000, 1e-10);
    printf("      Iterations: %d\n", iters);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < UNIFORM_RHS_TOL,
        "Uniform RHS L2 error exceeds tolerance");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(analytical);
}

void test_uniform_rhs_redblack(void) {
    printf("\n    Testing uniform RHS (quadratic solution) with Red-Black SOR solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_uniform_rhs(rhs, nx, ny);
    init_quadratic_analytical(analytical, nx, ny, dx, dy);
    apply_quadratic_bc(p, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-10;
    params.max_iterations = 10000;
    params.omega = 1.7;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    int iters = solve_quadratic_with_dirichlet_bc(solver, p, NULL, rhs,
                                                   nx, ny, dx, dy, 10000, 1e-10);
    printf("      Iterations: %d\n", iters);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < UNIFORM_RHS_TOL,
        "Uniform RHS L2 error exceeds tolerance");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(analytical);
}

/* ============================================================================
 * SINUSOIDAL RHS TESTS
 * ============================================================================
 *
 * Manufactured solution: p = sin(πx)sin(πy)
 * RHS: f = -2π²sin(πx)sin(πy)
 *
 * Use Dirichlet BCs (enforce analytical solution on boundary) for this test.
 */

/**
 * Custom solver loop that applies Dirichlet BCs after each iteration
 */
static int solve_with_dirichlet_bc(poisson_solver_t* solver,
                                    double* p, double* p_temp, const double* rhs,
                                    size_t nx, size_t ny, double dx, double dy,
                                    int max_iter, double tolerance) {
    double residual = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        /* Perform one iteration */
        cfd_status_t status = poisson_solver_iterate(solver, p, p_temp, rhs, &residual);
        if (status != CFD_SUCCESS) return -1;

        /* Apply Dirichlet BCs */
        apply_sinusoidal_bc(p, nx, ny, dx, dy);

        /* Check convergence */
        if (residual < tolerance) {
            return iter + 1;
        }
    }
    return max_iter;
}

void test_sinusoidal_rhs_jacobi(void) {
    printf("\n    Testing sinusoidal RHS with Jacobi solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* p_temp = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(p_temp);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    /* Initialize */
    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);
    init_sinusoidal_analytical(analytical, nx, ny, dx, dy);
    apply_sinusoidal_bc(p, nx, ny, dx, dy);  /* Start with correct BCs */

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create Jacobi solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    /* Solve with Dirichlet BCs */
    int iterations = solve_with_dirichlet_bc(solver, p, p_temp, rhs,
                                              nx, ny, dx, dy,
                                              MAX_ITERATIONS, SOLVER_TOLERANCE);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);

    printf("      Grid: %zux%zu, Iterations: %d\n", nx, ny, iterations);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);
    printf("      Tolerance: %.2e\n", SINUSOIDAL_RHS_TOL);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < SINUSOIDAL_RHS_TOL,
        "Jacobi solver L2 error exceeds tolerance for sinusoidal RHS");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
    cfd_free(analytical);
}

void test_sinusoidal_rhs_sor(void) {
    printf("\n    Testing sinusoidal RHS with SOR solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);
    init_sinusoidal_analytical(analytical, nx, ny, dx, dy);
    apply_sinusoidal_bc(p, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create SOR solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = MAX_ITERATIONS;
    params.omega = 1.7;  /* Near-optimal for this problem */

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    int iterations = solve_with_dirichlet_bc(solver, p, NULL, rhs,
                                              nx, ny, dx, dy,
                                              MAX_ITERATIONS, SOLVER_TOLERANCE);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);

    printf("      Grid: %zux%zu, Iterations: %d\n", nx, ny, iterations);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < SINUSOIDAL_RHS_TOL,
        "SOR solver L2 error exceeds tolerance for sinusoidal RHS");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(analytical);
}

void test_sinusoidal_rhs_redblack(void) {
    printf("\n    Testing sinusoidal RHS with Red-Black SOR solver...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);
    init_sinusoidal_analytical(analytical, nx, ny, dx, dy);
    apply_sinusoidal_bc(p, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Failed to create Red-Black SOR solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = MAX_ITERATIONS;
    params.omega = 1.7;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    int iterations = solve_with_dirichlet_bc(solver, p, NULL, rhs,
                                              nx, ny, dx, dy,
                                              MAX_ITERATIONS, SOLVER_TOLERANCE);

    double l2_error = compute_l2_error(p, analytical, nx, ny);
    double max_error = compute_max_error(p, analytical, nx, ny);

    printf("      Grid: %zux%zu, Iterations: %d\n", nx, ny, iterations);
    printf("      L2 error: %.6e, Max error: %.6e\n", l2_error, max_error);

    TEST_ASSERT_TRUE_MESSAGE(l2_error < SINUSOIDAL_RHS_TOL,
        "Red-Black SOR solver L2 error exceeds tolerance for sinusoidal RHS");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(analytical);
}

/* ============================================================================
 * GRID CONVERGENCE TESTS
 * ============================================================================
 *
 * Verify that error decreases as O(h²) with grid refinement.
 * This validates the 2nd-order accuracy of the finite difference stencil.
 */

void test_grid_convergence_jacobi(void) {
    printf("\n    Testing grid convergence with Jacobi solver...\n");

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        spacings[s] = dx;

        double* p = create_field(n, n);
        double* p_temp = create_field(n, n);
        double* rhs = create_field(n, n);
        double* analytical = create_field(n, n);

        if (!p || !p_temp || !rhs || !analytical) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            cfd_free(analytical);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_sinusoidal_rhs(rhs, n, n, dx, dy);
        init_sinusoidal_analytical(analytical, n, n, dx, dy);
        apply_sinusoidal_bc(p, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-10;
        params.max_iterations = 10000;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        solve_with_dirichlet_bc(solver, p, p_temp, rhs, n, n, dx, dy, 10000, 1e-10);

        errors[s] = compute_l2_error(p, analytical, n, n);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
        cfd_free(analytical);
    }

    /* Verify O(h²) convergence */
    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s-1], errors[s],
                                                spacings[s-1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s-1], sizes[s], rate);

        /* Allow some tolerance but expect roughly 2nd order */
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Grid convergence rate below O(h²)");
    }
}

void test_grid_convergence_sor(void) {
    printf("\n    Testing grid convergence with SOR solver...\n");

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        spacings[s] = dx;

        double* p = create_field(n, n);
        double* rhs = create_field(n, n);
        double* analytical = create_field(n, n);

        if (!p || !rhs || !analytical) {
            cfd_free(p);
            cfd_free(rhs);
            cfd_free(analytical);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_sinusoidal_rhs(rhs, n, n, dx, dy);
        init_sinusoidal_analytical(analytical, n, n, dx, dy);
        apply_sinusoidal_bc(p, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-10;
        params.max_iterations = 10000;
        params.omega = 1.7;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        solve_with_dirichlet_bc(solver, p, NULL, rhs, n, n, dx, dy, 10000, 1e-10);

        errors[s] = compute_l2_error(p, analytical, n, n);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
        cfd_free(analytical);
    }

    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s-1], errors[s],
                                                spacings[s-1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s-1], sizes[s], rate);

        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Grid convergence rate below O(h²)");
    }
}

void test_grid_convergence_redblack(void) {
    printf("\n    Testing grid convergence with Red-Black SOR solver...\n");

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        spacings[s] = dx;

        double* p = create_field(n, n);
        double* rhs = create_field(n, n);
        double* analytical = create_field(n, n);

        if (!p || !rhs || !analytical) {
            cfd_free(p);
            cfd_free(rhs);
            cfd_free(analytical);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_sinusoidal_rhs(rhs, n, n, dx, dy);
        init_sinusoidal_analytical(analytical, n, n, dx, dy);
        apply_sinusoidal_bc(p, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-10;
        params.max_iterations = 10000;
        params.omega = 1.7;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        solve_with_dirichlet_bc(solver, p, NULL, rhs, n, n, dx, dy, 10000, 1e-10);

        errors[s] = compute_l2_error(p, analytical, n, n);
        printf("      %zux%zu: L2 error = %.6e\n", n, n, errors[s]);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
        cfd_free(analytical);
    }

    for (int s = 1; s < num_sizes; s++) {
        double rate = compute_convergence_rate(errors[s-1], errors[s],
                                                spacings[s-1], spacings[s]);
        printf("      Convergence rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s-1], sizes[s], rate);

        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "Grid convergence rate below O(h²)");
    }
}

/* ============================================================================
 * RESIDUAL CONVERGENCE TESTS
 * ============================================================================
 *
 * Verify that residual decreases during iteration using the solver's native
 * Neumann boundary conditions (no manual BC overwriting).
 *
 * Uses uniform RHS which is compatible with Neumann BCs.
 */

void test_residual_convergence_jacobi(void) {
    printf("\n    Testing residual convergence with Jacobi solver...\n");

    size_t nx = 32, ny = 32;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    /* Use uniform RHS - compatible with Neumann BCs */
    double* p = create_field(nx, ny);
    double* p_temp = create_field(nx, ny);
    double* rhs = create_field_constant(nx, ny, 1.0);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(p_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Random initial guess to ensure convergence is needed */
    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = (double)(i % 7) * 0.1;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = 200;

    cfd_status_t init_status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, init_status);

    /* Run solver and track residual reduction */
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    printf("      Initial residual: %.6e\n", stats.initial_residual);
    printf("      Final residual: %.6e\n", stats.final_residual);
    printf("      Iterations: %d\n", stats.iterations);

    /* Verify residual decreased significantly */
    double reduction_ratio = stats.final_residual / stats.initial_residual;
    printf("      Reduction ratio: %.6e\n", reduction_ratio);

    /* Residual should decrease by at least 2 orders of magnitude or hit tolerance */
    int converged = (status == CFD_SUCCESS) ||
                    (reduction_ratio < 0.01) ||
                    (stats.final_residual < 1e-6);

    TEST_ASSERT_TRUE_MESSAGE(converged,
        "Jacobi solver should reduce residual significantly");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
}

void test_residual_convergence_sor(void) {
    printf("\n    Testing residual convergence with SOR solver...\n");

    size_t nx = 32, ny = 32;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field_constant(nx, ny, 1.0);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Random initial guess */
    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = (double)(i % 7) * 0.1;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = 200;
    params.omega = 1.5;

    cfd_status_t init_status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, init_status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    printf("      Initial residual: %.6e\n", stats.initial_residual);
    printf("      Final residual: %.6e\n", stats.final_residual);
    printf("      Iterations: %d\n", stats.iterations);

    double reduction_ratio = stats.final_residual / stats.initial_residual;
    printf("      Reduction ratio: %.6e\n", reduction_ratio);

    int converged = (status == CFD_SUCCESS) ||
                    (reduction_ratio < 0.01) ||
                    (stats.final_residual < 1e-6);

    TEST_ASSERT_TRUE_MESSAGE(converged,
        "SOR solver should reduce residual significantly");

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/* ============================================================================
 * SOLVER COMPARISON TEST
 * ============================================================================
 *
 * Compare all solver variants on the same problem to ensure consistency.
 */

void test_solver_comparison(void) {
    printf("\n    Comparing all Poisson solver variants...\n");

    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* rhs = create_field(nx, ny);
    double* analytical = create_field(nx, ny);

    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(analytical);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);
    init_sinusoidal_analytical(analytical, nx, ny, dx, dy);

    /* Test configurations */
    struct {
        poisson_solver_method_t method;
        const char* name;
        int needs_temp;
    } solvers[] = {
        { POISSON_METHOD_JACOBI, "Jacobi", 1 },
        { POISSON_METHOD_SOR, "SOR", 0 },
        { POISSON_METHOD_REDBLACK_SOR, "Red-Black SOR", 0 }
    };
    int num_solvers = sizeof(solvers) / sizeof(solvers[0]);

    double reference_error = 0.0;
    int have_reference = 0;

    for (int i = 0; i < num_solvers; i++) {
        double* p = create_field(nx, ny);
        double* p_temp = solvers[i].needs_temp ? create_field(nx, ny) : NULL;

        if (!p || (solvers[i].needs_temp && !p_temp)) {
            cfd_free(p);
            cfd_free(p_temp);
            continue;
        }

        apply_sinusoidal_bc(p, nx, ny, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            solvers[i].method, POISSON_BACKEND_SCALAR);

        if (!solver) {
            printf("      %s: SKIPPED (solver not available)\n", solvers[i].name);
            cfd_free(p);
            cfd_free(p_temp);
            continue;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-10;
        params.max_iterations = 10000;
        params.omega = 1.7;

        cfd_status_t status = poisson_solver_init(solver, nx, ny, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
        int iterations = solve_with_dirichlet_bc(solver, p, p_temp, rhs,
                                                  nx, ny, dx, dy, 10000, 1e-10);

        double l2_error = compute_l2_error(p, analytical, nx, ny);
        printf("      %s: L2 error = %.6e, iterations = %d\n",
               solvers[i].name, l2_error, iterations);

        if (!have_reference) {
            reference_error = l2_error;
            have_reference = 1;
        } else {
            /* All solvers should achieve similar accuracy */
            double diff = fabs(l2_error - reference_error);
            TEST_ASSERT_TRUE_MESSAGE(diff < 1e-3,
                "Solver accuracy differs significantly from reference");
        }

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
    }

    cfd_free(rhs);
    cfd_free(analytical);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("POISSON SOLVER ACCURACY TESTS\n");
    printf("========================================\n");

    /* Zero RHS tests */
    printf("\n--- Zero RHS Tests ---\n");
    RUN_TEST(test_zero_rhs_jacobi);
    RUN_TEST(test_zero_rhs_sor);
    RUN_TEST(test_zero_rhs_redblack);

    /* Uniform RHS tests (quadratic solution) */
    printf("\n--- Uniform RHS Tests (Quadratic Solution) ---\n");
    RUN_TEST(test_uniform_rhs_jacobi);
    RUN_TEST(test_uniform_rhs_sor);
    RUN_TEST(test_uniform_rhs_redblack);

    /* Sinusoidal RHS tests */
    printf("\n--- Sinusoidal RHS Tests ---\n");
    RUN_TEST(test_sinusoidal_rhs_jacobi);
    RUN_TEST(test_sinusoidal_rhs_sor);
    RUN_TEST(test_sinusoidal_rhs_redblack);

    /* Grid convergence tests */
    printf("\n--- Grid Convergence Tests ---\n");
    RUN_TEST(test_grid_convergence_jacobi);
    RUN_TEST(test_grid_convergence_sor);
    RUN_TEST(test_grid_convergence_redblack);

    /* Residual convergence tests */
    printf("\n--- Residual Convergence Tests ---\n");
    RUN_TEST(test_residual_convergence_jacobi);
    RUN_TEST(test_residual_convergence_sor);

    /* Solver comparison */
    printf("\n--- Solver Comparison ---\n");
    RUN_TEST(test_solver_comparison);

    return UNITY_END();
}
