/**
 * @file test_gmres.c
 * @brief Restarted GMRES(m) solver unit tests
 *
 * Tests the GMRES solver on the Poisson equation. GMRES is designed for
 * non-symmetric systems but must also solve the symmetric Poisson operator
 * used here, converging to the same solution as CG (a strong cross-check).
 *
 * Tests cover:
 *   - Creation / initialization / restart-parameter plumbing
 *   - Convergence on zero RHS (happy/early-return path)
 *   - Convergence on sinusoidal RHS (Neumann compatible)
 *   - Comparison with CG (max-norm and L2-norm) on the same SPD problem
 *   - Manufactured Dirichlet solution accuracy (O(h^2))
 *   - Hessenberg estimate vs independently recomputed true residual
 *   - Restart-no-stall: small m and default m reach the same solution
 *   - Jacobi preconditioner path converges (no iteration-count assertion)
 *   - Max-iteration exhaustion: GMRES(1) stops at the cap with POISSON_MAX_ITER
 *   - Error handling (unsupported backends, oversized restart rejected at init,
 *     solve before init, NULL destroy)
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <limits.h>
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

#define NX_SMALL  17
#define NY_SMALL  17
#define NX_MEDIUM 33
#define NY_MEDIUM 33

#define TOLERANCE 1e-6
#define ABS_TOLERANCE 1e-10
#define MAX_ITERATIONS 2000

/* ============================================================================
 * TEST FIXTURES
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * HELPER FUNCTIONS (mirror test_bicgstab.c)
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

/** Sinusoidal RHS f = cos(2πx)cos(2πy) with interior mean removed (Neumann compatible) */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            interior_sum += rhs[IDX_2D(i, j, nx)];
            interior_count++;
        }
    }
    double interior_mean = interior_sum / (double)interior_count;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            rhs[IDX_2D(i, j, nx)] -= interior_mean;
        }
    }
}

/** RMS difference between two fields over interior points */
static double compute_l2_error(const double* a, const double* b,
                               size_t nx, size_t ny) {
    double sum = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double diff = a[IDX_2D(i, j, nx)] - b[IDX_2D(i, j, nx)];
            sum += diff * diff;
        }
    }
    return sqrt(sum / ((nx - 2) * (ny - 2)));
}

/** RHS for manufactured p = sin(πx)sin(πy): ∇²p = -2π² sin(πx)sin(πy) */
static void init_dirichlet_rhs(double* rhs, size_t nx, size_t ny,
                               double dx, double dy) {
    (void)dx; (void)dy;
    double coeff = -2.0 * M_PI * M_PI;
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
            rhs[IDX_2D(i, j, nx)] = coeff * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

static void compute_exact_solution(double* exact, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_YMIN + j * (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_XMIN + i * (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
            exact[IDX_2D(i, j, nx)] = sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

static void remove_interior_mean(double* field, size_t nx, size_t ny) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += field[IDX_2D(i, j, nx)];
            count++;
        }
    }
    double mean = sum / count;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            field[IDX_2D(i, j, nx)] -= mean;
        }
    }
}

/**
 * Independently recompute the true residual max-norm ||rhs - ∇²x||_inf.
 *
 * Measured over the DEEP interior (indices 2..n-3), skipping the first interior
 * ring. The solver applies output boundary conditions to x after the solve, which
 * rewrites the boundary cells; that changes the stencil residual only at the ring
 * i==1 / j==1. The deep interior residual is exactly the solve-consistent residual
 * the solver reports, independently recomputed here without the Hessenberg estimate.
 */
static double true_residual_inf(const double* x, const double* rhs,
                                size_t nx, size_t ny, double dx, double dy) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double max_r = 0.0;
    for (size_t j = 2; j < ny - 2; j++) {
        for (size_t i = 2; i < nx - 2; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double lap = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / dx2
                       + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / dy2;
            double r = fabs(lap - rhs[idx]);
            if (r > max_r) max_r = r;
        }
    }
    return max_r;
}

/* ============================================================================
 * BASIC TESTS
 * ============================================================================ */

void test_gmres_create(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);

    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(POISSON_METHOD_GMRES, solver->method);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_SCALAR, solver->backend);
    TEST_ASSERT_EQUAL_STRING("gmres_scalar", solver->name);

    poisson_solver_destroy(solver);
}

void test_gmres_init(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, NULL);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(nx, solver->nx);
    TEST_ASSERT_EQUAL(ny, solver->ny);

    poisson_solver_destroy(solver);
}

/* ============================================================================
 * CONVERGENCE TESTS
 * ============================================================================ */

/**
 * Zero RHS with non-zero initial guess. The initial residual is tiny (the guess
 * is a constant, whose discrete Laplacian is zero on the interior), so this
 * exercises the already-converged early-return / happy path. Must NOT error.
 */
void test_gmres_zero_rhs(void) {
    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = 1.0;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE(stats.iterations < MAX_ITERATIONS);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

void test_gmres_sinusoidal_rhs(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/**
 * After GMRES converges, an independently recomputed true residual must be
 * small. This cross-checks that the solver's reported convergence corresponds
 * to a genuinely small unpreconditioned residual (no precond here). Note the
 * independent check uses the max-norm, whereas stats.final_residual is the
 * 2-norm the solver measures internally, so this validates smallness rather
 * than an exact norm-for-norm match.
 */
void test_gmres_residual_matches_true(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;
    poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    /* Independently recomputed max-norm residual must be small... */
    double true_res = true_residual_inf(p, rhs, nx, ny, dx, dy);
    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(true_res < 10.0 * relative_tol ||
                     true_res < 1e-6);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/**
 * GMRES vs CG on the SPD Poisson problem: same solution up to a Neumann
 * constant offset. Max-norm difference cross-check.
 */
void test_gmres_vs_cg(void) {
    size_t nx = NX_SMALL, ny = NY_SMALL;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p_gmres = create_field(nx, ny);
    double* p_cg = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p_gmres);
    TEST_ASSERT_NOT_NULL(p_cg);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    poisson_solver_t* gmres = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(gmres);
    poisson_solver_init(gmres, nx, ny, 1, dx, dy, 0.0, &params);
    poisson_solver_stats_t stats_gmres = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(gmres, p_gmres, NULL, rhs, &stats_gmres);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_t* cg = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(cg);
    poisson_solver_init(cg, nx, ny, 1, dx, dy, 0.0, &params);
    poisson_solver_stats_t stats_cg = poisson_solver_stats_default();
    status = poisson_solver_solve(cg, p_cg, NULL, rhs, &stats_cg);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_gmres.status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_cg.status);

    remove_interior_mean(p_gmres, nx, ny);
    remove_interior_mean(p_cg, nx, ny);

    double max_diff = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double diff = fabs(p_gmres[IDX_2D(i, j, nx)] - p_cg[IDX_2D(i, j, nx)]);
            if (diff > max_diff) max_diff = diff;
        }
    }

    TEST_ASSERT_TRUE(max_diff < 1e-4);

    poisson_solver_destroy(gmres);
    poisson_solver_destroy(cg);
    cfd_free(p_gmres);
    cfd_free(p_cg);
    cfd_free(rhs);
}

/** GMRES vs CG L2 (RMS) cross-check on the medium grid */
void test_gmres_vs_cg_l2(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p_gmres = create_field(nx, ny);
    double* p_cg = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p_gmres);
    TEST_ASSERT_NOT_NULL(p_cg);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    poisson_solver_t* gmres = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    poisson_solver_init(gmres, nx, ny, 1, dx, dy, 0.0, &params);
    poisson_solver_stats_t s1 = poisson_solver_stats_default();
    poisson_solver_solve(gmres, p_gmres, NULL, rhs, &s1);

    poisson_solver_t* cg = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    poisson_solver_init(cg, nx, ny, 1, dx, dy, 0.0, &params);
    poisson_solver_stats_t s2 = poisson_solver_stats_default();
    poisson_solver_solve(cg, p_cg, NULL, rhs, &s2);

    remove_interior_mean(p_gmres, nx, ny);
    remove_interior_mean(p_cg, nx, ny);

    double l2_error = compute_l2_error(p_gmres, p_cg, nx, ny);
    TEST_ASSERT_TRUE(l2_error < 1e-5);

    poisson_solver_destroy(gmres);
    poisson_solver_destroy(cg);
    cfd_free(p_gmres);
    cfd_free(p_cg);
    cfd_free(rhs);
}

/** Manufactured Dirichlet solution p = sin(πx)sin(πy): O(h²) accuracy */
void test_gmres_dirichlet(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    double* exact = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(exact);

    init_dirichlet_rhs(rhs, nx, ny, dx, dy);
    compute_exact_solution(exact, nx, ny);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    remove_interior_mean(p, nx, ny);
    remove_interior_mean(exact, nx, ny);

    double l2_error = compute_l2_error(p, exact, nx, ny);
    TEST_ASSERT_TRUE(l2_error < 0.01);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
    cfd_free(exact);
}

/**
 * Restart must not stall: GMRES(5) (many restarts) and GMRES(30, default) must
 * reach the same mean-removed solution, and the small-m run must finish in a
 * finite number of iterations below the cap.
 */
void test_gmres_restart_no_stall(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p_small = create_field(nx, ny);
    double* p_big = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p_small);
    TEST_ASSERT_NOT_NULL(p_big);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    /* Small restart m = 5 */
    poisson_solver_params_t params_small = poisson_solver_params_default();
    params_small.tolerance = TOLERANCE;
    params_small.max_iterations = MAX_ITERATIONS;
    params_small.restart = 5;

    poisson_solver_t* g_small = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    poisson_solver_init(g_small, nx, ny, 1, dx, dy, 0.0, &params_small);
    poisson_solver_stats_t stats_small = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(g_small, p_small, NULL, rhs, &stats_small);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_small.status);
    TEST_ASSERT_TRUE(stats_small.iterations < MAX_ITERATIONS);

    /* Default restart (m = 30) */
    poisson_solver_params_t params_big = poisson_solver_params_default();
    params_big.tolerance = TOLERANCE;
    params_big.max_iterations = MAX_ITERATIONS;
    params_big.restart = 0;  /* auto -> 30 */

    poisson_solver_t* g_big = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    poisson_solver_init(g_big, nx, ny, 1, dx, dy, 0.0, &params_big);
    poisson_solver_stats_t stats_big = poisson_solver_stats_default();
    status = poisson_solver_solve(g_big, p_big, NULL, rhs, &stats_big);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_big.status);

    remove_interior_mean(p_small, nx, ny);
    remove_interior_mean(p_big, nx, ny);

    double l2_error = compute_l2_error(p_small, p_big, nx, ny);
    TEST_ASSERT_TRUE(l2_error < 1e-6);

    poisson_solver_destroy(g_small);
    poisson_solver_destroy(g_big);
    cfd_free(p_small);
    cfd_free(p_big);
    cfd_free(rhs);
}

/**
 * Jacobi preconditioner path must converge. On a uniform grid the Laplacian
 * diagonal is constant, so Jacobi preconditioning is a scalar multiply and does
 * NOT reduce iteration count — we assert convergence only, not fewer iterations.
 */
void test_gmres_jacobi_precond(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;
    params.preconditioner = POISSON_PRECOND_JACOBI;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    double relative_tol = stats.initial_residual * params.tolerance;
    TEST_ASSERT_TRUE(stats.final_residual < relative_tol ||
                     stats.final_residual < params.absolute_tolerance);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/**
 * GMRES(1) cannot reach the tolerance in a handful of iterations: the solve must
 * stop exactly at max_iterations and report POISSON_MAX_ITER, and because GMRES
 * minimizes the residual at every step, the residual must not grow.
 */
void test_gmres_max_iter_exhaustion(void) {
    size_t nx = NX_MEDIUM, ny = NY_MEDIUM;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (nx - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (ny - 1);

    double* p = create_field(nx, ny);
    double* rhs = create_field(nx, ny);
    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = 10;
    params.restart = 1;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, NULL, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_ERROR_MAX_ITER, status);
    TEST_ASSERT_EQUAL(POISSON_MAX_ITER, stats.status);
    TEST_ASSERT_EQUAL_INT(10, (int)stats.iterations);
    TEST_ASSERT_TRUE(stats.final_residual <= stats.initial_residual);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(rhs);
}

/* ============================================================================
 * ERROR HANDLING TESTS
 * ============================================================================ */

void test_gmres_unsupported_backend(void) {
    /* GPU backend: create() returns the factory solver whenever the library was
     * built with CUDA (device check happens in init). GMRES has no GPU backend
     * yet, so it must return NULL regardless. */
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_GPU);
    TEST_ASSERT_NULL(solver);
}

/**
 * Restart lengths whose (m+1)*m Hessenberg index would overflow int are rejected
 * at init with CFD_ERROR_LIMIT_EXCEEDED, before anything is allocated. 46341 is
 * the smallest such m.
 */
void test_gmres_rejects_oversized_restart(void) {
    const int restarts[] = { 46341, INT_MAX };
    double h = (DOMAIN_XMAX - DOMAIN_XMIN) / (NX_SMALL - 1);

    for (size_t r = 0; r < sizeof(restarts) / sizeof(restarts[0]); r++) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL(solver);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.restart = restarts[r];

        cfd_status_t status = poisson_solver_init(solver, NX_SMALL, NY_SMALL, 1,
                                                  h, h, 0.0, &params);
        poisson_solver_destroy(solver);
        TEST_ASSERT_EQUAL(CFD_ERROR_LIMIT_EXCEEDED, status);
    }
}

/** solve() before a successful init must return an error, not dereference a NULL context */
void test_gmres_solve_before_init(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    double x[NX_SMALL] = {0.0};
    double rhs[NX_SMALL] = {0.0};
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, x, NULL, rhs, &stats);
    poisson_solver_destroy(solver);

    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
}

void test_gmres_destroy_null(void) {
    poisson_solver_destroy(NULL);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_gmres_create);
    RUN_TEST(test_gmres_init);

    RUN_TEST(test_gmres_zero_rhs);
    RUN_TEST(test_gmres_sinusoidal_rhs);
    RUN_TEST(test_gmres_residual_matches_true);
    RUN_TEST(test_gmres_vs_cg);
    RUN_TEST(test_gmres_vs_cg_l2);
    RUN_TEST(test_gmres_dirichlet);
    RUN_TEST(test_gmres_restart_no_stall);
    RUN_TEST(test_gmres_jacobi_precond);
    RUN_TEST(test_gmres_max_iter_exhaustion);

    RUN_TEST(test_gmres_unsupported_backend);
    RUN_TEST(test_gmres_rejects_oversized_restart);
    RUN_TEST(test_gmres_solve_before_init);
    RUN_TEST(test_gmres_destroy_null);

    return UNITY_END();
}
