/**
 * @file test_poisson_3d.c
 * @brief 3D Poisson solver tests with manufactured solutions
 *
 * Tests verify that all five scalar CPU linear solvers (Jacobi, SOR,
 * Red-Black SOR, CG, BiCGSTAB) correctly solve the 3D Poisson equation
 * on a [0,1]^3 domain with Dirichlet BCs.
 *
 * Manufactured solution: p = sin(pi*x) * sin(pi*y) * sin(pi*z)
 *   => nabla^2 p = -3*pi^2 * sin(pi*x) * sin(pi*y) * sin(pi*z)
 *
 * Also tests backward compatibility (nz=1 produces same results as 2D)
 * and grid convergence (O(h^2) rate).
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.0

/* Grid sizes */
#define N3D 17  /* 17^3 = 4913 points — fast enough for scalar solvers */

/* Tolerances */
#define L2_ERROR_TOL       1e-2   /* L2 error tolerance for sinusoidal */
#define COMPAT_TOL         1e-10  /* nz=1 backward compatibility */
#define SOLVER_COMPARE_TOL 1e-4   /* Cross-solver agreement */
#define CONVERGENCE_RATE_TOL 0.3  /* Allowance for convergence order check */

/* Solver parameters */
#define MAX_ITER_JACOBI  5000
#define MAX_ITER_SOR     3000
#define MAX_ITER_CG      2000
#define SOLVER_TOL       1e-8

/* ============================================================================
 * UNITY SETUP
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * 3D ANALYTICAL SOLUTION
 * ============================================================================ */

static double sinusoidal_3d(double x, double y, double z) {
    return sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

static double sinusoidal_3d_rhs(double x, double y, double z) {
    return -3.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
}

/* 2D version for backward-compat tests */
static double sinusoidal_2d(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

static double sinusoidal_2d_rhs(double x, double y) {
    return -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

/* ============================================================================
 * FIELD HELPERS
 * ============================================================================ */

static double* alloc_field(size_t n) {
    return (double*)cfd_calloc(n, sizeof(double));
}

static void init_3d_rhs(double* rhs, size_t nx, size_t ny, size_t nz,
                         double dx, double dy, double dz) {
    for (size_t k = 0; k < nz; k++) {
        double z = DOMAIN_MIN + k * dz;
        for (size_t j = 0; j < ny; j++) {
            double y = DOMAIN_MIN + j * dy;
            for (size_t i = 0; i < nx; i++) {
                double x = DOMAIN_MIN + i * dx;
                size_t idx = k * nx * ny + j * nx + i;
                rhs[idx] = sinusoidal_3d_rhs(x, y, z);
            }
        }
    }
}

static void init_3d_analytical(double* p, size_t nx, size_t ny, size_t nz,
                                double dx, double dy, double dz) {
    for (size_t k = 0; k < nz; k++) {
        double z = DOMAIN_MIN + k * dz;
        for (size_t j = 0; j < ny; j++) {
            double y = DOMAIN_MIN + j * dy;
            for (size_t i = 0; i < nx; i++) {
                double x = DOMAIN_MIN + i * dx;
                size_t idx = k * nx * ny + j * nx + i;
                p[idx] = sinusoidal_3d(x, y, z);
            }
        }
    }
}

static double compute_l2_error_3d(const double* numerical, const double* analytical,
                                   size_t nx, size_t ny, size_t nz) {
    double sum_sq = 0.0;
    size_t count = 0;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end   = (nz > 1) ? (nz - 1) : 1;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * nx * ny + j * nx + i;
                double err = numerical[idx] - analytical[idx];
                sum_sq += err * err;
                count++;
            }
        }
    }
    if (count == 0) return 0.0;
    return sqrt(sum_sq / count);
}

/* ============================================================================
 * 3D DIRICHLET BC CALLBACK
 *
 * Sets analytical values on all 6 faces.  Grid dimensions are passed
 * through the module-level g_bc_ctx struct, set before each solve.
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz;
    double dx, dy, dz;
} bc_3d_ctx_t;

/* Global context for BC callback (simple approach for tests) */
static bc_3d_ctx_t g_bc_ctx;

static void apply_dirichlet_3d(poisson_solver_t* solver, double* p) {
    (void)solver;
    size_t nx = g_bc_ctx.nx;
    size_t ny = g_bc_ctx.ny;
    size_t nz = g_bc_ctx.nz;
    double dx = g_bc_ctx.dx;
    double dy = g_bc_ctx.dy;
    double dz = g_bc_ctx.dz;
    size_t plane = nx * ny;

    /* x-faces (i=0 and i=nx-1) */
    for (size_t k = 0; k < nz; k++) {
        double z = DOMAIN_MIN + k * dz;
        for (size_t j = 0; j < ny; j++) {
            double y = DOMAIN_MIN + j * dy;
            size_t base = k * plane + j * nx;
            p[base + 0]      = sinusoidal_3d(DOMAIN_MIN, y, z);
            p[base + nx - 1] = sinusoidal_3d(DOMAIN_MAX, y, z);
        }
    }
    /* y-faces (j=0 and j=ny-1) */
    for (size_t k = 0; k < nz; k++) {
        double z = DOMAIN_MIN + k * dz;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_MIN + i * dx;
            p[k * plane + 0 * nx + i]          = sinusoidal_3d(x, DOMAIN_MIN, z);
            p[k * plane + (ny - 1) * nx + i]   = sinusoidal_3d(x, DOMAIN_MAX, z);
        }
    }
    /* z-faces (k=0 and k=nz-1) — only when nz>1 */
    if (nz > 1) {
        for (size_t j = 0; j < ny; j++) {
            double y = DOMAIN_MIN + j * dy;
            for (size_t i = 0; i < nx; i++) {
                double x = DOMAIN_MIN + i * dx;
                p[0 * plane + j * nx + i]          = sinusoidal_3d(x, y, DOMAIN_MIN);
                p[(nz - 1) * plane + j * nx + i]   = sinusoidal_3d(x, y, DOMAIN_MAX);
            }
        }
    }
}

/* 2D Dirichlet BC callback for backward-compat tests */
static void apply_dirichlet_2d(poisson_solver_t* solver, double* p) {
    (void)solver;
    size_t nx = g_bc_ctx.nx;
    size_t ny = g_bc_ctx.ny;
    double dx = g_bc_ctx.dx;
    double dy = g_bc_ctx.dy;

    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_MIN + j * dy;
        p[j * nx + 0]      = sinusoidal_2d(DOMAIN_MIN, y);
        p[j * nx + nx - 1] = sinusoidal_2d(DOMAIN_MAX, y);
    }
    for (size_t i = 0; i < nx; i++) {
        double x = DOMAIN_MIN + i * dx;
        p[0 * nx + i]          = sinusoidal_2d(x, DOMAIN_MIN);
        p[(ny - 1) * nx + i]   = sinusoidal_2d(x, DOMAIN_MAX);
    }
}

/* ============================================================================
 * GENERIC SOLVE HELPER
 *
 * Creates solver, sets Dirichlet BC callback, solves, returns L2 error.
 * ============================================================================ */

typedef struct {
    double l2_error;
    int    iterations;
    int    converged;
    int    solver_unavailable;
} solve_result_t;

static solve_result_t solve_3d_sinusoidal_backend(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend,
    size_t nx, size_t ny, size_t nz,
    int max_iter, double omega)
{
    solve_result_t result = {0};

    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (double)(nx - 1);
    double dy = (DOMAIN_MAX - DOMAIN_MIN) / (double)(ny - 1);
    double dz = (nz > 1) ? (DOMAIN_MAX - DOMAIN_MIN) / (double)(nz - 1) : 0.0;
    size_t total = nx * ny * nz;

    /* Set global BC context */
    g_bc_ctx = (bc_3d_ctx_t){ nx, ny, nz, dx, dy, dz };

    double* p          = alloc_field(total);
    double* p_temp     = alloc_field(total);  /* needed for Jacobi */
    double* rhs        = alloc_field(total);
    double* analytical = alloc_field(total);

    if (!p || !p_temp || !rhs || !analytical) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analytical);
        result.l2_error = -1.0;
        return result;
    }

    init_3d_rhs(rhs, nx, ny, nz, dx, dy, dz);
    init_3d_analytical(analytical, nx, ny, nz, dx, dy, dz);

    /* Check backend availability before creating solver */
    if (!poisson_solver_backend_available(backend)) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analytical);
        result.l2_error = -1.0;
        result.solver_unavailable = 1;
        return result;
    }

    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analytical);
        result.l2_error = -1.0;
        return result;
    }

    /* Install custom Dirichlet BC */
    solver->apply_bc = apply_dirichlet_3d;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = SOLVER_TOL;
    params.max_iterations = max_iter;
    if (omega > 0.0) params.omega = omega;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        poisson_solver_destroy(solver);
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analytical);
        result.l2_error = -1.0;
        result.solver_unavailable = 1;
        return result;
    }
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analytical);
        result.l2_error = -1.0;
        return result;
    }

    /* Apply initial Dirichlet BCs */
    apply_dirichlet_3d(solver, p);

    /* Solve */
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    result.l2_error   = compute_l2_error_3d(p, analytical, nx, ny, nz);
    result.iterations = stats.iterations;
    result.converged  = (status == CFD_SUCCESS);

    poisson_solver_destroy(solver);
    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
    cfd_free(analytical);
    return result;
}

/* Backward-compatible wrapper: defaults to SCALAR backend */
static solve_result_t solve_3d_sinusoidal(
    poisson_solver_method_t method,
    size_t nx, size_t ny, size_t nz,
    int max_iter, double omega)
{
    return solve_3d_sinusoidal_backend(method, POISSON_BACKEND_SCALAR,
                                       nx, ny, nz, max_iter, omega);
}

/* ============================================================================
 * 3D SINUSOIDAL TESTS — ONE PER SOLVER (SCALAR)
 * ============================================================================ */

void test_3d_cg_sinusoidal(void) {
    printf("\n    CG on 3D sinusoidal (%dx%dx%d)...\n", N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal(
        POISSON_METHOD_CG, N3D, N3D, N3D, MAX_ITER_CG, 0.0);
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "CG 3D L2 error exceeds tolerance");
}

void test_3d_jacobi_sinusoidal(void) {
    printf("\n    Jacobi on 3D sinusoidal (%dx%dx%d)...\n", N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal(
        POISSON_METHOD_JACOBI, N3D, N3D, N3D, MAX_ITER_JACOBI, 0.0);
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "Jacobi 3D L2 error exceeds tolerance");
}

void test_3d_sor_sinusoidal(void) {
    printf("\n    SOR on 3D sinusoidal (%dx%dx%d)...\n", N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal(
        POISSON_METHOD_SOR, N3D, N3D, N3D, MAX_ITER_SOR, 1.5);
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "SOR 3D L2 error exceeds tolerance");
}

void test_3d_redblack_sinusoidal(void) {
    printf("\n    Red-Black SOR on 3D sinusoidal (%dx%dx%d)...\n", N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal(
        POISSON_METHOD_REDBLACK_SOR, N3D, N3D, N3D, MAX_ITER_SOR, 1.5);
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "Red-Black SOR 3D L2 error exceeds tolerance");
}

void test_3d_bicgstab_sinusoidal(void) {
    printf("\n    BiCGSTAB on 3D sinusoidal (%dx%dx%d)...\n", N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal(
        POISSON_METHOD_BICGSTAB, N3D, N3D, N3D, MAX_ITER_CG, 0.0);
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "BiCGSTAB 3D L2 error exceeds tolerance");
}

/* ============================================================================
 * BACKWARD COMPATIBILITY — nz=1 must match 2D
 * ============================================================================ */

static double solve_2d_sinusoidal_l2(poisson_solver_method_t method,
                                      size_t nx, size_t ny, int max_iter) {
    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (double)(nx - 1);
    double dy = (DOMAIN_MAX - DOMAIN_MIN) / (double)(ny - 1);

    g_bc_ctx = (bc_3d_ctx_t){ nx, ny, 1, dx, dy, 0.0 };

    double* p      = alloc_field(nx * ny);
    double* p_temp = alloc_field(nx * ny);
    double* rhs    = alloc_field(nx * ny);
    double* analy  = alloc_field(nx * ny);

    if (!p || !p_temp || !rhs || !analy) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
        return -1.0;
    }

    /* 2D sinusoidal manufactured solution */
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_MIN + i * dx;
            rhs[j * nx + i]   = sinusoidal_2d_rhs(x, y);
            analy[j * nx + i] = sinusoidal_2d(x, y);
        }
    }

    poisson_solver_t* solver = poisson_solver_create(method, POISSON_BACKEND_SCALAR);
    if (!solver) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
        return -1.0;
    }

    solver->apply_bc = apply_dirichlet_2d;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = SOLVER_TOL;
    params.max_iterations = max_iter;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
        return -1.0;
    }

    apply_dirichlet_2d(solver, p);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    double l2 = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double err = p[j * nx + i] - analy[j * nx + i];
            l2 += err * err;
            count++;
        }
    }
    l2 = (count > 0) ? sqrt(l2 / count) : 0.0;

    poisson_solver_destroy(solver);
    cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
    return l2;
}

void test_3d_backward_compat_nz1_cg(void) {
    printf("\n    Backward compat nz=1 CG...\n");
    size_t n = 33;

    double l2_2d = solve_2d_sinusoidal_l2(POISSON_METHOD_CG, n, n, MAX_ITER_CG);
    TEST_ASSERT_TRUE_MESSAGE(l2_2d >= 0.0, "2D solve failed");

    /* Solve same problem via 3D path with nz=1, dz=0 */
    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (double)(n - 1);
    double dy = dx;

    g_bc_ctx = (bc_3d_ctx_t){ n, n, 1, dx, dy, 0.0 };

    double* p      = alloc_field(n * n);
    double* p_temp = alloc_field(n * n);
    double* rhs    = alloc_field(n * n);
    double* analy  = alloc_field(n * n);

    if (!p || !p_temp || !rhs || !analy) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }

    for (size_t j = 0; j < n; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 0; i < n; i++) {
            double x = DOMAIN_MIN + i * dx;
            rhs[j * n + i]   = sinusoidal_2d_rhs(x, y);
            analy[j * n + i] = sinusoidal_2d(x, y);
        }
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    solver->apply_bc = apply_dirichlet_2d;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = SOLVER_TOL;
    params.max_iterations = MAX_ITER_CG;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    apply_dirichlet_2d(solver, p);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    double l2_nz1 = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double err = p[j * n + i] - analy[j * n + i];
            l2_nz1 += err * err;
            count++;
        }
    }
    l2_nz1 = sqrt(l2_nz1 / count);

    printf("      2D L2: %.6e, nz=1 L2: %.6e, diff: %.6e\n",
           l2_2d, l2_nz1, fabs(l2_2d - l2_nz1));

    TEST_ASSERT_DOUBLE_WITHIN(COMPAT_TOL, l2_2d, l2_nz1);

    poisson_solver_destroy(solver);
    cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
}

void test_3d_backward_compat_nz1_jacobi(void) {
    printf("\n    Backward compat nz=1 Jacobi...\n");
    size_t n = 33;

    double l2_2d = solve_2d_sinusoidal_l2(POISSON_METHOD_JACOBI, n, n, MAX_ITER_JACOBI);
    TEST_ASSERT_TRUE_MESSAGE(l2_2d >= 0.0, "2D solve failed");

    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (double)(n - 1);
    double dy = dx;

    g_bc_ctx = (bc_3d_ctx_t){ n, n, 1, dx, dy, 0.0 };

    double* p      = alloc_field(n * n);
    double* p_temp = alloc_field(n * n);
    double* rhs    = alloc_field(n * n);
    double* analy  = alloc_field(n * n);

    if (!p || !p_temp || !rhs || !analy) {
        cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }

    for (size_t j = 0; j < n; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 0; i < n; i++) {
            double x = DOMAIN_MIN + i * dx;
            rhs[j * n + i]   = sinusoidal_2d_rhs(x, y);
            analy[j * n + i] = sinusoidal_2d(x, y);
        }
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    solver->apply_bc = apply_dirichlet_2d;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = SOLVER_TOL;
    params.max_iterations = MAX_ITER_JACOBI;

    cfd_status_t status = poisson_solver_init(solver, n, n, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    apply_dirichlet_2d(solver, p);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    double l2_nz1 = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            double err = p[j * n + i] - analy[j * n + i];
            l2_nz1 += err * err;
            count++;
        }
    }
    l2_nz1 = sqrt(l2_nz1 / count);

    printf("      2D L2: %.6e, nz=1 L2: %.6e, diff: %.6e\n",
           l2_2d, l2_nz1, fabs(l2_2d - l2_nz1));

    TEST_ASSERT_DOUBLE_WITHIN(COMPAT_TOL, l2_2d, l2_nz1);

    poisson_solver_destroy(solver);
    cfd_free(p); cfd_free(p_temp); cfd_free(rhs); cfd_free(analy);
}

/* ============================================================================
 * GRID CONVERGENCE — verify O(h^2) rate
 * ============================================================================ */

void test_3d_grid_convergence_cg(void) {
    printf("\n    3D grid convergence CG...\n");

    size_t sizes[] = {9, 17, 33};
    int num_sizes = 3;
    double errors[3];
    double spacings[3];

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double h = (DOMAIN_MAX - DOMAIN_MIN) / (double)(n - 1);
        spacings[s] = h;

        solve_result_t r = solve_3d_sinusoidal(POISSON_METHOD_CG, n, n, n, MAX_ITER_CG, 0.0);
        TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
        errors[s] = r.l2_error;
        printf("      %zux%zux%zu (h=%.4f): L2 = %.6e, iters = %d\n",
               n, n, n, h, errors[s], r.iterations);
    }

    for (int s = 1; s < num_sizes; s++) {
        double rate = log(errors[s-1] / errors[s]) / log(spacings[s-1] / spacings[s]);
        printf("      Rate %zu->%zu: %.2f (expected ~2.0)\n",
               sizes[s-1], sizes[s], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate > 2.0 - CONVERGENCE_RATE_TOL,
            "3D grid convergence rate below O(h^2)");
    }
}

/* ============================================================================
 * SOLVER COMPARISON — all 5 solvers on same 3D problem
 * ============================================================================ */

void test_3d_solver_comparison(void) {
    printf("\n    Comparing all 5 solvers on 3D problem...\n");

    struct {
        poisson_solver_method_t method;
        const char* name;
        int max_iter;
        double omega;
    } solvers[] = {
        { POISSON_METHOD_CG,            "CG",          MAX_ITER_CG,     0.0 },
        { POISSON_METHOD_BICGSTAB,      "BiCGSTAB",    MAX_ITER_CG,     0.0 },
        { POISSON_METHOD_JACOBI,        "Jacobi",      MAX_ITER_JACOBI, 0.0 },
        { POISSON_METHOD_SOR,           "SOR",         MAX_ITER_SOR,    1.5 },
        { POISSON_METHOD_REDBLACK_SOR,  "Red-Black",   MAX_ITER_SOR,    1.5 }
    };
    int num_solvers = (int)(sizeof(solvers) / sizeof(solvers[0]));

    double reference_error = -1.0;

    for (int i = 0; i < num_solvers; i++) {
        solve_result_t r = solve_3d_sinusoidal(
            solvers[i].method, N3D, N3D, N3D,
            solvers[i].max_iter, solvers[i].omega);

        printf("      %-12s: L2 = %.6e, iters = %d, converged = %d\n",
               solvers[i].name, r.l2_error, r.iterations, r.converged);

        TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
        TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
            "3D solver L2 error exceeds tolerance");

        if (reference_error < 0.0) {
            reference_error = r.l2_error;
        } else {
            double diff = fabs(r.l2_error - reference_error);
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "%s differs from reference by %.6e (tol %.6e)",
                     solvers[i].name, diff, SOLVER_COMPARE_TOL);
            TEST_ASSERT_TRUE_MESSAGE(diff < SOLVER_COMPARE_TOL, msg);
        }
    }
}

/* ============================================================================
 * 3D SINUSOIDAL TESTS — SIMD BACKEND
 *
 * These test SIMD (AVX2/NEON) solvers on 3D grids.
 * Skip gracefully if SIMD backend is unavailable.
 * ============================================================================ */

static void run_simd_3d_test(poisson_solver_method_t method, const char* name,
                              int max_iter, double omega) {
    printf("\n    %s SIMD on 3D sinusoidal (%dx%dx%d)...\n", name, N3D, N3D, N3D);
    solve_result_t r = solve_3d_sinusoidal_backend(
        method, POISSON_BACKEND_SIMD, N3D, N3D, N3D, max_iter, omega);
    if (r.solver_unavailable) {
        printf("      SIMD backend unavailable — skipping\n");
        TEST_PASS();
        return;
    }
    printf("      L2 error: %.6e, iters: %d, converged: %d\n",
           r.l2_error, r.iterations, r.converged);
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error >= 0.0, "Solver setup failed");
    TEST_ASSERT_TRUE_MESSAGE(r.l2_error < L2_ERROR_TOL,
        "SIMD 3D L2 error exceeds tolerance");
}

void test_3d_cg_simd_sinusoidal(void) {
    run_simd_3d_test(POISSON_METHOD_CG, "CG", MAX_ITER_CG, 0.0);
}

void test_3d_jacobi_simd_sinusoidal(void) {
    run_simd_3d_test(POISSON_METHOD_JACOBI, "Jacobi", MAX_ITER_JACOBI, 0.0);
}

void test_3d_redblack_simd_sinusoidal(void) {
    run_simd_3d_test(POISSON_METHOD_REDBLACK_SOR, "Red-Black", MAX_ITER_SOR, 1.5);
}

void test_3d_bicgstab_simd_sinusoidal(void) {
    run_simd_3d_test(POISSON_METHOD_BICGSTAB, "BiCGSTAB", MAX_ITER_CG, 0.0);
}

/* Cross-backend comparison: SIMD results should match scalar */
void test_3d_simd_vs_scalar_cg(void) {
    printf("\n    SIMD vs Scalar CG on 3D (%dx%dx%d)...\n", N3D, N3D, N3D);

    solve_result_t scalar = solve_3d_sinusoidal_backend(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, N3D, N3D, N3D, MAX_ITER_CG, 0.0);
    TEST_ASSERT_TRUE_MESSAGE(scalar.l2_error >= 0.0, "Scalar solve failed");

    solve_result_t simd = solve_3d_sinusoidal_backend(
        POISSON_METHOD_CG, POISSON_BACKEND_SIMD, N3D, N3D, N3D, MAX_ITER_CG, 0.0);
    if (simd.solver_unavailable) {
        printf("      SIMD backend unavailable — skipping\n");
        TEST_PASS();
        return;
    }
    TEST_ASSERT_TRUE_MESSAGE(simd.l2_error >= 0.0, "SIMD solve failed");

    double diff = fabs(scalar.l2_error - simd.l2_error);
    printf("      Scalar L2: %.6e, SIMD L2: %.6e, diff: %.6e\n",
           scalar.l2_error, simd.l2_error, diff);
    TEST_ASSERT_TRUE_MESSAGE(diff < SOLVER_COMPARE_TOL,
        "SIMD and Scalar CG solutions differ too much");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("3D POISSON SOLVER TESTS\n");
    printf("========================================\n");

    /* 3D sinusoidal tests */
    printf("\n--- 3D Sinusoidal Tests ---\n");
    RUN_TEST(test_3d_cg_sinusoidal);
    RUN_TEST(test_3d_jacobi_sinusoidal);
    RUN_TEST(test_3d_sor_sinusoidal);
    RUN_TEST(test_3d_redblack_sinusoidal);
    RUN_TEST(test_3d_bicgstab_sinusoidal);

    /* Backward compatibility */
    printf("\n--- Backward Compatibility (nz=1) ---\n");
    RUN_TEST(test_3d_backward_compat_nz1_cg);
    RUN_TEST(test_3d_backward_compat_nz1_jacobi);

    /* Grid convergence */
    printf("\n--- 3D Grid Convergence ---\n");
    RUN_TEST(test_3d_grid_convergence_cg);

    /* Solver comparison */
    printf("\n--- 3D Solver Comparison ---\n");
    RUN_TEST(test_3d_solver_comparison);

    /* SIMD backend tests */
    printf("\n--- 3D SIMD Backend Tests ---\n");
    RUN_TEST(test_3d_cg_simd_sinusoidal);
    RUN_TEST(test_3d_jacobi_simd_sinusoidal);
    RUN_TEST(test_3d_redblack_simd_sinusoidal);
    RUN_TEST(test_3d_bicgstab_simd_sinusoidal);
    RUN_TEST(test_3d_simd_vs_scalar_cg);

    return UNITY_END();
}
