/**
 * @file test_optimal_omega.c
 * @brief Tests for the automatic SOR relaxation factor
 *
 * The automatic omega (params.omega = 0) must be the documented optimum for the
 * walls in use, and must converge in close to the fewest sweeps any omega gives:
 * each "near best" test sweeps omega across a range on the same problem and
 * compares. The right-hand side is seeded noise with zero interior mean, so
 * every error mode is present, including the slowest.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2*pi*x) * cos(2*pi*y) with discrete interior mean subtracted.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy)
{
    for (size_t j = 0; j < ny; j++) {
        double y = j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Subtract interior mean for Neumann compatibility */
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += rhs[IDX_2D(i, j, nx)];
            count++;
        }
    }
    double mean = sum / (double)count;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            rhs[IDX_2D(i, j, nx)] -= mean;
        }
    }
}

/**
 * Helper: solve Poisson with given method/backend on NxN grid, return stats.
 * Returns CFD_SUCCESS if converged, or error status.
 */
static cfd_status_t solve_poisson_test(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend,
    size_t nx, size_t ny,
    int max_iterations,
    poisson_solver_stats_t* out_stats)
{
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    size_t n = nx * ny;

    double* x = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    if (!x || !x_temp || !rhs) {
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return CFD_ERROR_NOMEM;
    }

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-6;
    params.max_iterations = max_iterations;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        cfd_free(x); cfd_free(x_temp); cfd_free(rhs);
        return status;
    }

    *out_stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, out_stats);

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    return status;
}

/* ============================================================================
 * NOISE PROBLEMS AND OMEGA SWEEPS
 * ============================================================================ */

typedef struct {
    size_t nx, ny, nz;
    double dx, dy, dz;   /* dz = 0 for 2D */
    int fixed_walls;     /* 1: apply_bc holds the boundary at zero (Dirichlet) */
} grid_case_t;

/* splitmix64, mapped to [-1, 1) */
static double next_uniform(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return 2.0 * ((double)(z >> 11) / 9007199254740992.0) - 1.0;
}

/* Seeded noise on the interior, shifted to zero mean so zero-gradient walls
 * leave the problem solvable */
static double* create_noise_rhs(const grid_case_t* g) {
    size_t plane = (g->nz > 1) ? g->nx * g->ny : 0;
    size_t k_first = (g->nz > 1) ? 1 : 0;
    size_t k_end = (g->nz > 1) ? g->nz - 1 : 1;
    double* rhs = (double*)cfd_calloc(g->nx * g->ny * g->nz, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);

    uint64_t state = 20260913ULL;
    double sum = 0.0;
    size_t count = 0;
    for (size_t k = k_first; k < k_end; k++) {
        for (size_t j = 1; j < g->ny - 1; j++) {
            for (size_t i = 1; i < g->nx - 1; i++) {
                double v = next_uniform(&state);
                rhs[k * plane + IDX_2D(i, j, g->nx)] = v;
                sum += v;
                count++;
            }
        }
    }
    double shift = -sum / (double)count;
    for (size_t k = k_first; k < k_end; k++) {
        for (size_t j = 1; j < g->ny - 1; j++) {
            for (size_t i = 1; i < g->nx - 1; i++) {
                rhs[k * plane + IDX_2D(i, j, g->nx)] += shift;
            }
        }
    }
    return rhs;
}

static void hold_walls_at_zero(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

/* Sweeps Red-Black SOR takes from a zero start with this omega (0 = automatic) */
static int count_sweeps(poisson_solver_method_t method, const grid_case_t* g,
                        const double* rhs, double omega, double* final_residual)
{
    double* x = (double*)cfd_calloc(g->nx * g->ny * g->nz, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);

    poisson_solver_t* solver = poisson_solver_create(method, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    if (g->fixed_walls) {
        solver->apply_bc = hold_walls_at_zero;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-6;
    params.max_iterations = 100000;
    params.omega = omega;
    cfd_status_t init = poisson_solver_init(solver, g->nx, g->ny, g->nz,
                                            g->dx, g->dy, g->dz, &params);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    if (init == CFD_SUCCESS) {
        poisson_solver_solve(solver, x, NULL, rhs, &stats);
    }
    poisson_solver_destroy(solver);
    cfd_free(x);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, init);
    TEST_ASSERT_EQUAL_MESSAGE(POISSON_CONVERGED, stats.status, "solve did not converge");
    if (final_residual) {
        *final_residual = stats.final_residual;
    }
    return stats.iterations;
}

/* The fewest sweeps over omega = from, from + step, ..., to */
static int fewest_sweeps(const grid_case_t* g, const double* rhs,
                         double from, double to, double step)
{
    int best = INT_MAX;
    int count = (int)((to - from) / step + 0.5);
    for (int s = 0; s <= count; s++) {
        int sweeps = count_sweeps(POISSON_METHOD_REDBLACK_SOR, g, rhs, from + s * step, NULL);
        if (sweeps < best) {
            best = sweeps;
        }
    }
    return best;
}

/* The automatic omega converges in at most a quarter more sweeps than the best */
static void assert_auto_near_best(const grid_case_t* g, double from, double to, double step)
{
    double* rhs = create_noise_rhs(g);
    int best = fewest_sweeps(g, rhs, from, to, step);
    int automatic = count_sweeps(POISSON_METHOD_REDBLACK_SOR, g, rhs, 0.0, NULL);
    cfd_free(rhs);

    char msg[160];
    snprintf(msg, sizeof(msg), "%zux%zux%zu: automatic omega took %d sweeps, best omega %d",
             g->nx, g->ny, g->nz, automatic, best);
    TEST_ASSERT_LESS_OR_EQUAL_INT_MESSAGE(best + best / 4, automatic, msg);
}

/* The Neumann optimum as linear_solver_internal.h documents it */
static double documented_neumann_omega(const grid_case_t* g) {
    size_t n[3] = { g->nx, g->ny, g->nz };
    double w[3] = { 1.0 / (g->dx * g->dx), 1.0 / (g->dy * g->dy),
                    (g->nz > 1 && g->dz > 0.0) ? 1.0 / (g->dz * g->dz) : 0.0 };
    double factor = 2.0 * (w[0] + w[1] + w[2]);
    double interior[3];
    for (int a = 0; a < 3; a++) {
        interior[a] = (w[a] > 0.0) ? (double)(n[a] - 2) : 0.0;
    }
    double lambda = -1.0;
    for (int a = 0; a < 3; a++) {
        if (interior[a] < 2.0) {
            continue;
        }
        double half = cos(M_PI / (2.0 * interior[a]));
        double deficit = w[a] * (4.0 / interior[a]) * half * half;
        for (int b = 0; b < 3; b++) {
            if (b != a && interior[b] >= 1.0) {
                deficit += w[b] * (2.0 / interior[b]);
            }
        }
        double mode = w[a] * (2.0 - 2.0 * cos(M_PI / interior[a])) / (factor - deficit);
        if (lambda < 0.0 || mode < lambda) {
            lambda = mode;
        }
    }
    double rho_j = 1.0 - lambda;
    return 2.0 / (1.0 + sqrt(1.0 - (rho_j * rho_j)));
}

/* The Dirichlet optimum as linear_solver_internal.h documents it */
static double documented_dirichlet_omega(const grid_case_t* g) {
    double inv_dx2 = 1.0 / (g->dx * g->dx);
    double inv_dy2 = 1.0 / (g->dy * g->dy);
    double rho_j = (cos(M_PI / (double)(g->nx - 1)) * inv_dx2
                  + cos(M_PI / (double)(g->ny - 1)) * inv_dy2) / (inv_dx2 + inv_dy2);
    return 2.0 / (1.0 + sqrt(1.0 - (rho_j * rho_j)));
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

/**
 * Test: RB-SOR scalar converges on 33x33 within 1000 iterations with auto-omega.
 */
void test_redblack_scalar_33x33_converges(void) {
    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
        33, 33, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_THAN(500, stats.iterations);
}

/**
 * Test: RB-SOR OMP converges on 33x33 within 1000 iterations with auto-omega.
 */
void test_redblack_omp_33x33_converges(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP);
    if (!solver) {
        TEST_IGNORE_MESSAGE("OMP Red-Black SOR solver not available");
        return;
    }
    poisson_solver_destroy(solver);

    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP,
        33, 33, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_LESS_THAN(500, stats.iterations);
}

/**
 * Test: RB-SOR scalar converges across multiple grid sizes.
 */
void test_redblack_scalar_multi_grid(void) {
    size_t grid_sizes[] = {17, 33, 65};
    for (int g = 0; g < 3; g++) {
        size_t n = grid_sizes[g];
        poisson_solver_stats_t stats;
        cfd_status_t status = solve_poisson_test(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
            n, n, 1000, &stats);

        char msg[128];
        snprintf(msg, sizeof(msg), "Failed to converge on %zux%zu grid", n, n);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, msg);
        TEST_ASSERT_EQUAL_MESSAGE(POISSON_CONVERGED, stats.status, msg);
    }
}

/**
 * Test: RB-SOR converges on non-square grid (33x65).
 */
void test_redblack_scalar_nonsquare(void) {
    poisson_solver_stats_t stats;
    cfd_status_t status = solve_poisson_test(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
        33, 65, 1000, &stats);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
}

/**
 * Test: Explicit omega override is respected (omega=1.2 should converge
 * slower than auto-optimal).
 */
void test_explicit_omega_override(void) {
    size_t nx = 33, ny = 33;
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    size_t n = nx * ny;

    double* x1 = (double*)cfd_calloc(n, sizeof(double));
    double* x2 = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x1);
    TEST_ASSERT_NOT_NULL(x2);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    /* Solve with auto-omega (default) */
    poisson_solver_t* solver_auto = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_auto);

    poisson_solver_params_t params_auto = poisson_solver_params_default();
    params_auto.tolerance = 1e-6;
    params_auto.max_iterations = 2000;
    cfd_status_t status = poisson_solver_init(solver_auto, nx, ny, 1, dx, dy, 0.0, &params_auto);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_auto = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_auto, x1, x_temp, rhs, &stats_auto);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve with explicit suboptimal omega=1.2 */
    poisson_solver_t* solver_explicit = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_explicit);

    poisson_solver_params_t params_explicit = poisson_solver_params_default();
    params_explicit.tolerance = 1e-6;
    params_explicit.max_iterations = 2000;
    params_explicit.omega = 1.2;  /* Explicit suboptimal value */
    status = poisson_solver_init(solver_explicit, nx, ny, 1, dx, dy, 0.0, &params_explicit);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_explicit = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_explicit, x2, x_temp, rhs, &stats_explicit);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Explicit omega=1.2 should need more iterations than auto-optimal */
    TEST_ASSERT_GREATER_THAN(stats_auto.iterations, stats_explicit.iterations);

    poisson_solver_destroy(solver_auto);
    poisson_solver_destroy(solver_explicit);
    cfd_free(x1);
    cfd_free(x2);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/**
 * Test: with the default zero-gradient walls, the automatic omega converges
 * close to the fewest sweeps any omega gives, on square grids.
 */
void test_neumann_auto_omega_near_best_square(void) {
    grid_case_t g17 = { 17, 17, 1, 1.0 / 16.0, 1.0 / 16.0, 0.0, 0 };
    grid_case_t g33 = { 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, 0 };
    grid_case_t g65 = { 65, 65, 1, 1.0 / 64.0, 1.0 / 64.0, 0.0, 0 };
    assert_auto_near_best(&g17, 1.50, 1.99, 0.01);
    assert_auto_near_best(&g33, 1.70, 1.99, 0.005);
    assert_auto_near_best(&g65, 1.85, 1.99, 0.005);
}

/**
 * Test: the same on an anisotropic grid and in 3D.
 */
void test_neumann_auto_omega_near_best_anisotropic_and_3d(void) {
    grid_case_t wide = { 33, 65, 1, 1.0 / 32.0, 1.0 / 64.0, 0.0, 0 };
    grid_case_t cube = { 17, 17, 17, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 0 };
    grid_case_t slab = { 33, 33, 9, 1.0 / 32.0, 1.0 / 32.0, 1.0 / 32.0, 0 };
    assert_auto_near_best(&wide, 1.80, 1.99, 0.005);
    assert_auto_near_best(&cube, 1.60, 1.99, 0.01);
    assert_auto_near_best(&slab, 1.60, 1.99, 0.01);
}

/**
 * Test: with walls the caller holds fixed, the automatic omega is still near
 * the best, through the Dirichlet formula.
 */
void test_dirichlet_auto_omega_near_best(void) {
    grid_case_t g33 = { 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, 1 };
    grid_case_t g65 = { 65, 65, 1, 1.0 / 64.0, 1.0 / 64.0, 0.0, 1 };
    assert_auto_near_best(&g33, 1.70, 1.95, 0.005);
    assert_auto_near_best(&g65, 1.85, 1.98, 0.005);
}

/**
 * Test: the automatic omega is exactly the documented formula for the walls in
 * use: an explicit omega at that value gives the same sweeps and residual.
 */
void test_auto_omega_is_the_documented_formula(void) {
    grid_case_t cases[] = {
        { 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, 0 },
        { 33, 65, 1, 1.0 / 32.0, 1.0 / 64.0, 0.0, 0 },
        { 17, 17, 17, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 0 },
        { 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, 1 },
    };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const grid_case_t* g = &cases[c];
        double* rhs = create_noise_rhs(g);
        double documented = g->fixed_walls ? documented_dirichlet_omega(g) : documented_neumann_omega(g);
        double res_auto = 0.0, res_documented = 0.0;
        int sweeps_auto = count_sweeps(POISSON_METHOD_REDBLACK_SOR, g, rhs, 0.0, &res_auto);
        int sweeps_documented = count_sweeps(POISSON_METHOD_REDBLACK_SOR, g, rhs, documented, &res_documented);
        cfd_free(rhs);

        char msg[128];
        snprintf(msg, sizeof(msg), "case %zu (%zux%zux%zu, %s walls)", c, g->nx, g->ny, g->nz,
                 g->fixed_walls ? "fixed" : "zero-gradient");
        TEST_ASSERT_EQUAL_INT_MESSAGE(sweeps_documented, sweeps_auto, msg);
        TEST_ASSERT_EQUAL_DOUBLE_MESSAGE(res_documented, res_auto, msg);
    }
}

/**
 * Test: a fixed omega of 1.5 costs many times the automatic omega's sweeps on
 * a 65x65 grid, and the gap grows with the grid.
 */
void test_fixed_omega_1_5_falls_behind_as_grid_grows(void) {
    grid_case_t g33 = { 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, 0 };
    grid_case_t g65 = { 65, 65, 1, 1.0 / 64.0, 1.0 / 64.0, 0.0, 0 };

    double* rhs33 = create_noise_rhs(&g33);
    int ratio33 = count_sweeps(POISSON_METHOD_REDBLACK_SOR, &g33, rhs33, 1.5, NULL)
                / count_sweeps(POISSON_METHOD_REDBLACK_SOR, &g33, rhs33, 0.0, NULL);
    cfd_free(rhs33);

    double* rhs65 = create_noise_rhs(&g65);
    int ratio65 = count_sweeps(POISSON_METHOD_REDBLACK_SOR, &g65, rhs65, 1.5, NULL)
                / count_sweeps(POISSON_METHOD_REDBLACK_SOR, &g65, rhs65, 0.0, NULL);
    cfd_free(rhs65);

    TEST_ASSERT_GREATER_OR_EQUAL_INT(5, ratio65);
    TEST_ASSERT_GREATER_THAN_INT(ratio33, ratio65);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_redblack_scalar_33x33_converges);
    RUN_TEST(test_redblack_omp_33x33_converges);
    RUN_TEST(test_redblack_scalar_multi_grid);
    RUN_TEST(test_redblack_scalar_nonsquare);
    RUN_TEST(test_explicit_omega_override);
    RUN_TEST(test_neumann_auto_omega_near_best_square);
    RUN_TEST(test_neumann_auto_omega_near_best_anisotropic_and_3d);
    RUN_TEST(test_dirichlet_auto_omega_near_best);
    RUN_TEST(test_auto_omega_is_the_documented_formula);
    RUN_TEST(test_fixed_omega_1_5_falls_behind_as_grid_grows);
    return UNITY_END();
}
