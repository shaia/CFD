/**
 * @file test_poisson_roundoff_floor.c
 * @brief A Poisson solve started from its own converged solution must return
 *        at once, however large the pressure and however fine the grid.
 *
 * The discrete Laplacian of a field x cannot be evaluated more accurately than
 * about eps * |x| * (2/dx^2 + 2/dy^2): the stencil subtracts O(|x|/h^2) terms
 * that cancel. That is a floor under every residual a solver can measure. The
 * stopping rule asks for max(tolerance * r0, absolute_tolerance), and near a
 * steady state, where each solve warm-starts from the last pressure, r0 is
 * already at that floor. The target is then set by absolute_tolerance alone,
 * and once the floor exceeds it the solve cannot stop: it spends every
 * iteration it is allowed and reports CFD_ERROR_MAX_ITER.
 *
 * The floor grows as |x|/h^2, so a fixed absolute tolerance is met on coarse
 * grids and missed on fine ones. Measured with the multigrid pressure solve
 * (|p| ~ 5): converged every time on 129x129, stalled from 257x257 up. A
 * 513x513 Re=1000 cavity run failed this way at step 60,895.
 *
 * This test builds that situation directly at small cost: 129x129 with
 * |x| = 100 puts the floor near 1e-9, ten times the default absolute tolerance.
 * Every method and backend available is started from the converged field and
 * must succeed within a handful of iterations without moving it.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/poisson_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    cfd_init();
}
void tearDown(void) {
    cfd_finalize();
}

#define FLOOR_N   129
#define FLOOR_AMP 100.0
/* Enough for any method to leave a floor-level start; a stall spends them all */
#define FLOOR_MAX_ITER 2000
/* "At once": a solve from the answer needs no more than this */
#define FLOOR_ITER_LIMIT 10

/* ============================================================================
 * PROBLEM
 * ============================================================================ */

typedef struct {
    size_t n;
    double h;
    double* rhs;
    double* x; /* converged solution, residual at the round-off floor */
} floor_problem_t;

/* rhs = Laplacian of amp * cos(pi x) cos(pi y), shifted to zero interior mean so
 * the singular zero-gradient operator has a solution. */
static floor_problem_t floor_problem_create(void) {
    floor_problem_t p;
    p.n = FLOOR_N;
    p.h = 1.0 / (double)(p.n - 1);
    size_t total = p.n * p.n;
    p.rhs = (double*)cfd_calloc(total, sizeof(double));
    p.x = (double*)cfd_calloc(total, sizeof(double));
    TEST_ASSERT_NOT_NULL(p.rhs);
    TEST_ASSERT_NOT_NULL(p.x);
    for (size_t j = 0; j < p.n; j++) {
        for (size_t i = 0; i < p.n; i++) {
            p.rhs[IDX_2D(i, j, p.n)] = -2.0 * M_PI * M_PI * FLOOR_AMP *
                                       cos(M_PI * (double)i * p.h) * cos(M_PI * (double)j * p.h);
        }
    }
    poisson_make_rhs_compatible(p.rhs, p.n, p.n, 1);

    /* Converge with scalar multigrid, driven well past its tolerance so the
     * residual is as small as double precision allows. */
    poisson_solver_config_t cfg = poisson_solver_config_preset(POISSON_PRESET_MULTIGRID);
    poisson_solver_t* mg = poisson_solver_create(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(mg);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          poisson_solver_init(mg, p.n, p.n, 1, p.h, p.h, 0.0, &cfg.params));
    double* tmp = (double*)cfd_calloc(total, sizeof(double));
    TEST_ASSERT_NOT_NULL(tmp);
    for (int c = 0; c < 60; c++) {
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, poisson_solver_iterate(mg, p.x, tmp, p.rhs, NULL));
    }
    double floor_res = poisson_solver_compute_residual(mg, p.x, p.rhs);
    printf("\n    Converged field: residual %.2e (default absolute_tolerance %.0e)\n", floor_res,
           poisson_solver_params_default().absolute_tolerance);
    /* The premise: the floor really is above the fixed absolute tolerance */
    TEST_ASSERT_TRUE_MESSAGE(floor_res > 5.0 * poisson_solver_params_default().absolute_tolerance,
                             "Test premise: round-off floor must exceed absolute_tolerance");
    cfd_free(tmp);
    poisson_solver_destroy(mg);
    return p;
}

static void floor_problem_destroy(floor_problem_t* p) {
    cfd_free(p->rhs);
    cfd_free(p->x);
}

/* ============================================================================
 * CHECK
 * ============================================================================ */

static const char* method_name(poisson_solver_method_t m) {
    switch (m) {
        case POISSON_METHOD_JACOBI:
            return "Jacobi";
        case POISSON_METHOD_GAUSS_SEIDEL:
            return "Gauss-Seidel";
        case POISSON_METHOD_SOR:
            return "SOR";
        case POISSON_METHOD_REDBLACK_SOR:
            return "Red-Black SOR";
        case POISSON_METHOD_CG:
            return "CG";
        case POISSON_METHOD_BICGSTAB:
            return "BiCGSTAB";
        case POISSON_METHOD_GMRES:
            return "GMRES";
        case POISSON_METHOD_MULTIGRID:
            return "Multigrid";
    }
    return "?";
}

static const char* backend_name(poisson_solver_backend_t b) {
    switch (b) {
        case POISSON_BACKEND_SCALAR:
            return "scalar";
        case POISSON_BACKEND_OMP:
            return "omp";
        case POISSON_BACKEND_SIMD:
            return "simd";
        case POISSON_BACKEND_GPU:
            return "gpu";
        default:
            return "auto";
    }
}

/* Returns 1 on failure, 0 on pass or when the pair is not available. */
static int check_restart_from_solution(const floor_problem_t* p, poisson_solver_method_t method,
                                       poisson_solver_backend_t backend) {
    poisson_solver_t* s = poisson_solver_create(method, backend);
    if (!s) {
        return 0;
    }
    poisson_solver_params_t params = poisson_solver_params_default();
    params.max_iterations = FLOOR_MAX_ITER;
    if (poisson_solver_init(s, p->n, p->n, 1, p->h, p->h, 0.0, &params) != CFD_SUCCESS) {
        poisson_solver_destroy(s);
        return 0;
    }

    size_t total = p->n * p->n;
    double* x = (double*)cfd_malloc(total * sizeof(double));
    double* tmp = (double*)cfd_calloc(total, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(tmp);
    memcpy(x, p->x, total * sizeof(double));

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(s, x, tmp, p->rhs, &stats);

    /* The answer must come back: at most a round-off move, relative to |x| */
    double moved = 0.0;
    for (size_t k = 0; k < total; k++) {
        moved = fmax(moved, fabs(x[k] - p->x[k]));
    }
    moved /= FLOOR_AMP;

    int ok = (status == CFD_SUCCESS) && (stats.iterations <= FLOOR_ITER_LIMIT) && (moved < 1e-9);
    printf("      %-14s %-7s %-10s iters=%5d  r0=%.2e  final=%.2e  moved=%.1e  %s\n",
           method_name(method), backend_name(backend),
           status == CFD_SUCCESS ? "success" : cfd_get_error_string(status), stats.iterations,
           stats.initial_residual, stats.final_residual, moved, ok ? "ok" : "FAIL");

    cfd_free(x);
    cfd_free(tmp);
    poisson_solver_destroy(s);
    return !ok;
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

static const poisson_solver_method_t METHODS[] = {
    POISSON_METHOD_JACOBI,       POISSON_METHOD_GAUSS_SEIDEL, POISSON_METHOD_SOR,
    POISSON_METHOD_REDBLACK_SOR, POISSON_METHOD_CG,           POISSON_METHOD_BICGSTAB,
    POISSON_METHOD_GMRES,        POISSON_METHOD_MULTIGRID};

static void check_backend(poisson_solver_backend_t backend) {
    floor_problem_t p = floor_problem_create();
    int failures = 0;
    for (size_t m = 0; m < sizeof(METHODS) / sizeof(METHODS[0]); m++) {
        failures += check_restart_from_solution(&p, METHODS[m], backend);
    }
    floor_problem_destroy(&p);
    TEST_ASSERT_EQUAL_INT_MESSAGE(
        0, failures, "A solve started from its converged solution did not return at once");
}

void test_restart_from_solution_scalar(void) {
    check_backend(POISSON_BACKEND_SCALAR);
}
void test_restart_from_solution_omp(void) {
    check_backend(POISSON_BACKEND_OMP);
}
void test_restart_from_solution_simd(void) {
    check_backend(POISSON_BACKEND_SIMD);
}
void test_restart_from_solution_gpu(void) {
    check_backend(POISSON_BACKEND_GPU);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_restart_from_solution_scalar);
    RUN_TEST(test_restart_from_solution_omp);
    RUN_TEST(test_restart_from_solution_simd);
    RUN_TEST(test_restart_from_solution_gpu);
    return UNITY_END();
}
