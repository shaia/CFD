/**
 * @file test_poisson_sor_gpu.c
 * @brief GPU plain SOR Poisson solver: convergence + consistency vs CPU
 *
 * Verifies the standalone CUDA plain-SOR backend (POISSON_METHOD_SOR +
 * POISSON_BACKEND_GPU) that plugs into the poisson_solver_t interface. The GPU
 * backend is a "Block SOR": each thread sweeps a small tile sequentially, with
 * red-black *tile* coloring (red pass then black pass per iteration), so a tile's
 * halo is read from the opposite-color pass rather than written concurrently. It
 * is therefore NOT a
 * bit-for-bit reproduction of the CPU lexicographic sweep — but it solves the
 * same discrete linear system, so once converged it lands on the same field
 * (up to the additive Neumann constant) as the CPU reference.
 *
 * Manufactured solution: p = cos(pi x) cos(pi y) on [0,1]^2.
 *   nabla^2 p = -2 pi^2 cos(pi x) cos(pi y)  -> RHS
 *   dp/dn = 0 on every face  -> exactly satisfies the solver's native Neumann BC.
 * The pure-Neumann solution is unique only up to an additive constant, so all
 * field comparisons are made after subtracting the interior mean.
 *
 * The test skips gracefully (no failure) when CUDA is not compiled in (create
 * returns NULL) or no GPU device is present at runtime (init returns
 * CFD_ERROR_UNSUPPORTED), per the project's optional-backend testing policy.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.0

void setUp(void) {}
void tearDown(void) {}

/* ---- helpers ------------------------------------------------------------- */

static double* create_field(size_t n) {
    return (double*)cfd_calloc(n, sizeof(double));
}

static double manufactured_p(double x, double y) {
    return cos(M_PI * x) * cos(M_PI * y);
}

static double manufactured_rhs(double x, double y) {
    return -2.0 * M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y);
}

static void init_rhs(double* rhs, size_t nx, size_t ny, double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_MIN + i * dx;
            rhs[IDX_2D(i, j, nx)] = manufactured_rhs(x, y);
        }
    }
}

/* Mean of the interior points (used to remove the additive Neumann constant). */
static double interior_mean(const double* f, size_t nx, size_t ny) {
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += f[IDX_2D(i, j, nx)];
            count++;
        }
    }
    return count ? sum / (double)count : 0.0;
}

/* Max abs interior difference between two fields after removing each one's mean. */
static double max_diff_demeaned(const double* a, const double* b, size_t nx, size_t ny) {
    double ma = interior_mean(a, nx, ny);
    double mb = interior_mean(b, nx, ny);
    double max_d = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double d = fabs((a[idx] - ma) - (b[idx] - mb));
            if (d > max_d) max_d = d;
        }
    }
    return max_d;
}

/* L2 error vs the analytical manufactured solution (after mean removal). */
static double l2_error_vs_analytical(const double* p, size_t nx, size_t ny,
                                     double dx, double dy) {
    double mp = interior_mean(p, nx, ny);
    double sum = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            sum += manufactured_p(DOMAIN_MIN + i * dx, DOMAIN_MIN + j * dy);
            count++;
        }
    }
    double ma = count ? sum / (double)count : 0.0;

    double sq = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 1; i < nx - 1; i++) {
            double x = DOMAIN_MIN + i * dx;
            size_t idx = IDX_2D(i, j, nx);
            double err = (p[idx] - mp) - (manufactured_p(x, y) - ma);
            sq += err * err;
        }
    }
    return count ? sqrt(sq / (double)count) : 0.0;
}

/* Solve the manufactured problem with the given method+backend. Returns:
 *   1  on success (solved, *p populated)
 *   0  to SKIP (backend unavailable)
 *  -1  on hard failure (caller should assert). */
static int solve_backend(poisson_solver_method_t method, poisson_solver_backend_t backend,
                         double* p, const double* rhs, size_t nx, size_t ny,
                         double dx, double dy, poisson_solver_stats_t* stats) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        return 0;  /* backend not compiled in */
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-7;
    params.absolute_tolerance = 1e-12;
    params.max_iterations = 30000;  /* generous: Block SOR halos slow convergence */

    cfd_status_t st = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    if (st == CFD_ERROR_UNSUPPORTED) {
        poisson_solver_destroy(solver);
        return 0;  /* no GPU device at runtime */
    }
    if (st != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return -1;
    }

    double* p_temp = create_field(nx * ny);
    if (!p_temp) {
        poisson_solver_destroy(solver);
        return -1;
    }

    cfd_status_t solve_st = poisson_solver_solve(solver, p, p_temp, rhs, stats);
    cfd_free(p_temp);
    poisson_solver_destroy(solver);

    /* CFD_SUCCESS or CFD_ERROR_MAX_ITER (residual still good) both acceptable. */
    return (solve_st == CFD_SUCCESS || solve_st == CFD_ERROR_MAX_ITER) ? 1 : -1;
}

/* ---- tests --------------------------------------------------------------- */

/* GPU Block SOR converges with the auto-resolved optimal omega and drives the
 * residual down by many orders of magnitude. This is the key stability check:
 * the 2D-tile scheme keeps the stale-halo fraction small enough that omega ~1.8
 * does not diverge (a one-thread-per-row scheme would). */
void test_sor_gpu_converges(void) {
    printf("\n    GPU plain SOR: convergence with auto omega...\n");
    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (nx - 1);
    double dy = (DOMAIN_MAX - DOMAIN_MIN) / (ny - 1);

    double* p_gpu = create_field(nx * ny);
    double* rhs = create_field(nx * ny);
    TEST_ASSERT_NOT_NULL(p_gpu);
    TEST_ASSERT_NOT_NULL(rhs);
    init_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_stats_t sg = poisson_solver_stats_default();
    int rc = solve_backend(POISSON_METHOD_SOR, POISSON_BACKEND_GPU,
                           p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(p_gpu);
        cfd_free(rhs);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc, "GPU SOR solve failed");
    printf("      iters=%d  init_res=%.3e  final_res=%.3e  status=%d\n",
           sg.iterations, sg.initial_residual, sg.final_residual, sg.status);

    TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, sg.status,
        "GPU Block SOR did not converge with auto omega (possible instability)");
    TEST_ASSERT_TRUE_MESSAGE(sg.final_residual < 1e-3 * sg.initial_residual,
        "GPU SOR did not reduce the residual significantly");

    cfd_free(p_gpu);
    cfd_free(rhs);
}

/* GPU Block SOR reaches the same discretization-floor accuracy as the CPU plain
 * SOR reference. Both solve the identical discrete system, so on convergence they
 * agree on the field (up to the additive Neumann constant). The block-staleness
 * difference is bounded by the converged tolerance, so a loose ceiling applies. */
void test_sor_gpu_matches_cpu(void) {
    printf("\n    GPU plain SOR vs CPU plain SOR consistency...\n");
    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (nx - 1);
    double dy = (DOMAIN_MAX - DOMAIN_MIN) / (ny - 1);

    double* rhs = create_field(nx * ny);
    double* p_gpu = create_field(nx * ny);
    double* p_cpu = create_field(nx * ny);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(p_gpu);
    TEST_ASSERT_NOT_NULL(p_cpu);
    init_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_stats_t sg = poisson_solver_stats_default();
    int rc_gpu = solve_backend(POISSON_METHOD_SOR, POISSON_BACKEND_GPU,
                               p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc_gpu == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(rhs);
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_gpu, "GPU SOR solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU SOR reference solve failed");

    double diff = max_diff_demeaned(p_gpu, p_cpu, nx, ny);
    double l2_gpu = l2_error_vs_analytical(p_gpu, nx, ny, dx, dy);
    double l2_cpu = l2_error_vs_analytical(p_cpu, nx, ny, dx, dy);
    printf("      GPU iters=%d  CPU iters=%d  max|GPU-CPU|(demeaned)=%.3e\n",
           sg.iterations, sc.iterations, diff);
    printf("      L2_gpu=%.3e  L2_cpu=%.3e\n", l2_gpu, l2_cpu);

    /* Same discrete system: converged fields agree up to the Neumann constant.
     * Loose bounds absorb the block-staleness path difference at solver tol. */
    TEST_ASSERT_TRUE_MESSAGE(diff < 1e-3,
        "GPU and CPU SOR solutions diverge beyond 1e-3");
    TEST_ASSERT_TRUE_MESSAGE(fabs(l2_gpu - l2_cpu) < 1e-3,
        "GPU SOR accuracy differs from CPU reference");
    TEST_ASSERT_TRUE_MESSAGE(l2_gpu < 1e-1,
        "GPU SOR L2 error vs analytical unexpectedly large");

    cfd_free(rhs);
    cfd_free(p_gpu);
    cfd_free(p_cpu);
}

int main(void) {
    UNITY_BEGIN();
    printf("\n========================================\n");
    printf("GPU PLAIN SOR (POISSON) SOLVER TESTS\n");
    printf("========================================\n");
    RUN_TEST(test_sor_gpu_converges);
    RUN_TEST(test_sor_gpu_matches_cpu);
    return UNITY_END();
}
