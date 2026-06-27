/**
 * @file test_poisson_jacobi_gpu.c
 * @brief GPU Jacobi Poisson solver: consistency vs CPU and a manufactured solution
 *
 * Verifies the standalone CUDA Jacobi Poisson backend (POISSON_METHOD_JACOBI +
 * POISSON_BACKEND_GPU) that plugs into the poisson_solver_t interface.
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
    /* analytical mean over the interior */
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
 *   1  on success (solved, *out_field populated)
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
    params.max_iterations = 30000;  /* Jacobi is slow; give it room (CG needs few) */

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

/* GPU Jacobi reaches the same discretization-floor accuracy as the CPU reference.
 *
 * The absolute L2 error vs the continuous analytical solution is dominated by the
 * first-order Neumann boundary treatment (documented ~O(h^1.5) BC limitation), so
 * an arbitrary absolute bound would be brittle. Instead we calibrate against the
 * trusted CPU Jacobi on the identical discrete system: the GPU must reach the same
 * error floor, with a loose absolute ceiling as a sanity backstop. */
void test_jacobi_gpu_manufactured_accuracy(void) {
    printf("\n    GPU Jacobi: manufactured-solution accuracy...\n");
    size_t nx = 33, ny = 33;
    double dx = (DOMAIN_MAX - DOMAIN_MIN) / (nx - 1);
    double dy = (DOMAIN_MAX - DOMAIN_MIN) / (ny - 1);

    double* p_gpu = create_field(nx * ny);
    double* p_cpu = create_field(nx * ny);
    double* rhs = create_field(nx * ny);
    TEST_ASSERT_NOT_NULL(p_gpu);
    TEST_ASSERT_NOT_NULL(p_cpu);
    TEST_ASSERT_NOT_NULL(rhs);
    init_rhs(rhs, nx, ny, dx, dy);

    poisson_solver_stats_t sg = poisson_solver_stats_default();
    int rc = solve_backend(POISSON_METHOD_JACOBI, POISSON_BACKEND_GPU,
                           p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        cfd_free(rhs);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc, "GPU Jacobi solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU Jacobi reference solve failed");

    double l2_gpu = l2_error_vs_analytical(p_gpu, nx, ny, dx, dy);
    double l2_cpu = l2_error_vs_analytical(p_cpu, nx, ny, dx, dy);
    printf("      iters=%d  final_res=%.3e  L2_gpu=%.3e  L2_cpu=%.3e\n",
           sg.iterations, sg.final_residual, l2_gpu, l2_cpu);

    /* Residual must have been driven down by many orders of magnitude. */
    TEST_ASSERT_TRUE_MESSAGE(sg.final_residual < 1e-3 * sg.initial_residual,
        "GPU Jacobi did not reduce the residual significantly");
    /* GPU reaches the CPU's discretization-floor accuracy. */
    TEST_ASSERT_TRUE_MESSAGE(fabs(l2_gpu - l2_cpu) < 1e-3,
        "GPU Jacobi accuracy differs from CPU reference");
    /* Loose absolute backstop (BC-limited floor on a 33x33 grid). */
    TEST_ASSERT_TRUE_MESSAGE(l2_gpu < 1e-1,
        "GPU Jacobi L2 error vs analytical unexpectedly large");

    cfd_free(p_gpu);
    cfd_free(p_cpu);
    cfd_free(rhs);
}

/* GPU Jacobi agrees with the reference CPU Jacobi on the identical problem. */
void test_jacobi_gpu_matches_cpu(void) {
    printf("\n    GPU Jacobi vs CPU Jacobi consistency...\n");
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
    int rc_gpu = solve_backend(POISSON_METHOD_JACOBI, POISSON_BACKEND_GPU,
                               p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc_gpu == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(rhs);
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_gpu, "GPU Jacobi solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU Jacobi solve failed");

    double diff = max_diff_demeaned(p_gpu, p_cpu, nx, ny);
    printf("      GPU iters=%d  CPU iters=%d  max|GPU-CPU|(demeaned)=%.3e\n",
           sg.iterations, sc.iterations, diff);
    /* Both are the same Jacobi iteration on the same SPD system; they converge to
     * the same field (up to the additive Neumann constant) within solver tol. */
    TEST_ASSERT_TRUE_MESSAGE(diff < 1e-4,
        "GPU and CPU Jacobi solutions diverge beyond 1e-4");

    cfd_free(rhs);
    cfd_free(p_gpu);
    cfd_free(p_cpu);
}

/* GPU CG agrees with the reference CPU CG, and converges in far fewer iterations
 * than Jacobi (grid-size-independent-ish vs O(N) for Jacobi). */
void test_cg_gpu_matches_cpu(void) {
    printf("\n    GPU CG vs CPU CG consistency...\n");
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
    int rc_gpu = solve_backend(POISSON_METHOD_CG, POISSON_BACKEND_GPU,
                               p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc_gpu == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(rhs);
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_gpu, "GPU CG solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU CG reference solve failed");

    double diff = max_diff_demeaned(p_gpu, p_cpu, nx, ny);
    double l2_gpu = l2_error_vs_analytical(p_gpu, nx, ny, dx, dy);
    double l2_cpu = l2_error_vs_analytical(p_cpu, nx, ny, dx, dy);
    printf("      GPU iters=%d  CPU iters=%d  max|GPU-CPU|(demeaned)=%.3e\n",
           sg.iterations, sc.iterations, diff);
    printf("      L2_gpu=%.3e  L2_cpu=%.3e\n", l2_gpu, l2_cpu);

    TEST_ASSERT_TRUE_MESSAGE(diff < 1e-6,
        "GPU and CPU CG solutions diverge beyond 1e-6");
    TEST_ASSERT_TRUE_MESSAGE(fabs(l2_gpu - l2_cpu) < 1e-3,
        "GPU CG accuracy differs from CPU reference");
    /* CG must reach this tolerance in far fewer iterations than Jacobi (~thousands). */
    TEST_ASSERT_TRUE_MESSAGE(sg.iterations < 500,
        "GPU CG took unexpectedly many iterations");

    cfd_free(rhs);
    cfd_free(p_gpu);
    cfd_free(p_cpu);
}

/* GPU BiCGSTAB agrees with the reference CPU BiCGSTAB on the identical problem.
 * For the symmetric Poisson operator BiCGSTAB is not the natural choice (CG is),
 * but the backend must still solve it and match its CPU twin to solver tolerance. */
void test_bicgstab_gpu_matches_cpu(void) {
    printf("\n    GPU BiCGSTAB vs CPU BiCGSTAB consistency...\n");
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
    int rc_gpu = solve_backend(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_GPU,
                               p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc_gpu == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(rhs);
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_gpu, "GPU BiCGSTAB solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU BiCGSTAB reference solve failed");

    double diff = max_diff_demeaned(p_gpu, p_cpu, nx, ny);
    double l2_gpu = l2_error_vs_analytical(p_gpu, nx, ny, dx, dy);
    double l2_cpu = l2_error_vs_analytical(p_cpu, nx, ny, dx, dy);
    printf("      GPU iters=%d  CPU iters=%d  max|GPU-CPU|(demeaned)=%.3e\n",
           sg.iterations, sc.iterations, diff);
    printf("      L2_gpu=%.3e  L2_cpu=%.3e\n", l2_gpu, l2_cpu);

    TEST_ASSERT_TRUE_MESSAGE(diff < 1e-6,
        "GPU and CPU BiCGSTAB solutions diverge beyond 1e-6");
    TEST_ASSERT_TRUE_MESSAGE(fabs(l2_gpu - l2_cpu) < 1e-3,
        "GPU BiCGSTAB accuracy differs from CPU reference");
    /* Krylov method: must reach tolerance in far fewer iterations than Jacobi. */
    TEST_ASSERT_TRUE_MESSAGE(sg.iterations < 500,
        "GPU BiCGSTAB took unexpectedly many iterations");

    cfd_free(rhs);
    cfd_free(p_gpu);
    cfd_free(p_cpu);
}

/* GPU Red-Black SOR agrees with the reference CPU Red-Black SOR on the identical
 * problem. Both resolve the same optimal omega from the grid dimensions, so they
 * iterate the same SOR scheme and converge to the same field (up to the additive
 * Neumann constant). RB-SOR needs more iterations than the Krylov methods but far
 * fewer than Jacobi. */
void test_redblack_sor_gpu_matches_cpu(void) {
    printf("\n    GPU Red-Black SOR vs CPU Red-Black SOR consistency...\n");
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
    int rc_gpu = solve_backend(POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_GPU,
                               p_gpu, rhs, nx, ny, dx, dy, &sg);
    if (rc_gpu == 0) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        cfd_free(rhs);
        cfd_free(p_gpu);
        cfd_free(p_cpu);
        return;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_gpu, "GPU Red-Black SOR solve failed");

    poisson_solver_stats_t sc = poisson_solver_stats_default();
    int rc_cpu = solve_backend(POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
                               p_cpu, rhs, nx, ny, dx, dy, &sc);
    TEST_ASSERT_EQUAL_INT_MESSAGE(1, rc_cpu, "CPU Red-Black SOR reference solve failed");

    double diff = max_diff_demeaned(p_gpu, p_cpu, nx, ny);
    double l2_gpu = l2_error_vs_analytical(p_gpu, nx, ny, dx, dy);
    double l2_cpu = l2_error_vs_analytical(p_cpu, nx, ny, dx, dy);
    printf("      GPU iters=%d  CPU iters=%d  max|GPU-CPU|(demeaned)=%.3e\n",
           sg.iterations, sc.iterations, diff);
    printf("      L2_gpu=%.3e  L2_cpu=%.3e\n", l2_gpu, l2_cpu);

    TEST_ASSERT_TRUE_MESSAGE(diff < 1e-6,
        "GPU and CPU Red-Black SOR solutions diverge beyond 1e-6");
    TEST_ASSERT_TRUE_MESSAGE(fabs(l2_gpu - l2_cpu) < 1e-3,
        "GPU Red-Black SOR accuracy differs from CPU reference");

    cfd_free(rhs);
    cfd_free(p_gpu);
    cfd_free(p_cpu);
}

int main(void) {
    UNITY_BEGIN();
    printf("\n========================================\n");
    printf("GPU LINEAR (POISSON) SOLVER TESTS\n");
    printf("========================================\n");
    RUN_TEST(test_jacobi_gpu_manufactured_accuracy);
    RUN_TEST(test_jacobi_gpu_matches_cpu);
    RUN_TEST(test_cg_gpu_matches_cpu);
    RUN_TEST(test_bicgstab_gpu_matches_cpu);
    RUN_TEST(test_redblack_sor_gpu_matches_cpu);
    return UNITY_END();
}
