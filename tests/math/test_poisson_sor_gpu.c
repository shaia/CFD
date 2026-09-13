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

static void hold_walls_at_zero(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

/* The GPU SOR and Red-Black SOR solvers apply the zero-gradient walls on the
 * device and never call apply_bc, so init rejects a caller-supplied one rather
 * than solve the zero-gradient problem with the omega for the caller's walls. */
void test_sor_gpu_rejects_custom_apply_bc(void) {
    printf("\n    GPU SOR and Red-Black SOR: custom apply_bc rejected at init...\n");
    poisson_solver_method_t methods[] = { POISSON_METHOD_SOR, POISSON_METHOD_REDBLACK_SOR };
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        poisson_solver_t* solver = poisson_solver_create(methods[m], POISSON_BACKEND_GPU);
        if (!solver) {
            printf("      SKIPPED (GPU backend unavailable)\n");
            return;
        }
        cfd_status_t plain = poisson_solver_init(solver, 17, 17, 1, 1.0 / 16.0, 1.0 / 16.0, 0.0, NULL);
        poisson_solver_destroy(solver);
        if (plain == CFD_ERROR_UNSUPPORTED) {
            printf("      SKIPPED (no GPU device at runtime)\n");
            return;
        }
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, plain);

        solver = poisson_solver_create(methods[m], POISSON_BACKEND_GPU);
        TEST_ASSERT_NOT_NULL(solver);
        solver->apply_bc = hold_walls_at_zero;
        cfd_status_t custom = poisson_solver_init(solver, 17, 17, 1, 1.0 / 16.0, 1.0 / 16.0, 0.0, NULL);
        poisson_solver_destroy(solver);
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED, custom);
    }
}

/* ---- 3D zero-gradient walls ----------------------------------------------- */

/* Run exactly `sweeps` sweeps from a zero start with the default walls and the
 * automatic omega; a zero tolerance keeps the solve from stopping early. Returns 0
 * if the backend is unavailable, 1 once the sweeps have run. */
static int run_sweeps_3d(poisson_solver_method_t method, poisson_solver_backend_t backend,
                         size_t nx, size_t ny, size_t nz, double h, int sweeps,
                         double* x, const double* rhs) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        return 0;
    }
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 0.0;
    params.absolute_tolerance = 0.0;
    params.max_iterations = sweeps;

    cfd_status_t st = poisson_solver_init(solver, nx, ny, nz, h, h, h, &params);
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    if (st == CFD_SUCCESS) {
        poisson_solver_solve(solver, x, NULL, rhs, &stats);
    }
    poisson_solver_destroy(solver);
    if (st == CFD_ERROR_UNSUPPORTED) {
        return 0;
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);
    TEST_ASSERT_EQUAL_INT(sweeps, stats.iterations);
    return 1;
}

/* Index of cell (i, j, k), mirrored in x and/or z */
static size_t mirrored_index(size_t i, size_t j, size_t k, size_t nx, size_t ny, size_t nz,
                             int mirror_x, int mirror_z) {
    size_t mi = mirror_x ? nx - 1 - i : i;
    size_t mk = mirror_z ? nz - 1 - k : k;
    return (mk * ny + j) * nx + mi;
}

/* Compare 20 GPU sweeps with 20 scalar sweeps on a 10x10xnz grid, cell for cell.
 * A wrong wall factor at a z-face changes the field in the first sweep, so this
 * checks the GPU z-wall relaxation against the scalar reference.
 *
 * Both runs must visit the cells in the same order. The 8x8 interior of a 10x10
 * plane is a single GPU SOR tile, which it sweeps like scalar SOR, but it sweeps
 * even k-planes before odd ones: with nz = 3 there is one plane, and with nz = 4
 * the scalar run on the z-mirrored problem takes the planes in the GPU's order.
 * GPU Red-Black SOR updates the even (i + j + k) cells first and the scalar solver
 * the odd ones; mirroring x on an even nx swaps the two colours.
 * Returns 0 if the GPU backend is unavailable. */
static int compare_gpu_sweeps_with_scalar(poisson_solver_method_t method, size_t nz,
                                          int mirror_x, int mirror_z) {
    const size_t nx = 10, ny = 10;
    const double h = 1.0 / 9.0;
    const int sweeps = 20;
    size_t n = nx * ny * nz;

    double* rhs = create_field(n);
    double* rhs_mirrored = create_field(n);
    double* x_gpu = create_field(n);
    double* x_scalar = create_field(n);
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(rhs_mirrored);
    TEST_ASSERT_NOT_NULL(x_gpu);
    TEST_ASSERT_NOT_NULL(x_scalar);

    /* Any interior field without mirror symmetry */
    for (size_t k = 1; k < nz - 1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                double v = sin(1.3 * (double)i + 0.7 * (double)j + 2.1 * (double)k);
                rhs[(k * ny + j) * nx + i] = v;
                rhs_mirrored[mirrored_index(i, j, k, nx, ny, nz, mirror_x, mirror_z)] = v;
            }
        }
    }

    int ran = run_sweeps_3d(method, POISSON_BACKEND_GPU, nx, ny, nz, h, sweeps, x_gpu, rhs);
    if (ran) {
        TEST_ASSERT_EQUAL_INT(1, run_sweeps_3d(method, POISSON_BACKEND_SCALAR, nx, ny, nz, h,
                                               sweeps, x_scalar, rhs_mirrored));
        double max_diff = 0.0;
        double max_abs = 0.0;
        for (size_t k = 1; k < nz - 1; k++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    double g = x_gpu[(k * ny + j) * nx + i];
                    double d = fabs(g - x_scalar[mirrored_index(i, j, k, nx, ny, nz, mirror_x, mirror_z)]);
                    if (d > max_diff || isnan(d)) max_diff = d;
                    if (fabs(g) > max_abs) max_abs = fabs(g);
                }
            }
        }
        char msg[128];
        snprintf(msg, sizeof(msg), "%s, nz = %zu: max |GPU - scalar| %.3e of max |x| %.3e",
                 method == POISSON_METHOD_SOR ? "SOR" : "Red-Black SOR", nz, max_diff, max_abs);
        printf("      %s\n", msg);
        TEST_ASSERT_TRUE_MESSAGE(max_abs > 1e-6, "the sweeps did not move the field");
        TEST_ASSERT_TRUE_MESSAGE(max_diff <= 1e-10 * max_abs, msg);
    }

    cfd_free(rhs);
    cfd_free(rhs_mirrored);
    cfd_free(x_gpu);
    cfd_free(x_scalar);
    return ran;
}

/* The GPU SOR and Red-Black SOR sweeps relax beside the z-walls as the scalar ones
 * do, on a lone interior plane that touches both z-walls and on planes that touch
 * one each. */
void test_sor_gpu_3d_walls_match_scalar(void) {
    printf("\n    GPU SOR and Red-Black SOR vs scalar, 3D zero-gradient walls...\n");
    if (!compare_gpu_sweeps_with_scalar(POISSON_METHOD_SOR, 3, 0, 0)) {
        printf("      SKIPPED (GPU backend unavailable)\n");
        return;
    }
    compare_gpu_sweeps_with_scalar(POISSON_METHOD_SOR, 4, 0, 1);
    compare_gpu_sweeps_with_scalar(POISSON_METHOD_REDBLACK_SOR, 3, 1, 0);
    compare_gpu_sweeps_with_scalar(POISSON_METHOD_REDBLACK_SOR, 6, 1, 0);
}

int main(void) {
    UNITY_BEGIN();
    printf("\n========================================\n");
    printf("GPU PLAIN SOR (POISSON) SOLVER TESTS\n");
    printf("========================================\n");
    RUN_TEST(test_sor_gpu_converges);
    RUN_TEST(test_sor_gpu_matches_cpu);
    RUN_TEST(test_sor_gpu_rejects_custom_apply_bc);
    RUN_TEST(test_sor_gpu_3d_walls_match_scalar);
    return UNITY_END();
}
