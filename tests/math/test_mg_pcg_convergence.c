/**
 * @file test_mg_pcg_convergence.c
 * @brief Multigrid-preconditioned Conjugate Gradient convergence tests
 *
 * These tests verify that CG with the multigrid V-cycle preconditioner
 * (POISSON_PRECOND_MULTIGRID):
 *   - Converges to the same solution as standard CG
 *   - Needs a grid-size-independent iteration count (the MG payoff),
 *     far fewer iterations than plain CG on larger grids
 *   - Works in 3D
 *   - Rejects non-2^k+1 grid dimensions at init
 *   - Is explicitly rejected (CFD_ERROR_UNSUPPORTED) by SIMD CG and by GMRES,
 *     instead of being silently ignored
 *   - Is reachable through the poisson_solve_3d() convenience presets
 *     POISSON_SOLVER_PCG_MG_SCALAR and POISSON_SOLVER_PCG_MG_OMP with the
 *     legacy return contract
 *
 * The iteration-count, dimension-rejection and preset checks run on both the
 * scalar and the OpenMP CG backends; OpenMP cases are skipped when the build
 * has no OpenMP backend.
 *
 * The preconditioner runs one symmetric V(2,2) weighted-Jacobi cycle in
 * Dirichlet mode per apply (poisson_solver_create_mg_precond), so M is SPD and
 * CG theory applies.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.0

#define TOLERANCE 1e-8
#define MAX_ITERATIONS 5000

/* Grid-independence bound for MG-PCG: a working MG preconditioner keeps the
 * outer CG count roughly constant across grid sizes. */
#define MG_PCG_MAX_ITERS 25

/* On larger grids plain CG needs O(n) iterations while MG-PCG stays flat;
 * require at least a 2x reduction from 65x65 up. */
#define MG_PCG_MIN_SPEEDUP 2.0

/* CG backends that implement the MG preconditioner */
static const poisson_solver_backend_t MG_PCG_BACKENDS[] = {
    POISSON_BACKEND_SCALAR,
    POISSON_BACKEND_OMP,
};
#define NUM_MG_PCG_BACKENDS (sizeof(MG_PCG_BACKENDS) / sizeof(MG_PCG_BACKENDS[0]))

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static double* create_field_3d(size_t nx, size_t ny, size_t nz) {
    return (double*)cfd_calloc(nx * ny * nz, sizeof(double));
}

static const char* backend_label(poisson_solver_backend_t backend) {
    return (backend == POISSON_BACKEND_OMP) ? "OMP" : "scalar";
}

/** Nonzero when the backend is built; prints a skip line otherwise */
static int backend_or_skip(poisson_solver_backend_t backend) {
    if (backend == POISSON_BACKEND_SCALAR || poisson_solver_backend_available(backend)) {
        return 1;
    }
    printf("      %s: backend not available (skipping)\n", backend_label(backend));
    return 0;
}

static void init_nontrivial_guess(double* p, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            p[IDX_2D(i, j, nx)] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
    }
}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2πx)cos(2πy) with discrete interior mean subtracted.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = DOMAIN_MIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = DOMAIN_MIN + i * dx;
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

    if (interior_count > 0) {
        double interior_mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                rhs[IDX_2D(i, j, nx)] -= interior_mean;
            }
        }
    }

    for (size_t i = 0; i < nx; i++) {
        rhs[i] = 0.0;
        rhs[(ny - 1) * nx + i] = 0.0;
    }
    for (size_t j = 1; j < ny - 1; j++) {
        rhs[j * nx] = 0.0;
        rhs[j * nx + (nx - 1)] = 0.0;
    }
}

/**
 * 3D analog: f = cos(2πx)cos(2πy)cos(2πz), interior mean subtracted,
 * boundary RHS zeroed.
 */
static void init_sinusoidal_rhs_3d(double* rhs, size_t nx, size_t ny, size_t nz,
                                   double dx, double dy, double dz) {
    size_t plane = nx * ny;
    for (size_t k = 0; k < nz; k++) {
        double z = DOMAIN_MIN + k * dz;
        for (size_t j = 0; j < ny; j++) {
            double y = DOMAIN_MIN + j * dy;
            for (size_t i = 0; i < nx; i++) {
                double x = DOMAIN_MIN + i * dx;
                rhs[k * plane + IDX_2D(i, j, nx)] =
                    cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y) * cos(2.0 * M_PI * z);
            }
        }
    }

    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t k = 1; k < nz - 1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                interior_sum += rhs[k * plane + IDX_2D(i, j, nx)];
                interior_count++;
            }
        }
    }
    double interior_mean = (interior_count > 0)
        ? interior_sum / (double)interior_count : 0.0;

    for (size_t k = 0; k < nz; k++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t idx = k * plane + IDX_2D(i, j, nx);
                int interior = (k > 0 && k < nz - 1 && j > 0 && j < ny - 1 &&
                                i > 0 && i < nx - 1);
                rhs[idx] = interior ? rhs[idx] - interior_mean : 0.0;
            }
        }
    }
}

/** L2 norm of the interior difference of two fields (3D-aware; nz=1 for 2D) */
static double compute_l2_difference_3d(const double* a, const double* b,
                                       size_t nx, size_t ny, size_t nz) {
    size_t plane = nx * ny;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? nz - 1 : 1;
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * plane + IDX_2D(i, j, nx);
                double diff = a[idx] - b[idx];
                sum_sq += diff * diff;
                count++;
            }
        }
    }
    return sqrt(sum_sq / (double)count);
}

/**
 * Subtract the interior mean (Neumann solutions are defined up to a constant;
 * CG and MG-PCG may settle on different constants).
 */
static void subtract_interior_mean_3d(double* f, size_t nx, size_t ny, size_t nz) {
    size_t plane = nx * ny;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? nz - 1 : 1;
    double sum = 0.0;
    size_t count = 0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                sum += f[k * plane + IDX_2D(i, j, nx)];
                count++;
            }
        }
    }
    double mean = sum / (double)count;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                f[k * plane + IDX_2D(i, j, nx)] -= mean;
            }
        }
    }
}

/**
 * Run one CG solve on the given backend with the given preconditioner.
 * Returns iteration count; asserts convergence.
 */
static int run_cg_solve(poisson_solver_backend_t backend,
                        double* p, double* p_temp, const double* rhs,
                        size_t nx, size_t ny, size_t nz,
                        double dx, double dy, double dz,
                        poisson_precond_type_t precond) {
    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG, backend);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = MAX_ITERATIONS;
    params.preconditioner = precond;

    cfd_status_t status = poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &params);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL_INT(POISSON_CONVERGED, stats.status);

    poisson_solver_destroy(solver);
    return stats.iterations;
}

/* ============================================================================
 * TEST: MG-PCG CONVERGES TO THE SAME SOLUTION AS CG
 * ============================================================================ */

void test_mg_pcg_converges_correctly(void) {
    printf("\n    Testing MG-PCG converges to correct solution...\n");

    size_t n = 33;
    double h = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1);

    double* p_cg = create_field_3d(n, n, 1);
    double* p_pcg = create_field_3d(n, n, 1);
    double* p_temp = create_field_3d(n, n, 1);
    double* rhs = create_field_3d(n, n, 1);
    TEST_ASSERT_NOT_NULL_MESSAGE(p_cg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_pcg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs(rhs, n, n, h, h);

    init_nontrivial_guess(p_cg, n, n);
    int cg_iters = run_cg_solve(POISSON_BACKEND_SCALAR, p_cg, p_temp, rhs,
                                n, n, 1, h, h, 0.0, POISSON_PRECOND_NONE);

    init_nontrivial_guess(p_pcg, n, n);
    int pcg_iters = run_cg_solve(POISSON_BACKEND_SCALAR, p_pcg, p_temp, rhs,
                                 n, n, 1, h, h, 0.0, POISSON_PRECOND_MULTIGRID);

    printf("      CG:     %d iterations\n", cg_iters);
    printf("      MG-PCG: %d iterations\n", pcg_iters);

    /* Neumann solutions are defined up to a constant */
    subtract_interior_mean_3d(p_cg, n, n, 1);
    subtract_interior_mean_3d(p_pcg, n, n, 1);

    double l2_diff = compute_l2_difference_3d(p_cg, p_pcg, n, n, 1);
    printf("      L2 difference between CG and MG-PCG solutions: %.2e\n", l2_diff);

    TEST_ASSERT_TRUE_MESSAGE(l2_diff < 1e-6,
        "MG-PCG and CG should converge to the same solution");

    cfd_free(p_cg);
    cfd_free(p_pcg);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST: MG-PCG ITERATION COUNT IS GRID-SIZE-INDEPENDENT
 * ============================================================================ */

void test_mg_pcg_iteration_reduction(void) {
    printf("\n    Testing MG-PCG grid-independent iteration count...\n");

    size_t sizes[] = {33, 65, 129};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t b = 0; b < NUM_MG_PCG_BACKENDS; b++) {
        poisson_solver_backend_t backend = MG_PCG_BACKENDS[b];
        if (!backend_or_skip(backend)) {
            continue;
        }

        for (int s = 0; s < num_sizes; s++) {
            size_t n = sizes[s];
            double h = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1);

            double* p = create_field_3d(n, n, 1);
            double* p_temp = create_field_3d(n, n, 1);
            double* rhs = create_field_3d(n, n, 1);
            TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed");
            TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
            TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

            init_sinusoidal_rhs(rhs, n, n, h, h);

            init_nontrivial_guess(p, n, n);
            int cg_iters = run_cg_solve(backend, p, p_temp, rhs, n, n, 1, h, h, 0.0,
                                        POISSON_PRECOND_NONE);

            init_nontrivial_guess(p, n, n);
            int pcg_iters = run_cg_solve(backend, p, p_temp, rhs, n, n, 1, h, h, 0.0,
                                         POISSON_PRECOND_MULTIGRID);

            double speedup = (double)cg_iters / (double)pcg_iters;
            printf("      %-6s %3zux%-3zu: CG=%4d iters, MG-PCG=%3d iters (%.1fx)\n",
                   backend_label(backend), n, n, cg_iters, pcg_iters, speedup);

            TEST_ASSERT_TRUE_MESSAGE(pcg_iters <= MG_PCG_MAX_ITERS,
                "MG-PCG iteration count should be grid-size-independent");

            if (n >= 65) {
                TEST_ASSERT_TRUE_MESSAGE(speedup >= MG_PCG_MIN_SPEEDUP,
                    "MG-PCG should need far fewer iterations than CG on larger grids");
            }

            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
        }
    }
}

/* ============================================================================
 * TEST: MG-PCG IN 3D
 * ============================================================================ */

void test_mg_pcg_3d(void) {
    printf("\n    Testing MG-PCG on a 3D grid...\n");

    size_t n = 17;
    double h = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1);

    double* p_cg = create_field_3d(n, n, n);
    double* p_pcg = create_field_3d(n, n, n);
    double* p_temp = create_field_3d(n, n, n);
    double* rhs = create_field_3d(n, n, n);
    TEST_ASSERT_NOT_NULL_MESSAGE(p_cg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_pcg, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs_3d(rhs, n, n, n, h, h, h);

    int cg_iters = run_cg_solve(POISSON_BACKEND_SCALAR, p_cg, p_temp, rhs,
                                n, n, n, h, h, h, POISSON_PRECOND_NONE);
    int pcg_iters = run_cg_solve(POISSON_BACKEND_SCALAR, p_pcg, p_temp, rhs,
                                 n, n, n, h, h, h, POISSON_PRECOND_MULTIGRID);

    printf("      CG:     %d iterations\n", cg_iters);
    printf("      MG-PCG: %d iterations\n", pcg_iters);

    subtract_interior_mean_3d(p_cg, n, n, n);
    subtract_interior_mean_3d(p_pcg, n, n, n);

    double l2_diff = compute_l2_difference_3d(p_cg, p_pcg, n, n, n);
    printf("      L2 difference: %.2e\n", l2_diff);

    TEST_ASSERT_TRUE_MESSAGE(l2_diff < 1e-6,
        "3D MG-PCG and CG should converge to the same solution");

    cfd_free(p_cg);
    cfd_free(p_pcg);
    cfd_free(p_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * TEST: NON-2^K+1 DIMENSIONS ARE REJECTED AT INIT
 * ============================================================================ */

void test_mg_pcg_rejects_invalid_dims(void) {
    printf("\n    Testing MG-PCG rejects non-2^k+1 grid dimensions...\n");

    size_t n = 30;  /* not 2^k+1 */
    double h = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1);

    for (size_t b = 0; b < NUM_MG_PCG_BACKENDS; b++) {
        poisson_solver_backend_t backend = MG_PCG_BACKENDS[b];
        if (!backend_or_skip(backend)) {
            continue;
        }

        poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG, backend);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create CG solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.preconditioner = POISSON_PRECOND_MULTIGRID;

        cfd_status_t status = poisson_solver_init(solver, n, n, 1, h, h, 0.0, &params);
        printf("      %s: init status %d\n", backend_label(backend), (int)status);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID, status,
            "CG init must propagate the inner MG rejection of non-2^k+1 dims");

        poisson_solver_destroy(solver);
    }
}

/* ============================================================================
 * TEST: OTHER BACKENDS REJECT THE MG PRECONDITIONER
 * ============================================================================ */

static void assert_backend_rejects_mg_precond(poisson_solver_method_t method,
                                              poisson_solver_backend_t backend,
                                              const char* label) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        printf("      %s: not available (skipping)\n", label);
        return;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.preconditioner = POISSON_PRECOND_MULTIGRID;

    cfd_status_t status = poisson_solver_init(solver, 33, 33, 1,
                                              1.0 / 32.0, 1.0 / 32.0, 0.0, &params);
    printf("      %s: init status %d\n", label, (int)status);
    TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
        "Backends without an MG preconditioner must reject it explicitly");

    poisson_solver_destroy(solver);
}

void test_mg_precond_unsupported_on_other_backends(void) {
    printf("\n    Testing MG preconditioner is rejected on unsupported backends...\n");

    assert_backend_rejects_mg_precond(POISSON_METHOD_CG, POISSON_BACKEND_SIMD,
                                      "CG SIMD");
    assert_backend_rejects_mg_precond(POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR,
                                      "GMRES scalar");
    assert_backend_rejects_mg_precond(POISSON_METHOD_GMRES, POISSON_BACKEND_SIMD,
                                      "GMRES SIMD");
    assert_backend_rejects_mg_precond(POISSON_METHOD_GMRES, POISSON_BACKEND_OMP,
                                      "GMRES OMP");
}

/* ============================================================================
 * TEST: CONVENIENCE PRESETS (POISSON_SOLVER_PCG_MG_SCALAR / _OMP)
 * ============================================================================ */

/**
 * Solve the sinusoidal problem on an n x n grid through a convenience preset,
 * from a zero initial guess. Returns poisson_solve_3d's result: the iteration
 * count, or -1 on failure.
 */
static int run_preset(poisson_solver_type preset, size_t n) {
    double h = (DOMAIN_MAX - DOMAIN_MIN) / (n - 1);
    double* p = create_field_3d(n, n, 1);
    double* p_temp = create_field_3d(n, n, 1);
    double* rhs = create_field_3d(n, n, 1);
    TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed");
    TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed");

    init_sinusoidal_rhs(rhs, n, n, h, h);

    int iters = poisson_solve_3d(p, p_temp, rhs, n, n, 1, h, h, 0.0, preset);

    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
    return iters;
}

void test_pcg_mg_preset(void) {
    printf("\n    Testing poisson_solve_3d PCG_MG presets...\n");

    /* Conforming grid: converges, returns iteration count */
    int scalar_iters = run_preset(POISSON_SOLVER_PCG_MG_SCALAR, 33);
    printf("      scalar 33x33 preset solve: %d iterations\n", scalar_iters);
    TEST_ASSERT_TRUE_MESSAGE(scalar_iters > 0,
        "Preset must converge on a 2^k+1 grid and report iterations");

    /* Non-conforming grid: fails loudly with -1 (legacy contract) */
    int scalar_bad = run_preset(POISSON_SOLVER_PCG_MG_SCALAR, 30);
    printf("      scalar 30x30 preset solve: returned %d\n", scalar_bad);
    TEST_ASSERT_EQUAL_INT_MESSAGE(-1, scalar_bad,
        "Preset must return -1 on non-2^k+1 grids");

    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_ASSERT_EQUAL_INT_MESSAGE(-1, run_preset(POISSON_SOLVER_PCG_MG_OMP, 33),
            "PCG_MG_OMP preset must fail without an OpenMP backend");
        printf("      OMP: backend not available (PCG_MG_OMP preset returned -1)\n");
        return;
    }

    int omp_iters = run_preset(POISSON_SOLVER_PCG_MG_OMP, 33);
    printf("      OMP    33x33 preset solve: %d iterations\n", omp_iters);
    TEST_ASSERT_TRUE_MESSAGE(omp_iters > 0 && omp_iters <= MG_PCG_MAX_ITERS,
        "OMP preset must converge within the MG-PCG iteration bound");
    TEST_ASSERT_TRUE_MESSAGE(abs(omp_iters - scalar_iters) <= 2,
        "OMP and scalar PCG_MG presets must need the same iterations (+-2)");

    /* The preset must carry the MG preconditioner into its cached OMP solver,
     * not fall back to plain CG: on 65x65 it needs far fewer iterations than
     * the CG_OMP preset on the same problem. */
    int omp_pcg_65 = run_preset(POISSON_SOLVER_PCG_MG_OMP, 65);
    int omp_cg_65 = run_preset(POISSON_SOLVER_CG_OMP, 65);
    printf("      OMP    65x65: PCG_MG preset %d iterations, CG preset %d iterations\n",
           omp_pcg_65, omp_cg_65);
    TEST_ASSERT_TRUE_MESSAGE(omp_pcg_65 > 0 &&
                             (double)omp_cg_65 >= MG_PCG_MIN_SPEEDUP * (double)omp_pcg_65,
        "PCG_MG_OMP preset must be multigrid-preconditioned, not plain CG");

    int omp_bad = run_preset(POISSON_SOLVER_PCG_MG_OMP, 30);
    printf("      OMP    30x30 preset solve: returned %d\n", omp_bad);
    TEST_ASSERT_EQUAL_INT_MESSAGE(-1, omp_bad,
        "OMP preset must return -1 on non-2^k+1 grids");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("MG-PCG Convergence Tests\n");
    printf("========================================\n");

    RUN_TEST(test_mg_pcg_converges_correctly);
    RUN_TEST(test_mg_pcg_iteration_reduction);
    RUN_TEST(test_mg_pcg_3d);
    RUN_TEST(test_mg_pcg_rejects_invalid_dims);
    RUN_TEST(test_mg_precond_unsupported_on_other_backends);
    RUN_TEST(test_pcg_mg_preset);

    printf("\n========================================\n");
    return UNITY_END();
}
