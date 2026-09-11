/**
 * @file test_omp_consistency.c
 * @brief Consistency tests: OMP vs scalar linear solvers
 *
 * Verifies that OpenMP-parallelized Poisson solvers produce numerically
 * consistent results with their scalar reference implementations:
 * - CG OMP vs CG Scalar: L2 difference < 1e-9 (iterative math is identical)
 * - Red-Black SOR OMP vs Scalar: L2 difference < 1e-6 (parallel ordering
 *   causes minor rounding differences)
 * - GMRES OMP factory metadata (name, method, backend)
 * - GMRES OMP vs GMRES Scalar: L2 difference <= 1e-9 and |iteration delta| <= 2
 *   over a configuration matrix (2D/3D, non-square, restart default/5/1, Jacobi
 *   preconditioner, zero-RHS early return, max-iteration exhaustion), printed as
 *   one line per configuration with the measured margins
 *
 * The thread count comes from OMP_NUM_THREADS: CMake registers this executable at
 * 1, 2 and 4 threads. It is not set in-process because on MSVC this test binds
 * omp_set_num_threads() to a different OpenMP runtime than the library uses.
 *
 * Tests are ignored when the build has no OpenMP backend. Once the backend is
 * reported available, a NULL OMP solver is a failure rather than a skip: Unity
 * counts an ignored test as passed, so a regressed dispatch arm would go unnoticed.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Test problem parameters */
#define NX 33
#define NY 33
#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define ZMIN 0.0
#define ZMAX 1.0
#define TOLERANCE 1e-6

/* GMRES OMP vs scalar bounds: only the dot-product reduction differs, so the
 * solutions agree to rounding; +-2 iterations absorbs a convergence check that
 * flips by one inner step. */
#define GMRES_L2_TOL   1.0e-9
#define GMRES_ITER_TOL 2

void setUp(void) {}
void tearDown(void) {}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y,z) = cos(2πx)cos(2πy)cos(2πz) (the z factor is 1 in 2D) with the
 * discrete interior mean subtracted and boundary values zeroed.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny, size_t nz,
                                double dx, double dy, double dz) {
    size_t plane = nx * ny;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? (nz - 1) : 1;

    /* First pass: initialize sinusoidal values */
    for (size_t k = 0; k < nz; k++) {
        double z = ZMIN + k * dz;
        for (size_t j = 0; j < ny; j++) {
            double y = YMIN + j * dy;
            for (size_t i = 0; i < nx; i++) {
                double x = XMIN + i * dx;
                rhs[k * plane + IDX_2D(i, j, nx)] =
                    cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y) * cos(2.0 * M_PI * z);
            }
        }
    }

    /* Second pass: compute interior mean and subtract */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                interior_sum += rhs[k * plane + IDX_2D(i, j, nx)];
                interior_count++;
            }
        }
    }

    double interior_mean = interior_sum / (double)interior_count;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                rhs[k * plane + IDX_2D(i, j, nx)] -= interior_mean;
            }
        }
    }

    /* Zero boundary values (x and y faces, plus z faces in 3D) */
    for (size_t k = 0; k < nz; k++) {
        int z_face = (nz > 1) && (k == 0 || k == nz - 1);
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                if (z_face || i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    rhs[k * plane + IDX_2D(i, j, nx)] = 0.0;
                }
            }
        }
    }
}

/** RMS difference between two fields over interior points (2D or 3D) */
static double interior_rms_diff(const double* a, const double* b,
                                size_t nx, size_t ny, size_t nz) {
    size_t plane = nx * ny;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? (nz - 1) : 1;
    double sum = 0.0;
    size_t count = 0;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * plane + IDX_2D(i, j, nx);
                double diff = a[idx] - b[idx];
                sum += diff * diff;
                count++;
            }
        }
    }
    return sqrt(sum / (double)count);
}

/**
 * Test: CG OMP vs Scalar Consistency
 *
 * Solves the same Poisson problem with both scalar and OMP CG solvers,
 * then verifies the solutions are numerically identical.
 */
void test_cg_omp_vs_scalar(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    /* Allocate solution vectors, temp buffer, and RHS */
    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp   = (double*)cfd_calloc(n, sizeof(double));
    double* rhs      = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_omp);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Initialize RHS */
    init_sinusoidal_rhs(rhs, NX, NY, 1, dx, dy, 0.0);

    /* Create scalar solver */
    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    /* Initialize and solve with scalar solver */
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = TOLERANCE;
    params.max_iterations  = 1000;

    cfd_status_t status = poisson_solver_init(solver_scalar, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_scalar, x_scalar, x_temp, rhs, &stats_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_scalar.status);

    /* OMP backend requires OpenMP - skip at runtime if unavailable */
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        cfd_free(x_scalar);
        cfd_free(x_omp);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    /* The backend is available, so a NULL solver is a regression, not a skip */
    poisson_solver_t* solver_omp = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver_omp,
        "OMP backend available but OMP CG solver creation returned NULL");

    /* Initialize and solve with OMP solver */
    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, x_temp, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    /* L2 difference between solutions (interior only) */
    double l2_diff = interior_rms_diff(x_scalar, x_omp, NX, NY, 1);

    /* CG OMP and scalar use the same arithmetic operations in the same order
     * (only the dot-product reduction differs across threads), so agreement
     * should be excellent — better than 9 decimal places. */
    TEST_ASSERT_DOUBLE_WITHIN(1.0e-9, 0.0, l2_diff);

    /* Iteration counts should be essentially identical (±2 allowed for any
     * residual accumulation differences from parallel reduction). */
    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_omp.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(2, iter_diff);

    /* Cleanup */
    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_omp);
    cfd_free(x_scalar);
    cfd_free(x_omp);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/**
 * Test: Red-Black SOR OMP vs Scalar Consistency
 *
 * Solves the same Poisson problem with both scalar and OMP Red-Black SOR
 * solvers, then verifies the solutions agree within a relaxed tolerance.
 *
 * Parallel sweep ordering differs from scalar, producing larger rounding
 * differences than CG. Tolerances are relaxed accordingly.
 */
void test_redblack_omp_vs_scalar(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    /* Allocate solution vectors, temp buffer, and RHS */
    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp   = (double*)cfd_calloc(n, sizeof(double));
    double* rhs      = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_omp);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Initialize RHS */
    init_sinusoidal_rhs(rhs, NX, NY, 1, dx, dy, 0.0);

    /* Create scalar Red-Black SOR solver */
    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    /* Initialize and solve with scalar solver */
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = TOLERANCE;
    params.max_iterations  = 1000;

    cfd_status_t status = poisson_solver_init(solver_scalar, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_scalar, x_scalar, x_temp, rhs, &stats_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_scalar.status);

    /* OMP backend requires OpenMP - skip at runtime if unavailable */
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        cfd_free(x_scalar);
        cfd_free(x_omp);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    /* The backend is available, so a NULL solver is a regression, not a skip */
    poisson_solver_t* solver_omp = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver_omp,
        "OMP backend available but OMP Red-Black SOR solver creation returned NULL");

    /* Initialize and solve with OMP solver */
    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, x_temp, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    /* L2 difference between solutions (interior only) */
    double l2_diff = interior_rms_diff(x_scalar, x_omp, NX, NY, 1);

    /* Relaxed tolerance: OMP Red-Black SOR processes red/black points in
     * parallel with non-deterministic inter-thread ordering, causing larger
     * rounding differences than scalar sequential sweeps. */
    TEST_ASSERT_DOUBLE_WITHIN(1.0e-6, 0.0, l2_diff);

    /* SOR is more sensitive to parallel sweep ordering, so allow ±5 iterations. */
    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_omp.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(5, iter_diff);

    /* Cleanup */
    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_omp);
    cfd_free(x_scalar);
    cfd_free(x_omp);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * GMRES OMP
 * ============================================================================ */

/**
 * Test: GMRES OMP factory metadata
 *
 * The dispatcher must hand back the OMP GMRES solver, not another backend's.
 */
void test_gmres_omp_factory_metadata(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_GMRES, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver,
        "OMP backend available but OMP GMRES solver creation returned NULL");

    TEST_ASSERT_EQUAL_STRING("gmres_omp", solver->name);
    TEST_ASSERT_EQUAL(POISSON_METHOD_GMRES, solver->method);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_OMP, solver->backend);

    poisson_solver_destroy(solver);
}

typedef enum {
    RHS_SINUSOIDAL,
    RHS_ZERO
} rhs_kind_t;

typedef struct {
    size_t nx, ny, nz;
    int restart;                     /* 0 = solver default */
    poisson_precond_type_t precond;
    rhs_kind_t rhs;
    int max_iterations;
    int expect_converged;            /* 0: both runs must exhaust max_iterations */
} gmres_config_t;

/* Grids stay small: every primitive call opens one parallel region per k-plane,
 * and the executable is registered at three thread counts. */
static const gmres_config_t GMRES_CONFIGS[] = {
    /* nx  ny  nz  restart  precond                 rhs             max_it  conv */
    {  33, 33,  1, 0,       POISSON_PRECOND_NONE,   RHS_SINUSOIDAL, 1000,   1 }, /* baseline */
    {  17, 17,  1, 0,       POISSON_PRECOND_JACOBI, RHS_SINUSOIDAL, 1000,   1 }, /* Jacobi apply + update */
    {  17, 17,  1, 5,       POISSON_PRECOND_NONE,   RHS_SINUSOIDAL, 2000,   1 }, /* many restart cycles */
    {  33, 17,  1, 0,       POISSON_PRECOND_NONE,   RHS_SINUSOIDAL, 1000,   1 }, /* nx != ny row bounds */
    {   9,  9,  9, 0,       POISSON_PRECOND_NONE,   RHS_SINUSOIDAL, 1000,   1 }, /* 3D planes + z faces */
    {   9,  9,  9, 0,       POISSON_PRECOND_JACOBI, RHS_SINUSOIDAL, 1000,   1 }, /* 3D + Jacobi */
    {  17, 17,  1, 0,       POISSON_PRECOND_NONE,   RHS_ZERO,       1000,   1 }, /* zero-RHS early return */
    {  17, 17,  1, 1,       POISSON_PRECOND_NONE,   RHS_SINUSOIDAL,   20,   0 }, /* GMRES(1) hits max_iter */
};

static void gmres_config_spacing(const gmres_config_t* cfg,
                                 double* dx, double* dy, double* dz) {
    *dx = (XMAX - XMIN) / (double)(cfg->nx - 1);
    *dy = (YMAX - YMIN) / (double)(cfg->ny - 1);
    *dz = (cfg->nz > 1) ? (ZMAX - ZMIN) / (double)(cfg->nz - 1) : 0.0;
}

/**
 * Solve one GMRES configuration on `backend` into x (zeroed first), asserting
 * creation, init and the configuration's expected outcome.
 */
static poisson_solver_stats_t solve_gmres(poisson_solver_backend_t backend,
                                          const gmres_config_t* cfg,
                                          const double* rhs, double* x) {
    double dx, dy, dz;
    gmres_config_spacing(cfg, &dx, &dy, &dz);
    memset(x, 0, cfg->nx * cfg->ny * cfg->nz * sizeof(double));

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_GMRES, backend);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "GMRES solver creation failed on an available backend");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance      = TOLERANCE;
    params.max_iterations = cfg->max_iterations;
    params.restart        = cfg->restart;
    params.preconditioner = cfg->precond;

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t solve_status = CFD_ERROR;
    cfd_status_t init_status = poisson_solver_init(solver, cfg->nx, cfg->ny, cfg->nz,
                                                   dx, dy, dz, &params);
    if (init_status == CFD_SUCCESS) {
        solve_status = poisson_solver_solve(solver, x, NULL, rhs, &stats);
    }
    poisson_solver_destroy(solver);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);
    if (cfg->expect_converged) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_status);
        TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    } else {
        TEST_ASSERT_EQUAL(CFD_ERROR_MAX_ITER, solve_status);
        TEST_ASSERT_EQUAL(POISSON_MAX_ITER, stats.status);
        TEST_ASSERT_EQUAL_INT(cfg->max_iterations, (int)stats.iterations);
    }
    if (cfg->rhs == RHS_ZERO) {
        TEST_ASSERT_EQUAL_INT(0, (int)stats.iterations);
    }
    return stats;
}

/**
 * Test: GMRES OMP vs Scalar Consistency matrix
 *
 * Each configuration is solved with the scalar reference and the OMP backend. The
 * dense Givens/Hessenberg work is shared serial code, so only the parallel-reduction
 * dot products introduce rounding differences: the solutions are consistent, not
 * bit-for-bit identical (except 2D on one thread). Prints one line per configuration
 * with the measured margins.
 */
void test_gmres_omp_vs_scalar(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

#ifdef _OPENMP
    int threads = omp_get_max_threads();  /* follows OMP_NUM_THREADS, as the library does */
#else
    int threads = 0;                      /* unknown: test built without OpenMP flags */
#endif
    size_t num_configs = sizeof(GMRES_CONFIGS) / sizeof(GMRES_CONFIGS[0]);

    for (size_t c = 0; c < num_configs; c++) {
        const gmres_config_t* cfg = &GMRES_CONFIGS[c];
        size_t n = cfg->nx * cfg->ny * cfg->nz;

        double* rhs      = (double*)cfd_calloc(n, sizeof(double));
        double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
        double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
        TEST_ASSERT_NOT_NULL(rhs);
        TEST_ASSERT_NOT_NULL(x_scalar);
        TEST_ASSERT_NOT_NULL(x_omp);

        if (cfg->rhs == RHS_SINUSOIDAL) {
            double dx, dy, dz;
            gmres_config_spacing(cfg, &dx, &dy, &dz);
            init_sinusoidal_rhs(rhs, cfg->nx, cfg->ny, cfg->nz, dx, dy, dz);
        }

        poisson_solver_stats_t stats_ref = solve_gmres(POISSON_BACKEND_SCALAR, cfg, rhs, x_scalar);
        poisson_solver_stats_t stats_omp = solve_gmres(POISSON_BACKEND_OMP, cfg, rhs, x_omp);

        double l2_diff = interior_rms_diff(x_scalar, x_omp, cfg->nx, cfg->ny, cfg->nz);
        int iter_diff = abs((int)stats_ref.iterations - (int)stats_omp.iterations);

        char restart_label[16];
        if (cfg->restart > 0) {
            snprintf(restart_label, sizeof(restart_label), "%d", cfg->restart);
        } else {
            snprintf(restart_label, sizeof(restart_label), "default");
        }
        printf("%zux%zux%zu m=%s precond=%s rhs=%s threads=%d l2=%.3e "
               "iters_ref=%d iters_target=%d\n",
               cfg->nx, cfg->ny, cfg->nz, restart_label,
               (cfg->precond == POISSON_PRECOND_JACOBI) ? "jacobi" : "none",
               (cfg->rhs == RHS_ZERO) ? "zero" : "sin",
               threads, l2_diff,
               (int)stats_ref.iterations, (int)stats_omp.iterations);

        TEST_ASSERT_DOUBLE_WITHIN(GMRES_L2_TOL, 0.0, l2_diff);
        TEST_ASSERT_LESS_OR_EQUAL(GMRES_ITER_TOL, iter_diff);

        cfd_free(rhs);
        cfd_free(x_scalar);
        cfd_free(x_omp);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_cg_omp_vs_scalar);
    RUN_TEST(test_redblack_omp_vs_scalar);
    RUN_TEST(test_gmres_omp_factory_metadata);
    RUN_TEST(test_gmres_omp_vs_scalar);
    return UNITY_END();
}
