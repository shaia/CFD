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
 * - Jacobi OMP vs Scalar: L2 difference <= 1e-9 (element-wise update, no reduction)
 * - BiCGSTAB OMP vs Scalar: L2 difference <= 1e-6 (reduction order feeds the
 *   alpha/beta/omega coefficients)
 * - Multigrid OMP factory metadata, then OMP vs Scalar: interior RMS and boundary
 *   max-abs differences <= 1e-10 and |iteration delta| <= 1 over a V/W/F x
 *   Red-Black GS/Jacobi x Neumann/Dirichlet matrix (2D/3D, non-square, custom
 *   sweeps, level cap + max-iteration exhaustion, inhomogeneous Dirichlet data,
 *   zero RHS), printed one line per configuration; an explicit apply_bc
 *   comparison; and the POISSON_SOLVER_MG_OMP convenience preset. The OMP
 *   multigrid path has no parallel reductions, so the expected difference is 0.
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
#include "../../lib/src/solvers/linear/multigrid_internal.h"
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

/* Multigrid OMP vs scalar bounds: every OMP multigrid primitive is element-wise
 * and all sums (interior mean, residual norm) stay serial, so the solutions are
 * expected to be bit-identical; the bounds only absorb floating-point
 * contraction differences between translation units on GCC/Clang. */
#define MG_L2_TOL   1.0e-10
#define MG_ITER_TOL 1

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

/** Max absolute difference over non-interior points (x/y edges, plus z faces in 3D) */
static double boundary_max_abs_diff(const double* a, const double* b,
                                    size_t nx, size_t ny, size_t nz) {
    size_t plane = nx * ny;
    double max_diff = 0.0;

    for (size_t k = 0; k < nz; k++) {
        int z_face = (nz > 1) && (k == 0 || k == nz - 1);
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                if (z_face || i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                    size_t idx = k * plane + IDX_2D(i, j, nx);
                    double diff = fabs(a[idx] - b[idx]);
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
            }
        }
    }
    return max_diff;
}

/** Grid spacing on the unit domain (dz = 0 for 2D) */
static void config_spacing(size_t nx, size_t ny, size_t nz,
                           double* dx, double* dy, double* dz) {
    *dx = (XMAX - XMIN) / (double)(nx - 1);
    *dy = (YMAX - YMIN) / (double)(ny - 1);
    *dz = (nz > 1) ? (ZMAX - ZMIN) / (double)(nz - 1) : 0.0;
}

/** Thread count the library's parallel regions use (0 when the test lacks OpenMP flags) */
static int reported_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();  /* follows OMP_NUM_THREADS, as the library does */
#else
    return 0;                      /* unknown: test built without OpenMP flags */
#endif
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
    RHS_ZERO,
    RHS_DIRICHLET_DATA               /* rhs = 4, boundary x = x^2 + y^2, interior 0 */
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

/**
 * Solve one GMRES configuration on `backend` into x (zeroed first), asserting
 * creation, init and the configuration's expected outcome.
 */
static poisson_solver_stats_t solve_gmres(poisson_solver_backend_t backend,
                                          const gmres_config_t* cfg,
                                          const double* rhs, double* x) {
    double dx, dy, dz;
    config_spacing(cfg->nx, cfg->ny, cfg->nz, &dx, &dy, &dz);
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

    int threads = reported_threads();
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
            config_spacing(cfg->nx, cfg->ny, cfg->nz, &dx, &dy, &dz);
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

/**
 * Test: Jacobi OMP vs Scalar Consistency
 *
 * Jacobi's interior update is a pure element-wise map (reads the previous
 * iterate, writes the next) with no reduction, and both backends drive
 * convergence with the same shared scalar residual check — so the OMP and
 * scalar solutions are effectively bit-identical. Jacobi converges slowly, so a
 * larger iteration budget is needed at 33x33 (~2900 iterations).
 */
void test_jacobi_omp_vs_scalar(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp   = (double*)cfd_calloc(n, sizeof(double));
    double* rhs      = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_omp);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, NX, NY, 1, dx, dy, 0.0);

    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = TOLERANCE;
    params.max_iterations  = 5000;  /* Jacobi needs ~2900 iters at 33x33 */

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
        POISSON_METHOD_JACOBI, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver_omp,
        "OMP backend available but OMP Jacobi solver creation returned NULL");

    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, x_temp, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    double l2_diff = interior_rms_diff(x_scalar, x_omp, NX, NY, 1);

    /* Jacobi's element-wise update carries no reduction, so OMP and scalar are
     * effectively bit-identical. */
    TEST_ASSERT_DOUBLE_WITHIN(1.0e-9, 0.0, l2_diff);

    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_omp.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(2, iter_diff);

    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_omp);
    cfd_free(x_scalar);
    cfd_free(x_omp);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/**
 * Test: BiCGSTAB OMP vs Scalar Consistency
 *
 * The BiCGSTAB solve loop and every per-element vector update are identical
 * across backends; only the dot-product reductions accumulate in a different
 * order across threads. Those reductions feed the scalar coefficients
 * (alpha/beta/omega), so BiCGSTAB is more path-sensitive than CG — a relaxed
 * tolerance (matching the Red-Black SOR consistency test) is used, verifying
 * both backends converge to the same solution within solver tolerance.
 */
void test_bicgstab_omp_vs_scalar(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
    double* rhs      = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_omp);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, NX, NY, 1, dx, dy, 0.0);

    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance       = TOLERANCE;
    params.max_iterations  = 1000;

    cfd_status_t status = poisson_solver_init(solver_scalar, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_scalar, x_scalar, NULL, rhs, &stats_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_scalar.status);

    /* OMP backend requires OpenMP - skip at runtime if unavailable */
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        cfd_free(x_scalar);
        cfd_free(x_omp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    /* The backend is available, so a NULL solver is a regression, not a skip */
    poisson_solver_t* solver_omp = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver_omp,
        "OMP backend available but OMP BiCGSTAB solver creation returned NULL");

    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, NULL, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    double l2_diff = interior_rms_diff(x_scalar, x_omp, NX, NY, 1);

    /* Both backends converge to the same solution within solver tolerance;
     * reduction-order differences in alpha/beta/omega make BiCGSTAB more
     * path-sensitive than CG, so the tolerance is relaxed (as for RB-SOR). */
    TEST_ASSERT_DOUBLE_WITHIN(1.0e-6, 0.0, l2_diff);

    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_omp.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(5, iter_diff);

    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_omp);
    cfd_free(x_scalar);
    cfd_free(x_omp);
    cfd_free(rhs);
}

/* ============================================================================
 * MULTIGRID OMP
 * ============================================================================ */

/**
 * Test: Multigrid OMP factory metadata
 *
 * The dispatcher must hand back the OMP multigrid solver, not the scalar one.
 */
void test_multigrid_omp_factory_metadata(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_OMP);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver,
        "OMP backend available but OMP multigrid solver creation returned NULL");

    TEST_ASSERT_EQUAL_STRING("multigrid_omp", solver->name);
    TEST_ASSERT_EQUAL(POISSON_METHOD_MULTIGRID, solver->method);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_OMP, solver->backend);

    poisson_solver_destroy(solver);
}

typedef struct {
    size_t nx, ny, nz;
    mg_cycle_type_t cycle;
    mg_smoother_type_t smoother;
    mg_bc_type_t bc;
    int pre_smooth;                  /* sweeps/levels: 0 = solver default */
    int post_smooth;
    int coarse_max_iter;
    int max_levels;
    rhs_kind_t rhs;
    int max_iterations;
    int expect_converged;            /* 0: both runs must exhaust max_iterations */
} mg_config_t;

/* Levels follow the coarsening rule (Neumann floor 5, Dirichlet floor 3). Most
 * grids stay small, since the executable is registered at three thread counts;
 * their planes fall below MG_OMP_MIN_POINTS and run serially. The 257x257
 * configurations put the finest level above it, so the thread team runs too, and
 * exhaust a few cycles to stay cheap. */
static const mg_config_t MG_CONFIGS[] = {
    /* nx  ny  nz  cycle       smoother                 bc               pre post crs lvl rhs                 max_it conv */
    /* Full 2D product (4 Neumann / 5 Dirichlet levels) */
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_W, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_W, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_F, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_F, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_F, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_F, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    /* Full 3D product: z-face BCs, 3D transfers */
    {   9,  9,  9, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_V, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_V, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_W, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_W, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_F, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_F, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_F, MG_SMOOTHER_JACOBI,      MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {   9,  9,  9, MG_CYCLE_F, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    /* Deeper 3D hierarchies, non-square and mixed dimensions */
    {  17, 17, 17, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  17, 17, 17, MG_CYCLE_F, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  65, 17,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  17, 33,  1, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    {  17,  9,  5, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,     100,   1 },
    /* Non-default sweeps; level cap + max-iteration exhaustion */
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 1,  3,  10,  0, RHS_SINUSOIDAL,     100,   1 },
    {  33, 33,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   5,  2, RHS_SINUSOIDAL,       3,   0 },
    /* Inhomogeneous Dirichlet boundary data; zero-RHS early return */
    {  17, 17,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_DIRICHLET_DATA, 100,   1 },
    {  17, 17,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_ZERO,           100,   1 },
    /* Finest planes above MG_OMP_MIN_POINTS: parallel kernels, both smoothers, 2D and 3D */
    { 257,257,  1, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_NEUMANN,   0,  0,   0,  0, RHS_SINUSOIDAL,       3,   0 },
    { 257,257,  1, MG_CYCLE_W, MG_SMOOTHER_JACOBI,      MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,       3,   0 },
    { 257,257,  5, MG_CYCLE_V, MG_SMOOTHER_REDBLACK_GS, MG_BC_DIRICHLET, 0,  0,   0,  0, RHS_SINUSOIDAL,       2,   0 },
};

/** Fill the RHS and the initial guess x0 for one multigrid configuration */
static void init_mg_problem(const mg_config_t* cfg, double* rhs, double* x0) {
    size_t n = cfg->nx * cfg->ny * cfg->nz;
    double dx, dy, dz;
    config_spacing(cfg->nx, cfg->ny, cfg->nz, &dx, &dy, &dz);
    memset(rhs, 0, n * sizeof(double));
    memset(x0, 0, n * sizeof(double));

    if (cfg->rhs == RHS_SINUSOIDAL) {
        init_sinusoidal_rhs(rhs, cfg->nx, cfg->ny, cfg->nz, dx, dy, dz);
    } else if (cfg->rhs == RHS_DIRICHLET_DATA) {
        /* u = x^2 + y^2 has Laplacian 4: boundary values carry the data and
         * the interior starts from zero */
        size_t plane = cfg->nx * cfg->ny;
        for (size_t k = 0; k < cfg->nz; k++) {
            int z_face = (cfg->nz > 1) && (k == 0 || k == cfg->nz - 1);
            for (size_t j = 0; j < cfg->ny; j++) {
                double y = YMIN + (double)j * dy;
                for (size_t i = 0; i < cfg->nx; i++) {
                    double x = XMIN + (double)i * dx;
                    size_t idx = k * plane + IDX_2D(i, j, cfg->nx);
                    rhs[idx] = 4.0;
                    if (z_face || i == 0 || i == cfg->nx - 1 || j == 0 || j == cfg->ny - 1) {
                        x0[idx] = x * x + y * y;
                    }
                }
            }
        }
    }
}

/**
 * Solve one multigrid configuration on `backend` into x (copied from x0 first),
 * asserting creation, init and the configuration's expected outcome.
 */
static poisson_solver_stats_t solve_mg(poisson_solver_backend_t backend,
                                       const mg_config_t* cfg,
                                       const double* rhs, const double* x0,
                                       double* x, cfd_status_t* solve_status_out) {
    double dx, dy, dz;
    config_spacing(cfg->nx, cfg->ny, cfg->nz, &dx, &dy, &dz);
    memcpy(x, x0, cfg->nx * cfg->ny * cfg->nz * sizeof(double));

    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_MULTIGRID, backend);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Multigrid solver creation failed on an available backend");

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance          = TOLERANCE;
    params.max_iterations     = cfg->max_iterations;
    params.mg_cycle           = cfg->cycle;
    params.mg_smoother        = cfg->smoother;
    params.mg_bc              = cfg->bc;
    params.mg_pre_smooth      = cfg->pre_smooth;
    params.mg_post_smooth     = cfg->post_smooth;
    params.mg_coarse_max_iter = cfg->coarse_max_iter;
    params.mg_max_levels      = cfg->max_levels;

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
        /* Multigrid iterates through poisson_solver_solve_common, which
         * reports the max_iterations cycles it ran */
        TEST_ASSERT_EQUAL_INT(cfg->max_iterations, (int)stats.iterations);
    }
    if (cfg->rhs == RHS_ZERO) {
        TEST_ASSERT_EQUAL_INT(0, (int)stats.iterations);
    }
    *solve_status_out = solve_status;
    return stats;
}

/**
 * Test: Multigrid OMP vs Scalar Consistency matrix
 *
 * Each configuration is solved with the scalar reference and the OMP backend.
 * The OMP primitives are element-wise with serial sums, so interior and boundary
 * values are expected to match exactly (l2 = bnd = 0, equal iteration counts);
 * the tolerances only absorb compiler contraction differences. Prints one line
 * per configuration with the measured margins.
 */
void test_multigrid_omp_vs_scalar(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    int threads = reported_threads();
    size_t num_configs = sizeof(MG_CONFIGS) / sizeof(MG_CONFIGS[0]);

    /* Planes below MG_OMP_MIN_POINTS run serially: without a configuration above
     * it, this matrix would never exercise the parallel kernels */
    size_t parallel_configs = 0;
    for (size_t c = 0; c < num_configs; c++) {
        if ((MG_CONFIGS[c].nx - 2) * (MG_CONFIGS[c].ny - 2) >= MG_OMP_MIN_POINTS) {
            parallel_configs++;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(parallel_configs > 0,
        "No multigrid configuration reaches MG_OMP_MIN_POINTS");

    for (size_t c = 0; c < num_configs; c++) {
        const mg_config_t* cfg = &MG_CONFIGS[c];
        size_t n = cfg->nx * cfg->ny * cfg->nz;

        double* rhs      = (double*)cfd_calloc(n, sizeof(double));
        double* x0       = (double*)cfd_calloc(n, sizeof(double));
        double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
        double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
        TEST_ASSERT_NOT_NULL(rhs);
        TEST_ASSERT_NOT_NULL(x0);
        TEST_ASSERT_NOT_NULL(x_scalar);
        TEST_ASSERT_NOT_NULL(x_omp);

        init_mg_problem(cfg, rhs, x0);

        cfd_status_t status_ref = CFD_ERROR;
        cfd_status_t status_omp = CFD_ERROR;
        poisson_solver_stats_t stats_ref =
            solve_mg(POISSON_BACKEND_SCALAR, cfg, rhs, x0, x_scalar, &status_ref);
        poisson_solver_stats_t stats_omp =
            solve_mg(POISSON_BACKEND_OMP, cfg, rhs, x0, x_omp, &status_omp);

        double l2_diff = interior_rms_diff(x_scalar, x_omp, cfg->nx, cfg->ny, cfg->nz);
        double bnd_diff = boundary_max_abs_diff(x_scalar, x_omp, cfg->nx, cfg->ny, cfg->nz);
        int iter_diff = abs((int)stats_ref.iterations - (int)stats_omp.iterations);

        printf("%zux%zux%zu cycle=%s smoother=%s bc=%s pre=%d post=%d coarse=%d levels=%d "
               "rhs=%s threads=%d bnd=%.3e l2=%.3e iters_ref=%d iters_target=%d\n",
               cfg->nx, cfg->ny, cfg->nz,
               (cfg->cycle == MG_CYCLE_W) ? "W" : (cfg->cycle == MG_CYCLE_F) ? "F" : "V",
               (cfg->smoother == MG_SMOOTHER_JACOBI) ? "jacobi" : "rbgs",
               (cfg->bc == MG_BC_DIRICHLET) ? "dirichlet" : "neumann",
               cfg->pre_smooth, cfg->post_smooth, cfg->coarse_max_iter, cfg->max_levels,
               (cfg->rhs == RHS_ZERO) ? "zero" : (cfg->rhs == RHS_DIRICHLET_DATA) ? "data" : "sin",
               threads, bnd_diff, l2_diff,
               (int)stats_ref.iterations, (int)stats_omp.iterations);

        TEST_ASSERT_EQUAL(status_ref, status_omp);
        TEST_ASSERT_EQUAL(stats_ref.status, stats_omp.status);
        TEST_ASSERT_LESS_OR_EQUAL(MG_ITER_TOL, iter_diff);
        TEST_ASSERT_DOUBLE_WITHIN(MG_L2_TOL, 0.0, l2_diff);
        TEST_ASSERT_DOUBLE_WITHIN(MG_L2_TOL, 0.0, bnd_diff);

        cfd_free(rhs);
        cfd_free(x0);
        cfd_free(x_scalar);
        cfd_free(x_omp);
    }
}

/**
 * Test: Multigrid OMP apply_bc matches scalar
 *
 * Explicit coverage of the boundary primitive (bc_apply_scalar_omp per plane plus
 * the shared z-face copies): a field with distinct values everywhere is passed
 * through poisson_solver_apply_bc on both backends and compared entry by entry.
 * Dirichlet mode must leave the field untouched.
 */
void test_multigrid_omp_apply_bc_matches_scalar(void) {
    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    static const size_t dims[][3] = { { 17, 17, 1 }, { 9, 9, 9 } };
    static const mg_bc_type_t modes[] = { MG_BC_NEUMANN, MG_BC_DIRICHLET };
    int threads = reported_threads();

    for (size_t d = 0; d < sizeof(dims) / sizeof(dims[0]); d++) {
        for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
            size_t nx = dims[d][0];
            size_t ny = dims[d][1];
            size_t nz = dims[d][2];
            size_t n = nx * ny * nz;
            double dx, dy, dz;
            config_spacing(nx, ny, nz, &dx, &dy, &dz);

            double* field    = (double*)cfd_calloc(n, sizeof(double));
            double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
            double* x_omp    = (double*)cfd_calloc(n, sizeof(double));
            TEST_ASSERT_NOT_NULL(field);
            TEST_ASSERT_NOT_NULL(x_scalar);
            TEST_ASSERT_NOT_NULL(x_omp);

            for (size_t idx = 0; idx < n; idx++) {
                field[idx] = sin(0.37 * (double)idx) + 0.001 * (double)idx;
            }
            memcpy(x_scalar, field, n * sizeof(double));
            memcpy(x_omp, field, n * sizeof(double));

            poisson_solver_params_t params = poisson_solver_params_default();
            params.mg_bc = modes[m];

            poisson_solver_t* solver_scalar = poisson_solver_create(
                POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
            TEST_ASSERT_NOT_NULL(solver_scalar);
            poisson_solver_t* solver_omp = poisson_solver_create(
                POISSON_METHOD_MULTIGRID, POISSON_BACKEND_OMP);
            TEST_ASSERT_NOT_NULL_MESSAGE(solver_omp,
                "OMP backend available but OMP multigrid solver creation returned NULL");
            TEST_ASSERT_EQUAL(CFD_SUCCESS,
                poisson_solver_init(solver_scalar, nx, ny, nz, dx, dy, dz, &params));
            TEST_ASSERT_EQUAL(CFD_SUCCESS,
                poisson_solver_init(solver_omp, nx, ny, nz, dx, dy, dz, &params));

            poisson_solver_apply_bc(solver_scalar, x_scalar);
            poisson_solver_apply_bc(solver_omp, x_omp);
            poisson_solver_destroy(solver_scalar);
            poisson_solver_destroy(solver_omp);

            double max_diff = 0.0;
            for (size_t idx = 0; idx < n; idx++) {
                double diff = fabs(x_scalar[idx] - x_omp[idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
            int scalar_changed = memcmp(x_scalar, field, n * sizeof(double)) != 0;

            printf("apply_bc %zux%zux%zu bc=%s threads=%d max_abs=%.3e\n",
                   nx, ny, nz, (modes[m] == MG_BC_DIRICHLET) ? "dirichlet" : "neumann",
                   threads, max_diff);

            TEST_ASSERT_DOUBLE_WITHIN(MG_L2_TOL, 0.0, max_diff);
            if (modes[m] == MG_BC_DIRICHLET) {
                TEST_ASSERT_EQUAL_MEMORY(field, x_scalar, (unsigned int)(n * sizeof(double)));
                TEST_ASSERT_EQUAL_MEMORY(field, x_omp, (unsigned int)(n * sizeof(double)));
            } else {
                TEST_ASSERT_TRUE_MESSAGE(scalar_changed,
                    "Neumann apply_bc left the field unchanged, so the comparison proves nothing");
            }

            cfd_free(field);
            cfd_free(x_scalar);
            cfd_free(x_omp);
        }
    }
}

/**
 * Test: POISSON_SOLVER_MG_OMP convenience preset
 *
 * Without OpenMP the preset has no solver and returns -1. With it, the preset
 * matches POISSON_SOLVER_MG_SCALAR, reuses its cached instance, and a rejected
 * non-2^k+1 grid does not poison the cache for the next valid call.
 */
void test_multigrid_omp_convenience_preset(void) {
    double dx, dy, dz;
    config_spacing(NX, NY, 1, &dx, &dy, &dz);
    size_t n = NX * NY;
    const size_t n_bad = 32 * 32;
    double dx_bad = (XMAX - XMIN) / 31.0;

    double* rhs      = (double*)cfd_calloc(n, sizeof(double));
    double* p_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* p_omp    = (double*)cfd_calloc(n, sizeof(double));
    double* rhs_bad  = (double*)cfd_calloc(n_bad, sizeof(double));
    double* p_bad    = (double*)cfd_calloc(n_bad, sizeof(double));
    TEST_ASSERT_NOT_NULL(rhs);
    TEST_ASSERT_NOT_NULL(p_scalar);
    TEST_ASSERT_NOT_NULL(p_omp);
    TEST_ASSERT_NOT_NULL(rhs_bad);
    TEST_ASSERT_NOT_NULL(p_bad);

    init_sinusoidal_rhs(rhs, NX, NY, 1, dx, dy, 0.0);

    if (!poisson_solver_backend_available(POISSON_BACKEND_OMP)) {
        int iters = poisson_solve_3d(p_omp, NULL, rhs, NX, NY, 1, dx, dy, 0.0,
                                     POISSON_SOLVER_MG_OMP);
        cfd_free(rhs);
        cfd_free(p_scalar);
        cfd_free(p_omp);
        cfd_free(rhs_bad);
        cfd_free(p_bad);
        TEST_ASSERT_EQUAL_INT(-1, iters);
        TEST_IGNORE_MESSAGE("OMP backend not available on this platform");
        return;
    }

    int iters_omp = poisson_solve_3d(p_omp, NULL, rhs, NX, NY, 1, dx, dy, 0.0,
                                     POISSON_SOLVER_MG_OMP);
    int iters_scalar = poisson_solve_3d(p_scalar, NULL, rhs, NX, NY, 1, dx, dy, 0.0,
                                        POISSON_SOLVER_MG_SCALAR);
    double l2_diff = interior_rms_diff(p_scalar, p_omp, NX, NY, 1);

    printf("preset %dx%dx1 threads=%d l2=%.3e iters_ref=%d iters_target=%d\n",
           NX, NY, reported_threads(), l2_diff, iters_scalar, iters_omp);

    TEST_ASSERT_GREATER_THAN_INT(0, iters_omp);
    TEST_ASSERT_GREATER_THAN_INT(0, iters_scalar);
    TEST_ASSERT_LESS_OR_EQUAL(MG_ITER_TOL, abs(iters_scalar - iters_omp));
    TEST_ASSERT_DOUBLE_WITHIN(MG_L2_TOL, 0.0, l2_diff);

    /* Same inputs through the cached OMP instance reproduce the first solve */
    memset(p_omp, 0, n * sizeof(double));
    int iters_cached = poisson_solve_3d(p_omp, NULL, rhs, NX, NY, 1, dx, dy, 0.0,
                                        POISSON_SOLVER_MG_OMP);
    TEST_ASSERT_EQUAL_INT(iters_omp, iters_cached);

    /* 32x32 is not 2^k+1: init fails and the call returns -1 ... */
    TEST_ASSERT_EQUAL_INT(-1, poisson_solve_3d(p_bad, NULL, rhs_bad, 32, 32, 1,
                                               dx_bad, dx_bad, 0.0, POISSON_SOLVER_MG_OMP));

    /* ... without leaving a broken solver in the cache */
    memset(p_omp, 0, n * sizeof(double));
    int iters_after = poisson_solve_3d(p_omp, NULL, rhs, NX, NY, 1, dx, dy, 0.0,
                                       POISSON_SOLVER_MG_OMP);
    TEST_ASSERT_GREATER_THAN_INT(0, iters_after);

    cfd_free(rhs);
    cfd_free(p_scalar);
    cfd_free(p_omp);
    cfd_free(rhs_bad);
    cfd_free(p_bad);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_cg_omp_vs_scalar);
    RUN_TEST(test_redblack_omp_vs_scalar);
    RUN_TEST(test_gmres_omp_factory_metadata);
    RUN_TEST(test_gmres_omp_vs_scalar);
    RUN_TEST(test_jacobi_omp_vs_scalar);
    RUN_TEST(test_bicgstab_omp_vs_scalar);
    RUN_TEST(test_multigrid_omp_factory_metadata);
    RUN_TEST(test_multigrid_omp_vs_scalar);
    RUN_TEST(test_multigrid_omp_apply_bc_matches_scalar);
    RUN_TEST(test_multigrid_omp_convenience_preset);
    return UNITY_END();
}
