/**
 * @file test_omp_consistency.c
 * @brief Consistency tests: OMP vs scalar linear solvers
 *
 * Verifies that OpenMP-parallelized Poisson solvers produce numerically
 * consistent results with their scalar reference implementations:
 * - CG OMP vs CG Scalar: L2 difference < 1e-9 (iterative math is identical)
 * - Red-Black SOR OMP vs Scalar: L2 difference < 1e-6 (parallel ordering
 *   causes minor rounding differences)
 *
 * Both tests skip gracefully when OpenMP is not enabled or unavailable.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>

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
#define TOLERANCE 1e-6

void setUp(void) {}
void tearDown(void) {}

/**
 * Initialize sinusoidal RHS compatible with Neumann BCs.
 * f(x,y) = cos(2πx)cos(2πy) with discrete interior mean subtracted.
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy) {
    /* First pass: initialize sinusoidal values */
    for (size_t j = 0; j < ny; j++) {
        double y = YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = XMIN + i * dx;
            rhs[IDX_2D(i, j, nx)] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Second pass: compute interior mean and subtract */
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

    /* Zero boundary values */
    for (size_t i = 0; i < nx; i++) {
        rhs[i] = 0.0;
        rhs[(ny - 1) * nx + i] = 0.0;
    }
    for (size_t j = 0; j < ny; j++) {
        rhs[j * nx] = 0.0;
        rhs[j * nx + (nx - 1)] = 0.0;
    }
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
    init_sinusoidal_rhs(rhs, NX, NY, dx, dy);

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

    /* Create OMP solver */
    poisson_solver_t* solver_omp = poisson_solver_create(
        POISSON_METHOD_CG, POISSON_BACKEND_OMP);

    if (!solver_omp) {
        cfd_free(x_scalar);
        cfd_free(x_omp);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("OMP CG solver creation failed; backend may be unavailable");
        return;
    }

    /* Initialize and solve with OMP solver */
    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, x_temp, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    /* Compute L2 difference between solutions (interior only) */
    double l2_diff = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            size_t idx = IDX_2D(i, j, NX);
            double diff = x_scalar[idx] - x_omp[idx];
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = sqrt(l2_diff / count);

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
    init_sinusoidal_rhs(rhs, NX, NY, dx, dy);

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

    /* Create OMP Red-Black SOR solver.
     * solver_create may return NULL if the OMP RB-SOR factory
     * is not registered.  Skip gracefully rather than failing. */
    poisson_solver_t* solver_omp = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_OMP);

    if (!solver_omp) {
        cfd_free(x_scalar);
        cfd_free(x_omp);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("OMP Red-Black SOR solver creation failed; backend may be unavailable");
        return;
    }

    /* Initialize and solve with OMP solver */
    status = poisson_solver_init(solver_omp, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_omp = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_omp, x_omp, x_temp, rhs, &stats_omp);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_omp.status);

    /* Compute L2 difference between solutions (interior only) */
    double l2_diff = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            size_t idx = IDX_2D(i, j, NX);
            double diff = x_scalar[idx] - x_omp[idx];
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = sqrt(l2_diff / count);

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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_cg_omp_vs_scalar);
    RUN_TEST(test_redblack_omp_vs_scalar);
    return UNITY_END();
}
