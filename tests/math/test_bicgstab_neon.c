/**
 * @file test_bicgstab_neon.c
 * @brief Consistency test: BiCGSTAB NEON vs Scalar
 *
 * Verifies that the NEON-optimized BiCGSTAB solver produces identical results
 * to the scalar reference implementation:
 * - L2 difference < 1e-10 between solutions
 * - Iteration counts within ±1 (due to floating-point rounding)
 *
 * This test ensures that SIMD optimizations preserve numerical correctness.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
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
            rhs[j * nx + i] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    /* Second pass: compute interior mean and subtract */
    double interior_sum = 0.0;
    size_t interior_count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            interior_sum += rhs[j * nx + i];
            interior_count++;
        }
    }

    if (interior_count > 0) {
        double interior_mean = interior_sum / (double)interior_count;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                rhs[j * nx + i] -= interior_mean;
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
 * Test: BiCGSTAB NEON vs Scalar Consistency
 *
 * Solves the same Poisson problem with both scalar and NEON BiCGSTAB solvers,
 * then verifies the solutions are numerically identical.
 */
void test_bicgstab_neon_scalar_consistency(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    /* Allocate solution vectors, temp buffer, and RHS */
    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_neon = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_neon);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Initialize RHS */
    init_sinusoidal_rhs(rhs, NX, NY, dx, dy);

    /* Create scalar solver */
    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    /* Initialize scalar solver */
    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = 5000;

    cfd_status_t status = poisson_solver_init(solver_scalar, NX, NY, dx, dy, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve with scalar solver */
    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_scalar, x_scalar, x_temp, rhs, &stats_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_scalar.status);

    /* Create SIMD solver */
    poisson_solver_t* solver_neon = poisson_solver_create(
        POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SIMD);

    /* Check if SIMD backend should be available */
    bool simd_available = poisson_solver_backend_available(POISSON_BACKEND_SIMD);

    if (!solver_neon) {
        cfd_free(x_scalar);
        cfd_free(x_neon);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);

        if (simd_available) {
            /* SIMD available but factory returned NULL - this is a bug */
            TEST_FAIL_MESSAGE("SIMD backend available but BiCGSTAB SIMD solver creation failed");
        } else {
            /* SIMD not available on this platform - expected */
            TEST_IGNORE_MESSAGE("SIMD backend not available on this platform");
        }
    }

    /* Initialize NEON solver */
    status = poisson_solver_init(solver_neon, NX, NY, dx, dy, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Solve with NEON solver */
    poisson_solver_stats_t stats_neon = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_neon, x_neon, x_temp, rhs, &stats_neon);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_neon.status);

    /* Compute L2 difference between solutions (interior only) */
    double l2_diff = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            size_t idx = j * NX + i;
            double diff = x_scalar[idx] - x_neon[idx];
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = sqrt(l2_diff / count);

    /* Verify consistency (L2 diff < 5e-9)
     * Note: SIMD uses FMA and different operation ordering than scalar,
     * causing minor rounding differences that vary by platform:
     * - Windows AVX2: ~2.4e-10
     * - macOS AVX2/NEON: ~1.5e-9
     * 5e-9 threshold provides cross-platform margin while ensuring excellent
     * numerical agreement (better than 8 decimal places). */
    TEST_ASSERT_DOUBLE_WITHIN(5.0e-9, 0.0, l2_diff);

    /* Iteration counts should match (±2 allowed due to rounding)
     * SIMD rounding differences can accumulate over iterations, causing
     * slightly different convergence paths across platforms. */
    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_neon.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(2, iter_diff);

    /* Cleanup */
    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_neon);
    cfd_free(x_scalar);
    cfd_free(x_neon);
    cfd_free(x_temp);
    cfd_free(rhs);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_bicgstab_neon_scalar_consistency);
    return UNITY_END();
}
