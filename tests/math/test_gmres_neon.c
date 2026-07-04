/**
 * @file test_gmres_neon.c
 * @brief Consistency test: GMRES ARM NEON vs Scalar
 *
 * Verifies the NEON-optimized GMRES(m) solver produces numerically consistent
 * results with the scalar reference:
 * - L2 difference < 1e-7 between solutions
 * - Iteration counts within ±2 (SIMD rounding accumulation)
 *
 * Skips gracefully on non-ARM platforms. The dense Givens/Hessenberg/back-sub
 * work is byte-identical scalar in both backends.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NX 33
#define NY 33
#define XMIN 0.0
#define XMAX 1.0
#define YMIN 0.0
#define YMAX 1.0
#define TOLERANCE 1e-6

void setUp(void) {}
void tearDown(void) {}

static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = YMIN + j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = XMIN + i * dx;
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
    for (size_t j = 0; j < ny; j++) {
        rhs[j * nx] = 0.0;
        rhs[j * nx + (nx - 1)] = 0.0;
    }
}

void test_gmres_neon_scalar_consistency(void) {
    double dx = (XMAX - XMIN) / (NX - 1);
    double dy = (YMAX - YMIN) / (NY - 1);
    size_t n = NX * NY;

    double* x_scalar = (double*)cfd_calloc(n, sizeof(double));
    double* x_neon = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x_scalar);
    TEST_ASSERT_NOT_NULL(x_neon);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    init_sinusoidal_rhs(rhs, NX, NY, dx, dy);

    poisson_solver_t* solver_scalar = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver_scalar);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = TOLERANCE;
    params.max_iterations = 5000;

    cfd_status_t status = poisson_solver_init(solver_scalar, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_scalar = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_scalar, x_scalar, x_temp, rhs, &stats_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_scalar.status);

    /* GMRES SIMD backends require OpenMP - skip if not available */
#ifndef CFD_ENABLE_OPENMP
    cfd_free(x_scalar);
    cfd_free(x_neon);
    cfd_free(x_temp);
    cfd_free(rhs);
    poisson_solver_destroy(solver_scalar);
    TEST_IGNORE_MESSAGE("GMRES SIMD requires OpenMP (not enabled in this build)");
    return;
#endif

    bool simd_available = poisson_solver_backend_available(POISSON_BACKEND_SIMD);

    cfd_simd_arch_t arch = cfd_detect_simd_arch();
    if (simd_available && arch != CFD_SIMD_NEON) {
        cfd_free(x_scalar);
        cfd_free(x_neon);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        TEST_IGNORE_MESSAGE("NEON test skipped: platform uses different SIMD architecture");
        return;
    }

    poisson_solver_t* solver_neon = poisson_solver_create(
        POISSON_METHOD_GMRES, POISSON_BACKEND_SIMD);

    if (!solver_neon) {
        cfd_free(x_scalar);
        cfd_free(x_neon);
        cfd_free(x_temp);
        cfd_free(rhs);
        poisson_solver_destroy(solver_scalar);
        if (simd_available) {
            TEST_FAIL_MESSAGE("SIMD backend available but GMRES SIMD solver creation failed");
        } else {
            TEST_IGNORE_MESSAGE("SIMD backend not available on this platform");
        }
        return;
    }

    status = poisson_solver_init(solver_neon, NX, NY, 1, dx, dy, 0.0, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    poisson_solver_stats_t stats_neon = poisson_solver_stats_default();
    status = poisson_solver_solve(solver_neon, x_neon, x_temp, rhs, &stats_neon);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats_neon.status);

    double l2_diff = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            size_t idx = IDX_2D(i, j, NX);
            double diff = x_scalar[idx] - x_neon[idx];
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = sqrt(l2_diff / count);

    TEST_ASSERT_DOUBLE_WITHIN(1.0e-7, 0.0, l2_diff);

    int iter_diff = abs((int)stats_scalar.iterations - (int)stats_neon.iterations);
    TEST_ASSERT_LESS_OR_EQUAL(2, iter_diff);

    poisson_solver_destroy(solver_scalar);
    poisson_solver_destroy(solver_neon);
    cfd_free(x_scalar);
    cfd_free(x_neon);
    cfd_free(x_temp);
    cfd_free(rhs);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_gmres_neon_scalar_consistency);
    return UNITY_END();
}
