/**
 * Unit Tests for No-Slip Wall Boundary Conditions
 *
 * Tests the bc_apply_noslip* functions across all backends:
 * - Scalar (CPU baseline, single-threaded)
 * - SIMD + OpenMP (AVX2 on x86, NEON on ARM, multi-threaded + SIMD)
 * - OpenMP (multi-threaded, scalar inner loops)
 *
 * No-slip BCs enforce zero velocity (u=0, v=0) at all wall boundaries.
 */

#include "cfd/boundary/boundary_conditions.h"
#include "unity.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Test grid sizes */
#define TEST_NX_SMALL 4
#define TEST_NY_SMALL 4
#define TEST_NX_MEDIUM 16
#define TEST_NY_MEDIUM 16
#define TEST_NX_LARGE 64
#define TEST_NY_LARGE 64

/* Test tolerance for floating point comparison */
#define TOLERANCE 1e-14

void setUp(void) {
    /* Reset to auto backend before each test */
    bc_set_backend(BC_BACKEND_AUTO);
}

void tearDown(void) {
    /* Nothing to clean up */
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * Allocate and initialize a velocity field with non-zero values.
 * Interior and boundary values are set to a distinctive non-zero value
 * to verify that no-slip correctly sets boundaries to zero.
 */
static double* create_test_field(size_t nx, size_t ny) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;

    /* Fill with a non-zero pattern */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field[j * nx + i] = 999.0;
        }
    }
    return field;
}

/**
 * Verify that all boundary values are zero (no-slip condition).
 */
static int verify_noslip_bc(const double* field, size_t nx, size_t ny) {
    /* Check left boundary (column 0) */
    for (size_t j = 0; j < ny; j++) {
        if (fabs(field[j * nx]) > TOLERANCE) {
            return 0;
        }
    }

    /* Check right boundary (column nx-1) */
    for (size_t j = 0; j < ny; j++) {
        if (fabs(field[j * nx + (nx - 1)]) > TOLERANCE) {
            return 0;
        }
    }

    /* Check bottom boundary (row 0) */
    for (size_t i = 0; i < nx; i++) {
        if (fabs(field[i]) > TOLERANCE) {
            return 0;
        }
    }

    /* Check top boundary (row ny-1) */
    for (size_t i = 0; i < nx; i++) {
        if (fabs(field[(ny - 1) * nx + i]) > TOLERANCE) {
            return 0;
        }
    }

    return 1;
}

/**
 * Verify that interior points are unchanged (should still be 999.0).
 */
static int verify_interior_unchanged(const double* field, size_t nx, size_t ny) {
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            if (fabs(field[j * nx + i] - 999.0) > TOLERANCE) {
                return 0;
            }
        }
    }
    return 1;
}

/* ============================================================================
 * Scalar Backend Tests
 * ============================================================================ */

void test_noslip_scalar_basic(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_apply_noslip_cpu(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "u-velocity no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "v-velocity no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(u, nx, ny),
                              "u interior values were modified");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(v, nx, ny),
                              "v interior values were modified");

    free(u);
    free(v);
}

void test_noslip_scalar_large_grid(void) {
    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_apply_noslip_cpu(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "Large grid u no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "Large grid v no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(u, nx, ny),
                              "Large grid u interior values were modified");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(v, nx, ny),
                              "Large grid v interior values were modified");

    free(u);
    free(v);
}

/* ============================================================================
 * SIMD + OpenMP Backend Tests (AVX2 on x86, NEON on ARM)
 * ============================================================================ */

void test_noslip_simd_omp_basic(void) {
    if (!bc_backend_available(BC_BACKEND_SIMD_OMP)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_apply_noslip_simd_omp(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "SIMD u no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "SIMD v no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(u, nx, ny),
                              "Interior values were modified by SIMD");

    free(u);
    free(v);
}

void test_noslip_simd_omp_consistency_with_scalar(void) {
    if (!bc_backend_available(BC_BACKEND_SIMD_OMP)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u_scalar = create_test_field(nx, ny);
    double* v_scalar = create_test_field(nx, ny);
    double* u_simd_omp = create_test_field(nx, ny);
    double* v_simd_omp = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_scalar);
    TEST_ASSERT_NOT_NULL(v_scalar);
    TEST_ASSERT_NOT_NULL(u_simd_omp);
    TEST_ASSERT_NOT_NULL(v_simd_omp);

    bc_apply_noslip_cpu(u_scalar, v_scalar, nx, ny);
    bc_apply_noslip_simd_omp(u_simd_omp, v_simd_omp, nx, ny);

    /* Compare all boundary values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          u_scalar[j * nx + i],
                                          u_simd_omp[j * nx + i]);
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          v_scalar[j * nx + i],
                                          v_simd_omp[j * nx + i]);
            }
        }
    }

    free(u_scalar);
    free(v_scalar);
    free(u_simd_omp);
    free(v_simd_omp);
}

/* ============================================================================
 * OpenMP Backend Tests
 * ============================================================================ */

void test_noslip_omp_basic(void) {
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_apply_noslip_omp(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "OMP u no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "OMP v no-slip BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(u, nx, ny),
                              "Interior values were modified by OMP");

    free(u);
    free(v);
}

void test_noslip_omp_consistency_with_scalar(void) {
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;
    double* u_scalar = create_test_field(nx, ny);
    double* v_scalar = create_test_field(nx, ny);
    double* u_omp = create_test_field(nx, ny);
    double* v_omp = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_scalar);
    TEST_ASSERT_NOT_NULL(v_scalar);
    TEST_ASSERT_NOT_NULL(u_omp);
    TEST_ASSERT_NOT_NULL(v_omp);

    bc_apply_noslip_cpu(u_scalar, v_scalar, nx, ny);
    bc_apply_noslip_omp(u_omp, v_omp, nx, ny);

    /* Compare all boundary values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          u_scalar[j * nx + i],
                                          u_omp[j * nx + i]);
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          v_scalar[j * nx + i],
                                          v_omp[j * nx + i]);
            }
        }
    }

    free(u_scalar);
    free(v_scalar);
    free(u_omp);
    free(v_omp);
}

/* ============================================================================
 * Main Dispatch Function Tests
 * ============================================================================ */

void test_noslip_main_dispatch_auto(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_set_backend(BC_BACKEND_AUTO);
    bc_apply_noslip(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "Main dispatch (AUTO) u no-slip BC failed");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "Main dispatch (AUTO) v no-slip BC failed");

    free(u);
    free(v);
}

void test_noslip_convenience_macro(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Use the convenience macro */
    bc_apply_noslip_velocity(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "Convenience macro u no-slip BC failed");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "Convenience macro v no-slip BC failed");

    free(u);
    free(v);
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

void test_noslip_minimum_grid(void) {
    /* Minimum valid grid size is 3x3 */
    size_t nx = 3, ny = 3;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_apply_noslip(u, v, nx, ny);

    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(u, nx, ny),
                              "Minimum grid u no-slip BC failed");
    TEST_ASSERT_TRUE_MESSAGE(verify_noslip_bc(v, nx, ny),
                              "Minimum grid v no-slip BC failed");

    /* Only one interior point (1,1) should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, u[1 * nx + 1]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, v[1 * nx + 1]);

    free(u);
    free(v);
}

void test_noslip_null_fields(void) {
    /* Should return error and not crash with NULL fields */
    cfd_status_t status;

    status = bc_apply_noslip(NULL, NULL, 10, 10);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    double* u = create_test_field(10, 10);
    TEST_ASSERT_NOT_NULL(u);

    status = bc_apply_noslip(u, NULL, 10, 10);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    status = bc_apply_noslip(NULL, u, 10, 10);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(u);
}

void test_noslip_too_small_grid(void) {
    double* u = create_test_field(2, 2);
    double* v = create_test_field(2, 2);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Should return error for grid < 3x3 */
    cfd_status_t status = bc_apply_noslip(u, v, 2, 2);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    /* Fields should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, u[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, v[0]);

    free(u);
    free(v);
}

void test_noslip_returns_success(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    cfd_status_t status = bc_apply_noslip(u, v, nx, ny);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    status = bc_apply_noslip_cpu(u, v, nx, ny);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    free(u);
    free(v);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Scalar backend tests */
    RUN_TEST(test_noslip_scalar_basic);
    RUN_TEST(test_noslip_scalar_large_grid);

    /* SIMD + OpenMP backend tests (AVX2 on x86, NEON on ARM) */
    RUN_TEST(test_noslip_simd_omp_basic);
    RUN_TEST(test_noslip_simd_omp_consistency_with_scalar);

    /* OpenMP backend tests */
    RUN_TEST(test_noslip_omp_basic);
    RUN_TEST(test_noslip_omp_consistency_with_scalar);

    /* Main dispatch tests */
    RUN_TEST(test_noslip_main_dispatch_auto);
    RUN_TEST(test_noslip_convenience_macro);

    /* Edge case tests */
    RUN_TEST(test_noslip_minimum_grid);
    RUN_TEST(test_noslip_null_fields);
    RUN_TEST(test_noslip_too_small_grid);
    RUN_TEST(test_noslip_returns_success);

    return UNITY_END();
}
