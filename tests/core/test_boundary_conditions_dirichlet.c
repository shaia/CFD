/**
 * Unit Tests for Dirichlet Boundary Conditions
 *
 * Tests the bc_apply_dirichlet_* functions across all backends:
 * - Scalar (CPU baseline)
 * - SIMD (AVX2/SSE2)
 * - OpenMP
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
 * Allocate and initialize a field with a known pattern.
 * Interior values are set to a distinctive value to verify they're unchanged.
 */
static double* create_test_field(size_t nx, size_t ny) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;

    /* Fill with a pattern: interior = 999.0 to verify BC doesn't touch it */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field[j * nx + i] = 999.0;
        }
    }
    return field;
}

/**
 * Verify that Dirichlet BCs were correctly applied to a field.
 *
 * Note: Corners are owned by bottom/top boundaries since they are applied last.
 * The implementation order is: left, right, bottom, top.
 * Therefore corners have bottom/top values, not left/right values.
 */
static int verify_dirichlet_bc(const double* field, size_t nx, size_t ny,
                                const bc_dirichlet_values_t* values) {
    /* Check left boundary (column 0), excluding corners */
    for (size_t j = 1; j < ny - 1; j++) {
        if (fabs(field[j * nx] - values->left) > TOLERANCE) {
            return 0;
        }
    }

    /* Check right boundary (column nx-1), excluding corners */
    for (size_t j = 1; j < ny - 1; j++) {
        if (fabs(field[j * nx + (nx - 1)] - values->right) > TOLERANCE) {
            return 0;
        }
    }

    /* Check bottom boundary (row 0), including corners */
    for (size_t i = 0; i < nx; i++) {
        if (fabs(field[i] - values->bottom) > TOLERANCE) {
            return 0;
        }
    }

    /* Check top boundary (row ny-1), including corners */
    for (size_t i = 0; i < nx; i++) {
        if (fabs(field[(ny - 1) * nx + i] - values->top) > TOLERANCE) {
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

void test_dirichlet_scalar_basic(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 1.0,
        .right = 2.0,
        .top = 3.0,
        .bottom = 4.0
    };

    bc_apply_dirichlet_scalar_cpu(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Dirichlet BC values not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(field, nx, ny),
                              "Interior values were modified");

    free(field);
}

void test_dirichlet_scalar_zero_values(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 0.0,
        .right = 0.0,
        .top = 0.0,
        .bottom = 0.0
    };

    bc_apply_dirichlet_scalar_cpu(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Zero Dirichlet BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(field, nx, ny),
                              "Interior values were modified");

    free(field);
}

void test_dirichlet_scalar_negative_values(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = -1.5,
        .right = -2.5,
        .top = -3.5,
        .bottom = -4.5
    };

    bc_apply_dirichlet_scalar_cpu(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Negative Dirichlet BC not correctly applied");

    free(field);
}

void test_dirichlet_scalar_large_grid(void) {
    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 10.0,
        .right = 20.0,
        .top = 30.0,
        .bottom = 40.0
    };

    bc_apply_dirichlet_scalar_cpu(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Large grid Dirichlet BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(field, nx, ny),
                              "Interior values were modified on large grid");

    free(field);
}

/* ============================================================================
 * SIMD Backend Tests
 * ============================================================================ */

void test_dirichlet_simd_basic(void) {
    if (!bc_backend_available(BC_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 1.0,
        .right = 2.0,
        .top = 3.0,
        .bottom = 4.0
    };

    bc_apply_dirichlet_scalar_simd(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "SIMD Dirichlet BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(field, nx, ny),
                              "Interior values were modified by SIMD");

    free(field);
}

void test_dirichlet_simd_consistency_with_scalar(void) {
    if (!bc_backend_available(BC_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field(nx, ny);
    double* field_simd = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_dirichlet_values_t values = {
        .left = 1.5,
        .right = 2.5,
        .top = 3.5,
        .bottom = 4.5
    };

    bc_apply_dirichlet_scalar_cpu(field_scalar, nx, ny, &values);
    bc_apply_dirichlet_scalar_simd(field_simd, nx, ny, &values);

    /* Compare all boundary values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          field_scalar[j * nx + i],
                                          field_simd[j * nx + i]);
            }
        }
    }

    free(field_scalar);
    free(field_simd);
}

/* ============================================================================
 * OpenMP Backend Tests
 * ============================================================================ */

void test_dirichlet_omp_basic(void) {
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 1.0,
        .right = 2.0,
        .top = 3.0,
        .bottom = 4.0
    };

    bc_apply_dirichlet_scalar_omp(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "OMP Dirichlet BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_interior_unchanged(field, nx, ny),
                              "Interior values were modified by OMP");

    free(field);
}

void test_dirichlet_omp_consistency_with_scalar(void) {
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;
    double* field_scalar = create_test_field(nx, ny);
    double* field_omp = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);

    bc_dirichlet_values_t values = {
        .left = 11.0,
        .right = 22.0,
        .top = 33.0,
        .bottom = 44.0
    };

    bc_apply_dirichlet_scalar_cpu(field_scalar, nx, ny, &values);
    bc_apply_dirichlet_scalar_omp(field_omp, nx, ny, &values);

    /* Compare all boundary values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                          field_scalar[j * nx + i],
                                          field_omp[j * nx + i]);
            }
        }
    }

    free(field_scalar);
    free(field_omp);
}

/* ============================================================================
 * Velocity Field Tests
 * ============================================================================ */

void test_dirichlet_velocity_basic(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_dirichlet_values_t u_values = {
        .left = 0.0,
        .right = 0.0,
        .top = 1.0,   /* Moving lid */
        .bottom = 0.0
    };

    bc_dirichlet_values_t v_values = {
        .left = 0.0,
        .right = 0.0,
        .top = 0.0,
        .bottom = 0.0
    };

    bc_apply_dirichlet_velocity(u, v, nx, ny, &u_values, &v_values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(u, nx, ny, &u_values),
                              "u-velocity Dirichlet BC not correctly applied");
    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(v, nx, ny, &v_values),
                              "v-velocity Dirichlet BC not correctly applied");

    free(u);
    free(v);
}

void test_dirichlet_velocity_lid_driven_cavity(void) {
    /* Standard lid-driven cavity setup: top wall moves at u=1, all others stationary */
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_dirichlet_values_t u_bc = {
        .left = 0.0,
        .right = 0.0,
        .top = 1.0,
        .bottom = 0.0
    };

    bc_dirichlet_values_t v_bc = {
        .left = 0.0,
        .right = 0.0,
        .top = 0.0,
        .bottom = 0.0
    };

    bc_apply_dirichlet_velocity(u, v, nx, ny, &u_bc, &v_bc);

    /* Verify top boundary of u is 1.0 (moving lid) */
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, u[(ny - 1) * nx + i]);
    }

    /* Verify all v boundaries are 0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);           /* Left */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx + nx - 1]);  /* Right */
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Main Dispatch Function Tests
 * ============================================================================ */

void test_dirichlet_main_dispatch_auto(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 5.0,
        .right = 6.0,
        .top = 7.0,
        .bottom = 8.0
    };

    bc_set_backend(BC_BACKEND_AUTO);
    bc_apply_dirichlet_scalar(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Main dispatch (AUTO) Dirichlet BC failed");

    free(field);
}

void test_dirichlet_convenience_macro(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 100.0,
        .right = 200.0,
        .top = 300.0,
        .bottom = 400.0
    };

    /* Use the convenience macro */
    bc_apply_dirichlet(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Convenience macro Dirichlet BC failed");

    free(field);
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

void test_dirichlet_minimum_grid(void) {
    /* Minimum valid grid size is 3x3 */
    size_t nx = 3, ny = 3;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {
        .left = 1.0,
        .right = 2.0,
        .top = 3.0,
        .bottom = 4.0
    };

    bc_apply_dirichlet_scalar(field, nx, ny, &values);

    TEST_ASSERT_TRUE_MESSAGE(verify_dirichlet_bc(field, nx, ny, &values),
                              "Minimum grid Dirichlet BC failed");

    /* Only one interior point (1,1) should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, field[1 * nx + 1]);

    free(field);
}

void test_dirichlet_null_field(void) {
    bc_dirichlet_values_t values = {.left = 1.0, .right = 2.0, .top = 3.0, .bottom = 4.0};

    /* Should not crash with NULL field */
    bc_apply_dirichlet_scalar(NULL, 10, 10, &values);
    bc_apply_dirichlet_scalar_cpu(NULL, 10, 10, &values);
    bc_apply_dirichlet_scalar_simd(NULL, 10, 10, &values);
    bc_apply_dirichlet_scalar_omp(NULL, 10, 10, &values);

    TEST_PASS();
}

void test_dirichlet_null_values(void) {
    size_t nx = 10, ny = 10;
    double* field = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    /* Should not crash with NULL values */
    bc_apply_dirichlet_scalar(field, nx, ny, NULL);
    bc_apply_dirichlet_scalar_cpu(field, nx, ny, NULL);

    /* Field should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, field[0]);

    free(field);
}

void test_dirichlet_too_small_grid(void) {
    double* field = create_test_field(2, 2);
    TEST_ASSERT_NOT_NULL(field);

    bc_dirichlet_values_t values = {.left = 1.0, .right = 2.0, .top = 3.0, .bottom = 4.0};

    /* Should silently return without modification for grid < 3x3 */
    bc_apply_dirichlet_scalar(field, 2, 2, &values);

    /* Field should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, field[0]);

    free(field);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Scalar backend tests */
    RUN_TEST(test_dirichlet_scalar_basic);
    RUN_TEST(test_dirichlet_scalar_zero_values);
    RUN_TEST(test_dirichlet_scalar_negative_values);
    RUN_TEST(test_dirichlet_scalar_large_grid);

    /* SIMD backend tests */
    RUN_TEST(test_dirichlet_simd_basic);
    RUN_TEST(test_dirichlet_simd_consistency_with_scalar);

    /* OpenMP backend tests */
    RUN_TEST(test_dirichlet_omp_basic);
    RUN_TEST(test_dirichlet_omp_consistency_with_scalar);

    /* Velocity field tests */
    RUN_TEST(test_dirichlet_velocity_basic);
    RUN_TEST(test_dirichlet_velocity_lid_driven_cavity);

    /* Main dispatch tests */
    RUN_TEST(test_dirichlet_main_dispatch_auto);
    RUN_TEST(test_dirichlet_convenience_macro);

    /* Edge case tests */
    RUN_TEST(test_dirichlet_minimum_grid);
    RUN_TEST(test_dirichlet_null_field);
    RUN_TEST(test_dirichlet_null_values);
    RUN_TEST(test_dirichlet_too_small_grid);

    return UNITY_END();
}
