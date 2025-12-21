/**
 * Unit Tests for Outlet Boundary Conditions
 *
 * Tests the bc_apply_outlet_* functions across all backends:
 * - Scalar (CPU baseline, single-threaded)
 * - SIMD + OpenMP (AVX2 on x86, NEON on ARM, multi-threaded + SIMD)
 * - OpenMP (multi-threaded, scalar inner loops)
 *
 * Tests outlet types:
 * - Zero-gradient (Neumann): boundary = adjacent interior value
 * - Convective: advects values out of domain (currently falls back to zero-gradient)
 *
 * Tests all four boundary edges:
 * - Left, Right, Bottom, Top
 */

#include "cfd/boundary/boundary_conditions.h"
#include "unity.h"

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
#define TOLERANCE 1e-10

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
 * Allocate and initialize field with a known pattern.
 * Interior values are set to a ramp pattern to verify zero-gradient works.
 */
static double* create_test_field_ramp(size_t nx, size_t ny) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;

    /* Fill with ramp pattern: field[j][i] = i + j*10 + 100 */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field[j * nx + i] = (double)(i + j * 10 + 100);
        }
    }
    return field;
}

/**
 * Allocate and initialize field with uniform interior value.
 */
static double* create_test_field_uniform(size_t nx, size_t ny, double value) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field[j * nx + i] = value;
        }
    }
    return field;
}

/**
 * Helper: Verify outlet boundary matches adjacent interior for given edge.
 */
static void verify_outlet_boundary(const double* field, size_t nx, size_t ny, bc_edge_t edge) {
    switch (edge) {
        case BC_EDGE_LEFT:
            for (size_t j = 0; j < ny; j++) {
                double expected = field[j * nx + 1];
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx]);
            }
            break;
        case BC_EDGE_RIGHT:
            for (size_t j = 0; j < ny; j++) {
                double expected = field[j * nx + (nx - 2)];
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx + (nx - 1)]);
            }
            break;
        case BC_EDGE_BOTTOM:
            for (size_t i = 0; i < nx; i++) {
                double expected = field[nx + i];
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[i]);
            }
            break;
        case BC_EDGE_TOP:
            for (size_t i = 0; i < nx; i++) {
                double expected = field[(ny - 2) * nx + i];
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[(ny - 1) * nx + i]);
            }
            break;
        default:
            TEST_FAIL_MESSAGE("Invalid edge in verify_outlet_boundary");
    }
}

/**
 * Helper: Verify two fields are identical.
 */
static void verify_fields_equal(const double* field1, const double* field2, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field1[j * nx + i], field2[j * nx + i]);
        }
    }
}

/**
 * Helper: Verify only the specified edge was modified.
 */
static void verify_only_edge_modified(const double* field, const double* original,
                                       size_t nx, size_t ny, bc_edge_t edge) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            int on_modified_edge = 0;
            switch (edge) {
                case BC_EDGE_LEFT:   on_modified_edge = (i == 0); break;
                case BC_EDGE_RIGHT:  on_modified_edge = (i == nx - 1); break;
                case BC_EDGE_BOTTOM: on_modified_edge = (j == 0); break;
                case BC_EDGE_TOP:    on_modified_edge = (j == ny - 1); break;
                default: break;
            }
            if (!on_modified_edge) {
                TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, original[j * nx + i], field[j * nx + i]);
            }
        }
    }
}

/* ============================================================================
 * Factory Function Tests
 * ============================================================================ */

void test_outlet_config_zero_gradient(void) {
    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    TEST_ASSERT_EQUAL(BC_EDGE_RIGHT, config.edge);
    TEST_ASSERT_EQUAL(BC_OUTLET_ZERO_GRADIENT, config.type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.advection_velocity);
}

void test_outlet_config_convective(void) {
    bc_outlet_config_t config = bc_outlet_config_convective(1.5);

    TEST_ASSERT_EQUAL(BC_EDGE_RIGHT, config.edge);
    TEST_ASSERT_EQUAL(BC_OUTLET_CONVECTIVE, config.type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.5, config.advection_velocity);
}

void test_outlet_set_edge(void) {
    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    /* Default is right */
    TEST_ASSERT_EQUAL(BC_EDGE_RIGHT, config.edge);

    /* Change to left */
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);
    TEST_ASSERT_EQUAL(BC_EDGE_LEFT, config.edge);

    /* Change to top */
    bc_outlet_set_edge(&config, BC_EDGE_TOP);
    TEST_ASSERT_EQUAL(BC_EDGE_TOP, config.edge);

    /* Change to bottom */
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);
    TEST_ASSERT_EQUAL(BC_EDGE_BOTTOM, config.edge);
}

/* ============================================================================
 * Zero-Gradient Outlet Tests - All Edges
 * ============================================================================ */

void test_outlet_zero_gradient_right(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check right boundary (column nx-1) equals adjacent interior (column nx-2) */
    for (size_t j = 0; j < ny; j++) {
        double expected = field[j * nx + (nx - 2)];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx + (nx - 1)]);
    }

    free(field);
}

void test_outlet_zero_gradient_left(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check left boundary (column 0) equals adjacent interior (column 1) */
    for (size_t j = 0; j < ny; j++) {
        double expected = field[j * nx + 1];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx]);
    }

    free(field);
}

void test_outlet_zero_gradient_top(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check top boundary (row ny-1) equals adjacent interior (row ny-2) */
    for (size_t i = 0; i < nx; i++) {
        double expected = field[(ny - 2) * nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[(ny - 1) * nx + i]);
    }

    free(field);
}

void test_outlet_zero_gradient_bottom(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check bottom boundary (row 0) equals adjacent interior (row 1) */
    for (size_t i = 0; i < nx; i++) {
        double expected = field[nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[i]);
    }

    free(field);
}

/* ============================================================================
 * Convective Outlet Tests - All Edges
 * ============================================================================ */

void test_outlet_convective_right(void) {
    /* Convective outlet currently falls back to zero-gradient */
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_convective(1.0);
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check right boundary equals adjacent interior (zero-gradient behavior) */
    for (size_t j = 0; j < ny; j++) {
        double expected = field[j * nx + (nx - 2)];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx + (nx - 1)]);
    }

    free(field);
}

void test_outlet_convective_left(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_convective(1.0);
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check left boundary equals adjacent interior */
    for (size_t j = 0; j < ny; j++) {
        double expected = field[j * nx + 1];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[j * nx]);
    }

    free(field);
}

void test_outlet_convective_top(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_convective(1.0);
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check top boundary equals adjacent interior */
    for (size_t i = 0; i < nx; i++) {
        double expected = field[(ny - 2) * nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[(ny - 1) * nx + i]);
    }

    free(field);
}

void test_outlet_convective_bottom(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_convective(1.0);
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check bottom boundary equals adjacent interior */
    for (size_t i = 0; i < nx; i++) {
        double expected = field[nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, field[i]);
    }

    free(field);
}

/* ============================================================================
 * Velocity Field Tests
 * ============================================================================ */

void test_outlet_velocity_zero_gradient(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field_ramp(nx, ny);
    double* v = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Modify v to have different values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            v[j * nx + i] = -(double)(i + j * 10 + 100);
        }
    }

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_velocity_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check both u and v at right boundary */
    for (size_t j = 0; j < ny; j++) {
        double expected_u = u[j * nx + (nx - 2)];
        double expected_v = v[j * nx + (nx - 2)];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx + (nx - 1)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[j * nx + (nx - 1)]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Interior Unchanged Tests
 * ============================================================================ */

void test_outlet_interior_unchanged(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field = create_test_field_ramp(nx, ny);
    double* field_copy = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(field_copy);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Interior should be unchanged (all columns except nx-1) */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx - 1; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field_copy[j * nx + i], field[j * nx + i]);
        }
    }

    free(field);
    free(field_copy);
}

/* ============================================================================
 * Backend Consistency Tests
 * ============================================================================ */

void test_outlet_backend_consistency(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;

    /* Create identical fields for each backend */
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_auto = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_auto);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    /* Apply with scalar backend */
    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    /* Apply with auto backend */
    cfd_status_t status2 = bc_apply_outlet_scalar(field_auto, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status2);

    /* Results should match */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field_scalar[j * nx + i], field_auto[j * nx + i]);
        }
    }

    free(field_scalar);
    free(field_auto);
}

/* ============================================================================
 * Error Handling Tests
 * ============================================================================ */

void test_outlet_null_field(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    cfd_status_t status = bc_apply_outlet_scalar_cpu(NULL, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
}

void test_outlet_null_config(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, NULL);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

void test_outlet_grid_too_small(void) {
    size_t nx = 2, ny = 2;
    double* field = create_test_field_uniform(nx, ny, 1.0);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

void test_outlet_invalid_edge(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    config.edge = (bc_edge_t)0;  /* Invalid edge */

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

void test_outlet_invalid_type(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    config.type = (bc_outlet_type_t)99;  /* Invalid type */

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

/* ============================================================================
 * Large Grid Test
 * ============================================================================ */

void test_outlet_large_grid(void) {
    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    verify_outlet_boundary(field, nx, ny, BC_EDGE_RIGHT);

    free(field);
}

void test_outlet_large_grid_all_backends(void) {
    size_t nx = TEST_NX_LARGE, ny = TEST_NY_LARGE;

    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_omp = create_test_field_ramp(nx, ny);
    double* field_simd = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field_omp, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_omp, nx, ny);
    }

    cfd_status_t status3 = bc_apply_outlet_scalar_simd_omp(field_simd, nx, ny, &config);
    if (status3 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_simd, nx, ny);
    }

    free(field_scalar);
    free(field_omp);
    free(field_simd);
}

/* ============================================================================
 * OMP Backend Consistency Tests
 * ============================================================================ */

void test_outlet_omp_backend_right(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_omp = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field_omp, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_omp, nx, ny);
    }

    free(field_scalar);
    free(field_omp);
}

void test_outlet_omp_backend_left(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_omp = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field_omp, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_omp, nx, ny);
    }

    free(field_scalar);
    free(field_omp);
}

void test_outlet_omp_backend_top(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_omp = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field_omp, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_omp, nx, ny);
    }

    free(field_scalar);
    free(field_omp);
}

void test_outlet_omp_backend_bottom(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_omp = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_omp);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field_omp, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_omp, nx, ny);
    }

    free(field_scalar);
    free(field_omp);
}

/* ============================================================================
 * SIMD Backend Consistency Tests
 * ============================================================================ */

void test_outlet_simd_omp_backend_right(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_simd = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_simd_omp(field_simd, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_simd, nx, ny);
    }

    free(field_scalar);
    free(field_simd);
}

void test_outlet_simd_omp_backend_left(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_simd = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_simd_omp(field_simd, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_simd, nx, ny);
    }

    free(field_scalar);
    free(field_simd);
}

void test_outlet_simd_omp_backend_top(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_simd = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_simd_omp(field_simd, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_simd, nx, ny);
    }

    free(field_scalar);
    free(field_simd);
}

void test_outlet_simd_omp_backend_bottom(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_scalar = create_test_field_ramp(nx, ny);
    double* field_simd = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_scalar);
    TEST_ASSERT_NOT_NULL(field_simd);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_simd_omp(field_simd, nx, ny, &config);
    if (status2 == CFD_SUCCESS) {
        verify_fields_equal(field_scalar, field_simd, nx, ny);
    }

    free(field_scalar);
    free(field_simd);
}

/* ============================================================================
 * Only Specified Edge Modified Tests
 * ============================================================================ */

void test_outlet_only_left_edge_modified(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    double* original = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(original);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    verify_only_edge_modified(field, original, nx, ny, BC_EDGE_LEFT);

    free(field);
    free(original);
}

void test_outlet_only_right_edge_modified(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    double* original = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(original);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    verify_only_edge_modified(field, original, nx, ny, BC_EDGE_RIGHT);

    free(field);
    free(original);
}

void test_outlet_only_top_edge_modified(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    double* original = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(original);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    verify_only_edge_modified(field, original, nx, ny, BC_EDGE_TOP);

    free(field);
    free(original);
}

void test_outlet_only_bottom_edge_modified(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    double* original = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(original);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    verify_only_edge_modified(field, original, nx, ny, BC_EDGE_BOTTOM);

    free(field);
    free(original);
}

/* ============================================================================
 * Velocity Field Tests - All Edges
 * ============================================================================ */

void test_outlet_velocity_left(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field_ramp(nx, ny);
    double* v = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Make v have different values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            v[j * nx + i] = -(double)(i + j * 10 + 100);
        }
    }

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_outlet_velocity_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check both u and v at left boundary */
    for (size_t j = 0; j < ny; j++) {
        double expected_u = u[j * nx + 1];
        double expected_v = v[j * nx + 1];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[j * nx]);
    }

    free(u);
    free(v);
}

void test_outlet_velocity_top(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field_ramp(nx, ny);
    double* v = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            v[j * nx + i] = -(double)(i + j * 10 + 100);
        }
    }

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_outlet_velocity_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    for (size_t i = 0; i < nx; i++) {
        double expected_u = u[(ny - 2) * nx + i];
        double expected_v = v[(ny - 2) * nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[(ny - 1) * nx + i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[(ny - 1) * nx + i]);
    }

    free(u);
    free(v);
}

void test_outlet_velocity_bottom(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field_ramp(nx, ny);
    double* v = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            v[j * nx + i] = -(double)(i + j * 10 + 100);
        }
    }

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_outlet_velocity_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    for (size_t i = 0; i < nx; i++) {
        double expected_u = u[nx + i];
        double expected_v = v[nx + i];
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[i]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Minimum Grid Size Test
 * ============================================================================ */

void test_outlet_minimum_grid_size(void) {
    /* 3x3 is the minimum valid grid size */
    size_t nx = 3, ny = 3;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    /* Test all edges on minimum grid */
    bc_edge_t edges[] = { BC_EDGE_LEFT, BC_EDGE_RIGHT, BC_EDGE_TOP, BC_EDGE_BOTTOM };
    for (int e = 0; e < 4; e++) {
        bc_outlet_set_edge(&config, edges[e]);
        cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    free(field);
}

/* ============================================================================
 * Invalid Edge Variations Tests
 * ============================================================================ */

void test_outlet_invalid_edge_combined_flags(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    /* Combined flags are invalid - must be exactly one edge */
    config.edge = (bc_edge_t)(BC_EDGE_LEFT | BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

void test_outlet_invalid_edge_out_of_range(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    config.edge = (bc_edge_t)0x10;  /* Out of range */

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(field);
}

void test_outlet_all_backends_invalid_edge(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    config.edge = (bc_edge_t)0;

    /* All backends should reject invalid edge */
    cfd_status_t status1 = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status1);

    cfd_status_t status2 = bc_apply_outlet_scalar_omp(field, nx, ny, &config);
    if (status2 != CFD_ERROR_UNSUPPORTED) {
        TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status2);
    }

    cfd_status_t status3 = bc_apply_outlet_scalar_simd_omp(field, nx, ny, &config);
    if (status3 != CFD_ERROR_UNSUPPORTED) {
        TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status3);
    }

    free(field);
}

/* ============================================================================
 * Main Dispatch Test
 * ============================================================================ */

void test_outlet_main_dispatch(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* field_dispatch = create_test_field_ramp(nx, ny);
    double* field_scalar = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field_dispatch);
    TEST_ASSERT_NOT_NULL(field_scalar);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();

    /* Test all edges through main dispatch vs scalar */
    bc_edge_t edges[] = { BC_EDGE_LEFT, BC_EDGE_RIGHT, BC_EDGE_TOP, BC_EDGE_BOTTOM };
    for (int e = 0; e < 4; e++) {
        /* Reset fields */
        free(field_dispatch);
        free(field_scalar);
        field_dispatch = create_test_field_ramp(nx, ny);
        field_scalar = create_test_field_ramp(nx, ny);
        TEST_ASSERT_NOT_NULL(field_dispatch);
        TEST_ASSERT_NOT_NULL(field_scalar);

        bc_outlet_set_edge(&config, edges[e]);

        cfd_status_t status1 = bc_apply_outlet_scalar(field_dispatch, nx, ny, &config);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

        cfd_status_t status2 = bc_apply_outlet_scalar_cpu(field_scalar, nx, ny, &config);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status2);

        verify_fields_equal(field_dispatch, field_scalar, nx, ny);
    }

    free(field_dispatch);
    free(field_scalar);
}

/* ============================================================================
 * Correct Indices Tests
 * ============================================================================ */

void test_outlet_correct_indices_left(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify specific indices: field[j][0] = field[j][1] */
    for (size_t j = 0; j < ny; j++) {
        size_t dst_idx = j * nx + 0;
        size_t src_idx = j * nx + 1;
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field[src_idx], field[dst_idx]);
    }

    free(field);
}

void test_outlet_correct_indices_right(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify specific indices: field[j][nx-1] = field[j][nx-2] */
    for (size_t j = 0; j < ny; j++) {
        size_t dst_idx = j * nx + (nx - 1);
        size_t src_idx = j * nx + (nx - 2);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field[src_idx], field[dst_idx]);
    }

    free(field);
}

void test_outlet_correct_indices_bottom(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify specific indices: field[0][i] = field[1][i] */
    for (size_t i = 0; i < nx; i++) {
        size_t dst_idx = 0 * nx + i;
        size_t src_idx = 1 * nx + i;
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field[src_idx], field[dst_idx]);
    }

    free(field);
}

void test_outlet_correct_indices_top(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* field = create_test_field_ramp(nx, ny);
    TEST_ASSERT_NOT_NULL(field);

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_outlet_scalar_cpu(field, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify specific indices: field[ny-1][i] = field[ny-2][i] */
    for (size_t i = 0; i < nx; i++) {
        size_t dst_idx = (ny - 1) * nx + i;
        size_t src_idx = (ny - 2) * nx + i;
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, field[src_idx], field[dst_idx]);
    }

    free(field);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Factory function tests */
    RUN_TEST(test_outlet_config_zero_gradient);
    RUN_TEST(test_outlet_config_convective);
    RUN_TEST(test_outlet_set_edge);

    /* Zero-gradient tests for all edges */
    RUN_TEST(test_outlet_zero_gradient_right);
    RUN_TEST(test_outlet_zero_gradient_left);
    RUN_TEST(test_outlet_zero_gradient_top);
    RUN_TEST(test_outlet_zero_gradient_bottom);

    /* Convective outlet tests for all edges */
    RUN_TEST(test_outlet_convective_right);
    RUN_TEST(test_outlet_convective_left);
    RUN_TEST(test_outlet_convective_top);
    RUN_TEST(test_outlet_convective_bottom);

    /* Velocity field tests for all edges */
    RUN_TEST(test_outlet_velocity_zero_gradient);
    RUN_TEST(test_outlet_velocity_left);
    RUN_TEST(test_outlet_velocity_top);
    RUN_TEST(test_outlet_velocity_bottom);

    /* Interior unchanged tests */
    RUN_TEST(test_outlet_interior_unchanged);

    /* Backend consistency tests */
    RUN_TEST(test_outlet_backend_consistency);

    /* OMP backend consistency tests */
    RUN_TEST(test_outlet_omp_backend_right);
    RUN_TEST(test_outlet_omp_backend_left);
    RUN_TEST(test_outlet_omp_backend_top);
    RUN_TEST(test_outlet_omp_backend_bottom);

    /* SIMD backend consistency tests */
    RUN_TEST(test_outlet_simd_omp_backend_right);
    RUN_TEST(test_outlet_simd_omp_backend_left);
    RUN_TEST(test_outlet_simd_omp_backend_top);
    RUN_TEST(test_outlet_simd_omp_backend_bottom);

    /* Only specified edge modified tests */
    RUN_TEST(test_outlet_only_left_edge_modified);
    RUN_TEST(test_outlet_only_right_edge_modified);
    RUN_TEST(test_outlet_only_top_edge_modified);
    RUN_TEST(test_outlet_only_bottom_edge_modified);

    /* Correct indices tests */
    RUN_TEST(test_outlet_correct_indices_left);
    RUN_TEST(test_outlet_correct_indices_right);
    RUN_TEST(test_outlet_correct_indices_bottom);
    RUN_TEST(test_outlet_correct_indices_top);

    /* Error handling tests */
    RUN_TEST(test_outlet_null_field);
    RUN_TEST(test_outlet_null_config);
    RUN_TEST(test_outlet_grid_too_small);
    RUN_TEST(test_outlet_invalid_edge);
    RUN_TEST(test_outlet_invalid_type);
    RUN_TEST(test_outlet_invalid_edge_combined_flags);
    RUN_TEST(test_outlet_invalid_edge_out_of_range);
    RUN_TEST(test_outlet_all_backends_invalid_edge);

    /* Minimum grid size test */
    RUN_TEST(test_outlet_minimum_grid_size);

    /* Main dispatch test */
    RUN_TEST(test_outlet_main_dispatch);

    /* Large grid tests */
    RUN_TEST(test_outlet_large_grid);
    RUN_TEST(test_outlet_large_grid_all_backends);

    return UNITY_END();
}
