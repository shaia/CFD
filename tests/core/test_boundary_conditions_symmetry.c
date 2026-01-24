/**
 * Unit Tests for Symmetry Boundary Conditions
 *
 * Tests the bc_apply_symmetry_* functions:
 * - X-symmetry (left/right): u = 0, dv/dx = 0
 * - Y-symmetry (top/bottom): v = 0, du/dy = 0
 * - Combined edges
 * - Error handling
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"
#include "unity.h"

#include <math.h>
#include <string.h>

/* Test grid sizes */
#define TEST_NX 16
#define TEST_NY 16

/* Test tolerance for floating point comparison */
#define TOLERANCE 1e-10

void setUp(void) {
    bc_set_backend(BC_BACKEND_AUTO);
}

void tearDown(void) {
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static double* create_test_field(size_t nx, size_t ny) {
    double* field = (double*)cfd_malloc(nx * ny * sizeof(double));
    if (!field) return NULL;
    for (size_t i = 0; i < nx * ny; i++) {
        field[i] = 999.0;
    }
    return field;
}

/**
 * Initialize velocity fields with non-zero values for testing.
 * u = 1.0 + 0.1*i + 0.01*j
 * v = 2.0 + 0.1*i + 0.01*j
 */
static void init_velocity_fields(double* u, double* v, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            u[j * nx + i] = 1.0 + 0.1 * (double)i + 0.01 * (double)j;
            v[j * nx + i] = 2.0 + 0.1 * (double)i + 0.01 * (double)j;
        }
    }
}

/* ============================================================================
 * Left Edge Symmetry Tests (X-symmetry plane at x=0)
 * ============================================================================ */

void test_symmetry_left_edge_u_zero(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be zero at left boundary (column 0) */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_left_edge_v_neumann(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* v should be copied from interior (dv/dx = 0) */
    for (size_t j = 0; j < ny; j++) {
        double expected_v = v[j * nx + 1];  /* Interior value at column 1 */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[j * nx]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Right Edge Symmetry Tests (X-symmetry plane at x=Lx)
 * ============================================================================ */

void test_symmetry_right_edge_u_zero(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_RIGHT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be zero at right boundary (column nx-1) */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx + (nx - 1)]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_right_edge_v_neumann(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_RIGHT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* v should be copied from interior (dv/dx = 0) */
    for (size_t j = 0; j < ny; j++) {
        double expected_v = v[j * nx + (nx - 2)];  /* Interior value at column nx-2 */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[j * nx + (nx - 1)]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Bottom Edge Symmetry Tests (Y-symmetry plane at y=0)
 * ============================================================================ */

void test_symmetry_bottom_edge_v_zero(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* v should be zero at bottom boundary (row 0) */
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_bottom_edge_u_neumann(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be copied from interior (du/dy = 0) */
    for (size_t i = 0; i < nx; i++) {
        double expected_u = u[nx + i];  /* Interior value at row 1 */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Top Edge Symmetry Tests (Y-symmetry plane at y=Ly)
 * ============================================================================ */

void test_symmetry_top_edge_v_zero(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_TOP };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* v should be zero at top boundary (row ny-1) */
    double* v_top = v + (ny - 1) * nx;
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v_top[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_top_edge_u_neumann(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_TOP };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be copied from interior (du/dy = 0) */
    double* u_top = u + (ny - 1) * nx;
    double* u_interior = u + (ny - 2) * nx;
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_interior[i], u_top[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Combined Edge Tests
 * ============================================================================ */

void test_symmetry_left_right_edges(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT | BC_EDGE_RIGHT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be zero at both left and right boundaries */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx + (nx - 1)]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_top_bottom_edges(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_TOP | BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* v should be zero at both top and bottom boundaries */
    double* v_top = v + (ny - 1) * nx;
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v_top[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_all_edges(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT | BC_EDGE_RIGHT | BC_EDGE_TOP | BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be zero at left and right boundaries */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx + (nx - 1)]);
    }

    /* v should be zero at top and bottom boundaries */
    double* v_top = v + (ny - 1) * nx;
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v_top[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Dispatcher Tests
 * ============================================================================ */

void test_symmetry_main_dispatcher(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* u should be zero at left boundary */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Error Handling Tests
 * ============================================================================ */

void test_symmetry_null_u(void) {
    double* v = create_test_field(TEST_NX, TEST_NY);
    TEST_ASSERT_NOT_NULL(v);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_cpu(NULL, v, TEST_NX, TEST_NY, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    cfd_free(v);
}

void test_symmetry_null_v(void) {
    double* u = create_test_field(TEST_NX, TEST_NY);
    TEST_ASSERT_NOT_NULL(u);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, NULL, TEST_NX, TEST_NY, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    cfd_free(u);
}

void test_symmetry_null_config(void) {
    double* u = create_test_field(TEST_NX, TEST_NY);
    double* v = create_test_field(TEST_NX, TEST_NY);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    cfd_status_t status = bc_apply_symmetry_cpu(u, v, TEST_NX, TEST_NY, NULL);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_too_small_grid(void) {
    double* u = create_test_field(2, 2);
    double* v = create_test_field(2, 2);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, 2, 2, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_no_edges(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    /* Save original values for comparison */
    double u_orig = u[0];
    double v_orig = v[0];

    bc_symmetry_config_t config = { .edges = 0 };  /* No edges */
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Fields should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_orig, u[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v_orig, v[0]);

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Convenience Macro Test
 * ============================================================================ */

void test_symmetry_all_macro(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    cfd_status_t status = bc_apply_symmetry_all(u, v, nx, ny);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify all boundary conditions applied */
    /* u = 0 at left and right */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx + (nx - 1)]);
    }

    /* v = 0 at top and bottom */
    double* v_top = v + (ny - 1) * nx;
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v_top[i]);
    }

    cfd_free(u);
    cfd_free(v);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Left edge tests */
    RUN_TEST(test_symmetry_left_edge_u_zero);
    RUN_TEST(test_symmetry_left_edge_v_neumann);

    /* Right edge tests */
    RUN_TEST(test_symmetry_right_edge_u_zero);
    RUN_TEST(test_symmetry_right_edge_v_neumann);

    /* Bottom edge tests */
    RUN_TEST(test_symmetry_bottom_edge_v_zero);
    RUN_TEST(test_symmetry_bottom_edge_u_neumann);

    /* Top edge tests */
    RUN_TEST(test_symmetry_top_edge_v_zero);
    RUN_TEST(test_symmetry_top_edge_u_neumann);

    /* Combined edge tests */
    RUN_TEST(test_symmetry_left_right_edges);
    RUN_TEST(test_symmetry_top_bottom_edges);
    RUN_TEST(test_symmetry_all_edges);

    /* Dispatcher tests */
    RUN_TEST(test_symmetry_main_dispatcher);

    /* Error handling tests */
    RUN_TEST(test_symmetry_null_u);
    RUN_TEST(test_symmetry_null_v);
    RUN_TEST(test_symmetry_null_config);
    RUN_TEST(test_symmetry_too_small_grid);
    RUN_TEST(test_symmetry_no_edges);

    /* Macro tests */
    RUN_TEST(test_symmetry_all_macro);

    return UNITY_END();
}
