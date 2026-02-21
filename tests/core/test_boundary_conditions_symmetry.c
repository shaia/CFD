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
#include "cfd/core/indexing.h"
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
            u[IDX_2D(i, j, nx)] = 1.0 + 0.1 * (double)i + 0.01 * (double)j;
            v[IDX_2D(i, j, nx)] = 2.0 + 0.1 * (double)i + 0.01 * (double)j;
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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, j, nx)]);
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
        double expected_v = v[IDX_2D(1, j, nx)];  /* Interior value at column 1 */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[IDX_2D(0, j, nx)]);
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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(nx - 1, j, nx)]);
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
        double expected_v = v[IDX_2D(nx - 2, j, nx)];  /* Interior value at column nx-2 */
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[IDX_2D(nx - 1, j, nx)]);
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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, j, nx)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(nx - 1, j, nx)]);
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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, j, nx)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(nx - 1, j, nx)]);
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

void test_symmetry_corner_points(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    /* Apply all edges - corners should have both u=0 and v=0 */
    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT | BC_EDGE_RIGHT | BC_EDGE_TOP | BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Bottom-left corner (0,0): u=0 from LEFT, v=0 from BOTTOM */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[0]);

    /* Bottom-right corner (nx-1,0): u=0 from RIGHT, v=0 from BOTTOM */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[nx - 1]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[nx - 1]);

    /* Top-left corner (0,ny-1): u=0 from LEFT, v=0 from TOP */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, ny - 1, nx)]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[IDX_2D(0, ny - 1, nx)]);

    /* Top-right corner (nx-1,ny-1): u=0 from RIGHT, v=0 from TOP */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(nx - 1, ny - 1, nx)]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[IDX_2D(nx - 1, ny - 1, nx)]);

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_perpendicular_edges(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    /* Apply only left and bottom edges - test one corner */
    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT | BC_EDGE_BOTTOM };
    cfd_status_t status = bc_apply_symmetry_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Bottom-left corner (0,0): u=0 from LEFT, v=0 from BOTTOM */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[0]);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[0]);

    /* Bottom-right corner should only have v=0 (no right edge applied) */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[nx - 1]);
    /* u at bottom-right should be copied from interior (Neumann from BOTTOM) */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u[nx + (nx - 1)], u[nx - 1]);

    /* Top-left corner should only have u=0 (no top edge applied) */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, ny - 1, nx)]);
    /* v at top-left should be copied from interior (Neumann from LEFT) */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v[IDX_2D(1, ny - 1, nx)], v[IDX_2D(0, ny - 1, nx)]);

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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, j, nx)]);
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
 * Backend Consistency Tests
 *
 * These tests verify that different backends produce identical results.
 * The OMP and SIMD backends may not have symmetry implementations, in which
 * case they fall back to scalar or return UNSUPPORTED.
 * ============================================================================ */

void test_symmetry_omp_consistency(void) {
    /* Check if OMP backend is available at all */
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    /* OMP symmetry is not implemented - it returns UNSUPPORTED */
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    init_velocity_fields(u, v, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT };
    cfd_status_t status = bc_apply_symmetry_omp(u, v, nx, ny, &config);

    /* OMP symmetry returns UNSUPPORTED since it's not implemented */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, status);

    cfd_free(u);
    cfd_free(v);
}

void test_symmetry_simd_consistency(void) {
    /* Check if SIMD backend is available at all */
    if (!bc_backend_available(BC_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend not available");
        return;
    }

    /* SIMD symmetry falls back to scalar internally, so it should work */
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u_scalar = create_test_field(nx, ny);
    double* v_scalar = create_test_field(nx, ny);
    double* u_simd = create_test_field(nx, ny);
    double* v_simd = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_scalar);
    TEST_ASSERT_NOT_NULL(v_scalar);
    TEST_ASSERT_NOT_NULL(u_simd);
    TEST_ASSERT_NOT_NULL(v_simd);

    init_velocity_fields(u_scalar, v_scalar, nx, ny);
    init_velocity_fields(u_simd, v_simd, nx, ny);

    bc_symmetry_config_t config = { .edges = BC_EDGE_LEFT | BC_EDGE_RIGHT | BC_EDGE_TOP | BC_EDGE_BOTTOM };

    cfd_status_t status_scalar = bc_apply_symmetry_cpu(u_scalar, v_scalar, nx, ny, &config);
    cfd_status_t status_simd = bc_apply_symmetry_simd(u_simd, v_simd, nx, ny, &config);

    /* Both should succeed - SIMD falls back to scalar internally */
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status_scalar);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status_simd);

    /* Compare all field values - should be identical */
    for (size_t idx = 0; idx < nx * ny; idx++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_scalar[idx], u_simd[idx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v_scalar[idx], v_simd[idx]);
    }

    cfd_free(u_scalar);
    cfd_free(v_scalar);
    cfd_free(u_simd);
    cfd_free(v_simd);
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
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(0, j, nx)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[IDX_2D(nx - 1, j, nx)]);
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
    RUN_TEST(test_symmetry_corner_points);
    RUN_TEST(test_symmetry_perpendicular_edges);

    /* Dispatcher tests */
    RUN_TEST(test_symmetry_main_dispatcher);

    /* Backend consistency tests */
    RUN_TEST(test_symmetry_omp_consistency);
    RUN_TEST(test_symmetry_simd_consistency);

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
