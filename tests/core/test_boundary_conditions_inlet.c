/**
 * Unit Tests for Inlet Velocity Boundary Conditions
 *
 * Tests the bc_apply_inlet_* functions across all backends:
 * - Scalar (CPU baseline, single-threaded)
 * - SIMD + OpenMP (AVX2 on x86, NEON on ARM, multi-threaded + SIMD)
 * - OpenMP (multi-threaded, scalar inner loops)
 *
 * Tests profile types:
 * - Uniform velocity (constant across inlet)
 * - Parabolic velocity (fully-developed flow)
 * - Custom user-defined profiles
 *
 * Tests specification types:
 * - Fixed velocity components (u, v)
 * - Velocity magnitude + direction
 * - Mass flow rate
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
#define TOLERANCE 1e-10

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
 * Allocate and initialize velocity fields with a known pattern.
 * Interior values are set to a distinctive value to verify they're unchanged.
 */
static double* create_test_field(size_t nx, size_t ny) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;

    /* Fill with a pattern: all = 999.0 */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field[j * nx + i] = 999.0;
        }
    }
    return field;
}

/**
 * Custom inlet profile callback for testing.
 * Creates a sinusoidal velocity profile.
 */
static void custom_profile_sine(double position, double* u_out, double* v_out, void* user_data) {
    double amplitude = *(double*)user_data;
    *u_out = amplitude * sin(M_PI * position);
    *v_out = 0.0;
}

/* ============================================================================
 * Factory Function Tests
 * ============================================================================ */

void test_inlet_config_uniform(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.5, 0.5);

    TEST_ASSERT_EQUAL(BC_EDGE_LEFT, config.edge);
    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_UNIFORM, config.profile);
    TEST_ASSERT_EQUAL(BC_INLET_SPEC_VELOCITY, config.spec_type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.5, config.spec.velocity.u);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.spec.velocity.v);
    TEST_ASSERT_NULL(config.custom_profile);
}

void test_inlet_config_parabolic(void) {
    bc_inlet_config_t config = bc_inlet_config_parabolic(2.0);

    TEST_ASSERT_EQUAL(BC_EDGE_LEFT, config.edge);
    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_PARABOLIC, config.profile);
    TEST_ASSERT_EQUAL(BC_INLET_SPEC_VELOCITY, config.spec_type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, config.spec.velocity.u);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.spec.velocity.v);
}

void test_inlet_config_magnitude_dir(void) {
    bc_inlet_config_t config = bc_inlet_config_magnitude_dir(3.0, M_PI / 4.0);

    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_UNIFORM, config.profile);
    TEST_ASSERT_EQUAL(BC_INLET_SPEC_MAGNITUDE_DIR, config.spec_type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 3.0, config.spec.magnitude_dir.magnitude);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, M_PI / 4.0, config.spec.magnitude_dir.direction);
}

void test_inlet_config_mass_flow(void) {
    bc_inlet_config_t config = bc_inlet_config_mass_flow(10.0, 1000.0, 0.5);

    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_UNIFORM, config.profile);
    TEST_ASSERT_EQUAL(BC_INLET_SPEC_MASS_FLOW, config.spec_type);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 10.0, config.spec.mass_flow.mass_flow_rate);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1000.0, config.spec.mass_flow.density);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.spec.mass_flow.inlet_length);
}

void test_inlet_config_custom(void) {
    double amplitude = 2.5;
    bc_inlet_config_t config = bc_inlet_config_custom(custom_profile_sine, &amplitude);

    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_CUSTOM, config.profile);
    TEST_ASSERT_NOT_NULL(config.custom_profile);
    TEST_ASSERT_EQUAL(&amplitude, config.custom_profile_user_data);
}

void test_inlet_set_edge(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);

    /* Default is left */
    TEST_ASSERT_EQUAL(BC_EDGE_LEFT, config.edge);

    /* Change to right */
    bc_inlet_set_edge(&config, BC_EDGE_RIGHT);
    TEST_ASSERT_EQUAL(BC_EDGE_RIGHT, config.edge);

    /* Change to top */
    bc_inlet_set_edge(&config, BC_EDGE_TOP);
    TEST_ASSERT_EQUAL(BC_EDGE_TOP, config.edge);

    /* Change to bottom */
    bc_inlet_set_edge(&config, BC_EDGE_BOTTOM);
    TEST_ASSERT_EQUAL(BC_EDGE_BOTTOM, config.edge);
}

/* ============================================================================
 * Uniform Inlet Tests
 * ============================================================================ */

void test_inlet_uniform_left_boundary(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(2.0, 0.5);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check left boundary (column 0) has uniform velocity */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, v[j * nx]);
    }

    /* Check interior is unchanged */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 1; i < nx; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, u[j * nx + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, v[j * nx + i]);
        }
    }

    free(u);
    free(v);
}

void test_inlet_uniform_right_boundary(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(-1.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_RIGHT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check right boundary (column nx-1) has uniform velocity */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, -1.0, u[j * nx + (nx - 1)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx + (nx - 1)]);
    }

    free(u);
    free(v);
}

void test_inlet_uniform_bottom_boundary(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(0.0, 1.5);
    bc_inlet_set_edge(&config, BC_EDGE_BOTTOM);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check bottom boundary (row 0) has uniform velocity */
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.5, v[i]);
    }

    free(u);
    free(v);
}

void test_inlet_uniform_top_boundary(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(0.0, -2.0);
    bc_inlet_set_edge(&config, BC_EDGE_TOP);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check top boundary (row ny-1) has uniform velocity */
    for (size_t i = 0; i < nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[(ny - 1) * nx + i]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, -2.0, v[(ny - 1) * nx + i]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Parabolic Inlet Tests
 * ============================================================================ */

void test_inlet_parabolic_left_boundary(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    double u_max = 2.0;
    bc_inlet_config_t config = bc_inlet_config_parabolic(u_max);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check parabolic profile: u(y) = u_max * 4 * y * (1 - y) */
    for (size_t j = 0; j < ny; j++) {
        double position = (ny > 1) ? (double)j / (double)(ny - 1) : 0.5;
        double expected_u = u_max * 4.0 * position * (1.0 - position);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);
    }

    /* Check boundaries (position = 0 and 1) have zero velocity */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[0]);               /* j=0, position=0 */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[(ny - 1) * nx]);   /* j=ny-1, position=1 */

    /* Check center has maximum velocity */
    size_t j_center = ny / 2;
    double pos_center = (double)j_center / (double)(ny - 1);
    double expected_center = u_max * 4.0 * pos_center * (1.0 - pos_center);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_center, u[j_center * nx]);

    free(u);
    free(v);
}

void test_inlet_parabolic_symmetry(void) {
    /* Test that parabolic profile is symmetric about center */
    size_t nx = 10, ny = 11;  /* Odd ny ensures center point */
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_parabolic(1.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_apply_inlet_cpu(u, v, nx, ny, &config);

    /* Check symmetry: u[j] should equal u[ny-1-j] */
    for (size_t j = 0; j < ny / 2; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE,
                                   u[j * nx],
                                   u[(ny - 1 - j) * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Magnitude + Direction Tests
 * ============================================================================ */

void test_inlet_magnitude_direction(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* 45 degree angle, magnitude 2.0 */
    double magnitude = 2.0;
    double direction = M_PI / 4.0;  /* 45 degrees */
    bc_inlet_config_t config = bc_inlet_config_magnitude_dir(magnitude, direction);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2 * cos(45) = sqrt(2), v = 2 * sin(45) = sqrt(2) */
    double expected_u = magnitude * cos(direction);
    double expected_v = magnitude * sin(direction);

    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_v, v[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_magnitude_direction_horizontal(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* 0 degree angle = pure x direction */
    bc_inlet_config_t config = bc_inlet_config_magnitude_dir(5.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_apply_inlet_cpu(u, v, nx, ny, &config);

    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 5.0, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Mass Flow Rate Tests
 * ============================================================================ */

void test_inlet_mass_flow_left(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Mass flow rate = 10 kg/s, density = 1000 kg/m^3, inlet length = 0.5 m */
    double mass_flow = 10.0;
    double density = 1000.0;
    double length = 0.5;
    double expected_velocity = mass_flow / (density * length);  /* = 0.02 m/s */

    bc_inlet_config_t config = bc_inlet_config_mass_flow(mass_flow, density, length);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Left boundary should have positive u (into domain) */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_velocity, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_mass_flow_right(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    double mass_flow = 10.0;
    double density = 1000.0;
    double length = 0.5;
    double expected_velocity = -mass_flow / (density * length);  /* Negative for right */

    bc_inlet_config_t config = bc_inlet_config_mass_flow(mass_flow, density, length);
    bc_inlet_set_edge(&config, BC_EDGE_RIGHT);

    bc_apply_inlet_cpu(u, v, nx, ny, &config);

    /* Right boundary should have negative u (into domain) */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_velocity, u[j * nx + (nx - 1)]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx + (nx - 1)]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Custom Profile Tests
 * ============================================================================ */

void test_inlet_custom_profile(void) {
    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    double amplitude = 3.0;
    bc_inlet_config_t config = bc_inlet_config_custom(custom_profile_sine, &amplitude);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check sinusoidal profile: u = amplitude * sin(pi * position) */
    for (size_t j = 0; j < ny; j++) {
        double position = (ny > 1) ? (double)j / (double)(ny - 1) : 0.5;
        double expected_u = amplitude * sin(M_PI * position);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Backend Consistency Tests
 * ============================================================================ */

void test_inlet_omp_consistency(void) {
    if (!bc_backend_available(BC_BACKEND_OMP)) {
        TEST_IGNORE_MESSAGE("OpenMP backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u_scalar = create_test_field(nx, ny);
    double* v_scalar = create_test_field(nx, ny);
    double* u_omp = create_test_field(nx, ny);
    double* v_omp = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_scalar);
    TEST_ASSERT_NOT_NULL(v_scalar);
    TEST_ASSERT_NOT_NULL(u_omp);
    TEST_ASSERT_NOT_NULL(v_omp);

    bc_inlet_config_t config = bc_inlet_config_parabolic(2.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_apply_inlet_cpu(u_scalar, v_scalar, nx, ny, &config);
    bc_apply_inlet_omp(u_omp, v_omp, nx, ny, &config);

    /* Compare left boundary values */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_scalar[j * nx], u_omp[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v_scalar[j * nx], v_omp[j * nx]);
    }

    free(u_scalar);
    free(v_scalar);
    free(u_omp);
    free(v_omp);
}

void test_inlet_simd_omp_consistency(void) {
    if (!bc_backend_available(BC_BACKEND_SIMD_OMP)) {
        TEST_IGNORE_MESSAGE("SIMD+OMP backend not available");
        return;
    }

    size_t nx = TEST_NX_MEDIUM, ny = TEST_NY_MEDIUM;
    double* u_scalar = create_test_field(nx, ny);
    double* v_scalar = create_test_field(nx, ny);
    double* u_simd = create_test_field(nx, ny);
    double* v_simd = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_scalar);
    TEST_ASSERT_NOT_NULL(v_scalar);
    TEST_ASSERT_NOT_NULL(u_simd);
    TEST_ASSERT_NOT_NULL(v_simd);

    bc_inlet_config_t config = bc_inlet_config_uniform(1.5, 0.5);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_apply_inlet_cpu(u_scalar, v_scalar, nx, ny, &config);
    bc_apply_inlet_simd_omp(u_simd, v_simd, nx, ny, &config);

    /* Compare left boundary values */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_scalar[j * nx], u_simd[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v_scalar[j * nx], v_simd[j * nx]);
    }

    free(u_scalar);
    free(v_scalar);
    free(u_simd);
    free(v_simd);
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

void test_inlet_null_fields(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);

    /* Should return error with NULL fields */
    cfd_status_t status = bc_apply_inlet_cpu(NULL, NULL, 10, 10, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
}

void test_inlet_null_config(void) {
    size_t nx = 10, ny = 10;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Should return error with NULL config */
    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, NULL);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    /* Field should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, u[0]);

    free(u);
    free(v);
}

void test_inlet_too_small_grid(void) {
    double* u = create_test_field(2, 2);
    double* v = create_test_field(2, 2);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);

    /* Should return error for grid < 3x3 */
    cfd_status_t status = bc_apply_inlet_cpu(u, v, 2, 2, &config);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    /* Field should be unchanged */
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 999.0, u[0]);

    free(u);
    free(v);
}

void test_inlet_minimum_grid(void) {
    size_t nx = 3, ny = 3;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(5.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    cfd_status_t status = bc_apply_inlet_cpu(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check left boundary */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 5.0, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Main Dispatch Tests
 * ============================================================================ */

void test_inlet_main_dispatch(void) {
    size_t nx = TEST_NX_SMALL, ny = TEST_NY_SMALL;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    /* Use main dispatch function */
    cfd_status_t status = bc_apply_inlet(u, v, nx, ny, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Check left boundary */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Factory function tests */
    RUN_TEST(test_inlet_config_uniform);
    RUN_TEST(test_inlet_config_parabolic);
    RUN_TEST(test_inlet_config_magnitude_dir);
    RUN_TEST(test_inlet_config_mass_flow);
    RUN_TEST(test_inlet_config_custom);
    RUN_TEST(test_inlet_set_edge);

    /* Uniform inlet tests */
    RUN_TEST(test_inlet_uniform_left_boundary);
    RUN_TEST(test_inlet_uniform_right_boundary);
    RUN_TEST(test_inlet_uniform_bottom_boundary);
    RUN_TEST(test_inlet_uniform_top_boundary);

    /* Parabolic inlet tests */
    RUN_TEST(test_inlet_parabolic_left_boundary);
    RUN_TEST(test_inlet_parabolic_symmetry);

    /* Magnitude + direction tests */
    RUN_TEST(test_inlet_magnitude_direction);
    RUN_TEST(test_inlet_magnitude_direction_horizontal);

    /* Mass flow rate tests */
    RUN_TEST(test_inlet_mass_flow_left);
    RUN_TEST(test_inlet_mass_flow_right);

    /* Custom profile tests */
    RUN_TEST(test_inlet_custom_profile);

    /* Backend consistency tests */
    RUN_TEST(test_inlet_omp_consistency);
    RUN_TEST(test_inlet_simd_omp_consistency);

    /* Edge case tests */
    RUN_TEST(test_inlet_null_fields);
    RUN_TEST(test_inlet_null_config);
    RUN_TEST(test_inlet_too_small_grid);
    RUN_TEST(test_inlet_minimum_grid);

    /* Main dispatch tests */
    RUN_TEST(test_inlet_main_dispatch);

    return UNITY_END();
}
