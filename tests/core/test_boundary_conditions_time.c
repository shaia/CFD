/**
 * Unit Tests for Time-Varying Inlet Boundary Conditions
 *
 * Tests the bc_apply_inlet_time_* functions and time profile computations:
 * - Sinusoidal time modulation
 * - Ramp time modulation
 * - Step time modulation
 * - Custom time callback
 * - Backward compatibility (no time variation = standard inlet)
 */

#include "cfd/boundary/boundary_conditions.h"
#include "unity.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Test grid sizes */
#define TEST_NX 16
#define TEST_NY 16

/* Test tolerance for floating point comparison */
#define TOLERANCE 1e-10

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    bc_set_backend(BC_BACKEND_AUTO);
}

void tearDown(void) {
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static double* create_test_field(size_t nx, size_t ny) {
    double* field = (double*)malloc(nx * ny * sizeof(double));
    if (!field) return NULL;
    for (size_t i = 0; i < nx * ny; i++) {
        field[i] = 999.0;
    }
    return field;
}

/* Custom time callback for testing */
static double custom_time_fn(double time, double dt, void* user_data) {
    (void)dt;
    double* scale = (double*)user_data;
    return (*scale) * time;  /* Returns scale * t */
}

/* Custom time-varying profile callback for testing */
static void custom_time_profile(double position, double time, double dt,
                                 double* u_out, double* v_out, void* user_data) {
    (void)dt;
    (void)user_data;
    /* Velocity varies with both position and time */
    *u_out = sin(M_PI * position) * cos(2.0 * M_PI * time);
    *v_out = 0.0;
}

/* ============================================================================
 * Time Profile Factory Tests
 * ============================================================================ */

void test_inlet_config_time_sinusoidal(void) {
    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        1.0, 0.0,     /* base velocity */
        2.0,          /* frequency */
        0.5,          /* amplitude */
        M_PI / 4.0,   /* phase */
        1.0);         /* offset */

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_SINUSOIDAL, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, config.time_config.params.sinusoidal.frequency);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.time_config.params.sinusoidal.amplitude);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, M_PI / 4.0, config.time_config.params.sinusoidal.phase);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.sinusoidal.offset);

    /* Base config should be uniform */
    TEST_ASSERT_EQUAL(BC_INLET_PROFILE_UNIFORM, config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.spec.velocity.u);
}

void test_inlet_config_time_ramp(void) {
    bc_inlet_config_t config = bc_inlet_config_time_ramp(
        2.0, 0.0,     /* base velocity */
        0.5,          /* t_start */
        1.5,          /* t_end */
        0.0,          /* value_start */
        1.0);         /* value_end */

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_RAMP, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.time_config.params.ramp.t_start);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.5, config.time_config.params.ramp.t_end);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.time_config.params.ramp.value_start);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.ramp.value_end);
}

void test_inlet_config_time_step(void) {
    bc_inlet_config_t config = bc_inlet_config_time_step(
        1.0, 0.0,     /* base velocity */
        0.5,          /* t_step */
        0.0,          /* value_before */
        1.0);         /* value_after */

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_STEP, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.time_config.params.step.t_step);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.time_config.params.step.value_before);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.step.value_after);
}

void test_inlet_config_time_custom(void) {
    double scale = 2.5;
    bc_inlet_config_t config = bc_inlet_config_time_custom(custom_time_profile, &scale);

    TEST_ASSERT_NOT_NULL(config.custom_profile_time);
    TEST_ASSERT_EQUAL(&scale, config.custom_profile_time_user_data);
}

/* ============================================================================
 * Time Profile Setter Tests
 * ============================================================================ */

void test_inlet_set_time_sinusoidal(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);
    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_CONSTANT, config.time_config.profile);

    bc_inlet_set_time_sinusoidal(&config, 1.0, 0.3, 0.0, 1.0);

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_SINUSOIDAL, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.sinusoidal.frequency);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.3, config.time_config.params.sinusoidal.amplitude);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.time_config.params.sinusoidal.phase);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.sinusoidal.offset);
}

void test_inlet_set_time_ramp(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);

    bc_inlet_set_time_ramp(&config, 0.0, 1.0, 0.5, 1.5);

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_RAMP, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.time_config.params.ramp.t_start);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, config.time_config.params.ramp.t_end);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.time_config.params.ramp.value_start);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.5, config.time_config.params.ramp.value_end);
}

void test_inlet_set_time_step(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);

    bc_inlet_set_time_step(&config, 0.5, 0.0, 2.0);

    TEST_ASSERT_EQUAL(BC_TIME_PROFILE_STEP, config.time_config.profile);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.5, config.time_config.params.step.t_step);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, config.time_config.params.step.value_before);
    TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, config.time_config.params.step.value_after);
}

/* ============================================================================
 * Sinusoidal Profile Tests
 * ============================================================================ */

void test_inlet_time_sinusoidal_at_t_zero(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Sinusoidal: offset + amplitude * sin(2*pi*freq*t + phase) */
    /* At t=0 with phase=0: modulator = 1.0 + 0.5 * sin(0) = 1.0 */
    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0,     /* base velocity */
        1.0,          /* frequency */
        0.5,          /* amplitude */
        0.0,          /* phase */
        1.0);         /* offset */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.0, 0.01};
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * (1.0 + 0.5*sin(0)) = 2.0 * 1.0 = 2.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_sinusoidal_at_quarter_period(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* At t=T/4 = 0.25 for freq=1Hz: sin(2*pi*1*0.25) = sin(pi/2) = 1.0 */
    /* modulator = 1.0 + 0.5 * 1.0 = 1.5 */
    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0,     /* base velocity */
        1.0,          /* frequency = 1 Hz, period = 1s */
        0.5,          /* amplitude */
        0.0,          /* phase */
        1.0);         /* offset */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.25, 0.01};  /* t = T/4 */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 1.5 = 3.0 */
    double expected = 2.0 * (1.0 + 0.5 * 1.0);
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_sinusoidal_at_half_period(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* At t=T/2 = 0.5 for freq=1Hz: sin(2*pi*1*0.5) = sin(pi) = 0 */
    /* modulator = 1.0 + 0.5 * 0 = 1.0 */
    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0,     /* base velocity */
        1.0,          /* frequency = 1 Hz */
        0.5,          /* amplitude */
        0.0,          /* phase */
        1.0);         /* offset */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.5, 0.01};  /* t = T/2 */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 1.0 = 2.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_sinusoidal_at_three_quarter_period(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* At t=3T/4 = 0.75: sin(2*pi*1*0.75) = sin(3*pi/2) = -1.0 */
    /* modulator = 1.0 + 0.5 * (-1) = 0.5 */
    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0,     /* base velocity */
        1.0,          /* frequency = 1 Hz */
        0.5,          /* amplitude */
        0.0,          /* phase */
        1.0);         /* offset */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.75, 0.01};  /* t = 3T/4 */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 0.5 = 1.0 */
    double expected = 2.0 * (1.0 + 0.5 * (-1.0));
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Ramp Profile Tests
 * ============================================================================ */

void test_inlet_time_ramp_before_start(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Ramp from t=0.5 to t=1.5, value from 0 to 1 */
    /* Before t=0.5: modulator = 0.0 */
    bc_inlet_config_t config = bc_inlet_config_time_ramp(
        2.0, 0.0,     /* base velocity */
        0.5, 1.5,     /* t_start, t_end */
        0.0, 1.0);    /* value_start, value_end */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.2, 0.01};  /* t < t_start */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 0.0 = 0.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_ramp_at_midpoint(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Ramp from t=0.5 to t=1.5, value from 0 to 1 */
    /* At t=1.0 (midpoint): frac = (1.0-0.5)/(1.5-0.5) = 0.5 */
    /* modulator = 0.0 + 0.5 * (1.0 - 0.0) = 0.5 */
    bc_inlet_config_t config = bc_inlet_config_time_ramp(
        2.0, 0.0,     /* base velocity */
        0.5, 1.5,     /* t_start, t_end */
        0.0, 1.0);    /* value_start, value_end */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {1.0, 0.01};  /* t = midpoint */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 0.5 = 1.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 1.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_ramp_after_end(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Ramp from t=0.5 to t=1.5, value from 0 to 1 */
    /* After t=1.5: modulator = 1.0 */
    bc_inlet_config_t config = bc_inlet_config_time_ramp(
        2.0, 0.0,     /* base velocity */
        0.5, 1.5,     /* t_start, t_end */
        0.0, 1.0);    /* value_start, value_end */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {2.0, 0.01};  /* t > t_end */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 1.0 = 2.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Step Profile Tests
 * ============================================================================ */

void test_inlet_time_step_before(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Step at t=0.5, value_before=0.0, value_after=1.0 */
    bc_inlet_config_t config = bc_inlet_config_time_step(
        2.0, 0.0,     /* base velocity */
        0.5,          /* t_step */
        0.0,          /* value_before */
        1.0);         /* value_after */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.3, 0.01};  /* t < t_step */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 0.0 = 0.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_step_after(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Step at t=0.5, value_before=0.0, value_after=1.0 */
    bc_inlet_config_t config = bc_inlet_config_time_step(
        2.0, 0.0,     /* base velocity */
        0.5,          /* t_step */
        0.0,          /* value_before */
        1.0);         /* value_after */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.7, 0.01};  /* t >= t_step */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 1.0 = 2.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_step_at_transition(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Step at t=0.5, value_before=0.0, value_after=1.0 */
    /* At exactly t=0.5: t >= t_step, so value_after */
    bc_inlet_config_t config = bc_inlet_config_time_step(
        2.0, 0.0,     /* base velocity */
        0.5,          /* t_step */
        0.0,          /* value_before */
        1.0);         /* value_after */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.5, 0.01};  /* t == t_step */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Expected: u = 2.0 * 1.0 = 2.0 (at transition uses value_after) */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Custom Time Callback Tests
 * ============================================================================ */

void test_inlet_time_custom_callback(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Custom callback: velocity varies with both position and time */
    bc_inlet_config_t config = bc_inlet_config_time_custom(custom_time_profile, NULL);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    double time = 0.125;  /* cos(2*pi*0.125) = cos(pi/4) = sqrt(2)/2 */
    bc_time_context_t ctx = {time, 0.01};
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify velocity at each position */
    double cos_val = cos(2.0 * M_PI * time);
    for (size_t j = 0; j < ny; j++) {
        double position = (ny > 1) ? (double)j / (double)(ny - 1) : 0.5;
        double expected_u = sin(M_PI * position) * cos_val;
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 0.0, v[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Backward Compatibility Tests
 * ============================================================================ */

void test_inlet_time_constant_profile_matches_standard(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u_standard = create_test_field(nx, ny);
    double* v_standard = create_test_field(nx, ny);
    double* u_time = create_test_field(nx, ny);
    double* v_time = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u_standard);
    TEST_ASSERT_NOT_NULL(v_standard);
    TEST_ASSERT_NOT_NULL(u_time);
    TEST_ASSERT_NOT_NULL(v_time);

    /* Standard inlet without time variation */
    bc_inlet_config_t config_standard = bc_inlet_config_uniform(2.0, 0.5);
    bc_inlet_set_edge(&config_standard, BC_EDGE_LEFT);

    /* Time-varying with CONSTANT profile (should match standard) */
    bc_inlet_config_t config_time = bc_inlet_config_uniform(2.0, 0.5);
    bc_inlet_set_edge(&config_time, BC_EDGE_LEFT);
    /* time_config.profile is BC_TIME_PROFILE_CONSTANT by default (zero-initialized) */

    cfd_status_t status1 = bc_apply_inlet_cpu(u_standard, v_standard, nx, ny, &config_standard);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    bc_time_context_t ctx = {1.0, 0.01};  /* Arbitrary time */
    cfd_status_t status2 = bc_apply_inlet_time_cpu(u_time, v_time, nx, ny, &config_time, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status2);

    /* Results should be identical */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, u_standard[j * nx], u_time[j * nx]);
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, v_standard[j * nx], v_time[j * nx]);
    }

    free(u_standard);
    free(v_standard);
    free(u_time);
    free(v_time);
}

void test_inlet_time_dispatch_constant_delegates_to_standard(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Config with CONSTANT profile should delegate to standard inlet */
    bc_inlet_config_t config = bc_inlet_config_uniform(2.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {1.0, 0.01};
    cfd_status_t status = bc_apply_inlet_time(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Should have applied uniform velocity */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Combined Spatial and Time Profiles Tests
 * ============================================================================ */

void test_inlet_time_parabolic_with_sinusoidal(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    /* Parabolic spatial profile with sinusoidal time modulation */
    bc_inlet_config_t config = bc_inlet_config_parabolic(2.0);  /* max_velocity = 2.0 */
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);
    bc_inlet_set_time_sinusoidal(&config, 1.0, 0.5, 0.0, 1.0);  /* modulator = 1 + 0.5*sin(...) */

    /* At t=0.25 (T/4): sin(2*pi*0.25) = 1, modulator = 1.5 */
    bc_time_context_t ctx = {0.25, 0.01};
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    double modulator = 1.0 + 0.5 * 1.0;
    for (size_t j = 0; j < ny; j++) {
        double position = (ny > 1) ? (double)j / (double)(ny - 1) : 0.5;
        double parabolic = 4.0 * position * (1.0 - position);
        double expected_u = 2.0 * parabolic * modulator;
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected_u, u[j * nx]);
    }

    free(u);
    free(v);
}

/* ============================================================================
 * Edge Case Tests
 * ============================================================================ */

void test_inlet_time_null_time_context(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0, 1.0, 0.5, 0.0, 1.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    /* NULL time_ctx should use t=0, dt=0 as default */
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, &config, NULL);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* At t=0: modulator = 1.0 + 0.5 * sin(0) = 1.0 */
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, 2.0, u[j * nx]);
    }

    free(u);
    free(v);
}

void test_inlet_time_null_fields(void) {
    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);
    bc_time_context_t ctx = {0.0, 0.01};

    cfd_status_t status = bc_apply_inlet_time_cpu(NULL, NULL, 10, 10, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
}

void test_inlet_time_null_config(void) {
    size_t nx = 10, ny = 10;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_time_context_t ctx = {0.0, 0.01};
    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, nx, ny, NULL, &ctx);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(u);
    free(v);
}

void test_inlet_time_too_small_grid(void) {
    double* u = create_test_field(2, 2);
    double* v = create_test_field(2, 2);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);
    bc_time_context_t ctx = {0.0, 0.01};

    cfd_status_t status = bc_apply_inlet_time_cpu(u, v, 2, 2, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);

    free(u);
    free(v);
}

/* ============================================================================
 * Main Dispatch Tests
 * ============================================================================ */

void test_inlet_time_main_dispatch(void) {
    size_t nx = TEST_NX, ny = TEST_NY;
    double* u = create_test_field(nx, ny);
    double* v = create_test_field(nx, ny);
    TEST_ASSERT_NOT_NULL(u);
    TEST_ASSERT_NOT_NULL(v);

    bc_inlet_config_t config = bc_inlet_config_time_sinusoidal(
        2.0, 0.0, 1.0, 0.5, 0.0, 1.0);
    bc_inlet_set_edge(&config, BC_EDGE_LEFT);

    bc_time_context_t ctx = {0.25, 0.01};
    cfd_status_t status = bc_apply_inlet_time(u, v, nx, ny, &config, &ctx);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* At t=0.25: modulator = 1.0 + 0.5 * 1.0 = 1.5 */
    double expected = 2.0 * 1.5;
    for (size_t j = 0; j < ny; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOLERANCE, expected, u[j * nx]);
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
    RUN_TEST(test_inlet_config_time_sinusoidal);
    RUN_TEST(test_inlet_config_time_ramp);
    RUN_TEST(test_inlet_config_time_step);
    RUN_TEST(test_inlet_config_time_custom);

    /* Setter function tests */
    RUN_TEST(test_inlet_set_time_sinusoidal);
    RUN_TEST(test_inlet_set_time_ramp);
    RUN_TEST(test_inlet_set_time_step);

    /* Sinusoidal profile tests */
    RUN_TEST(test_inlet_time_sinusoidal_at_t_zero);
    RUN_TEST(test_inlet_time_sinusoidal_at_quarter_period);
    RUN_TEST(test_inlet_time_sinusoidal_at_half_period);
    RUN_TEST(test_inlet_time_sinusoidal_at_three_quarter_period);

    /* Ramp profile tests */
    RUN_TEST(test_inlet_time_ramp_before_start);
    RUN_TEST(test_inlet_time_ramp_at_midpoint);
    RUN_TEST(test_inlet_time_ramp_after_end);

    /* Step profile tests */
    RUN_TEST(test_inlet_time_step_before);
    RUN_TEST(test_inlet_time_step_after);
    RUN_TEST(test_inlet_time_step_at_transition);

    /* Custom callback tests */
    RUN_TEST(test_inlet_time_custom_callback);

    /* Backward compatibility tests */
    RUN_TEST(test_inlet_time_constant_profile_matches_standard);
    RUN_TEST(test_inlet_time_dispatch_constant_delegates_to_standard);

    /* Combined profiles tests */
    RUN_TEST(test_inlet_time_parabolic_with_sinusoidal);

    /* Edge case tests */
    RUN_TEST(test_inlet_time_null_time_context);
    RUN_TEST(test_inlet_time_null_fields);
    RUN_TEST(test_inlet_time_null_config);
    RUN_TEST(test_inlet_time_too_small_grid);

    /* Main dispatch tests */
    RUN_TEST(test_inlet_time_main_dispatch);

    return UNITY_END();
}
