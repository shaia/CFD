/**
 * @file test_poiseuille_3d.c
 * @brief 3D Poiseuille (channel) flow validation tests
 *
 * Extends the 2D Poiseuille validation to 3D using periodic BCs in z.
 * The analytical solution remains the 2D parabola u(y), independent of z.
 * This validates that 3D solver code paths produce correct results.
 *
 * Tests verify:
 *   - Velocity profile remains parabolic
 *   - Transverse velocities v and w remain near zero
 *   - Mass conservation across y-z planes
 *   - Pressure gradient matches analytical value
 *   - Inlet BC accuracy
 *   - Z-uniformity: solution is identical across z-planes
 */

#include "poiseuille_3d_reference.h"

void setUp(void) {}
void tearDown(void) {}

/* Cache simulation result — run once, check many metrics */
static pois3d_result_t s_result;
static int s_result_valid = 0;

static void ensure_simulation_run(void) {
    if (!s_result_valid) {
        printf("\n    Running 3D Poiseuille flow simulation...\n");
        printf("    Re=%.0f, nx=%d, ny=%d, nz=%d, dt=%.4f, steps=%d\n",
               POIS3D_RE, POIS3D_NX, POIS3D_NY, POIS3D_NZ, POIS3D_DT, POIS3D_STEPS);
        printf("    Periodic BCs in z, parabolic inlet, zero-gradient outlet\n");
        s_result = pois3d_run_simulation();
        s_result_valid = 1;

        if (s_result.success) {
            pois3d_print_result(&s_result);
        }
    }
}

/* ============================================================================
 * TEST: Velocity profile accuracy
 * ============================================================================ */

void test_3d_velocity_profile_accuracy(void) {
    printf("\n    Testing 3D velocity profile accuracy at x=75%%L...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Profile RMS error: %.6f (tolerance: %.4f)\n",
           s_result.profile_rms_error, POIS3D_PROFILE_RMS_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.profile_rms_error < POIS3D_PROFILE_RMS_TOL,
        "Velocity profile RMS error exceeds tolerance");
}

/* ============================================================================
 * TEST: Transverse velocity
 * ============================================================================ */

void test_3d_transverse_velocity(void) {
    printf("\n    Testing transverse velocities v and w...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Max |v|: %.6f (tolerance: %.4f)\n",
           s_result.max_v_magnitude, POIS3D_MAX_VW_TOL);
    printf("      Max |w|: %.6f (tolerance: %.4f)\n",
           s_result.max_w_magnitude, POIS3D_MAX_VW_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.max_v_magnitude < POIS3D_MAX_VW_TOL,
        "Transverse velocity |v| exceeds tolerance");
    TEST_ASSERT_TRUE_MESSAGE(s_result.max_w_magnitude < POIS3D_MAX_VW_TOL,
        "Transverse velocity |w| exceeds tolerance");
}

/* ============================================================================
 * TEST: Mass conservation
 * ============================================================================ */

void test_3d_mass_conservation(void) {
    printf("\n    Testing 3D mass conservation...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Mass flux in:  %.6f\n", s_result.mass_flux_in);
    printf("      Mass flux mid: %.6f\n", s_result.mass_flux_mid);
    printf("      Mass flux out: %.6f\n", s_result.mass_flux_out);

    double q_in = fabs(s_result.mass_flux_in);
    TEST_ASSERT_TRUE_MESSAGE(q_in > 1e-10, "Inlet mass flux is zero");

    double err_out = fabs(s_result.mass_flux_in - s_result.mass_flux_out) / q_in;
    double err_mid = fabs(s_result.mass_flux_in - s_result.mass_flux_mid) / q_in;

    printf("      |Q_in - Q_out|/Q_in: %.6f (tolerance: %.4f)\n",
           err_out, POIS3D_MASS_FLUX_TOL);
    printf("      |Q_in - Q_mid|/Q_in: %.6f (tolerance: %.4f)\n",
           err_mid, POIS3D_MASS_FLUX_TOL);

    TEST_ASSERT_TRUE_MESSAGE(err_out < POIS3D_MASS_FLUX_TOL,
        "Mass flux not conserved between inlet and outlet");
    TEST_ASSERT_TRUE_MESSAGE(err_mid < POIS3D_MASS_FLUX_TOL,
        "Mass flux not conserved between inlet and mid-channel");
}

/* ============================================================================
 * TEST: Pressure gradient
 * ============================================================================ */

void test_3d_pressure_gradient(void) {
    printf("\n    Testing 3D pressure gradient...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Measured dp/dx: %.6f\n", s_result.measured_dpdx);
    printf("      Expected dp/dx: %.6f\n", s_result.expected_dpdx);

    double expected_abs = fabs(s_result.expected_dpdx);
    TEST_ASSERT_TRUE_MESSAGE(expected_abs > 1e-10, "Expected dp/dx is zero");

    double rel_err = fabs(s_result.measured_dpdx - s_result.expected_dpdx) / expected_abs;
    printf("      Relative error: %.4f (tolerance: %.4f)\n",
           rel_err, POIS3D_PRESSURE_GRAD_TOL);

    TEST_ASSERT_TRUE_MESSAGE(rel_err < POIS3D_PRESSURE_GRAD_TOL,
        "Pressure gradient does not match analytical value");
}

/* ============================================================================
 * TEST: Inlet BC accuracy
 * ============================================================================ */

void test_3d_inlet_bc_accuracy(void) {
    printf("\n    Testing 3D inlet BC accuracy...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Inlet max error: %.2e (tolerance: %.2e)\n",
           s_result.inlet_max_error, POIS3D_INLET_BC_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.inlet_max_error < POIS3D_INLET_BC_TOL,
        "Inlet BC does not match analytical parabolic profile");
}

/* ============================================================================
 * TEST: Z-uniformity (key 3D-specific test)
 * ============================================================================ */

void test_3d_z_uniformity(void) {
    printf("\n    Testing z-uniformity (periodic z should give z-independent solution)...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Z-uniformity error: %.2e (tolerance: %.2e)\n",
           s_result.z_uniformity_error, POIS3D_Z_UNIFORMITY_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.z_uniformity_error < POIS3D_Z_UNIFORMITY_TOL,
        "Solution varies across z-planes (should be z-independent with periodic BCs)");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("3D POISEUILLE FLOW VALIDATION\n");
    printf("========================================\n");
    printf("\nAnalytical: u(y) = 4*U_max*(y/H)*(1-y/H), periodic in z\n");
    printf("Domain: [0, %.0f] x [0, %.0f] x [0, %.0f]\n",
           POIS3D_DOMAIN_LENGTH, POIS3D_CHANNEL_HEIGHT, POIS3D_CHANNEL_DEPTH);
    printf("Re = %.0f, U_max = %.1f, nu = %.3f\n", POIS3D_RE, POIS3D_U_MAX, POIS3D_NU);

    RUN_TEST(test_3d_velocity_profile_accuracy);
    RUN_TEST(test_3d_transverse_velocity);
    RUN_TEST(test_3d_mass_conservation);
    RUN_TEST(test_3d_pressure_gradient);
    RUN_TEST(test_3d_inlet_bc_accuracy);
    RUN_TEST(test_3d_z_uniformity);

    return UNITY_END();
}
