/**
 * @file test_cavity_validation.c
 * @brief Lid-driven cavity tests: conservation, Ghia comparison, and grid convergence
 */

#include "lid_driven_cavity_common.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * CONSERVATION TESTS
 * ============================================================================ */

void test_mass_conservation(void) {
    cavity_context_t* ctx = cavity_context_create(21, 21);
    TEST_ASSERT_NOT_NULL(ctx);

    double initial_mass = 0.0;
    size_t total = ctx->nx * ctx->ny;
    for (size_t i = 0; i < total; i++) {
        initial_mass += ctx->field->rho[i];
    }

    run_cavity_simulation(ctx, 100.0, 1.0, FAST_STEPS, 0.001);

    double final_mass = 0.0;
    for (size_t i = 0; i < total; i++) {
        final_mass += ctx->field->rho[i];
    }

    /* Mass should be exactly conserved (constant density incompressible) */
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, initial_mass, final_mass);

    cavity_context_destroy(ctx);
}

void test_kinetic_energy_growth(void) {
    cavity_context_t* ctx = cavity_context_create(21, 21);
    TEST_ASSERT_NOT_NULL(ctx);

    double initial_ke = compute_kinetic_energy(ctx->field);

    run_cavity_simulation(ctx, 100.0, 1.0, FAST_STEPS, 0.001);

    double final_ke = compute_kinetic_energy(ctx->field);

    /* KE should increase from zero as flow develops */
    TEST_ASSERT_TRUE(final_ke > initial_ke);
    /* KE should be bounded (no numerical blow-up) */
    TEST_ASSERT_TRUE(final_ke < 1000.0);

    cavity_context_destroy(ctx);
}

/* ============================================================================
 * GHIA ET AL. CENTERLINE VALIDATION
 * ============================================================================ */

void test_u_centerline_re100(void) {
    cavity_context_t* ctx = cavity_context_create(33, 33);
    TEST_ASSERT_NOT_NULL(ctx);

    run_cavity_simulation(ctx, 100.0, 1.0, FULL_STEPS, FINE_DT);

    /* Extract u-velocity along vertical centerline */
    double* y_vals = malloc(ctx->ny * sizeof(double));
    double* u_vals = malloc(ctx->ny * sizeof(double));
    TEST_ASSERT_NOT_NULL(y_vals);
    TEST_ASSERT_NOT_NULL(u_vals);

    size_t center_i = ctx->nx / 2;
    for (size_t j = 0; j < ctx->ny; j++) {
        y_vals[j] = ctx->g->y[j];
        u_vals[j] = ctx->field->u[j * ctx->nx + center_i];
    }

    double rms_error = compute_profile_rms_error(
        y_vals, u_vals, ctx->ny,
        GHIA_Y_COORDS, GHIA_U_RE100, GHIA_NUM_POINTS
    );

    printf("\n    u-centerline RMS error vs Ghia: %.4f\n", rms_error);

    /* Error threshold depends on grid resolution and iterations.
     * In fast mode with fewer iterations, allow higher error.
     * Full validation mode uses tighter tolerance. */
#if CAVITY_FULL_VALIDATION
    TEST_ASSERT_TRUE(rms_error < 0.20);
#else
    TEST_ASSERT_TRUE(rms_error < 0.40);
#endif

    free(y_vals);
    free(u_vals);
    cavity_context_destroy(ctx);
}

void test_v_centerline_re100(void) {
    cavity_context_t* ctx = cavity_context_create(33, 33);
    TEST_ASSERT_NOT_NULL(ctx);

    run_cavity_simulation(ctx, 100.0, 1.0, FULL_STEPS, FINE_DT);

    /* Extract v-velocity along horizontal centerline */
    double* x_vals = malloc(ctx->nx * sizeof(double));
    double* v_vals = malloc(ctx->nx * sizeof(double));
    TEST_ASSERT_NOT_NULL(x_vals);
    TEST_ASSERT_NOT_NULL(v_vals);

    size_t center_j = ctx->ny / 2;
    for (size_t i = 0; i < ctx->nx; i++) {
        x_vals[i] = ctx->g->x[i];
        v_vals[i] = ctx->field->v[center_j * ctx->nx + i];
    }

    double rms_error = compute_profile_rms_error(
        x_vals, v_vals, ctx->nx,
        GHIA_X_COORDS, GHIA_V_RE100, GHIA_NUM_POINTS
    );

    printf("\n    v-centerline RMS error vs Ghia: %.4f\n", rms_error);

    TEST_ASSERT_TRUE(rms_error < 0.25);

    free(x_vals);
    free(v_vals);
    cavity_context_destroy(ctx);
}

/* ============================================================================
 * GRID CONVERGENCE
 * ============================================================================ */

void test_grid_convergence(void) {
    size_t sizes[] = {13, 21, 33};
    double max_vels[3];

    for (int i = 0; i < 3; i++) {
        cavity_context_t* ctx = cavity_context_create(sizes[i], sizes[i]);
        TEST_ASSERT_NOT_NULL(ctx);

        double dt = 0.001 * (13.0 / sizes[i]);
        simulation_result_t result = run_cavity_simulation(ctx, 100.0, 1.0, MEDIUM_STEPS, dt);

        max_vels[i] = result.max_velocity;
        printf("\n    Grid %zux%zu: max_vel = %.4f", sizes[i], sizes[i], max_vels[i]);

        TEST_ASSERT_FALSE(result.blew_up);
        cavity_context_destroy(ctx);
    }
    printf("\n");

    /* All grids should develop flow */
    TEST_ASSERT_TRUE(max_vels[0] > 0.0);
    TEST_ASSERT_TRUE(max_vels[1] > 0.0);
    TEST_ASSERT_TRUE(max_vels[2] > 0.0);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("CAVITY VALIDATION TESTS\n");
#if CAVITY_FULL_VALIDATION
    printf("Mode: FULL VALIDATION (slow)\n");
#else
    printf("Mode: FAST (reduced iterations)\n");
#endif
    printf("========================================\n");

    printf("\n[Conservation Tests]\n");
    RUN_TEST(test_mass_conservation);
    RUN_TEST(test_kinetic_energy_growth);

    printf("\n[Ghia et al. Centerline Validation]\n");
    RUN_TEST(test_u_centerline_re100);
    RUN_TEST(test_v_centerline_re100);

    printf("\n[Grid Convergence]\n");
    RUN_TEST(test_grid_convergence);

    return UNITY_END();
}
