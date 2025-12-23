/**
 * @file test_cavity_flow.c
 * @brief Lid-driven cavity tests: flow development and numerical stability
 */

#include "lid_driven_cavity_common.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * FLOW DEVELOPMENT TESTS
 * ============================================================================ */

void test_flow_develops(void) {
    cavity_context_t* ctx = cavity_context_create(17, 17);
    TEST_ASSERT_NOT_NULL(ctx);

    /* Use 500 steps for fast CI execution */
    simulation_result_t result = run_cavity_simulation(ctx, 100.0, 1.0, 500, 0.001);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_TRUE(result.max_velocity > 0.01);
    TEST_ASSERT_TRUE(result.steps_completed > 0);

    cavity_context_destroy(ctx);
}

void test_vortex_circulation(void) {
    cavity_context_t* ctx = cavity_context_create(17, 17);
    TEST_ASSERT_NOT_NULL(ctx);

    /* Use 500 steps for fast CI execution */
    run_cavity_simulation(ctx, 100.0, 1.0, 500, 0.001);

    /* Near lid (just below top), u should be positive (flow with lid) */
    size_t near_lid = (ctx->ny - 3) * ctx->nx + ctx->nx / 2;
    double u_near_lid = ctx->field->u[near_lid];
    TEST_ASSERT_TRUE_MESSAGE(u_near_lid > 0.0,
                             "Flow near lid should move with lid (positive u)");

    /* Verify the velocity field has developed a gradient
     * The lid drags fluid, so velocity should decrease moving away from lid */
    size_t mid_height = (ctx->ny / 2) * ctx->nx + ctx->nx / 2;
    double u_mid = ctx->field->u[mid_height];
    TEST_ASSERT_TRUE_MESSAGE(u_mid < u_near_lid,
                             "Velocity gradient should exist (u decreases away from lid)");

    cavity_context_destroy(ctx);
}

void test_quiescent_with_zero_lid(void) {
    cavity_context_t* ctx = cavity_context_create(16, 16);
    TEST_ASSERT_NOT_NULL(ctx);

    simulation_result_t result = run_cavity_simulation(ctx, 100.0, 0.0, 200, 0.001);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, result.max_velocity);

    cavity_context_destroy(ctx);
}

void test_high_lid_velocity(void) {
    cavity_context_t* ctx = cavity_context_create(17, 17);
    TEST_ASSERT_NOT_NULL(ctx);

    /* High velocity lid test - use 300 steps for fast CI */
    simulation_result_t result = run_cavity_simulation(ctx, 100.0, 5.0, 300, 0.0005);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_TRUE(result.max_velocity > 0.1);

    cavity_context_destroy(ctx);
}

/* ============================================================================
 * NUMERICAL STABILITY TESTS
 * ============================================================================ */

void test_stability_re100(void) {
    cavity_context_t* ctx = cavity_context_create(21, 21);
    TEST_ASSERT_NOT_NULL(ctx);

    /* Use 500 steps for fast CI */
    simulation_result_t result = run_cavity_simulation(ctx, 100.0, 1.0, 500, 0.001);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_TRUE(check_field_finite(ctx->field));
    TEST_ASSERT_TRUE(result.max_velocity < 10.0);

    cavity_context_destroy(ctx);
}

void test_stability_re400(void) {
    cavity_context_t* ctx = cavity_context_create(25, 25);
    TEST_ASSERT_NOT_NULL(ctx);

    /* Re=400 needs smaller timestep - use 500 steps for fast CI */
    simulation_result_t result = run_cavity_simulation(ctx, 400.0, 1.0, 500, 0.0002);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_TRUE(check_field_finite(ctx->field));
    TEST_ASSERT_TRUE(result.max_velocity < 10.0);

    cavity_context_destroy(ctx);
}

void test_small_grid_stability(void) {
    cavity_context_t* ctx = cavity_context_create(10, 10);
    TEST_ASSERT_NOT_NULL(ctx);

    /* Use 300 steps for fast CI */
    simulation_result_t result = run_cavity_simulation(ctx, 100.0, 1.0, 300, 0.001);

    TEST_ASSERT_FALSE(result.blew_up);
    TEST_ASSERT_TRUE(check_field_finite(ctx->field));

    cavity_context_destroy(ctx);
}

/* ============================================================================
 * REYNOLDS NUMBER DEPENDENCY
 * ============================================================================ */

void test_reynolds_dependency(void) {
    double reynolds[] = {50.0, 100.0, 200.0};
    double max_vels[3];

    for (int i = 0; i < 3; i++) {
        cavity_context_t* ctx = cavity_context_create(17, 17);
        TEST_ASSERT_NOT_NULL(ctx);

        /* Use 400 steps for fast CI */
        simulation_result_t result = run_cavity_simulation(ctx, reynolds[i], 1.0, 400, 0.001);

        max_vels[i] = result.max_velocity;
        printf("\n    Re=%.0f: max_vel = %.4f", reynolds[i], max_vels[i]);

        TEST_ASSERT_FALSE(result.blew_up);
        cavity_context_destroy(ctx);
    }
    printf("\n");

    /* Flow should develop at all Re */
    TEST_ASSERT_TRUE(max_vels[0] > 0.01);
    TEST_ASSERT_TRUE(max_vels[1] > 0.01);
    TEST_ASSERT_TRUE(max_vels[2] > 0.01);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n[Flow Development Tests]\n");
    RUN_TEST(test_flow_develops);
    RUN_TEST(test_vortex_circulation);
    RUN_TEST(test_quiescent_with_zero_lid);
    RUN_TEST(test_high_lid_velocity);

    printf("\n[Numerical Stability Tests]\n");
    RUN_TEST(test_stability_re100);
    RUN_TEST(test_stability_re400);
    RUN_TEST(test_small_grid_stability);

    printf("\n[Reynolds Number Dependency]\n");
    RUN_TEST(test_reynolds_dependency);

    return UNITY_END();
}
