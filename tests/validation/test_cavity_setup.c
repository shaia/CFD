/**
 * @file test_cavity_setup.c
 * @brief Lid-driven cavity tests: setup and boundary conditions
 */

#include "lid_driven_cavity_common.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * BASIC SETUP TESTS
 * ============================================================================ */

void test_grid_creation(void) {
    grid* g = grid_create(32, 32, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_EQUAL_size_t(32, g->nx);
    TEST_ASSERT_EQUAL_size_t(32, g->ny);
    grid_destroy(g);
}

void test_flow_field_creation(void) {
    flow_field* field = flow_field_create(32, 32);
    TEST_ASSERT_NOT_NULL(field);
    TEST_ASSERT_NOT_NULL(field->u);
    TEST_ASSERT_NOT_NULL(field->v);
    TEST_ASSERT_NOT_NULL(field->p);
    flow_field_destroy(field);
}

void test_context_creation(void) {
    cavity_context_t* ctx = cavity_context_create(16, 16);
    TEST_ASSERT_NOT_NULL(ctx);
    TEST_ASSERT_NOT_NULL(ctx->g);
    TEST_ASSERT_NOT_NULL(ctx->field);
    cavity_context_destroy(ctx);
}

/* ============================================================================
 * BOUNDARY CONDITION TESTS
 * ============================================================================ */

void test_bc_lid_velocity(void) {
    cavity_context_t* ctx = cavity_context_create(16, 16);
    TEST_ASSERT_NOT_NULL(ctx);

    apply_cavity_bc(ctx->field, 1.0);

    /* Top row should have u = 1.0 */
    size_t top_j = ctx->ny - 1;
    for (size_t i = 0; i < ctx->nx; i++) {
        size_t idx = top_j * ctx->nx + i;
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, ctx->field->u[idx]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->v[idx]);
    }

    cavity_context_destroy(ctx);
}

void test_bc_walls_noslip(void) {
    cavity_context_t* ctx = cavity_context_create(16, 16);
    TEST_ASSERT_NOT_NULL(ctx);

    apply_cavity_bc(ctx->field, 1.0);

    /* Bottom wall: u = v = 0 */
    for (size_t i = 0; i < ctx->nx; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->u[i]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->v[i]);
    }

    /* Left wall: u = v = 0 (excluding top corner which has lid BC) */
    for (size_t j = 0; j < ctx->ny - 1; j++) {
        size_t idx = j * ctx->nx;
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->u[idx]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->v[idx]);
    }

    /* Right wall: u = v = 0 (excluding top corner which has lid BC) */
    for (size_t j = 0; j < ctx->ny - 1; j++) {
        size_t idx = j * ctx->nx + (ctx->nx - 1);
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->u[idx]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, ctx->field->v[idx]);
    }

    cavity_context_destroy(ctx);
}

void test_bc_various_velocities(void) {
    cavity_context_t* ctx = cavity_context_create(16, 16);
    TEST_ASSERT_NOT_NULL(ctx);

    double velocities[] = {0.5, 1.0, 2.0, 5.0};
    size_t top_j = ctx->ny - 1;

    for (size_t v = 0; v < 4; v++) {
        apply_cavity_bc(ctx->field, velocities[v]);

        size_t idx = top_j * ctx->nx + ctx->nx / 2;
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, velocities[v], ctx->field->u[idx]);
    }

    cavity_context_destroy(ctx);
}

void test_rectangular_domain(void) {
    /* 2:1 aspect ratio */
    grid* g = grid_create(32, 16, 0.0, 2.0, 0.0, 1.0);
    flow_field* field = flow_field_create(32, 16);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);

    grid_initialize_uniform(g);

    size_t total = 32 * 16;
    for (size_t i = 0; i < total; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->p[i] = 0.0;
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }

    apply_cavity_bc(field, 1.0);

    /* Check lid BC applied correctly */
    size_t top_j = 15;
    for (size_t i = 0; i < 32; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 1.0, field->u[top_j * 32 + i]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n[Setup Tests]\n");
    RUN_TEST(test_grid_creation);
    RUN_TEST(test_flow_field_creation);
    RUN_TEST(test_context_creation);

    printf("\n[Boundary Condition Tests]\n");
    RUN_TEST(test_bc_lid_velocity);
    RUN_TEST(test_bc_walls_noslip);
    RUN_TEST(test_bc_various_velocities);
    RUN_TEST(test_rectangular_domain);

    return UNITY_END();
}
