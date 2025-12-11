#include "cfd/core/derived_fields.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"
#include "cfd/core/logging.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/math_utils.h"

#include <math.h>

// Test fixtures
static FlowField* test_field = NULL;
static const size_t NX = 10;
static const size_t NY = 10;

void setUp(void) {
    test_field = flow_field_create(NX, NY);

    // Initialize with known values for testing
    for (size_t i = 0; i < NX * NY; i++) {
        test_field->u[i] = 3.0;  // u = 3
        test_field->v[i] = 4.0;  // v = 4, so magnitude = 5
    }
}

void tearDown(void) {
    if (test_field) {
        flow_field_destroy(test_field);
        test_field = NULL;
    }
}

//=============================================================================
// DERIVED FIELDS LIFECYCLE TESTS
//=============================================================================

void test_derived_fields_create(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);

    TEST_ASSERT_NOT_NULL(derived);
    TEST_ASSERT_NULL(derived->velocity_magnitude);
    TEST_ASSERT_EQUAL(NX, derived->nx);
    TEST_ASSERT_EQUAL(NY, derived->ny);

    derived_fields_destroy(derived);
}

void test_derived_fields_destroy_null(void) {
    // Should not crash
    derived_fields_destroy(NULL);
}

void test_derived_fields_clear(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);
    derived_fields_compute_velocity_magnitude(derived, test_field);

    TEST_ASSERT_NOT_NULL(derived->velocity_magnitude);

    derived_fields_clear(derived);

    TEST_ASSERT_NULL(derived->velocity_magnitude);

    derived_fields_destroy(derived);
}

void test_derived_fields_clear_null(void) {
    // Should not crash
    derived_fields_clear(NULL);
}

//=============================================================================
// VELOCITY MAGNITUDE COMPUTATION TESTS
//=============================================================================

void test_velocity_magnitude_computation(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);
    derived_fields_compute_velocity_magnitude(derived, test_field);

    TEST_ASSERT_NOT_NULL(derived->velocity_magnitude);

    // Check that magnitude = sqrt(3^2 + 4^2) = 5 for all points
    for (size_t i = 0; i < NX * NY; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, (float)derived->velocity_magnitude[i]);
    }

    derived_fields_destroy(derived);
}

void test_velocity_magnitude_varying_values(void) {
    // Set up varying velocity field
    for (size_t i = 0; i < NX * NY; i++) {
        test_field->u[i] = (double)i;
        test_field->v[i] = (double)(NX * NY - i);
    }

    DerivedFields* derived = derived_fields_create(NX, NY);
    derived_fields_compute_velocity_magnitude(derived, test_field);

    // Verify each point
    for (size_t i = 0; i < NX * NY; i++) {
        double expected = sqrt((double)i * (double)i +
                               (double)(NX * NY - i) * (double)(NX * NY - i));
        TEST_ASSERT_FLOAT_WITHIN(1e-5f, (float)expected, (float)derived->velocity_magnitude[i]);
    }

    derived_fields_destroy(derived);
}

void test_velocity_magnitude_zero_velocity(void) {
    // Set velocity to zero
    for (size_t i = 0; i < NX * NY; i++) {
        test_field->u[i] = 0.0;
        test_field->v[i] = 0.0;
    }

    DerivedFields* derived = derived_fields_create(NX, NY);
    derived_fields_compute_velocity_magnitude(derived, test_field);

    for (size_t i = 0; i < NX * NY; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, (float)derived->velocity_magnitude[i]);
    }

    derived_fields_destroy(derived);
}

void test_velocity_magnitude_recompute(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);

    // First computation
    derived_fields_compute_velocity_magnitude(derived, test_field);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 5.0f, (float)derived->velocity_magnitude[0]);

    // Change velocity field
    for (size_t i = 0; i < NX * NY; i++) {
        test_field->u[i] = 5.0;
        test_field->v[i] = 12.0;  // magnitude = 13
    }

    // Recompute - should reuse allocated memory
    derived_fields_compute_velocity_magnitude(derived, test_field);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 13.0f, (float)derived->velocity_magnitude[0]);

    derived_fields_destroy(derived);
}

//=============================================================================
// NULL SAFETY TESTS
//=============================================================================

void test_velocity_magnitude_null_derived(void) {
    // Should not crash
    derived_fields_compute_velocity_magnitude(NULL, test_field);
}

void test_velocity_magnitude_null_field(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);

    // Should not crash
    derived_fields_compute_velocity_magnitude(derived, NULL);
    TEST_ASSERT_NULL(derived->velocity_magnitude);

    derived_fields_destroy(derived);
}

void test_velocity_magnitude_null_u(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);

    // Temporarily set u to NULL
    double* original_u = test_field->u;
    test_field->u = NULL;

    derived_fields_compute_velocity_magnitude(derived, test_field);
    TEST_ASSERT_NULL(derived->velocity_magnitude);

    test_field->u = original_u;
    derived_fields_destroy(derived);
}

void test_velocity_magnitude_null_v(void) {
    DerivedFields* derived = derived_fields_create(NX, NY);

    // Temporarily set v to NULL
    double* original_v = test_field->v;
    test_field->v = NULL;

    derived_fields_compute_velocity_magnitude(derived, test_field);
    TEST_ASSERT_NULL(derived->velocity_magnitude);

    test_field->v = original_v;
    derived_fields_destroy(derived);
}

int main(void) {
    UNITY_BEGIN();

    // Lifecycle tests
    RUN_TEST(test_derived_fields_create);
    RUN_TEST(test_derived_fields_destroy_null);
    RUN_TEST(test_derived_fields_clear);
    RUN_TEST(test_derived_fields_clear_null);

    // Velocity magnitude computation tests
    RUN_TEST(test_velocity_magnitude_computation);
    RUN_TEST(test_velocity_magnitude_varying_values);
    RUN_TEST(test_velocity_magnitude_zero_velocity);
    RUN_TEST(test_velocity_magnitude_recompute);

    // NULL safety tests
    RUN_TEST(test_velocity_magnitude_null_derived);
    RUN_TEST(test_velocity_magnitude_null_field);
    RUN_TEST(test_velocity_magnitude_null_u);
    RUN_TEST(test_velocity_magnitude_null_v);

    return UNITY_END();
}
