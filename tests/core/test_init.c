#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "unity.h"


void setUp(void) {
    // Ensure we start fresh for each test if possible
    cfd_finalize();
}

void tearDown(void) {
    cfd_finalize();
}

void test_initialization_status(void) {
    TEST_ASSERT_FALSE(cfd_is_initialized());

    cfd_status_t status = cfd_init();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    TEST_ASSERT_TRUE(cfd_is_initialized());
}

void test_initialization_idempotency(void) {
    // First init
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_init());
    TEST_ASSERT_TRUE(cfd_is_initialized());

    // Second init - should be OK and still initialized
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_init());
    TEST_ASSERT_TRUE(cfd_is_initialized());
}

void test_finalize(void) {
    cfd_init();
    TEST_ASSERT_TRUE(cfd_is_initialized());

    cfd_finalize();
    TEST_ASSERT_FALSE(cfd_is_initialized());
}

void test_finalize_idempotency(void) {
    cfd_init();
    cfd_finalize();
    TEST_ASSERT_FALSE(cfd_is_initialized());

    // safe to call again
    cfd_finalize();
    TEST_ASSERT_FALSE(cfd_is_initialized());
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_initialization_status);
    RUN_TEST(test_initialization_idempotency);
    RUN_TEST(test_finalize);
    RUN_TEST(test_finalize_idempotency);
    return UNITY_END();
}
