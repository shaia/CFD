#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "unity.h"


#include <stdio.h>
#include <string.h>


// Mock function to test status return
cfd_status_t mock_allocator(size_t size) {
    if (size > 1024 * 1024 * 1024) {  // 1GB
        // Simulate failure
        cfd_set_error(CFD_ERROR_NOMEM, "Mock allocation failure");
        return CFD_ERROR_NOMEM;
    }
    return CFD_SUCCESS;
}

void setUp(void) {
    cfd_clear_error();
}

void tearDown(void) {
    cfd_clear_error();
}

void test_set_get_error(void) {
    TEST_ASSERT_NULL(cfd_get_last_error());

    cfd_set_error(CFD_ERROR, "Test error message");

    const char* err = cfd_get_last_error();
    TEST_ASSERT_NOT_NULL(err);
    TEST_ASSERT_EQUAL_STRING("Test error message", err);
    TEST_ASSERT_EQUAL(CFD_ERROR, cfd_get_last_status());
}

void test_clear_error(void) {
    cfd_set_error(CFD_ERROR, "Test error message");
    TEST_ASSERT_NOT_NULL(cfd_get_last_error());

    cfd_clear_error();
    TEST_ASSERT_NULL(cfd_get_last_error());
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_get_last_status());
}

void test_overwrite_error(void) {
    cfd_set_error(CFD_ERROR, "Initial error");
    cfd_set_error(CFD_ERROR_NOMEM, "New error");

    TEST_ASSERT_EQUAL_STRING("New error", cfd_get_last_error());
    TEST_ASSERT_EQUAL(CFD_ERROR_NOMEM, cfd_get_last_status());
}

void test_mock_allocation_failure(void) {
    cfd_status_t status = mock_allocator(2000000000);  // > 1GB

    TEST_ASSERT_EQUAL(CFD_ERROR_NOMEM, status);
    TEST_ASSERT_EQUAL_STRING("Mock allocation failure", cfd_get_last_error());
}

void test_error_string_mapping(void) {
    TEST_ASSERT_EQUAL_STRING("Success", cfd_get_error_string(CFD_SUCCESS));
    TEST_ASSERT_EQUAL_STRING("Generic error", cfd_get_error_string(CFD_ERROR));
    TEST_ASSERT_EQUAL_STRING("Out of memory", cfd_get_error_string(CFD_ERROR_NOMEM));
    TEST_ASSERT_EQUAL_STRING("Invalid argument", cfd_get_error_string(CFD_ERROR_INVALID));
    TEST_ASSERT_EQUAL_STRING("I/O error", cfd_get_error_string(CFD_ERROR_IO));
    TEST_ASSERT_EQUAL_STRING("Operation not supported",
                             cfd_get_error_string(CFD_ERROR_UNSUPPORTED));
    TEST_ASSERT_EQUAL_STRING("Solver diverged", cfd_get_error_string(CFD_ERROR_DIVERGED));
    TEST_ASSERT_EQUAL_STRING("Max iterations reached", cfd_get_error_string(CFD_ERROR_MAX_ITER));
    TEST_ASSERT_EQUAL_STRING("Unknown error", cfd_get_error_string((cfd_status_t)999));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_set_get_error);
    RUN_TEST(test_clear_error);
    RUN_TEST(test_overwrite_error);
    RUN_TEST(test_mock_allocation_failure);
    RUN_TEST(test_error_string_mapping);
    return UNITY_END();
}
