#include "cfd/core/logging.h"
#include "unity.h"
#include <string.h>

// Mock callback state
static int s_callback_called = 0;
static cfd_log_level_t s_last_level;
static char s_last_message[256];

void setUp(void) {
    s_callback_called = 0;
    s_last_level = -1;
    memset(s_last_message, 0, sizeof(s_last_message));
    cfd_set_log_callback(NULL);  // Reset to default
}

void tearDown(void) {
    cfd_set_log_callback(NULL);
}

// Mock callback function
void mock_log_callback(cfd_log_level_t level, const char* message) {
    s_callback_called++;
    s_last_level = level;
    if (message) {
        snprintf(s_last_message, sizeof(s_last_message), "%s", message);
    }
}

void test_error_logging_callback(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_error("Test error message");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_ERROR, s_last_level);
    TEST_ASSERT_EQUAL_STRING("Test error message", s_last_message);

    // Verify error state is also set
    TEST_ASSERT_EQUAL_INT(CFD_ERROR, cfd_get_last_status());
    TEST_ASSERT_EQUAL_STRING("Test error message", cfd_get_last_error());
}

void test_warning_logging_callback(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_warning("Test warning message");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_WARNING, s_last_level);
    TEST_ASSERT_EQUAL_STRING("Test warning message", s_last_message);
}

void test_info_logging_callback(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_info("Test info message");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_INFO, s_last_level);
    TEST_ASSERT_EQUAL_STRING("Test info message", s_last_message);
}

void test_null_handling(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_error(NULL);
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_ERROR, s_last_level);
    TEST_ASSERT_EQUAL_STRING("(null)", s_last_message);

    cfd_warning(NULL);
    TEST_ASSERT_EQUAL_INT(2, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_WARNING, s_last_level);
    TEST_ASSERT_EQUAL_STRING("(null)", s_last_message);

    cfd_info(NULL);
    TEST_ASSERT_EQUAL_INT(3, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_INFO, s_last_level);
    TEST_ASSERT_EQUAL_STRING("(null)", s_last_message);
}

void test_callback_reset(void) {
    cfd_set_log_callback(mock_log_callback);
    cfd_warning("First");
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);

    cfd_set_log_callback(NULL);
    cfd_warning("Second");
    // Callback should NOT be called
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_error_logging_callback);
    RUN_TEST(test_warning_logging_callback);
    RUN_TEST(test_info_logging_callback);
    RUN_TEST(test_null_handling);
    RUN_TEST(test_callback_reset);
    return UNITY_END();
}
