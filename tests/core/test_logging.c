#include "cfd/core/cfd_status.h"
#include "cfd/core/logging.h"
#include "unity.h"
#include <string.h>


// Mock callback state (legacy per-thread callback)
static int s_callback_called = 0;
static cfd_log_level_t s_last_level;
static char s_last_message[256];

// Mock extended callback state
static int s_ex_callback_called = 0;
static cfd_log_level_t s_ex_last_level;
static char s_ex_last_component[64];
static char s_ex_last_message[256];

void setUp(void) {
    s_callback_called = 0;
    s_last_level = (cfd_log_level_t)-1;
    memset(s_last_message, 0, sizeof(s_last_message));

    s_ex_callback_called = 0;
    s_ex_last_level = (cfd_log_level_t)-1;
    memset(s_ex_last_component, 0, sizeof(s_ex_last_component));
    memset(s_ex_last_message, 0, sizeof(s_ex_last_message));

    cfd_set_log_callback(NULL);
    cfd_set_log_callback_ex(NULL);
    cfd_set_log_level(CFD_LOG_LEVEL_INFO);  // Reset to default
    cfd_clear_error();
}

void tearDown(void) {
    cfd_set_log_callback(NULL);
    cfd_set_log_callback_ex(NULL);
    cfd_set_log_level(CFD_LOG_LEVEL_INFO);
}

// Mock callback function (legacy)
void mock_log_callback(cfd_log_level_t level, const char* message) {
    s_callback_called++;
    s_last_level = level;
    if (message) {
        snprintf(s_last_message, sizeof(s_last_message), "%s", message);
    }
}

// Mock extended callback function
void mock_log_callback_ex(cfd_log_level_t level, const char* component, const char* message) {
    s_ex_callback_called++;
    s_ex_last_level = level;
    if (component) {
        snprintf(s_ex_last_component, sizeof(s_ex_last_component), "%s", component);
    } else {
        s_ex_last_component[0] = '\0';
    }
    if (message) {
        snprintf(s_ex_last_message, sizeof(s_ex_last_message), "%s", message);
    }
}

/* ============================================================================
 * LEGACY API TESTS (backward compatibility)
 * ============================================================================ */

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

/* ============================================================================
 * cfd_log() TESTS
 * ============================================================================ */

void test_cfd_log_basic(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_log(CFD_LOG_LEVEL_WARNING, "test", "hello %s", "world");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_WARNING, s_last_level);
    TEST_ASSERT_EQUAL_STRING("hello world", s_last_message);
}

void test_cfd_log_formatting(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_log(CFD_LOG_LEVEL_INFO, "solver", "iter=%d residual=%.2e grid=%zux%zu", 42, 1.5e-6,
            (size_t)33, (size_t)33);

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_STRING("iter=42 residual=1.50e-06 grid=33x33", s_last_message);
}

void test_cfd_log_null_component(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_log(CFD_LOG_LEVEL_INFO, NULL, "no component");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_STRING("no component", s_last_message);
}

/* ============================================================================
 * LOG LEVEL FILTERING TESTS
 * ============================================================================ */

void test_log_level_default_suppresses_debug(void) {
    cfd_set_log_callback(mock_log_callback);

    // Default level is INFO, so DEBUG should be suppressed
    CFD_LOG_DEBUG("test", "should not appear");
    TEST_ASSERT_EQUAL_INT(0, s_callback_called);

    // INFO should pass
    CFD_LOG_INFO("test", "should appear");
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
}

void test_log_level_filtering_suppresses(void) {
    cfd_set_log_callback(mock_log_callback);
    cfd_set_log_level(CFD_LOG_LEVEL_WARNING);

    // DEBUG and INFO should be suppressed
    cfd_log(CFD_LOG_LEVEL_DEBUG, "test", "debug msg");
    TEST_ASSERT_EQUAL_INT(0, s_callback_called);

    cfd_log(CFD_LOG_LEVEL_INFO, "test", "info msg");
    TEST_ASSERT_EQUAL_INT(0, s_callback_called);
}

void test_log_level_filtering_passes(void) {
    cfd_set_log_callback(mock_log_callback);
    cfd_set_log_level(CFD_LOG_LEVEL_WARNING);

    // WARNING and ERROR should pass
    cfd_log(CFD_LOG_LEVEL_WARNING, "test", "warn msg");
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);

    cfd_log(CFD_LOG_LEVEL_ERROR, "test", "error msg");
    TEST_ASSERT_EQUAL_INT(2, s_callback_called);
}

void test_log_level_get_set(void) {
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_INFO, cfd_get_log_level());

    cfd_set_log_level(CFD_LOG_LEVEL_DEBUG);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_DEBUG, cfd_get_log_level());

    cfd_set_log_level(CFD_LOG_LEVEL_ERROR);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_ERROR, cfd_get_log_level());
}

/* ============================================================================
 * EXTENDED CALLBACK TESTS
 * ============================================================================ */

void test_extended_callback(void) {
    cfd_set_log_callback_ex(mock_log_callback_ex);

    cfd_log(CFD_LOG_LEVEL_INFO, "mycomp", "test message");

    TEST_ASSERT_EQUAL_INT(1, s_ex_callback_called);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_INFO, s_ex_last_level);
    TEST_ASSERT_EQUAL_STRING("mycomp", s_ex_last_component);
    TEST_ASSERT_EQUAL_STRING("test message", s_ex_last_message);
}

void test_per_thread_callback_takes_priority(void) {
    cfd_set_log_callback(mock_log_callback);
    cfd_set_log_callback_ex(mock_log_callback_ex);

    cfd_log(CFD_LOG_LEVEL_INFO, "comp", "msg");

    // Per-thread callback should be called, NOT extended
    TEST_ASSERT_EQUAL_INT(1, s_callback_called);
    TEST_ASSERT_EQUAL_INT(0, s_ex_callback_called);

    // Clear per-thread callback — extended should now be called
    cfd_set_log_callback(NULL);
    cfd_log(CFD_LOG_LEVEL_INFO, "comp", "msg2");

    TEST_ASSERT_EQUAL_INT(1, s_callback_called);  // unchanged
    TEST_ASSERT_EQUAL_INT(1, s_ex_callback_called);
    TEST_ASSERT_EQUAL_STRING("msg2", s_ex_last_message);
}

/* ============================================================================
 * ERROR STATE PRESERVATION TESTS
 * ============================================================================ */

void test_cfd_error_preserves_error_state(void) {
    cfd_set_log_callback(mock_log_callback);

    cfd_error("test error");

    TEST_ASSERT_EQUAL_INT(CFD_ERROR, cfd_get_last_status());
    TEST_ASSERT_EQUAL_STRING("test error", cfd_get_last_error());
}

/* ============================================================================
 * CONVENIENCE MACRO TESTS
 * ============================================================================ */

void test_convenience_macros(void) {
    cfd_set_log_callback(mock_log_callback);

    CFD_LOG_ERROR("comp", "val=%d", 42);
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_ERROR, s_last_level);
    TEST_ASSERT_EQUAL_STRING("val=42", s_last_message);

    CFD_LOG_WARNING("comp", "warn");
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_WARNING, s_last_level);

    CFD_LOG_INFO("comp", "info");
    TEST_ASSERT_EQUAL_INT(CFD_LOG_LEVEL_INFO, s_last_level);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    // Legacy API tests
    RUN_TEST(test_error_logging_callback);
    RUN_TEST(test_warning_logging_callback);
    RUN_TEST(test_info_logging_callback);
    RUN_TEST(test_null_handling);
    RUN_TEST(test_callback_reset);

    // cfd_log() tests
    RUN_TEST(test_cfd_log_basic);
    RUN_TEST(test_cfd_log_formatting);
    RUN_TEST(test_cfd_log_null_component);

    // Log level filtering tests
    RUN_TEST(test_log_level_default_suppresses_debug);
    RUN_TEST(test_log_level_filtering_suppresses);
    RUN_TEST(test_log_level_filtering_passes);
    RUN_TEST(test_log_level_get_set);

    // Extended callback tests
    RUN_TEST(test_extended_callback);
    RUN_TEST(test_per_thread_callback_takes_priority);

    // Error state preservation
    RUN_TEST(test_cfd_error_preserves_error_state);

    // Convenience macros
    RUN_TEST(test_convenience_macros);

    return UNITY_END();
}
