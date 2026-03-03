#ifndef CFD_TEST_MACROS_H
#define CFD_TEST_MACROS_H

#include "unity.h"

/**
 * TEST_FAIL_PRINTF(fmt, ...) — fail a test with a printf-style message.
 * This macro allows formatted failure messages without fixed-buffer truncation.
 * Usage:
 *   TEST_FAIL_PRINTF("step %d failed with status %d", step, status);
 *
 * When UNITY_INCLUDE_PRINT_FORMATTED is defined (set in CMakeLists.txt),
 * Unity's own TEST_PRINTF handles formatting with no fixed-buffer truncation.
 * The #else branch provides a fallback for standalone builds.
 */
#ifdef UNITY_INCLUDE_PRINT_FORMATTED
#  define TEST_FAIL_PRINTF(fmt, ...) \
     do { TEST_PRINTF(fmt, ##__VA_ARGS__); TEST_FAIL(); } while (0)
#else
#  define TEST_FAIL_PRINTF(fmt, ...) \
     do { \
         char _tfp_buf_[512]; \
         snprintf(_tfp_buf_, sizeof(_tfp_buf_), fmt, ##__VA_ARGS__); \
         TEST_FAIL_MESSAGE(_tfp_buf_); \
     } while (0)
#endif

#endif /* CFD_TEST_MACROS_H */
