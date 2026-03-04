#ifndef CFD_TEST_MACROS_H
#define CFD_TEST_MACROS_H

#include <stdio.h>
#include "unity.h"

/**
 * TEST_FAIL_PRINTF(fmt, ...) — fail a test with a printf-style message.
 * Usage:
 *   TEST_FAIL_PRINTF("step %d failed with status %d", step, status);
 *
 * When UNITY_INCLUDE_PRINT_FORMATTED is defined (set in CMakeLists.txt),
 * Unity's own TEST_PRINTF handles formatting.
 * The #else branch provides a fallback using a 512-byte stack buffer.
 *
 * Uses an argument-counting dispatcher instead of ##__VA_ARGS__ for
 * C99/MSVC portability (##__VA_ARGS__ is a GNU extension).
 *
 * Limit: supports fmt plus up to 7 additional arguments (8 slots total).
 * Calls with more arguments will mis-dispatch with a compile-time error.
 */
#ifdef UNITY_INCLUDE_PRINT_FORMATTED
/* Internal helpers — format into a buffer, then call TEST_PRINTF for real-time
 * output AND TEST_FAIL_MESSAGE to attach the text as the failure message.
 * Using TEST_FAIL() alone would drop the message from failure summaries. */
#  define TEST_FAIL_PRINTF_FMT_ONLY_(fmt) \
     do { TEST_PRINTF("%s", (fmt)); TEST_FAIL_MESSAGE(fmt); } while (0)
#  define TEST_FAIL_PRINTF_VAR_(fmt, ...) \
     do { \
         char _tfp_buf_[512]; \
         snprintf(_tfp_buf_, sizeof(_tfp_buf_), (fmt), __VA_ARGS__); \
         TEST_PRINTF("%s", _tfp_buf_); \
         TEST_FAIL_MESSAGE(_tfp_buf_); \
     } while (0)
/* Dispatcher: _1..._8 consume the arguments; NAME resolves to the right helper. */
#  define TEST_FAIL_PRINTF_PICK_(_1,_2,_3,_4,_5,_6,_7,_8,NAME,...) NAME
#  define TEST_FAIL_PRINTF(...) \
     TEST_FAIL_PRINTF_PICK_(__VA_ARGS__, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_FMT_ONLY_)(__VA_ARGS__)
#else
/* Internal helpers — same dispatcher pattern using snprintf fallback. */
#  define TEST_FAIL_PRINTF_FMT_ONLY_(fmt) \
     do { TEST_FAIL_MESSAGE(fmt); } while (0)
#  define TEST_FAIL_PRINTF_VAR_(fmt, ...) \
     do { \
         char _tfp_buf_[512]; \
         snprintf(_tfp_buf_, sizeof(_tfp_buf_), (fmt), __VA_ARGS__); \
         TEST_FAIL_MESSAGE(_tfp_buf_); \
     } while (0)
#  define TEST_FAIL_PRINTF_PICK_(_1,_2,_3,_4,_5,_6,_7,_8,NAME,...) NAME
#  define TEST_FAIL_PRINTF(...) \
     TEST_FAIL_PRINTF_PICK_(__VA_ARGS__, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_VAR_, \
         TEST_FAIL_PRINTF_VAR_, TEST_FAIL_PRINTF_FMT_ONLY_)(__VA_ARGS__)
#endif

#endif /* CFD_TEST_MACROS_H */
