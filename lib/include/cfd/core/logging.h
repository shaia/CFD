#ifndef CFD_LOGGING_H
#define CFD_LOGGING_H

#include "cfd/cfd_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * LOG LEVELS
 * ============================================================================ */

typedef enum {
    CFD_LOG_LEVEL_DEBUG = 0,
    CFD_LOG_LEVEL_INFO = 1,
    CFD_LOG_LEVEL_WARNING = 2,
    CFD_LOG_LEVEL_ERROR = 3
} cfd_log_level_t;

/* ============================================================================
 * CALLBACK TYPES
 * ============================================================================ */

/** Per-thread callback (legacy — no component info) */
typedef void (*cfd_log_callback_t)(cfd_log_level_t level, const char* message);

/** Extended callback with component tag */
typedef void (*cfd_log_callback_ex_t)(cfd_log_level_t level, const char* component,
                                      const char* message);

/* ============================================================================
 * CORE LOGGING API
 * ============================================================================ */

/**
 * Log a message with printf-style formatting, a log level, and a component tag.
 *
 * Messages below the global log level are suppressed (fast atomic check).
 *
 * Dispatch priority:
 *   1. Per-thread callback (set via cfd_set_log_callback)
 *   2. Global extended callback (set via cfd_set_log_callback_ex)
 *   3. Default handler (stderr for WARNING/ERROR, stdout for DEBUG/INFO)
 *
 * @param level     Log level (DEBUG, INFO, WARNING, ERROR)
 * @param component Component tag (e.g., "poisson", "boundary") or NULL
 * @param fmt       printf-style format string
 */
CFD_LIBRARY_EXPORT void cfd_log(cfd_log_level_t level, const char* component, const char* fmt,
                                ...);

/** Set the global minimum log level (default: CFD_LOG_LEVEL_INFO). Thread-safe. */
CFD_LIBRARY_EXPORT void cfd_set_log_level(cfd_log_level_t level);

/** Get the current global minimum log level. Thread-safe. */
CFD_LIBRARY_EXPORT cfd_log_level_t cfd_get_log_level(void);

/** Set a global extended callback (receives component info). Pass NULL to reset. */
CFD_LIBRARY_EXPORT void cfd_set_log_callback_ex(cfd_log_callback_ex_t callback);

/* ============================================================================
 * CONVENIENCE MACROS
 * ============================================================================ */

#define CFD_LOG_DEBUG(component, ...) cfd_log(CFD_LOG_LEVEL_DEBUG, (component), __VA_ARGS__)
#define CFD_LOG_INFO(component, ...) cfd_log(CFD_LOG_LEVEL_INFO, (component), __VA_ARGS__)
#define CFD_LOG_WARNING(component, ...) cfd_log(CFD_LOG_LEVEL_WARNING, (component), __VA_ARGS__)
#define CFD_LOG_ERROR(component, ...) cfd_log(CFD_LOG_LEVEL_ERROR, (component), __VA_ARGS__)

/* ============================================================================
 * LEGACY API (backward-compatible convenience wrappers)
 * ============================================================================ */

/** Set a per-thread logging callback (pass NULL to reset to default stderr output) */
CFD_LIBRARY_EXPORT void cfd_set_log_callback(cfd_log_callback_t callback);

/** Log error message and set thread-local error state (does NOT exit) */
CFD_LIBRARY_EXPORT void cfd_error(const char* message);

/** Print warning message */
CFD_LIBRARY_EXPORT void cfd_warning(const char* message);

/** Print info message */
CFD_LIBRARY_EXPORT void cfd_info(const char* message);

#ifdef __cplusplus
}
#endif

#endif  // CFD_LOGGING_H
