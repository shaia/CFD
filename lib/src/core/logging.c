#include "cfd/core/logging.h"
#include "cfd/core/cfd_status.h"
#include "cfd_threading_internal.h"

#include <stdarg.h>
#include <stdio.h>

/* ============================================================================
 * THREAD-LOCAL STATE
 * ============================================================================ */

#ifdef _WIN32
static __declspec(thread) cfd_status_t g_last_status = CFD_SUCCESS;
static __declspec(thread) char g_last_error_msg[256] = {0};
static __declspec(thread) cfd_log_callback_t s_log_callback = NULL;
#else
static __thread cfd_status_t g_last_status = CFD_SUCCESS;
static __thread char g_last_error_msg[256] = {0};
static __thread cfd_log_callback_t s_log_callback = NULL;
#endif

/* ============================================================================
 * GLOBAL STATE
 * ============================================================================ */

/* Default log level: INFO (suppresses DEBUG) */
static cfd_atomic_int s_global_log_level = 1; /* CFD_LOG_LEVEL_INFO */

/* Global extended callback (set once at startup, read on every log call) */
static cfd_atomic_ptr s_global_callback_ex;

/* Level name table (const, thread-safe) */
static const char* const s_level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

/* ============================================================================
 * ERROR CONTEXT
 * ============================================================================ */

void cfd_set_error(cfd_status_t status, const char* message) {
    g_last_status = status;
    if (message) {
        snprintf(g_last_error_msg, sizeof(g_last_error_msg), "%s", message);
    } else {
        g_last_error_msg[0] = '\0';
    }
}

const char* cfd_get_last_error(void) {
    if (g_last_error_msg[0] == '\0') {
        return NULL;
    }
    return g_last_error_msg;
}

cfd_status_t cfd_get_last_status(void) {
    return g_last_status;
}

const char* cfd_get_error_string(cfd_status_t status) {
    switch (status) {
        case CFD_SUCCESS:
            return "Success";
        case CFD_ERROR:
            return "Generic error";
        case CFD_ERROR_NOMEM:
            return "Out of memory";
        case CFD_ERROR_INVALID:
            return "Invalid argument";
        case CFD_ERROR_IO:
            return "I/O error";
        case CFD_ERROR_UNSUPPORTED:
            return "Operation not supported";
        case CFD_ERROR_DIVERGED:
            return "NSSolver diverged";
        case CFD_ERROR_MAX_ITER:
            return "Max iterations reached";
        case CFD_ERROR_LIMIT_EXCEEDED:
            return "Resource limit exceeded";
        case CFD_ERROR_NOT_FOUND:
            return "Resource not found";
        default:
            return "Unknown error";
    }
}

void cfd_clear_error(void) {
    g_last_status = CFD_SUCCESS;
    g_last_error_msg[0] = '\0';
}

/* ============================================================================
 * CORE LOGGING
 * ============================================================================ */

void cfd_log(cfd_log_level_t level, const char* component, const char* fmt, ...) {
    /* Fast path: suppress messages below global log level */
    if ((int)level < cfd_atomic_load(&s_global_log_level)) {
        return;
    }

    /* Format message on stack */
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    /* Dispatch: per-thread callback > global extended callback > default handler */
    if (s_log_callback) {
        s_log_callback(level, buf);
        return;
    }

    cfd_log_callback_ex_t cb_ex =
        (cfd_log_callback_ex_t)cfd_atomic_ptr_load(&s_global_callback_ex);
    if (cb_ex) {
        cb_ex(level, component, buf);
    } else {
        const char* level_str =
            ((int)level >= 0 && (int)level <= 3) ? s_level_names[(int)level] : "UNKNOWN";
        FILE* stream = (level >= CFD_LOG_LEVEL_WARNING) ? stderr : stdout;
        if (component) {
            fprintf(stream, "%s [%s]: %s\n", level_str, component, buf);
        } else {
            fprintf(stream, "%s: %s\n", level_str, buf);
        }
    }
}

void cfd_set_log_level(cfd_log_level_t level) {
    cfd_atomic_store(&s_global_log_level, (int)level);
}

cfd_log_level_t cfd_get_log_level(void) {
    return (cfd_log_level_t)cfd_atomic_load(&s_global_log_level);
}

void cfd_set_log_callback_ex(cfd_log_callback_ex_t callback) {
    cfd_atomic_ptr_store(&s_global_callback_ex, (void*)callback);
}

/* ============================================================================
 * LEGACY API
 * ============================================================================ */

void cfd_set_log_callback(cfd_log_callback_t callback) {
    s_log_callback = callback;
}

void cfd_error(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    /* Always set error state, regardless of log level filter */
    cfd_set_error(CFD_ERROR, safe_msg);
    cfd_log(CFD_LOG_LEVEL_ERROR, NULL, "%s", safe_msg);
}

void cfd_warning(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    cfd_log(CFD_LOG_LEVEL_WARNING, NULL, "%s", safe_msg);
}

void cfd_info(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    cfd_log(CFD_LOG_LEVEL_INFO, NULL, "%s", safe_msg);
}
