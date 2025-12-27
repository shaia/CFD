#include "cfd/core/logging.h"
#include "cfd/core/cfd_status.h"
#include <stdio.h>


#ifdef _WIN32
static __declspec(thread) cfd_status_t g_last_status = CFD_SUCCESS;
static __declspec(thread) char g_last_error_msg[256] = {0};
static __declspec(thread) cfd_log_callback_t s_log_callback = NULL;
#else
#include <pthread.h>
static __thread cfd_status_t g_last_status = CFD_SUCCESS;
static __thread char g_last_error_msg[256] = {0};
static __thread cfd_log_callback_t s_log_callback = NULL;
#endif

//=============================================================================
// ERROR CONTEXT
//=============================================================================

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

//=============================================================================
// LOGGING
//=============================================================================

void cfd_set_log_callback(cfd_log_callback_t callback) {
    s_log_callback = callback;
}

void cfd_error(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    if (s_log_callback) {
        s_log_callback(CFD_LOG_LEVEL_ERROR, safe_msg);
    } else {
        fprintf(stderr, "ERROR: %s\n", safe_msg);
    }
    cfd_set_error(CFD_ERROR, safe_msg);
}

void cfd_warning(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    if (s_log_callback) {
        s_log_callback(CFD_LOG_LEVEL_WARNING, safe_msg);
    } else {
        fprintf(stderr, "WARNING: %s\n", safe_msg);
    }
}

void cfd_info(const char* message) {
    const char* safe_msg = message ? message : "(null)";
    if (s_log_callback) {
        s_log_callback(CFD_LOG_LEVEL_INFO, safe_msg);
    } else {
        fprintf(stdout, "INFO: %s\n", safe_msg);
    }
}
