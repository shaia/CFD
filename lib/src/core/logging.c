#include "cfd/core/logging.h"
#include <stdio.h>


#ifdef _WIN32
#include <windows.h>
static __declspec(thread) cfd_status_t g_last_status = CFD_SUCCESS;
static __declspec(thread) char g_last_error_msg[256] = {0};
#else
#include <pthread.h>
static __thread cfd_status_t g_last_status = CFD_SUCCESS;
static __thread char g_last_error_msg[256] = {0};
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
            return "Solver diverged";
        case CFD_ERROR_MAX_ITER:
            return "Max iterations reached";
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

void cfd_error(const char* message) {
    fprintf(stderr, "ERROR: %s\n", message);
    cfd_set_error(CFD_ERROR, message);
}

void cfd_warning(const char* message) {
    fprintf(stderr, "WARNING: %s\n", message);
}
