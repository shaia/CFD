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
