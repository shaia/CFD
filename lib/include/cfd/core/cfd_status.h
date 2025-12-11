#ifndef CFD_STATUS_H
#define CFD_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Standard Status Codes for CFD Library
 */
typedef enum {
    CFD_SUCCESS = 0,
    CFD_ERROR = -1,              // Generic error
    CFD_ERROR_NOMEM = -2,        // Out of memory
    CFD_ERROR_INVALID = -3,      // Invalid argument/input
    CFD_ERROR_IO = -4,           // File I/O error
    CFD_ERROR_UNSUPPORTED = -5,  // Operation not supported
    CFD_ERROR_DIVERGED = -6,     // Solver diverged
    CFD_ERROR_MAX_ITER = -7      // Solver reached max iterations
} cfd_status_t;

/**
 * Error Reporting API
 */

// Set the last error message for the current thread
void cfd_set_error(cfd_status_t status, const char* message);

// Get the last error message for the current thread
const char* cfd_get_last_error(void);

// Get the last error code
cfd_status_t cfd_get_last_status(void);

// Get string description of status code
const char* cfd_get_error_string(cfd_status_t status);

// Clear the error state
void cfd_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif  // CFD_STATUS_H
