#ifndef CFD_LOGGING_H
#define CFD_LOGGING_H

#include "cfd/core/cfd_status.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// ERROR CONTEXT
//=============================================================================

// Set the last error message for the current thread
void cfd_set_error(cfd_status_t status, const char* message);

// Get the last error message for the current thread
const char* cfd_get_last_error(void);

// Get the last error code
cfd_status_t cfd_get_last_status(void);

// Clear the error state
void cfd_clear_error(void);

//=============================================================================
// LOGGING
//=============================================================================

// Log levels
typedef enum { CFD_LOG_LEVEL_INFO, CFD_LOG_LEVEL_WARNING, CFD_LOG_LEVEL_ERROR } cfd_log_level_t;

// Log callback function type
typedef void (*cfd_log_callback_t)(cfd_log_level_t level, const char* message);

// Set a custom logging callback (pass NULL to reset to default stderr output)
void cfd_set_log_callback(cfd_log_callback_t callback);

// Log error message and set thread-local error state (does NOT exit)
void cfd_error(const char* message);

// Print warning message
void cfd_warning(const char* message);

#ifdef __cplusplus
}
#endif

#endif  // CFD_LOGGING_H
