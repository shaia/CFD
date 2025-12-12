#ifndef CFD_LOGGING_H
#define CFD_LOGGING_H

#ifdef __cplusplus
extern "C" {
#endif

// Log levels
typedef enum { CFD_LOG_LEVEL_INFO, CFD_LOG_LEVEL_WARNING, CFD_LOG_LEVEL_ERROR } cfd_log_level_t;

// Log callback function type
typedef void (*cfd_log_callback_t)(cfd_log_level_t level, const char* message);

// Set a custom logging callback for the CURRENT THREAD (pass NULL to reset to default stderr
// output)
void cfd_set_log_callback(cfd_log_callback_t callback);

// Log error message and set thread-local error state (does NOT exit)
void cfd_error(const char* message);

// Print warning message
void cfd_warning(const char* message);

// Print info message
void cfd_info(const char* message);

#ifdef __cplusplus
}
#endif

#endif  // CFD_LOGGING_H
