#ifndef CFD_FILESYSTEM_H
#define CFD_FILESYSTEM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// FILE SYSTEM
//=============================================================================

// Cross-platform path separators
#ifdef _WIN32
#define PATH_SEPARATOR      "\\"
#define PATH_SEPARATOR_CHAR '\\'
#else
#define PATH_SEPARATOR      "/"
#define PATH_SEPARATOR_CHAR '/'
#endif

// Create directory if it doesn't exist (returns 1 on success)
int ensure_directory_exists(const char* path);

//=============================================================================
// OUTPUT PATH CONFIGURATION
//=============================================================================

// Path mode options for default output location
typedef enum {
    CFD_PATH_CURRENT_DIR,    // "./output" (default)
    CFD_PATH_TEMP_DIR,       // System temp directory
    CFD_PATH_RELATIVE_BUILD  // "../../artifacts" (for build tree)
} cfd_default_path_mode_t;

// Set custom base directory for all output (e.g., "../../artifacts")
void cfd_set_output_base_dir(const char* path);

// Set default path mode when custom base not specified
void cfd_set_default_path_mode(cfd_default_path_mode_t mode);

// Get current base path (returns custom or default based on mode)
const char* cfd_get_artifacts_path(void);

// Reset to default path mode
void cfd_reset_artifacts_path(void);

//=============================================================================
// PATH CONSTRUCTION
//=============================================================================

// Construct path in output directory: "{base}/output/{filename}"
void make_output_path(char* buffer, size_t buffer_size, const char* filename);

// Construct path in artifacts subdirectory: "{base}/{subdir}"
void make_artifacts_path(char* buffer, size_t buffer_size, const char* subdir);

//=============================================================================
// RUN DIRECTORY MANAGEMENT
//=============================================================================

// Create timestamped run directory with default prefix
// Example: "output/run_20250127_153045"
void cfd_create_run_directory(char* buffer, size_t buffer_size);

// Create timestamped run directory with custom prefix
// Example: "output/{prefix}_20250127_153045"
void cfd_create_run_directory_with_prefix(char* buffer, size_t buffer_size, const char* prefix);

// Create run directory with simulation context
// Example: "output/explicit_euler_100x50_20250127_153045"
void cfd_create_run_directory_ex(char* buffer, size_t buffer_size, const char* solver_name,
                                 size_t nx, size_t ny);

// Get current run directory (NULL if not yet created)
const char* cfd_get_run_directory(void);

// Reset run directory for new simulation
void cfd_reset_run_directory(void);


#ifdef __cplusplus
}
#endif

#endif  // CFD_FILESYSTEM_H
