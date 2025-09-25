#ifndef CFD_UTILS_H
#define CFD_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error handling
void cfd_error(const char* message);
void cfd_warning(const char* message);

// Memory management
void* cfd_malloc(size_t size);
void* cfd_calloc(size_t count, size_t size);
void cfd_free(void* ptr);

// Aligned memory allocation for SIMD operations (32-byte aligned)
void* cfd_aligned_malloc(size_t size);
void* cfd_aligned_calloc(size_t count, size_t size);
void cfd_aligned_free(void* ptr);

// Mathematical utilities
double min_double(double a, double b);
double max_double(double a, double b);
double sign(double x);

// File system utilities
int ensure_directory_exists(const char* path);

// Cross-platform path utilities
#ifdef _WIN32
    #define PATH_SEPARATOR "\\"
    #define PATH_SEPARATOR_CHAR '\\'
#else
    #define PATH_SEPARATOR "/"
    #define PATH_SEPARATOR_CHAR '/'
#endif

// Configurable artifacts path management
void cfd_set_artifacts_path(const char* path);
const char* cfd_get_artifacts_path(void);
void cfd_reset_artifacts_path(void); // Reset to default

// Default path options
typedef enum {
    CFD_PATH_CURRENT_DIR,     // "./output" (default)
    CFD_PATH_TEMP_DIR,        // System temp directory + "/cfd_output"
    CFD_PATH_RELATIVE_BUILD   // "../../artifacts" (for build tree compatibility)
} cfd_default_path_mode_t;

void cfd_set_default_path_mode(cfd_default_path_mode_t mode);

// Cross-platform path construction (uses configurable artifacts path)
void make_output_path(char* buffer, size_t buffer_size, const char* filename);
void make_artifacts_path(char* buffer, size_t buffer_size, const char* subdir);

#endif // CFD_UTILS_H 