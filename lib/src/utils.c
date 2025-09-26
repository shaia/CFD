#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
    #include <direct.h>
    #include <io.h>
    #include <malloc.h>  // For _aligned_malloc
#else
    #include <sys/stat.h>
    #include <unistd.h>
#endif

void cfd_error(const char* message) {
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

void cfd_warning(const char* message) {
    fprintf(stderr, "WARNING: %s\n", message);
}

void* cfd_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        cfd_error("Memory allocation failed");
    }
    return ptr;
}

void* cfd_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL) {
        cfd_error("Memory allocation failed");
    }
    return ptr;
}

void cfd_free(void* ptr) {
    if (ptr != NULL) {
        free(ptr);
    }
}

// Aligned memory allocation functions for SIMD operations
void* cfd_aligned_malloc(size_t size) {
    void* ptr;

#ifndef _WIN32
    // Linux/Unix - use robust posix_memalign
    size_t alignment = 32;  // 32-byte alignment for AVX2

    // Ensure alignment is valid (power of 2 and >= sizeof(void*))
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

    // Round alignment up to next power of 2 if needed
    if ((alignment & (alignment - 1)) != 0) {
        size_t power = 1;
        while (power < alignment) power <<= 1;
        alignment = power;
    }

    if (posix_memalign(&ptr, alignment, size) != 0) {
        cfd_error("Aligned memory allocation failed");
        return NULL;
    }
#else
    // Windows - use _aligned_malloc
    ptr = _aligned_malloc(size, 32);
    if (ptr == NULL) {
        cfd_error("Aligned memory allocation failed");
    }
#endif

    return ptr;
}

void* cfd_aligned_calloc(size_t count, size_t size) {
    size_t total_size = count * size;
    void* ptr = cfd_aligned_malloc(total_size);
    if (ptr != NULL) {
        memset(ptr, 0, total_size);
    }
    return ptr;
}

void cfd_aligned_free(void* ptr) {
    if (ptr != NULL) {
#ifndef _WIN32
        free(ptr);  // Linux/Unix - regular free works with posix_memalign
#else
        _aligned_free(ptr);  // Windows - use _aligned_free
#endif
    }
}

double min_double(double a, double b) {
    return (a < b) ? a : b;
}

double max_double(double a, double b) {
    return (a > b) ? a : b;
}

double sign(double x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
}

int ensure_directory_exists(const char* path) {
#ifdef _WIN32
    if (_access(path, 0) == 0) {
        return 1; // Directory exists
    }
    return _mkdir(path) == 0;
#else
    struct stat st = {0};
    if (stat(path, &st) == 0) {
        return 1; // Directory exists
    }
    return mkdir(path, 0755) == 0;
#endif
}

// Static variables for configurable artifacts path
static char artifacts_base_path[512] = ""; // Empty means use default
static cfd_default_path_mode_t default_path_mode = CFD_PATH_CURRENT_DIR;

// Configurable artifacts path management
void cfd_set_artifacts_path(const char* path) {
    if (path && strlen(path) > 0) {
        strncpy(artifacts_base_path, path, sizeof(artifacts_base_path) - 1);
        artifacts_base_path[sizeof(artifacts_base_path) - 1] = '\0';
    } else {
        artifacts_base_path[0] = '\0'; // Reset to default
    }
}

void cfd_set_default_path_mode(cfd_default_path_mode_t mode) {
    default_path_mode = mode;
}

const char* cfd_get_artifacts_path(void) {
    static char default_path_buffer[512];

    if (strlen(artifacts_base_path) > 0) {
        return artifacts_base_path;
    }

    // Generate default path based on mode
    switch (default_path_mode) {
        case CFD_PATH_CURRENT_DIR:
#ifdef _WIN32
            // Safe: use memcpy for compile-time known string length (includes null terminator)
            memcpy(default_path_buffer, ".\\output", 9); // Copy 8 chars + null terminator = 9 bytes
#else
            memcpy(default_path_buffer, "./output", 9); // Copy 8 chars + null terminator = 9 bytes
#endif
            break;

        case CFD_PATH_TEMP_DIR:
#ifdef _WIN32
            {
                char* temp_dir = getenv("TEMP");
                if (!temp_dir) temp_dir = getenv("TMP");
                if (!temp_dir) temp_dir = "C:\\temp";
                snprintf(default_path_buffer, sizeof(default_path_buffer),
                        "%s\\cfd_output", temp_dir);
            }
#else
            {
                char* temp_dir = getenv("TMPDIR");
                if (!temp_dir) temp_dir = "/tmp";
                snprintf(default_path_buffer, sizeof(default_path_buffer),
                        "%s/cfd_output", temp_dir);
            }
#endif
            break;

        case CFD_PATH_RELATIVE_BUILD:
#ifdef _WIN32
            // Safe: use memcpy for compile-time known string length (includes null terminator)
            memcpy(default_path_buffer, "..\\..\\artifacts", 17); // Copy 16 chars + null terminator = 17 bytes
#else
            memcpy(default_path_buffer, "../../artifacts", 16); // Copy 15 chars + null terminator = 16 bytes
#endif
            break;

        default:
#ifdef _WIN32
            // Safe: use memcpy for compile-time known string length (includes null terminator)
            memcpy(default_path_buffer, ".\\output", 9); // Copy 8 chars + null terminator = 9 bytes
#else
            memcpy(default_path_buffer, "./output", 9); // Copy 8 chars + null terminator = 9 bytes
#endif
            break;
    }

    return default_path_buffer;
}

void cfd_reset_artifacts_path(void) {
    artifacts_base_path[0] = '\0';
}

// Cross-platform path construction functions (using configurable path)
void make_output_path(char* buffer, size_t buffer_size, const char* filename) {
    const char* base_path = cfd_get_artifacts_path();

#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\output\\%s", base_path, filename);
#else
    snprintf(buffer, buffer_size, "%s/output/%s", base_path, filename);
#endif
}

void make_artifacts_path(char* buffer, size_t buffer_size, const char* subdir) {
    const char* base_path = cfd_get_artifacts_path();

    if (subdir && strlen(subdir) > 0) {
#ifdef _WIN32
        snprintf(buffer, buffer_size, "%s\\%s", base_path, subdir);
#else
        snprintf(buffer, buffer_size, "%s/%s", base_path, subdir);
#endif
    } else {
        strncpy(buffer, base_path, buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    }
}
