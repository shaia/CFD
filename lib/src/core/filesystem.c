#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

//=============================================================================
// FILE SYSTEM
//=============================================================================

int ensure_directory_exists(const char* path) {
#ifdef _WIN32
    if (_access(path, 0) == 0) {
        return 1;  // Directory exists
    }
    return _mkdir(path) == 0;
#else
    struct stat st = {0};
    if (stat(path, &st) == 0) {
        return 1;  // Directory exists
    }
    return mkdir(path, 0755) == 0;
#endif
}

//=============================================================================
// OUTPUT PATH CONFIGURATION
//=============================================================================

static char artifacts_base_path[512] = "";
static cfd_default_path_mode_t default_path_mode = CFD_PATH_CURRENT_DIR;

void cfd_set_output_base_dir(const char* path) {
    if (path && strlen(path) > 0) {
        strncpy(artifacts_base_path, path, sizeof(artifacts_base_path) - 1);
        artifacts_base_path[sizeof(artifacts_base_path) - 1] = '\0';
    } else {
        artifacts_base_path[0] = '\0';  // Reset to default
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
            strncpy(default_path_buffer, ".", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#else
            strncpy(default_path_buffer, ".", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#endif
            break;

        case CFD_PATH_TEMP_DIR:
#ifdef _WIN32
        {
            char* temp_dir = getenv("TEMP");
            if (!temp_dir)
                temp_dir = getenv("TMP");
            if (!temp_dir)
                temp_dir = "C:\\temp";
            snprintf(default_path_buffer, sizeof(default_path_buffer), "%s", temp_dir);
        }
#else
        {
            char* temp_dir = getenv("TMPDIR");
            if (!temp_dir)
                temp_dir = "/tmp";
            snprintf(default_path_buffer, sizeof(default_path_buffer), "%s", temp_dir);
        }
#endif
        break;

        case CFD_PATH_RELATIVE_BUILD:
#ifdef _WIN32
            strncpy(default_path_buffer, "..\\..\\artifacts", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#else
            strncpy(default_path_buffer, "../../artifacts", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#endif
            break;

        default:
#ifdef _WIN32
            strncpy(default_path_buffer, ".", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#else
            strncpy(default_path_buffer, ".", sizeof(default_path_buffer) - 1);
            default_path_buffer[sizeof(default_path_buffer) - 1] = '\0';
#endif
            break;
    }

    return default_path_buffer;
}

void cfd_reset_artifacts_path(void) {
    artifacts_base_path[0] = '\0';
}

//=============================================================================
// PATH CONSTRUCTION
//=============================================================================
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

//=============================================================================
// RUN DIRECTORY MANAGEMENT
//=============================================================================

static char current_run_directory[512] = {0};

void cfd_create_run_directory(char* buffer, size_t buffer_size) {
    cfd_create_run_directory_with_prefix(buffer, buffer_size, "run");
}

void cfd_create_run_directory_with_prefix(char* buffer, size_t buffer_size, const char* prefix) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    // Format: prefix_YYYYMMDD_HHMMSS
    char timestamp[64];
    snprintf(timestamp, sizeof(timestamp), "%s_%04d%02d%02d_%02d%02d%02d", prefix,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    // Create full path
    const char* base_path = cfd_get_artifacts_path();
#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\output\\%s", base_path, timestamp);
#else
    snprintf(buffer, buffer_size, "%s/output/%s", base_path, timestamp);
#endif

    // Ensure base output directory exists
    char output_base[512];
#ifdef _WIN32
    snprintf(output_base, sizeof(output_base), "%s\\output", base_path);
#else
    snprintf(output_base, sizeof(output_base), "%s/output", base_path);
#endif
    ensure_directory_exists(output_base);

    // Create run-specific directory
    if (!ensure_directory_exists(buffer)) {
        cfd_warning("Failed to create run directory, using base output directory");
        strncpy(buffer, output_base, buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    }

    // Store in global state
    strncpy(current_run_directory, buffer, sizeof(current_run_directory) - 1);
    current_run_directory[sizeof(current_run_directory) - 1] = '\0';
}

void cfd_create_run_directory_ex(char* buffer, size_t buffer_size, const char* solver_name,
                                 size_t nx, size_t ny) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    // Format: solvername_gridsize_YYYYMMDD_HHMMSS
    char timestamp[128];
    snprintf(timestamp, sizeof(timestamp), "%s_%zux%zu_%04d%02d%02d_%02d%02d%02d",
             solver_name ? solver_name : "sim", nx, ny, t->tm_year + 1900, t->tm_mon + 1,
             t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    // Create full path
    const char* base_path = cfd_get_artifacts_path();
#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\output\\%s", base_path, timestamp);
#else
    snprintf(buffer, buffer_size, "%s/output/%s", base_path, timestamp);
#endif

    // Ensure base output directory exists
    char output_base[512];
#ifdef _WIN32
    snprintf(output_base, sizeof(output_base), "%s\\output", base_path);
#else
    snprintf(output_base, sizeof(output_base), "%s/output", base_path);
#endif
    ensure_directory_exists(output_base);

    // Create run-specific directory
    if (!ensure_directory_exists(buffer)) {
        cfd_warning("Failed to create run directory, using base output directory");
        strncpy(buffer, output_base, buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    }

    // Store in global state
    strncpy(current_run_directory, buffer, sizeof(current_run_directory) - 1);
    current_run_directory[sizeof(current_run_directory) - 1] = '\0';
}

const char* cfd_get_run_directory(void) {
    return current_run_directory[0] != '\0' ? current_run_directory : NULL;
}

void cfd_reset_run_directory(void) {
    current_run_directory[0] = '\0';
}

//=============================================================================
// DEPRECATED ALIASES
//=============================================================================

void cfd_set_artifacts_path(const char* path) {
    cfd_set_output_base_dir(path);
}
