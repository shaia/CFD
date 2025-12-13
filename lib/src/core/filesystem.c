#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include <corecrt_io.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
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
        snprintf(artifacts_base_path, sizeof(artifacts_base_path), "%s", path);
    } else {
        artifacts_base_path[0] = '\0';  // Reset to default
    }
}

void cfd_set_default_path_mode(cfd_default_path_mode_t mode) {
    default_path_mode = mode;
}


void cfd_get_artifacts_path(char* buffer, size_t size) {
    if (!buffer || size == 0) {
        return;
    }

    if (strlen(artifacts_base_path) > 0) {
        snprintf(buffer, size, "%s", artifacts_base_path);
        return;
    }

    // Generate default path based on mode
    switch (default_path_mode) {
        case CFD_PATH_CURRENT_DIR:
            snprintf(buffer, size, ".");
            break;

        case CFD_PATH_TEMP_DIR:
#ifdef _WIN32
        {
            char* temp_dir = getenv("TEMP");
            if (!temp_dir) {
                temp_dir = getenv("TMP");
            }
            if (!temp_dir) {
                temp_dir = "C:\\temp";
            }
            snprintf(buffer, size, "%s", temp_dir);
        }
#else
        {
            char* temp_dir = getenv("TMPDIR");
            if (!temp_dir)
                temp_dir = "/tmp";
            snprintf(buffer, size, "%s", temp_dir);
        }
#endif
        break;

        case CFD_PATH_RELATIVE_BUILD:
#ifdef _WIN32
            snprintf(buffer, size, "..\\..\\artifacts");
#else
            snprintf(buffer, size, "../../artifacts");
#endif
            break;

        default:
            snprintf(buffer, size, ".");
            break;
    }
    buffer[size - 1] = '\0';
}

void cfd_reset_artifacts_path(void) {
    artifacts_base_path[0] = '\0';
}

//=============================================================================
// PATH CONSTRUCTION
//=============================================================================
void make_output_path(char* buffer, size_t buffer_size, const char* filename) {
    char base_path[512];
    cfd_get_artifacts_path(base_path, sizeof(base_path));

#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\output\\%s", base_path, filename);
#else
    snprintf(buffer, buffer_size, "%s/output/%s", base_path, filename);
#endif
}

void make_artifacts_path(char* buffer, size_t buffer_size, const char* subdir) {
    char base_path[512];
    cfd_get_artifacts_path(base_path, sizeof(base_path));

    if (subdir && strlen(subdir) > 0) {
#ifdef _WIN32
        snprintf(buffer, buffer_size, "%s\\%s", base_path, subdir);
#else
        snprintf(buffer, buffer_size, "%s/%s", base_path, subdir);
#endif
    } else {
        snprintf(buffer, buffer_size, "%s", base_path);
    }
}

//=============================================================================
// RUN DIRECTORY MANAGEMENT
//=============================================================================

static char current_run_directory[512] = {0};  // Keep state, but no direct access

static void create_run_directory_internal(char* buffer, size_t buffer_size, const char* base_dir,
                                          const char* timestamp_name) {
    // Use provided base directory
    const char* root_path = (base_dir && strlen(base_dir) > 0) ? base_dir : ".";

#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\output\\%s", root_path, timestamp_name);
#else
    snprintf(buffer, buffer_size, "%s/output/%s", root_path, timestamp_name);
#endif

    // Ensure base output directory exists
    char output_base[512];
#ifdef _WIN32
    snprintf(output_base, sizeof(output_base), "%s\\output", root_path);
#else
    snprintf(output_base, sizeof(output_base), "%s/output", root_path);
#endif
    ensure_directory_exists(output_base);

    // Create run-specific directory
    if (!ensure_directory_exists(buffer)) {
        cfd_warning("Failed to create run directory, using base output directory");
        snprintf(buffer, buffer_size, "%s", output_base);
    }
}

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
    char base_path[512];
    cfd_get_artifacts_path(base_path, sizeof(base_path));

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
        snprintf(buffer, buffer_size, "%s", output_base);
    }

    // Store in global state
    snprintf(current_run_directory, sizeof(current_run_directory), "%s", buffer);
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
    char base_path[512];
    cfd_get_artifacts_path(base_path, sizeof(base_path));

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
        snprintf(buffer, buffer_size, "%s", output_base);
    }

    // Store in global state
    snprintf(current_run_directory, sizeof(current_run_directory), "%s", buffer);
}

void cfd_create_run_directory_with_base(char* buffer, size_t buffer_size, const char* base_dir,
                                        const char* prefix) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    // Format: prefix_YYYYMMDD_HHMMSS
    char timestamp[64];
    snprintf(timestamp, sizeof(timestamp), "%s_%04d%02d%02d_%02d%02d%02d", prefix,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    create_run_directory_internal(buffer, buffer_size, base_dir, timestamp);
}

void cfd_create_run_directory_ex_with_base(char* buffer, size_t buffer_size, const char* base_dir,
                                           const char* solver_name, size_t nx, size_t ny) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    // Format: solvername_gridsize_YYYYMMDD_HHMMSS
    char timestamp[128];
    snprintf(timestamp, sizeof(timestamp), "%s_%zux%zu_%04d%02d%02d_%02d%02d%02d",
             solver_name ? solver_name : "sim", nx, ny, t->tm_year + 1900, t->tm_mon + 1,
             t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);

    create_run_directory_internal(buffer, buffer_size, base_dir, timestamp);
}

void cfd_get_run_directory(char* buffer, size_t size) {
    if (!buffer || size == 0) {
        return;
    }

    if (current_run_directory[0] != '\0') {
        snprintf(buffer, size, "%s", current_run_directory);
    } else {
        buffer[0] = '\0';
    }
    buffer[size - 1] = '\0';
}

void cfd_reset_run_directory(void) {
    current_run_directory[0] = '\0';
}
