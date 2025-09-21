#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
    #include <direct.h>
    #include <io.h>
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
