#include "cfd/core/memory.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/logging.h"

#include <stdlib.h>
#include <string.h>


#ifdef _WIN32
#else
#include <unistd.h>
#endif

//=============================================================================
// MEMORY MANAGEMENT
//=============================================================================

void* cfd_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        cfd_error("Memory allocation failed");
        cfd_set_error(CFD_ERROR_NOMEM, "Memory allocation failed");
    }
    return ptr;
}

void* cfd_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL) {
        cfd_error("Memory allocation failed");
        cfd_set_error(CFD_ERROR_NOMEM, "Memory allocation failed");
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
        while (power < alignment)
            power <<= 1;
        alignment = power;
    }

    if (posix_memalign(&ptr, alignment, size) != 0) {
        cfd_error("Aligned memory allocation failed");
        cfd_set_error(CFD_ERROR_NOMEM, "Aligned memory allocation failed");
        return NULL;
    }
#else
    // Windows - use _aligned_malloc
    ptr = _aligned_malloc(size, 32);
    if (ptr == NULL) {
        cfd_error("Aligned memory allocation failed");
        cfd_set_error(CFD_ERROR_NOMEM, "Aligned memory allocation failed");
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
        free(ptr);
#else
        _aligned_free(ptr);
#endif
    }
}
