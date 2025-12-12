#ifndef CFD_MEMORY_H
#define CFD_MEMORY_H

#include "cfd/cfd_export.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// MEMORY MANAGEMENT
//=============================================================================

// Standard memory allocation with error checking
CFD_LIBRARY_EXPORT void* cfd_malloc(size_t size);
CFD_LIBRARY_EXPORT void* cfd_calloc(size_t count, size_t size);
CFD_LIBRARY_EXPORT void cfd_free(void* ptr);

// Aligned memory allocation for SIMD operations (32-byte aligned)
CFD_LIBRARY_EXPORT void* cfd_aligned_malloc(size_t size);
CFD_LIBRARY_EXPORT void* cfd_aligned_calloc(size_t count, size_t size);
CFD_LIBRARY_EXPORT void cfd_aligned_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif  // CFD_MEMORY_H
