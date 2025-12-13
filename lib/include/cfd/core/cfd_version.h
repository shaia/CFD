#ifndef CFD_VERSION_H
#define CFD_VERSION_H

#include "cfd/cfd_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Version macros
#define CFD_VERSION_MAJOR 0
#define CFD_VERSION_MINOR 7
#define CFD_VERSION_PATCH 0

// Helper macro to stringify version definition
#define CFD_STR_HELPER(x) #x
#define CFD_STR(x)        CFD_STR_HELPER(x)
#define CFD_VERSION_STRING \
    CFD_STR(CFD_VERSION_MAJOR) "." CFD_STR(CFD_VERSION_MINOR) "." CFD_STR(CFD_VERSION_PATCH)

/**
 * @brief Get the library version string.
 *
 * @return const char* Version string in format "MAJOR.MINOR.PATCH"
 */
CFD_LIBRARY_EXPORT const char* cfd_get_version_string(void);

#ifdef __cplusplus
}
#endif

#endif  // CFD_VERSION_H
