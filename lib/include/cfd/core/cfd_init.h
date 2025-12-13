#ifndef CFD_CORE_INIT_H
#define CFD_CORE_INIT_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the CFD library.
 *
 * This function initializes any global state, logging defaults, and
 * hardware interactions if needed.
 *
 * It is safe to call this function multiple times; subsequent calls
 * will return CFD_SUCCESS immediately.
 *
 * @return CFD_SUCCESS on success, or an error code on failure.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_init(void);

/**
 * @brief Finalize the CFD library.
 *
 * This function cleans up any global resources allocated by cfd_init().
 * It should be called when the library is no longer needed to prevent memory leaks,
 * especially in repeated library load/unload scenarios.
 */
CFD_LIBRARY_EXPORT void cfd_finalize(void);

/**
 * @brief Check if the library is initialized.
 *
 * @return true if initialized, false otherwise.
 */
CFD_LIBRARY_EXPORT int cfd_is_initialized(void);

#ifdef __cplusplus
}
#endif

#endif  // CFD_CORE_INIT_H
