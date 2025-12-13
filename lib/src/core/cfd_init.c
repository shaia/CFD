#include "cfd/core/cfd_init.h"
#include "cfd/core/logging.h"
#include <stdbool.h>

/* Global initialization state */
static bool s_is_initialized = false;

cfd_status_t cfd_init(void) {
    if (s_is_initialized) {
        return CFD_SUCCESS;
    }

    /*
     * Perform global initialization tasks here.
     * Examples:
     * - Initialize default logging level if needed
     * - Check/Init CUDA context (if we were managing it explicitly)
     * - Initialize memory pools (if added in future)
     */

    // For now, logging defaults are handled statically/lazily, but we can log that we are starting.
    // However, if logging isn't safe before init, we shouldn't.
    // Assuming logging is safe (stateless/default).

    s_is_initialized = true;
    return CFD_SUCCESS;
}

void cfd_finalize(void) {
    if (!s_is_initialized) {
        return;
    }

    /*
     * Perform global cleanup here.
     */

    s_is_initialized = false;
}

int cfd_is_initialized(void) {
    return s_is_initialized ? 1 : 0;
}
