#include "cfd/core/cfd_init.h"
#include "cfd/core/logging.h"
#include "cfd_threading_internal.h"
#include <stdbool.h>


/* Global initialization state */
static cfd_atomic_int s_is_initialized = 0;

cfd_status_t cfd_init(void) {
    // Check if already initialized (fast path)
    if (cfd_atomic_load(&s_is_initialized)) {
        return CFD_SUCCESS;
    }

    // Attempt to transition from 0 (uninitialized) to 1 (initialized)
    if (cfd_atomic_cas(&s_is_initialized, 0, 1)) {
        /*
         * This thread won the race to initialize.
         * Perform global initialization tasks here.
         */
        return CFD_SUCCESS;
    } else {
        // Another thread beat us to it or initialization is already done
        return CFD_SUCCESS;
    }
}

void cfd_finalize(void) {
    // Attempt to transition from 1 to 0
    cfd_atomic_cas(&s_is_initialized, 1, 0);
}

int cfd_is_initialized(void) {
    return cfd_atomic_load(&s_is_initialized) ? 1 : 0;
}
