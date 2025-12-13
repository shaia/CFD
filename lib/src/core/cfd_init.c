#include "cfd/core/cfd_init.h"
#include "cfd/core/logging.h"
#include <stdbool.h>

#if defined(_WIN32)
#include <windows.h>
/* Use Windows Interlocked API */
static volatile LONG s_is_initialized = 0;

cfd_status_t cfd_init(void) {
    if (s_is_initialized) {
        return CFD_SUCCESS;
    }

    /* InterlockedCompareExchange returns the initial value.
       If it was 0, we successfully set it to 1 and proceed.
       If it was already 1, we do nothing. */
    if (InterlockedCompareExchange(&s_is_initialized, 1, 0) == 0) {
        /* We won the race */
        /* Perform initialization here */
        return CFD_SUCCESS;
    } else {
        /* Already initialized */
        return CFD_SUCCESS;
    }
}

void cfd_finalize(void) {
    /* If it is 1, set to 0 */
    InterlockedCompareExchange(&s_is_initialized, 0, 1);
}

int cfd_is_initialized(void) {
    return (s_is_initialized == 1) ? 1 : 0;
}

#else
/* Standard C11 Atomics */
#include <stdatomic.h>

static atomic_int s_is_initialized = 0;

cfd_status_t cfd_init(void) {
    if (atomic_load(&s_is_initialized)) {
        return CFD_SUCCESS;
    }

    int expected = 0;
    if (atomic_compare_exchange_strong(&s_is_initialized, &expected, 1)) {
        return CFD_SUCCESS;
    } else {
        return CFD_SUCCESS;
    }
}

void cfd_finalize(void) {
    int expected = 1;
    atomic_compare_exchange_strong(&s_is_initialized, &expected, 0);
}

int cfd_is_initialized(void) {
    return atomic_load(&s_is_initialized) ? 1 : 0;
}
#endif
