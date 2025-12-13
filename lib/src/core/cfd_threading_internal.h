#ifndef CFD_THREADING_INTERNAL_H
#define CFD_THREADING_INTERNAL_H

#ifdef _WIN32
#include <windows.h>

typedef volatile LONG cfd_atomic_int;

static inline int cfd_atomic_load(cfd_atomic_int* ptr) {
    return *ptr;
}

/**
 * @brief Compare and Swap
 * @return 1 if swap occurred (success), 0 otherwise
 */
static inline int cfd_atomic_cas(cfd_atomic_int* ptr, int expected, int desired) {
    // InterlockedCompareExchange(dest, exchange, comparand)
    // Returns the initial value of dest.
    // If return value == comparand (expected), then success.
    return (InterlockedCompareExchange(ptr, desired, expected) == expected);
}

static inline void cfd_atomic_store(cfd_atomic_int* ptr, int value) {
    InterlockedExchange(ptr, value);
}

#else
#include <stdatomic.h>

typedef atomic_int cfd_atomic_int;

static inline int cfd_atomic_load(cfd_atomic_int* ptr) {
    return atomic_load(ptr);
}

static inline int cfd_atomic_cas(cfd_atomic_int* ptr, int expected, int desired) {
    // atomic_compare_exchange_strong takes &expected
    int exp = expected;
    return atomic_compare_exchange_strong(ptr, &exp, desired);
}

static inline void cfd_atomic_store(cfd_atomic_int* ptr, int value) {
    atomic_store(ptr, value);
}

#endif

#endif  // CFD_THREADING_INTERNAL_H
