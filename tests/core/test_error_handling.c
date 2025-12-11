#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"


#include <assert.h>
#include <stdio.h>
#include <string.h>


// Mock function to test status return
cfd_status_t mock_allocator(size_t size) {
    if (size > 1024 * 1024 * 1024) {  // 1GB
        // Simulate failure
        cfd_set_error(CFD_ERROR_NOMEM, "Mock allocation failure");
        return CFD_ERROR_NOMEM;
    }
    return CFD_SUCCESS;
}

int test_error_reporting() {
    printf("Testing error reporting...\n");

    // Test 1: Set and Get Error
    cfd_clear_error();
    assert(cfd_get_last_error() == NULL);

    cfd_set_error(CFD_ERROR, "Test error message");
    const char* err = cfd_get_last_error();
    assert(err != NULL);
    assert(strcmp(err, "Test error message") == 0);
    printf("  Set/Get Error: OK\n");

    // Test 2: Clear Error
    cfd_clear_error();
    assert(cfd_get_last_error() == NULL);
    printf("  Clear Error: OK\n");

    // Test 3: Thread-local behavior (simulated in single thread for now)
    cfd_set_error(CFD_ERROR_INVALID, "Invalid status");
    // Ensure status is stored? current API only stores message, status is separate context?
    // Actually cfd_set_error only sets thread-local message. Status is up to caller to propagate.
    // The previous implementation in utils.c stores message and status.
    // Let's verify status retrieval if API exists.
    // There is no cfd_get_last_status() in my previous implementation plan, just message.

    // Check internal behavior:
    // If I call cfd_set_error, does it overwrite?
    cfd_set_error(CFD_ERROR_NOMEM, "New error");
    assert(strcmp(cfd_get_last_error(), "New error") == 0);
    printf("  Overwrite Error: OK\n");

    // Test 4: Mock allocation failure
    cfd_status_t status = mock_allocator(2000000000);  // > 1GB
    assert(status == CFD_ERROR_NOMEM);
    assert(strcmp(cfd_get_last_error(), "Mock allocation failure") == 0);
    printf("  Mock Failure: OK\n");

    // Test 5: cfd_malloc failure check (if we could force it)
    // Hard to force malloc failure without a custom allocator or massive size.
    // We will trust the mock for now.

    // Test 6: Error String Mapping
    assert(strcmp(cfd_get_error_string(CFD_SUCCESS), "Success") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR), "Generic error") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_NOMEM), "Out of memory") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_INVALID), "Invalid argument") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_IO), "I/O error") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_UNSUPPORTED), "Operation not supported") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_DIVERGED), "Solver diverged") == 0);
    assert(strcmp(cfd_get_error_string(CFD_ERROR_MAX_ITER), "Max iterations reached") == 0);
    assert(strcmp(cfd_get_error_string((cfd_status_t)999), "Unknown error") == 0);
    printf("  Error String Mapping: OK\n");

    printf("Error reporting tests passed.\n");
    return 0;
}

int main() {
    test_error_reporting();
    return 0;
}
