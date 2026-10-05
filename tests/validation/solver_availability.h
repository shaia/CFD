/**
 * @file solver_availability.h
 * @brief Create a validation solver, telling a missing backend apart from a failure
 *
 * Validation tests skip a solver whose backend this build or CPU lacks, and must fail
 * on anything else. A NULL from cfd_solver_create() cannot tell the two apart: it also
 * covers an unregistered name (CFD_ERROR_NOT_FOUND) and an allocation failure
 * (CFD_ERROR_NOMEM), and a solver left unregistered because OpenMP or CUDA is not
 * compiled in reports CFD_ERROR_NOT_FOUND too. cfd_solver_create_checked() refuses an
 * unavailable backend with CFD_ERROR_UNSUPPORTED before the name lookup, so that one
 * status is the skip.
 */

#ifndef CFD_VALIDATION_SOLVER_AVAILABILITY_H
#define CFD_VALIDATION_SOLVER_AVAILABILITY_H

#include "cfd/core/cfd_status.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <stdio.h>

/**
 * Create `type` from `registry`.
 *
 * On success returns the solver and sets *unavailable to 0. On failure returns NULL
 * and sets *unavailable to 1 only when the backend is unavailable
 * (CFD_ERROR_UNSUPPORTED), 0 for any other cause, which the caller must report as a
 * failure. When msg is non-NULL, a failure writes a description into it.
 */
static inline ns_solver_t* validation_create_solver(ns_solver_registry_t* registry,
                                                    const char* type, int* unavailable,
                                                    char* msg, size_t msg_size) {
    ns_solver_t* solver = cfd_solver_create_checked(registry, type);
    *unavailable = 0;
    if (!solver) {
        cfd_status_t status = cfd_get_last_status();
        *unavailable = (status == CFD_ERROR_UNSUPPORTED);
        if (msg && msg_size > 0) {
            snprintf(msg, msg_size, "Solver '%s' not created: %s", type,
                     cfd_get_error_string(status));
        }
    }
    return solver;
}

#endif /* CFD_VALIDATION_SOLVER_AVAILABILITY_H */
