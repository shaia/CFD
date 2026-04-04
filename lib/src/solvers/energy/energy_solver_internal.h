/**
 * @file energy_solver_internal.h
 * @brief Internal workspace-aware energy solver interface
 *
 * Not part of the public API. Used by scalar CPU NS solvers to avoid
 * per-step allocation of the T_new scratch buffer.
 */

#ifndef CFD_ENERGY_SOLVER_INTERNAL_H
#define CFD_ENERGY_SOLVER_INTERNAL_H

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <stddef.h>

/**
 * Workspace-aware energy step.
 *
 * When T_workspace is non-NULL and workspace_size >= nx*ny*nz, uses it
 * as scratch instead of allocating. Caller owns the buffer lifetime.
 * When T_workspace is NULL, allocates internally (same as public API).
 */
cfd_status_t energy_step_explicit_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* T_workspace, size_t workspace_size);

#endif /* CFD_ENERGY_SOLVER_INTERNAL_H */
