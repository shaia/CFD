/**
 * @file turbulence_solver_internal.h
 * @brief Internal workspace-aware turbulence solver interface and model constants
 *
 * Not part of the public API. Used by the NS solvers to avoid per-step
 * allocation of scratch buffers. Each backend (scalar, OMP, AVX2) provides a
 * workspace-aware turbulence step with identical numerics; the matching NS
 * backend calls its own variant so optimized solvers never fall back to the
 * scalar turbulence step.
 */

#ifndef CFD_TURBULENCE_SOLVER_INTERNAL_H
#define CFD_TURBULENCE_SOLVER_INTERNAL_H

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <stddef.h>

/* --- Standard k-epsilon (Launder-Spalding) constants --- */
#define TURB_C_MU      0.09
#define TURB_C1_EPS    1.44
#define TURB_C2_EPS    1.92
#define TURB_SIGMA_K   1.0
#define TURB_SIGMA_EPS 1.3

/* --- Spalart-Allmaras (no-ft2 variant) constants --- */
#define SA_CB1   0.1355
#define SA_CB2   0.622
#define SA_SIGMA (2.0 / 3.0)
#define SA_KAPPA 0.41
#define SA_CW1   (SA_CB1 / (SA_KAPPA * SA_KAPPA) + (1.0 + SA_CB2) / SA_SIGMA)
#define SA_CW2   0.3
#define SA_CW3   2.0
#define SA_CV1   7.1

/* --- Log-law wall function constants --- */
#define WALL_KAPPA         0.41
#define WALL_B             5.2
#define WALL_YPLUS_LAMINAR 11.63  /* linear/log law crossover y+ */

/* --- Positivity floors and realizability limits --- */
#define TURB_K_MIN             1e-10  /* floor for k */
#define TURB_EPS_MIN           1e-10  /* floor for epsilon */
#define TURB_PROD_LIMIT_FACTOR 10.0   /* P_k <= factor * eps */
#define TURB_NU_T_MAX_FACTOR   1e5    /* nu_t <= factor * nu */

/* Workspace requirement: 2*nx*ny*nz doubles (k-epsilon uses both halves,
 * Spalart-Allmaras uses the first half only). */
#define TURB_WORKSPACE_SIZE(total) (2 * (total))

/**
 * Workspace-aware turbulence step (scalar CPU).
 *
 * When workspace is non-NULL and workspace_size >= 2*nx*ny*nz, uses it as
 * scratch instead of allocating. Caller owns the buffer lifetime.
 * When workspace is NULL, allocates internally (same as public API).
 */
cfd_status_t turbulence_step_explicit_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size);

/** OpenMP-parallelized turbulence step (cfd_omp). Same numerics as the scalar
 *  variant; the interior stencil loop is parallelized over j. */
cfd_status_t turbulence_step_explicit_omp_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size);

/** AVX2-vectorized turbulence step (cfd_simd). Same numerics as the scalar
 *  variant; the interior stencil is vectorized along i with a scalar tail. */
cfd_status_t turbulence_step_explicit_avx2_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size);

/** Scalar k-epsilon transport kernel: writes updated k/eps into k_new/eps_new
 *  (interior points only; boundary values are carried over by the caller). */
void turb_kepsilon_step_scalar(const flow_field* field, const grid* grid,
                               const ns_solver_params_t* params, double dt,
                               double* k_new, double* eps_new);

/** Scalar Spalart-Allmaras transport kernel: writes updated nu_tilde into
 *  nt_new (interior points only). */
void turb_sa_step_scalar(const flow_field* field, const grid* grid,
                         const ns_solver_params_t* params, double dt,
                         double* nt_new);

/**
 * Minimum wall distance for point (i,j): distance to the nearest face marked
 * BC_TYPE_NOSLIP (wall-function wall) in tbc. Sets *has_wall to 0 and returns
 * 0.0 when no face is a wall (SA destruction term must then be disabled).
 */
double turb_wall_distance(const grid* grid, const ns_turbulence_bc_config_t* tbc,
                          size_t i, size_t j, int* has_wall);

/** Recompute nu_t from the active model fields with realizability clipping
 *  (nu_t <= TURB_NU_T_MAX_FACTOR * nu). Used after transport and by BCs. */
void turb_update_nu_t(flow_field* field, const ns_solver_params_t* params);

/** Shared argument/grid validation for the turbulence step (all backends):
 *  non-NULL args and fields, known model, 2D only, nx/ny >= 3, uniform
 *  spacing. Assumes turb_model != TURB_MODEL_NONE was already checked. */
cfd_status_t turb_validate_step_args(const flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params);

#endif /* CFD_TURBULENCE_SOLVER_INTERNAL_H */
