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

/* --- Optional eddy-viscosity corrections (algebraic or learned) --- */

/** Features per cell: ln S*, ln Re_t, ln(nu_t/nu). The model's input width must
 *  equal this exactly; a model of any other shape is rejected at solver init. */
#define TURB_CLOSURE_FEATURES 3
/** The model predicts one multiplier per cell. */
#define TURB_CLOSURE_OUTPUTS 1
/** Bounds on the predicted multiplier. Both directions matter: the correction
 *  the channel DNS asks for is below 1 (k-epsilon over-predicts turbulent
 *  energy in the outer layer), so this is NOT a dissipation-only correction and
 *  the floor is what keeps a reduced nu_t away from zero. */
#define TURB_CLOSURE_BETA_MIN 0.1
#define TURB_CLOSURE_BETA_MAX 10.0
/** Cells per inference call. The correction walks the grid in tiles so it needs
 *  no per-step allocation and no context sized to the grid; a smaller context
 *  just means smaller tiles. */
#define TURB_CLOSURE_TILE 256

/* --- Algebraic correction: beta = A * (S*)^B, i.e. a strain-dependent C_mu ---
 *
 * Fitted by least squares on ln(k_dns / k_model) against ln S* at Re_tau =
 * 392.24 (9 comparison nodes), then evaluated at the held-out Re_tau = 587.19.
 * The target is the TKE ratio rather than the shear-stress ratio because in
 * fully developed channel flow the total shear stress is fixed by the mean
 * momentum balance, so -uv+ is nearly closure-independent and its apparent
 * error is dominated by discretisation. k is a genuine closure output.
 *
 * THE SIGN IS NOT THE INTUITIVE ONE, and the first version of these constants
 * had it backwards. Multiplying nu_t by beta is the same as using
 * C_mu' = beta * C_mu. In a stress-constrained flow the shear stress is set by
 * the mean momentum balance, not by the closure, so the strain rate absorbs any
 * change in nu_t: tau = nu_t |S| is fixed, P_k = tau^2 / nu_t, and local
 * equilibrium (P_k = eps) with nu_t = C_mu' k^2/eps gives
 *
 *     k = tau / sqrt(C_mu')  =>  k scales as beta^(-1/2).
 *
 * LOWERING nu_t therefore RAISES k. Measured: beta = 0.90 predicts +5.4% in k
 * and the run gave +6.0%. So to scale k by a factor r the multiplier must be
 * beta = r^(-2), which is how the fitted TKE ratio is converted below.
 *
 * Provenance, the failed first attempt and the a posteriori numbers are in
 * docs/technical-notes/ml-integration-design.md 2.7. These are fitted
 * constants, not derived ones -- treat them as a measured baseline for a
 * learned closure to beat, not as physics. */
#define TURB_ALG_BETA_A 1.6945
#define TURB_ALG_BETA_B (-0.2778)

/**
 * Apply whichever optional eddy-viscosity correction params selects: the
 * algebraic strain-rate power law, or the learned closure.
 *
 * Call immediately after turb_update_nu_t(). Returns CFD_SUCCESS and touches
 * nothing when neither is configured, so the un-corrected path stays
 * bit-identical. Additive by design: turb_update_nu_t keeps its signature and
 * its behaviour.
 *
 * @return CFD_SUCCESS; CFD_ERROR_UNSUPPORTED if a correction is configured with
 *         any model other than k-epsilon (S*, Re_t and nu_t/nu are built from k
 *         and epsilon, so they do not exist for Spalart-Allmaras), or if both
 *         corrections are configured at once; CFD_ERROR_INVALID for an unknown
 *         correction value, a NULL field/grid, a missing turbulence field or a
 *         model of the wrong shape; CFD_ERROR_DIVERGED if the model predicts a
 *         non-finite value. Never a silent skip: a caller that asked for a
 *         correction gets one or an error.
 */
cfd_status_t turb_apply_nu_t_correction(flow_field* field, const grid* grid,
                                        const ns_solver_params_t* params);

/**
 * Validate params->turb_closure and params->turb_nut_correction against
 * params->turb_model at solver init.
 *
 * Checked here rather than per step so a caller learns before init returns,
 * while they can still choose a different configuration.
 *
 * @return CFD_SUCCESS when no correction is set or the configuration is usable;
 *         CFD_ERROR_UNSUPPORTED when a correction is set without k-epsilon, or
 *         when both corrections are set at once;
 *         CFD_ERROR_INVALID for a value no enum defines, or when the model's
 *         input or output width does not match what the closure feeds it.
 */
cfd_status_t turb_check_closure_config(const ns_solver_params_t* params);

/** Shared argument/grid validation for the turbulence step (all backends):
 *  non-NULL args and fields, known model, 2D only, nx/ny >= 3, uniform
 *  spacing. Assumes turb_model != TURB_MODEL_NONE was already checked. */
cfd_status_t turb_validate_step_args(const flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params);

#endif /* CFD_TURBULENCE_SOLVER_INTERNAL_H */
