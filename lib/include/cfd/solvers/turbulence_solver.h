/**
 * @file turbulence_solver.h
 * @brief RANS turbulence models: standard k-epsilon and Spalart-Allmaras
 *
 * Solves the turbulence transport equations as a post-step after velocity
 * advancement (mirroring the energy equation), using the updated velocity
 * field. The resulting eddy viscosity nu_t enters the momentum equations
 * through a conservative face-averaged effective viscosity nu + nu_t.
 *
 * k-epsilon (Launder-Spalding, high-Reynolds form, wall functions required):
 *   dk/dt   + u*grad(k)   = div((nu + nu_t/sigma_k) grad k)   + P_k - eps
 *   deps/dt + u*grad(eps) = div((nu + nu_t/sigma_e) grad eps) + (eps/k)(C1*P_k - C2*eps)
 *   nu_t = C_mu * k^2 / eps
 *
 * Spalart-Allmaras (no-ft2 variant, fully turbulent):
 *   dnt/dt + u*grad(nt) = (1/sigma)[div((nu+nt) grad nt) + cb2*|grad nt|^2]
 *                         + cb1*S_tilde*nt - cw1*fw*(nt/d)^2
 *   nu_t = nt * fv1
 *
 * Discretization: first-order upwind advection (positivity), conservative
 * face-averaged diffusion, semi-implicit (Patankar) sink treatment so the
 * destruction terms can never drive k/eps/nt negative. Sources use central
 * gradients. 2D uniform grids only in this version; the momentum-equation
 * Boussinesq-stress transpose term and the -(2/3)k*delta_ij term are omitted
 * (the latter is absorbed into a modified pressure, standard practice).
 *
 * Wall functions (faces marked BC_TYPE_NOSLIP in params->turb_bc): u_tau from
 * the law of the wall in params->turb_bc.wall_law -- the linear/log law
 * (default) or Spalding's smooth law; see ns_wall_law_t.
 */

#ifndef CFD_TURBULENCE_SOLVER_H
#define CFD_TURBULENCE_SOLVER_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Advance the turbulence transport equations by one explicit step.
 *
 * Model chosen by params->turb_model; no-op returning CFD_SUCCESS when
 * TURB_MODEL_NONE. Updates turb_k/turb_eps (k-epsilon) or turb_nu_tilde
 * (Spalart-Allmaras) in-place using the current velocity field, then
 * recomputes nu_t (clipped for realizability).
 *
 * @param field  Flow field (turbulence arrays updated, u/v read-only)
 * @param grid   Computational grid (2D uniform spacing required)
 * @param params Solver parameters (turb_model, turb_bc, mu)
 * @param dt     Time step size
 * @param time   Current physical time
 * @return CFD_SUCCESS; CFD_ERROR_UNSUPPORTED for 3D or non-uniform grids;
 *         CFD_ERROR_DIVERGED if NaN/Inf detected
 */
CFD_LIBRARY_EXPORT cfd_status_t turbulence_step_explicit(flow_field* field, const grid* grid,
                                                         const ns_solver_params_t* params,
                                                         double dt, double time);

/**
 * Apply per-face turbulence boundary conditions, including wall functions.
 *
 * Face types (params->turb_bc): PERIODIC (default), NEUMANN (zero-gradient),
 * DIRICHLET (fixed values), NOSLIP (wall function on the law in
 * params->turb_bc.wall_law: sets equilibrium k/eps or nu_tilde at the first
 * interior node and a first-node nu_t chosen so the discrete wall shear is
 * exactly u_tau^2, with u_tau from turbulence_wall_u_tau()).
 *
 * Faces are applied in the order left, right, bottom, top; later faces
 * overwrite shared corner cells (same precedence as energy_apply_thermal_bcs).
 *
 * No-op returning CFD_SUCCESS when params->turb_model == TURB_MODEL_NONE.
 *
 * @return CFD_SUCCESS, CFD_ERROR_INVALID for NULL args / unsupported face
 *         types / an unknown turb_bc.wall_law / too-small grids,
 *         CFD_ERROR_UNSUPPORTED for 3D grids.
 */
CFD_LIBRARY_EXPORT cfd_status_t turbulence_apply_bcs(flow_field* field, const grid* grid,
                                                     const ns_solver_params_t* params);

/**
 * Initialize the active turbulence fields uniformly and set nu_t consistently.
 *
 * k-epsilon: k0 and eps0 are used (nu_tilde0 ignored); nu_t = C_mu*k0^2/eps0.
 * Spalart-Allmaras: nu_tilde0 is used (k0/eps0 ignored); nu_t = nu_tilde0*fv1.
 * No-op returning CFD_SUCCESS when TURB_MODEL_NONE.
 */
CFD_LIBRARY_EXPORT cfd_status_t turbulence_init_uniform(flow_field* field,
                                                        const ns_solver_params_t* params,
                                                        double k0, double eps0,
                                                        double nu_tilde0);

/**
 * Friction velocity u_tau from a law of the wall -- the value the wall function
 * imposes for that law.
 *
 * Given the wall-parallel speed u_p at wall distance y_p and kinematic
 * viscosity nu, returns u_tau such that u+ = u_p/u_tau and y+ = u_tau*y_p/nu
 * satisfy the selected law (kappa = 0.41, B = 5.2):
 *
 *  - NS_WALL_LAW_LOG: u+ = y+ while u_p*y_p/nu <= y+_c^2, else
 *    u+ = ln(y+)/kappa + B (Newton iteration), with y+_c = 11.06 where the two
 *    meet, so u_tau is continuous in u_p.
 *  - NS_WALL_LAW_SPALDING:
 *      y+ = u+ + e^{-kappa B} [e^{kappa u+} - 1 - kappa u+ - (kappa u+)^2/2 - (kappa u+)^3/6]
 *    (safeguarded Newton iteration), smooth through all three layers.
 *
 * Both are increasing in u_p. Returns 0.0 for non-positive inputs or an
 * unknown law. See ns_wall_law_t for which to choose.
 */
CFD_LIBRARY_EXPORT double turbulence_wall_u_tau(ns_wall_law_t law, double u_p, double y_p,
                                                double nu);

#ifdef __cplusplus
}
#endif

#endif /* CFD_TURBULENCE_SOLVER_H */
