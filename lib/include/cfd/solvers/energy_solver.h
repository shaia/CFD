/**
 * @file energy_solver.h
 * @brief Energy equation solver for temperature advection-diffusion
 *
 * Solves the energy equation for incompressible flow:
 *   dT/dt + u*nabla(T) = alpha * nabla^2(T) + Q
 *
 * where alpha = k/(rho*cp) is thermal diffusivity and Q is a heat source.
 *
 * Boussinesq buoyancy coupling adds a body force to the momentum equations:
 *   F_buoy = -rho_0 * beta * (T - T_ref) * g
 *
 * The energy equation is solved as a post-step after velocity advancement,
 * using the updated velocity field for advection.
 */

#ifndef CFD_ENERGY_SOLVER_H
#define CFD_ENERGY_SOLVER_H

#include "cfd/cfd_export.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Advance the temperature field by one explicit Euler step.
 *
 * Computes: T^{n+1} = T^n + dt * (-u*nabla(T) + alpha*nabla^2(T) + Q)
 *
 * Only active when params->alpha > 0. Returns immediately otherwise.
 *
 * @param field  Flow field (T is updated in-place, u/v/w are read-only)
 * @param grid   Computational grid
 * @param params Solver parameters (alpha, heat_source_func, dt)
 * @param dt     Time step size
 * @return CFD_SUCCESS, or CFD_ERROR_DIVERGED if NaN detected in T
 */
CFD_LIBRARY_EXPORT cfd_status_t energy_step_explicit(flow_field* field, const grid* grid,
                                                      const ns_solver_params_t* params,
                                                      double dt);

/**
 * Compute Boussinesq buoyancy source terms from the temperature field.
 *
 * Adds buoyancy force: source_{u,v,w} += -beta * (T[idx] - T_ref) * g_{x,y,z}
 *
 * Only active when params->beta != 0. Does nothing otherwise.
 *
 * @param T_local   Temperature at the current grid point
 * @param params    Solver parameters (beta, T_ref, gravity)
 * @param source_u  Output: added u-momentum buoyancy source
 * @param source_v  Output: added v-momentum buoyancy source
 * @param source_w  Output: added w-momentum buoyancy source
 */
CFD_LIBRARY_EXPORT void energy_compute_buoyancy(double T_local, const ns_solver_params_t* params,
                                                 double* source_u, double* source_v,
                                                 double* source_w);

#ifdef __cplusplus
}
#endif

#endif /* CFD_ENERGY_SOLVER_H */
