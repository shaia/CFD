/**
 * @file ns_dt_internal.h
 * @brief Per-process time-step stability constraints.
 *
 * `compute_time_step()` is the minimum of these. Each returns the largest
 * stable dt for one physical process, or INFINITY when that process imposes no
 * limit, so callers compose them with a plain minimum and a solver that treats
 * a term implicitly simply leaves that term's constraint out.
 *
 * Defined in cpu/solver_explicit_euler.c alongside compute_time_step().
 */
#ifndef CFD_NS_DT_INTERNAL_H
#define CFD_NS_DT_INTERNAL_H

#include "cfd/api/simulation_api.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

/** Smallest grid spacing over the active dimensions. */
double ns_grid_min_spacing(const grid* grid);

/** Advective/acoustic CFL limit. */
double ns_dt_convective(const flow_field* field, const grid* grid,
                        const ns_solver_params_t* params);

/** Viscous diffusion limit (molecular + eddy viscosity); INFINITY if inviscid. */
double ns_dt_viscous(const flow_field* field, const grid* grid,
                     const ns_solver_params_t* params);

/** Thermal diffusion limit; INFINITY when the energy equation is inactive. */
double ns_dt_thermal(const flow_field* field, const grid* grid,
                     const ns_solver_params_t* params);

#endif /* CFD_NS_DT_INTERNAL_H */
