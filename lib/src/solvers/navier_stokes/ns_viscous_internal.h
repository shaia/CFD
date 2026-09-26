/**
 * @file ns_viscous_internal.h
 * @brief The implicit viscous solve a projection solver owns.
 *
 * With an implicit ns_solver_params_t.viscous_scheme, the projection predictor
 * turns its explicit increment b = u*_explicit - u^n into
 *
 *     (I - theta * nu * dt * lap) delta = b,   u* = u^n + delta
 *
 * which is the shifted Poisson problem
 *
 *     lap(delta) - sigma * delta = -sigma * b,   sigma = 1 / (theta * nu * dt)
 *
 * with delta = 0 on every boundary node: the predictor holds boundary values at
 * their step-start values for the whole step, so the increment there is zero by
 * construction rather than by assumption. The Poisson Dirichlet walls are
 * node-based (the halo node itself takes the value), which is the velocity
 * grid's boundary node exactly.
 *
 * The mapping from params to a solver configuration lives here, shared by the
 * scalar and OpenMP projections, for the same reason ns_pressure_internal.h
 * exists: two copies of it would drift. The per-point loops stay in each
 * backend, like every other kernel.
 */
#ifndef CFD_NS_VISCOUS_INTERNAL_H
#define CFD_NS_VISCOUS_INTERNAL_H

#include "cfd/core/cfd_status.h"
#include "cfd/core/logging.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include <math.h>

/**
 * The theta of the viscous scheme: 0 for explicit, 1 for backward Euler, 1/2 for
 * Crank-Nicolson.
 *
 * Read on every step, not only at init, so a turbulence model switched on after
 * init is still refused: nu + nu_t varies in space, and the shifted operator
 * has one constant coefficient.
 *
 * @return CFD_SUCCESS, CFD_ERROR_INVALID for an unknown scheme, or
 *         CFD_ERROR_UNSUPPORTED for an implicit scheme with a turbulence model
 */
static inline cfd_status_t ns_viscous_theta(const ns_solver_params_t* params, double* theta) {
    *theta = 0.0;
    if (!params) {
        return CFD_SUCCESS;
    }
    switch (params->viscous_scheme) {
        case NS_VISCOUS_SCHEME_EXPLICIT:
            return CFD_SUCCESS;
        case NS_VISCOUS_SCHEME_BACKWARD_EULER:
            *theta = 1.0;
            break;
        case NS_VISCOUS_SCHEME_CRANK_NICOLSON:
            *theta = 0.5;
            break;
        default:
            cfd_set_error(CFD_ERROR_INVALID, "Unknown ns_solver_params_t.viscous_scheme");
            return CFD_ERROR_INVALID;
    }
    if (params->turb_model != TURB_MODEL_NONE) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "An implicit params.viscous_scheme cannot be combined with a "
                      "turbulence model: nu + nu_t varies in space and the implicit "
                      "operator has constant coefficients");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/**
 * The Helmholtz shift sigma = 1 / (theta * nu * dt), or 0 when there is nothing
 * to solve: an explicit scheme, or no viscous term (nu <= 0), where sigma would
 * be infinite. A non-positive dt is left to the caller's own dt checks.
 */
static inline double ns_viscous_shift(double theta, double nu, double dt) {
    double denom = theta * nu * dt;
    return (denom > 0.0 && isfinite(1.0 / denom)) ? 1.0 / denom : 0.0;
}

/**
 * The Poisson configuration for the implicit viscous solve on `backend`.
 *
 * CG, because it is the one method the shift is implemented for, at the
 * ACCURATE tolerance: the shifted operator is well conditioned enough that the
 * extra digits cost a few iterations, and a looser solve would show up as a
 * floor in temporal-convergence measurements. Homogeneous Dirichlet on every
 * face that exists; the z-faces stay zero-gradient on a 2D grid, which has none.
 */
static inline poisson_solver_config_t ns_viscous_config(poisson_solver_backend_t backend,
                                                        double sigma, size_t nz) {
    poisson_solver_config_t cfg = poisson_solver_config_preset(POISSON_PRESET_ACCURATE);
    cfg.method = POISSON_METHOD_CG;
    cfg.backend = backend;
    cfg.params.walls = poisson_walls_default();
    cfg.params.walls.left = cfg.params.walls.right = POISSON_WALL_DIRICHLET;
    cfg.params.walls.bottom = cfg.params.walls.top = POISSON_WALL_DIRICHLET;
    if (nz > 1) {
        cfg.params.walls.front = cfg.params.walls.back = POISSON_WALL_DIRICHLET;
    }
    cfg.params.helmholtz_shift = sigma;
    return cfg;
}

#endif /* CFD_NS_VISCOUS_INTERNAL_H */
