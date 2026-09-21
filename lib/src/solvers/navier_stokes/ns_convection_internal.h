/**
 * @file ns_convection_internal.h
 * @brief Convection-scheme validation and scalar upwind kernels shared by the
 *        Navier-Stokes solvers.
 *
 * Every solver keeps its own central convective derivatives. When
 * `params->convection_scheme` is NS_CONVECTION_SCHEME_UPWIND it overwrites them
 * with the first-order upwind derivatives below, so the central path is left
 * exactly as it was. Divergence and pressure gradients stay central either way.
 */
#ifndef CFD_NS_CONVECTION_INTERNAL_H
#define CFD_NS_CONVECTION_INTERNAL_H

#include "cfd/core/cfd_status.h"
#include "cfd/math/stencils.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stddef.h>

/**
 * Validate the convection scheme at solver init.
 *
 * @param params                   Solver parameters (NULL means defaults)
 * @param backend_supports_upwind  Nonzero if the backend implements upwind
 * @return CFD_SUCCESS, CFD_ERROR_INVALID for an unknown scheme, or
 *         CFD_ERROR_UNSUPPORTED for upwind on a backend without it
 */
static inline cfd_status_t ns_check_convection_scheme(const ns_solver_params_t* params,
                                                      int backend_supports_upwind) {
    if (!params || params->convection_scheme == NS_CONVECTION_SCHEME_CENTRAL) {
        return CFD_SUCCESS;
    }
    if (params->convection_scheme != NS_CONVECTION_SCHEME_UPWIND) {
        cfd_set_error(CFD_ERROR_INVALID, "Unknown convection scheme");
        return CFD_ERROR_INVALID;
    }
    if (!backend_supports_upwind) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "Upwind convection is only supported by the scalar, OpenMP and AVX2 solvers");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/**
 * Validate the pressure walls at solver init.
 *
 * Default (all zero-gradient) walls are what every projection solver has always
 * used and always pass. A solver that cannot honour a prescribed face must say
 * so rather than solve the zero-gradient problem instead of the caller's -- and
 * that includes the time integrators, which solve no Poisson equation at all, so
 * a pressure BC configured on one would silently do nothing.
 *
 * @param params                      Solver parameters (NULL means defaults)
 * @param backend_supports_per_face   Nonzero if this solver honours params.pressure_bc
 * @return CFD_SUCCESS, or CFD_ERROR_UNSUPPORTED
 */
static inline cfd_status_t ns_check_pressure_bc(const ns_solver_params_t* params,
                                                int backend_supports_per_face) {
    if (!params) {
        return CFD_SUCCESS;
    }
    const poisson_walls_t* w = &params->pressure_bc;
    if (w->left == POISSON_WALL_ZERO_GRADIENT
        && w->right == POISSON_WALL_ZERO_GRADIENT
        && w->bottom == POISSON_WALL_ZERO_GRADIENT
        && w->top == POISSON_WALL_ZERO_GRADIENT
        && w->front == POISSON_WALL_ZERO_GRADIENT
        && w->back == POISSON_WALL_ZERO_GRADIENT) {
        return CFD_SUCCESS;
    }
    if (!backend_supports_per_face) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "params.pressure_bc is honoured by the scalar, OpenMP and AVX2 "
                      "projection solvers only");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/**
 * Validate the pressure-solver selection at solver init.
 *
 * The analogue of ns_check_pressure_bc, and missing until now, which is why
 * NS_PRESSURE_SOLVER_MULTIGRID on rk4 or explicit_euler was accepted and did
 * nothing: those solvers run no Poisson solve at all.
 *
 * @param params                          Solver parameters (NULL means defaults)
 * @param backend_supports_pressure_solver Nonzero if this solver runs a pressure
 *                                        solve it can choose the method for
 * @return CFD_SUCCESS, CFD_ERROR_INVALID for an unknown value, or
 *         CFD_ERROR_UNSUPPORTED
 */
static inline cfd_status_t ns_check_pressure_solver(const ns_solver_params_t* params,
                                                    int backend_supports_pressure_solver) {
    if (!params || params->pressure_solver == NS_PRESSURE_SOLVER_DEFAULT) {
        return CFD_SUCCESS;
    }
    if (params->pressure_solver != NS_PRESSURE_SOLVER_MULTIGRID &&
        params->pressure_solver != NS_PRESSURE_SOLVER_PCG_MG) {
        cfd_set_error(CFD_ERROR_INVALID, "Unknown ns_solver_params_t.pressure_solver");
        return CFD_ERROR_INVALID;
    }
    if (!backend_supports_pressure_solver) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "params.pressure_solver is honoured by the scalar and OpenMP "
                      "projection solvers only");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/**
 * Validate the turbulence model at solver init.
 *
 * The GPU backends have no RANS kernels and were rejecting this per step, from
 * inside solve_projection_method_gpu and its siblings. That is late: a caller
 * gets the refusal after init has reported success, which is the one place they
 * could still have chosen a different backend.
 *
 * @param params                       Solver parameters (NULL means defaults)
 * @param backend_supports_turbulence  Nonzero if this backend implements RANS
 * @return CFD_SUCCESS, CFD_ERROR_INVALID for an unknown model, or
 *         CFD_ERROR_UNSUPPORTED
 */
static inline cfd_status_t ns_check_turbulence_model(const ns_solver_params_t* params,
                                                     int backend_supports_turbulence) {
    if (!params || params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }
    if (params->turb_model != TURB_MODEL_K_EPSILON &&
        params->turb_model != TURB_MODEL_SPALART_ALLMARAS) {
        cfd_set_error(CFD_ERROR_INVALID, "Unknown ns_solver_params_t.turb_model");
        return CFD_ERROR_INVALID;
    }
    if (!backend_supports_turbulence) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "RANS turbulence models are implemented on the scalar, OpenMP and "
                      "AVX2 solvers only; the GPU backends have no RANS kernels");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/** Convective first derivatives of (u, v, w) at one grid point */
typedef struct {
    double du_dx, du_dy, du_dz;
    double dv_dx, dv_dy, dv_dz;
    double dw_dx, dw_dy, dw_dz;
} ns_conv_derivs_t;

/**
 * First-order upwind convective derivatives of (u, v, w) at point idx, each
 * advected by the point velocity (u[idx], v[idx], w[idx]).
 *
 * @param il,ir,jd,ju,kd,ku  Neighbor indices (ghost-cell or periodic, as the
 *                           calling kernel uses for its central stencil)
 * @param dx,dy              Grid spacing at the point (must be > 0)
 * @param inv_dz             1/dz, or 0.0 when the z-axis is inactive
 */
static inline void ns_upwind_conv_derivs(const double* u, const double* v, const double* w,
                                         size_t idx, size_t il, size_t ir,
                                         size_t jd, size_t ju, size_t kd, size_t ku,
                                         double dx, double dy, double inv_dz,
                                         ns_conv_derivs_t* cd) {
    const double u_c = u[idx];
    const double v_c = v[idx];
    const double w_c = w[idx];

    cd->du_dx = stencil_upwind_deriv_x(u[ir], u_c, u[il], dx, u_c);
    cd->du_dy = stencil_upwind_deriv_y(u[ju], u_c, u[jd], dy, v_c);
    cd->du_dz = stencil_upwind_diff(u[ku], u_c, u[kd], w_c) * inv_dz;

    cd->dv_dx = stencil_upwind_deriv_x(v[ir], v_c, v[il], dx, u_c);
    cd->dv_dy = stencil_upwind_deriv_y(v[ju], v_c, v[jd], dy, v_c);
    cd->dv_dz = stencil_upwind_diff(v[ku], v_c, v[kd], w_c) * inv_dz;

    cd->dw_dx = stencil_upwind_deriv_x(w[ir], w_c, w[il], dx, u_c);
    cd->dw_dy = stencil_upwind_deriv_y(w[ju], w_c, w[jd], dy, v_c);
    cd->dw_dz = stencil_upwind_diff(w[ku], w_c, w[kd], w_c) * inv_dz;
}

/** Clamp every convective derivative to [-limit, limit] */
static inline void ns_clamp_conv_derivs(ns_conv_derivs_t* cd, double limit) {
    cd->du_dx = fmax(-limit, fmin(limit, cd->du_dx));
    cd->du_dy = fmax(-limit, fmin(limit, cd->du_dy));
    cd->du_dz = fmax(-limit, fmin(limit, cd->du_dz));
    cd->dv_dx = fmax(-limit, fmin(limit, cd->dv_dx));
    cd->dv_dy = fmax(-limit, fmin(limit, cd->dv_dy));
    cd->dv_dz = fmax(-limit, fmin(limit, cd->dv_dz));
    cd->dw_dx = fmax(-limit, fmin(limit, cd->dw_dx));
    cd->dw_dy = fmax(-limit, fmin(limit, cd->dw_dy));
    cd->dw_dz = fmax(-limit, fmin(limit, cd->dw_dz));
}

#endif /* CFD_NS_CONVECTION_INTERNAL_H */
