/**
 * @file turbulence_solver.c
 * @brief Scalar CPU implementation of the RANS turbulence solver
 *
 * Public entry points, model dispatch, boundary conditions, and the standard
 * log-law wall functions. The per-model transport kernels live in
 * turbulence_kepsilon.c and turbulence_sa.c.
 */

#include "cfd/solvers/turbulence_solver.h"
#include "../turbulence_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

/* Kinematic viscosity at a grid point (density-floored, like the RHS kernel). */
static double local_nu(const ns_solver_params_t* params, const flow_field* field,
                       size_t idx) {
    return params->mu / fmax(field->rho[idx], 1e-10);
}

/* SA viscous damping function fv1(chi). */
static double sa_fv1(double chi) {
    double chi3 = chi * chi * chi;
    return chi3 / (chi3 + SA_CV1 * SA_CV1 * SA_CV1);
}

void turb_update_nu_t(flow_field* field, const ns_solver_params_t* params) {
    size_t total = field->nx * field->ny * field->nz;

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        for (size_t n = 0; n < total; n++) {
            double nu = local_nu(params, field, n);
            double k_c = fmax(field->turb_k[n], 0.0);
            double eps_c = fmax(field->turb_eps[n], TURB_EPS_MIN);
            double nu_t = TURB_C_MU * k_c * k_c / eps_c;
            field->nu_t[n] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
        }
    } else if (params->turb_model == TURB_MODEL_SPALART_ALLMARAS) {
        for (size_t n = 0; n < total; n++) {
            double nu = local_nu(params, field, n);
            double nt = fmax(field->turb_nu_tilde[n], 0.0);
            double nu_t = nt * sa_fv1(nt / nu);
            field->nu_t[n] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
        }
    }
}

double turb_wall_distance(const grid* grid, const ns_turbulence_bc_config_t* tbc,
                          size_t i, size_t j, int* has_wall) {
    double d = 0.0;
    int found = 0;

    if (tbc->left == BC_TYPE_NOSLIP) {
        double c = grid->x[i] - grid->x[0];
        d = found ? fmin(d, c) : c;
        found = 1;
    }
    if (tbc->right == BC_TYPE_NOSLIP) {
        double c = grid->x[grid->nx - 1] - grid->x[i];
        d = found ? fmin(d, c) : c;
        found = 1;
    }
    if (tbc->bottom == BC_TYPE_NOSLIP) {
        double c = grid->y[j] - grid->y[0];
        d = found ? fmin(d, c) : c;
        found = 1;
    }
    if (tbc->top == BC_TYPE_NOSLIP) {
        double c = grid->y[grid->ny - 1] - grid->y[j];
        d = found ? fmin(d, c) : c;
        found = 1;
    }

    *has_wall = found;
    return d;
}

double turbulence_wall_u_tau(double u_p, double y_p, double nu) {
    if (u_p <= 0.0 || y_p <= 0.0 || nu <= 0.0) {
        return 0.0;
    }

    /* Viscous sublayer (linear law u+ = y+): u_tau = sqrt(nu*u_p/y_p) */
    double ut = sqrt(nu * u_p / y_p);
    if (ut * y_p / nu < WALL_YPLUS_LAMINAR) {
        return ut;
    }

    /* Log law: solve ut*(ln(ut*y_p/nu)/kappa + B) = u_p by Newton iteration */
    for (int it = 0; it < 20; it++) {
        double yplus = ut * y_p / nu;
        double f = ut * (log(yplus) / WALL_KAPPA + WALL_B) - u_p;
        double fp = (log(yplus) + 1.0) / WALL_KAPPA + WALL_B;
        double ut_new = ut - f / fp;
        if (ut_new < 1e-12) {
            ut_new = 1e-12;
        }
        double converged = fabs(ut_new - ut) < 1e-12 * fmax(ut, 1.0);
        ut = ut_new;
        if (converged) {
            break;
        }
    }
    return ut;
}

/* Validate uniform spacing (the transport stencils assume it, like energy). */
static cfd_status_t validate_uniform_spacing(const grid* grid, size_t nx, size_t ny) {
    const double dx0 = grid->dx[0];
    const double tol_x = 1e-12 * fmax(1.0, fabs(dx0));
    for (size_t i = 1; i < nx - 1; i++) {
        if (fabs(grid->dx[i] - dx0) > tol_x) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                          "turbulence_solver: non-uniform dx not supported");
            return CFD_ERROR_UNSUPPORTED;
        }
    }
    const double dy0 = grid->dy[0];
    const double tol_y = 1e-12 * fmax(1.0, fabs(dy0));
    for (size_t j = 1; j < ny - 1; j++) {
        if (fabs(grid->dy[j] - dy0) > tol_y) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                          "turbulence_solver: non-uniform dy not supported");
            return CFD_ERROR_UNSUPPORTED;
        }
    }
    return CFD_SUCCESS;
}

/* Shared validation for step and BC application. */
static cfd_status_t validate_turbulence_args(const flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params) {
    if (!field || !grid || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: field, grid, and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (!field->turb_k || !field->turb_eps || !field->turb_nu_tilde || !field->nu_t) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: missing turbulence fields");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model != TURB_MODEL_K_EPSILON &&
        params->turb_model != TURB_MODEL_SPALART_ALLMARAS) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: unknown turbulence model");
        return CFD_ERROR_INVALID;
    }
    /* nu = mu/rho appears in denominators (SA chi, wall functions); mu <= 0
     * would produce Inf/NaN rather than a diagnosable error. */
    if (params->mu <= 0.0) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: mu must be positive when a "
                      "turbulence model is active");
        return CFD_ERROR_INVALID;
    }
    if (field->nz > 1) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "turbulence_solver: 3D turbulence not supported");
        return CFD_ERROR_UNSUPPORTED;
    }
    if (!grid->dx || !grid->dy || field->nx < 3 || field->ny < 3) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver: grid too small or missing dx/dy");
        return CFD_ERROR_INVALID;
    }
    return CFD_SUCCESS;
}

cfd_status_t turb_validate_step_args(const flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params) {
    cfd_status_t status = validate_turbulence_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return validate_uniform_spacing(grid, field->nx, field->ny);
}

cfd_status_t turbulence_step_explicit_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size) {
    (void)time;

    if (!params) {
        cfd_set_error(CFD_ERROR_INVALID, "turbulence_solver: params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    /* Turbulence disabled: a legitimate no-op. */
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }

    cfd_status_t status = turb_validate_step_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }

    size_t total = field->nx * field->ny * field->nz;

    /* Use caller's workspace or allocate internally */
    int owns_buffer = 0;
    double* buf;
    if (workspace && workspace_size >= TURB_WORKSPACE_SIZE(total)) {
        buf = workspace;
    } else {
        buf = (double*)cfd_calloc(TURB_WORKSPACE_SIZE(total), sizeof(double));
        if (!buf) {
            return CFD_ERROR_NOMEM;
        }
        owns_buffer = 1;
    }

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        double* k_new = buf;
        double* eps_new = buf + total;
        memcpy(k_new, field->turb_k, total * sizeof(double));
        memcpy(eps_new, field->turb_eps, total * sizeof(double));

        turb_kepsilon_step_scalar(field, grid, params, dt, k_new, eps_new);

        for (size_t n = 0; n < total; n++) {
            if (!isfinite(k_new[n]) || !isfinite(eps_new[n])) {
                cfd_set_error(CFD_ERROR_DIVERGED,
                              "NaN/Inf detected in turbulence_step_explicit (k-epsilon)");
                if (owns_buffer) cfd_free(buf);
                return CFD_ERROR_DIVERGED;
            }
        }
        memcpy(field->turb_k, k_new, total * sizeof(double));
        memcpy(field->turb_eps, eps_new, total * sizeof(double));
    } else {
        double* nt_new = buf;
        memcpy(nt_new, field->turb_nu_tilde, total * sizeof(double));

        turb_sa_step_scalar(field, grid, params, dt, nt_new);

        for (size_t n = 0; n < total; n++) {
            if (!isfinite(nt_new[n])) {
                cfd_set_error(CFD_ERROR_DIVERGED,
                              "NaN/Inf detected in turbulence_step_explicit (SA)");
                if (owns_buffer) cfd_free(buf);
                return CFD_ERROR_DIVERGED;
            }
        }
        memcpy(field->turb_nu_tilde, nt_new, total * sizeof(double));
    }

    turb_update_nu_t(field, params);

    if (owns_buffer) cfd_free(buf);
    return CFD_SUCCESS;
}

cfd_status_t turbulence_step_explicit(flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params,
                                      double dt, double time) {
    return turbulence_step_explicit_with_workspace(field, grid, params,
                                                   dt, time, NULL, 0);
}

/* A face may only request a turbulence BC type the solver implements. */
static int is_supported_turb_bc(bc_type_t type) {
    return type == BC_TYPE_PERIODIC || type == BC_TYPE_NEUMANN ||
           type == BC_TYPE_DIRICHLET || type == BC_TYPE_NOSLIP;
}

/*
 * Standard log-law wall function at one wall node.
 *
 * idx_w: wall node, idx_p: first interior node at distance y_p, u_p: wall-
 * parallel speed at idx_p. Sets equilibrium turbulence values at the first
 * interior node (fixed-value wall-function variant):
 *   k_p  = u_tau^2 / sqrt(C_mu)      eps_p = u_tau^3 / (kappa*y_p)
 *   nt_p = kappa*u_tau*y_p
 * and imposes the log-law momentum sink through the wall-face viscosity: the
 * discrete wall shear is (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p, so storing
 *   nu_t_w = 0,   nu_t_p = max(2*(u_tau^2*y_p/u_p - nu), 0)
 * makes it exactly u_tau^2. The equilibrium eddy viscosity kappa*u_tau*y_p is
 * deliberately NOT stored at idx_p: the coarse one-sided gradient u_p/y_p
 * cannot represent the log profile's curvature, and pairing it with the
 * physical nu_t would overpredict the wall shear several-fold. In the viscous
 * sublayer u_tau^2 = nu*u_p/y_p, so nu_t_p reduces to 0 (pure laminar shear).
 */
static void apply_wall_function_node(flow_field* field,
                                     const ns_solver_params_t* params,
                                     size_t idx_w, size_t idx_p,
                                     double y_p, double u_p) {
    double nu = local_nu(params, field, idx_p);
    double ut = turbulence_wall_u_tau(u_p, y_p, nu);

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        field->turb_k[idx_p] = fmax(ut * ut / sqrt(TURB_C_MU), TURB_K_MIN);
        field->turb_eps[idx_p] = fmax(ut * ut * ut / (WALL_KAPPA * y_p), TURB_EPS_MIN);
        field->turb_k[idx_w] = 0.0;
        field->turb_eps[idx_w] = field->turb_eps[idx_p];
    } else {
        field->turb_nu_tilde[idx_p] = WALL_KAPPA * ut * y_p;
        field->turb_nu_tilde[idx_w] = 0.0;
    }

    /* Wall-shear-matching viscosity (see function comment) */
    if (u_p > 1e-12) {
        double nu_wall_eff = ut * ut * y_p / u_p;
        field->nu_t[idx_p] = fmax(2.0 * (nu_wall_eff - nu), 0.0);
    } else {
        field->nu_t[idx_p] = 0.0;
    }
    field->nu_t[idx_w] = 0.0;
}

/* Apply one BC type to one scalar array cell (periodic/neumann/dirichlet). */
static void apply_scalar_face_bc(double* arr, bc_type_t type, size_t idx,
                                 size_t idx_interior, size_t idx_periodic,
                                 double dirichlet_value) {
    if (type == BC_TYPE_DIRICHLET) {
        arr[idx] = dirichlet_value;
    } else if (type == BC_TYPE_NEUMANN) {
        arr[idx] = arr[idx_interior];
    } else if (type == BC_TYPE_PERIODIC) {
        arr[idx] = arr[idx_periodic];
    }
    /* BC_TYPE_NOSLIP handled by the wall-function path, not here. */
}

/* Set nu_t at a Dirichlet boundary node consistently with the fixed values. */
static void set_dirichlet_nu_t(flow_field* field, const ns_solver_params_t* params,
                               size_t idx) {
    double nu = local_nu(params, field, idx);
    double nu_t;
    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        double k_c = fmax(field->turb_k[idx], 0.0);
        double eps_c = fmax(field->turb_eps[idx], TURB_EPS_MIN);
        nu_t = TURB_C_MU * k_c * k_c / eps_c;
    } else {
        double nt = fmax(field->turb_nu_tilde[idx], 0.0);
        nu_t = nt * sa_fv1(nt / nu);
    }
    field->nu_t[idx] = fmin(nu_t, TURB_NU_T_MAX_FACTOR * nu);
}

cfd_status_t turbulence_apply_bcs(flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params) {
    if (!params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_apply_bcs: params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }

    cfd_status_t status = validate_turbulence_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }

    const ns_turbulence_bc_config_t* tbc = &params->turb_bc;
    size_t nx = field->nx;
    size_t ny = field->ny;

    if (!is_supported_turb_bc(tbc->left) || !is_supported_turb_bc(tbc->right) ||
        !is_supported_turb_bc(tbc->bottom) || !is_supported_turb_bc(tbc->top)) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_apply_bcs: unsupported turbulence BC type on a face "
                      "(only PERIODIC, NEUMANN, DIRICHLET, NOSLIP are valid)");
        return CFD_ERROR_INVALID;
    }

    const int is_ke = (params->turb_model == TURB_MODEL_K_EPSILON);

    /* Left face (i=0) */
    if (tbc->left == BC_TYPE_NOSLIP) {
        double y_p = grid->x[1] - grid->x[0];
        for (size_t j = 0; j < ny; j++) {
            size_t idx_w = j * nx;
            size_t idx_p = idx_w + 1;
            apply_wall_function_node(field, params, idx_w, idx_p, y_p,
                                     fabs(field->v[idx_p]));
        }
    } else {
        for (size_t j = 0; j < ny; j++) {
            size_t idx = j * nx;
            size_t idx_int = idx + 1;
            size_t idx_per = j * nx + (nx - 2);
            if (is_ke) {
                apply_scalar_face_bc(field->turb_k, tbc->left, idx, idx_int, idx_per,
                                     tbc->k_values.left);
                apply_scalar_face_bc(field->turb_eps, tbc->left, idx, idx_int, idx_per,
                                     tbc->eps_values.left);
            } else {
                apply_scalar_face_bc(field->turb_nu_tilde, tbc->left, idx, idx_int,
                                     idx_per, tbc->nu_tilde_values.left);
            }
            if (tbc->left == BC_TYPE_DIRICHLET) {
                set_dirichlet_nu_t(field, params, idx);
            } else {
                apply_scalar_face_bc(field->nu_t, tbc->left, idx, idx_int, idx_per, 0.0);
            }
        }
    }

    /* Right face (i=nx-1) */
    if (tbc->right == BC_TYPE_NOSLIP) {
        double y_p = grid->x[nx - 1] - grid->x[nx - 2];
        for (size_t j = 0; j < ny; j++) {
            size_t idx_w = j * nx + (nx - 1);
            size_t idx_p = idx_w - 1;
            apply_wall_function_node(field, params, idx_w, idx_p, y_p,
                                     fabs(field->v[idx_p]));
        }
    } else {
        for (size_t j = 0; j < ny; j++) {
            size_t idx = j * nx + (nx - 1);
            size_t idx_int = idx - 1;
            size_t idx_per = j * nx + 1;
            if (is_ke) {
                apply_scalar_face_bc(field->turb_k, tbc->right, idx, idx_int, idx_per,
                                     tbc->k_values.right);
                apply_scalar_face_bc(field->turb_eps, tbc->right, idx, idx_int, idx_per,
                                     tbc->eps_values.right);
            } else {
                apply_scalar_face_bc(field->turb_nu_tilde, tbc->right, idx, idx_int,
                                     idx_per, tbc->nu_tilde_values.right);
            }
            if (tbc->right == BC_TYPE_DIRICHLET) {
                set_dirichlet_nu_t(field, params, idx);
            } else {
                apply_scalar_face_bc(field->nu_t, tbc->right, idx, idx_int, idx_per, 0.0);
            }
        }
    }

    /* Bottom face (j=0) — runs after left/right, overwrites shared corners */
    if (tbc->bottom == BC_TYPE_NOSLIP) {
        double y_p = grid->y[1] - grid->y[0];
        for (size_t i = 0; i < nx; i++) {
            size_t idx_w = i;
            size_t idx_p = idx_w + nx;
            apply_wall_function_node(field, params, idx_w, idx_p, y_p,
                                     fabs(field->u[idx_p]));
        }
    } else {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = i;
            size_t idx_int = idx + nx;
            size_t idx_per = (ny - 2) * nx + i;
            if (is_ke) {
                apply_scalar_face_bc(field->turb_k, tbc->bottom, idx, idx_int, idx_per,
                                     tbc->k_values.bottom);
                apply_scalar_face_bc(field->turb_eps, tbc->bottom, idx, idx_int, idx_per,
                                     tbc->eps_values.bottom);
            } else {
                apply_scalar_face_bc(field->turb_nu_tilde, tbc->bottom, idx, idx_int,
                                     idx_per, tbc->nu_tilde_values.bottom);
            }
            if (tbc->bottom == BC_TYPE_DIRICHLET) {
                set_dirichlet_nu_t(field, params, idx);
            } else {
                apply_scalar_face_bc(field->nu_t, tbc->bottom, idx, idx_int, idx_per, 0.0);
            }
        }
    }

    /* Top face (j=ny-1) */
    if (tbc->top == BC_TYPE_NOSLIP) {
        double y_p = grid->y[ny - 1] - grid->y[ny - 2];
        for (size_t i = 0; i < nx; i++) {
            size_t idx_w = (ny - 1) * nx + i;
            size_t idx_p = idx_w - nx;
            apply_wall_function_node(field, params, idx_w, idx_p, y_p,
                                     fabs(field->u[idx_p]));
        }
    } else {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = (ny - 1) * nx + i;
            size_t idx_int = idx - nx;
            size_t idx_per = nx + i;
            if (is_ke) {
                apply_scalar_face_bc(field->turb_k, tbc->top, idx, idx_int, idx_per,
                                     tbc->k_values.top);
                apply_scalar_face_bc(field->turb_eps, tbc->top, idx, idx_int, idx_per,
                                     tbc->eps_values.top);
            } else {
                apply_scalar_face_bc(field->turb_nu_tilde, tbc->top, idx, idx_int,
                                     idx_per, tbc->nu_tilde_values.top);
            }
            if (tbc->top == BC_TYPE_DIRICHLET) {
                set_dirichlet_nu_t(field, params, idx);
            } else {
                apply_scalar_face_bc(field->nu_t, tbc->top, idx, idx_int, idx_per, 0.0);
            }
        }
    }

    return CFD_SUCCESS;
}

cfd_status_t turbulence_init_uniform(flow_field* field,
                                     const ns_solver_params_t* params,
                                     double k0, double eps0, double nu_tilde0) {
    if (!field || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_init_uniform: field and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }
    if (!field->turb_k || !field->turb_eps || !field->turb_nu_tilde || !field->nu_t) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_init_uniform: missing turbulence fields");
        return CFD_ERROR_INVALID;
    }

    size_t total = field->nx * field->ny * field->nz;

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        if (k0 < 0.0 || eps0 <= 0.0) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_init_uniform: k0 must be >= 0 and eps0 > 0");
            return CFD_ERROR_INVALID;
        }
        for (size_t n = 0; n < total; n++) {
            field->turb_k[n] = k0;
            field->turb_eps[n] = eps0;
        }
    } else {
        if (nu_tilde0 < 0.0) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "turbulence_init_uniform: nu_tilde0 must be >= 0");
            return CFD_ERROR_INVALID;
        }
        for (size_t n = 0; n < total; n++) {
            field->turb_nu_tilde[n] = nu_tilde0;
        }
    }

    turb_update_nu_t(field, params);
    return CFD_SUCCESS;
}
