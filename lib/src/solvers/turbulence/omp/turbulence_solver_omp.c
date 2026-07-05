/**
 * @file turbulence_solver_omp.c
 * @brief OpenMP-parallelized RANS turbulence transport step
 *
 * Same numerics as the scalar reference (turbulence/cpu/turbulence_kepsilon.c
 * and turbulence_sa.c); the interior stencil loops are parallelized over j.
 * Read/write separation (reads field arrays, writes the workspace buffers)
 * makes the loops race-free. Validation, nu_t update, boundary conditions,
 * and wall functions are shared with the scalar backend (boundary-only work).
 */

#include "../turbulence_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <string.h>

/* OMP k-epsilon transport kernel — numerics identical to
 * turb_kepsilon_step_scalar, j-loop parallelized. */
static void turb_kepsilon_step_omp(const flow_field* field, const grid* grid,
                                   const ns_solver_params_t* params, double dt,
                                   double* k_new, double* eps_new) {
    size_t nx = field->nx;
    size_t ny = field->ny;

    const double dx = grid->dx[0];
    const double dy = grid->dy[0];
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_2dx = 0.5 * inv_dx;
    const double inv_2dy = 0.5 * inv_dy;
    const double inv_dx2 = inv_dx * inv_dx;
    const double inv_dy2 = inv_dy * inv_dy;

    const double* u = field->u;
    const double* v = field->v;
    const double* k = field->turb_k;
    const double* eps = field->turb_eps;
    const double* nu_t = field->nu_t;

    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < (int)ny - 1; j++) {
        for (int i = 1; i < (int)nx - 1; i++) {
            size_t idx = IDX_2D((size_t)i, (size_t)j, nx);

            double nu = params->mu / fmax(field->rho[idx], 1e-10);
            double u_c = u[idx];
            double v_c = v[idx];
            double k_c = fmax(k[idx], TURB_K_MIN);
            double eps_c = fmax(eps[idx], TURB_EPS_MIN);

            /* First-order upwind advection */
            double adv_k = (u_c > 0.0 ? u_c * (k[idx] - k[idx - 1]) * inv_dx
                                      : u_c * (k[idx + 1] - k[idx]) * inv_dx) +
                           (v_c > 0.0 ? v_c * (k[idx] - k[idx - nx]) * inv_dy
                                      : v_c * (k[idx + nx] - k[idx]) * inv_dy);
            double adv_e = (u_c > 0.0 ? u_c * (eps[idx] - eps[idx - 1]) * inv_dx
                                      : u_c * (eps[idx + 1] - eps[idx]) * inv_dx) +
                           (v_c > 0.0 ? v_c * (eps[idx] - eps[idx - nx]) * inv_dy
                                      : v_c * (eps[idx + nx] - eps[idx]) * inv_dy);

            /* Conservative face-averaged diffusion */
            double gam_e_k = nu + 0.5 * (nu_t[idx] + nu_t[idx + 1]) / TURB_SIGMA_K;
            double gam_w_k = nu + 0.5 * (nu_t[idx] + nu_t[idx - 1]) / TURB_SIGMA_K;
            double gam_n_k = nu + 0.5 * (nu_t[idx] + nu_t[idx + nx]) / TURB_SIGMA_K;
            double gam_s_k = nu + 0.5 * (nu_t[idx] + nu_t[idx - nx]) / TURB_SIGMA_K;
            double diff_k = (gam_e_k * (k[idx + 1] - k[idx]) -
                             gam_w_k * (k[idx] - k[idx - 1])) * inv_dx2 +
                            (gam_n_k * (k[idx + nx] - k[idx]) -
                             gam_s_k * (k[idx] - k[idx - nx])) * inv_dy2;

            double gam_e_e = nu + 0.5 * (nu_t[idx] + nu_t[idx + 1]) / TURB_SIGMA_EPS;
            double gam_w_e = nu + 0.5 * (nu_t[idx] + nu_t[idx - 1]) / TURB_SIGMA_EPS;
            double gam_n_e = nu + 0.5 * (nu_t[idx] + nu_t[idx + nx]) / TURB_SIGMA_EPS;
            double gam_s_e = nu + 0.5 * (nu_t[idx] + nu_t[idx - nx]) / TURB_SIGMA_EPS;
            double diff_e = (gam_e_e * (eps[idx + 1] - eps[idx]) -
                             gam_w_e * (eps[idx] - eps[idx - 1])) * inv_dx2 +
                            (gam_n_e * (eps[idx + nx] - eps[idx]) -
                             gam_s_e * (eps[idx] - eps[idx - nx])) * inv_dy2;

            /* Production from central velocity gradients, limited */
            double du_dx = (u[idx + 1] - u[idx - 1]) * inv_2dx;
            double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
            double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
            double dv_dy = (v[idx + nx] - v[idx - nx]) * inv_2dy;
            double S2 = 2.0 * du_dx * du_dx + 2.0 * dv_dy * dv_dy +
                        (du_dy + dv_dx) * (du_dy + dv_dx);
            double P_k = fmin(nu_t[idx] * S2, TURB_PROD_LIMIT_FACTOR * eps_c);

            /* Patankar semi-implicit updates: sink terms cannot cross zero */
            double eps_over_k = eps_c / k_c;
            double k_upd = (k[idx] + dt * (-adv_k + diff_k + P_k)) /
                           (1.0 + dt * eps_over_k);
            double e_upd = (eps[idx] + dt * (-adv_e + diff_e +
                                             TURB_C1_EPS * eps_over_k * P_k)) /
                           (1.0 + dt * TURB_C2_EPS * eps_over_k);

            k_new[idx] = fmax(k_upd, TURB_K_MIN);
            eps_new[idx] = fmax(e_upd, TURB_EPS_MIN);
        }
    }
}

/* OMP Spalart-Allmaras transport kernel — numerics identical to
 * turb_sa_step_scalar, j-loop parallelized. */
static void turb_sa_step_omp(const flow_field* field, const grid* grid,
                             const ns_solver_params_t* params, double dt,
                             double* nt_new) {
    size_t nx = field->nx;
    size_t ny = field->ny;

    const double dx = grid->dx[0];
    const double dy = grid->dy[0];
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_2dx = 0.5 * inv_dx;
    const double inv_2dy = 0.5 * inv_dy;
    const double inv_dx2 = inv_dx * inv_dx;
    const double inv_dy2 = inv_dy * inv_dy;

    const double* u = field->u;
    const double* v = field->v;
    const double* nt = field->turb_nu_tilde;
    const double cv1_3 = SA_CV1 * SA_CV1 * SA_CV1;

    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < (int)ny - 1; j++) {
        for (int i = 1; i < (int)nx - 1; i++) {
            size_t idx = IDX_2D((size_t)i, (size_t)j, nx);

            double nu = params->mu / fmax(field->rho[idx], 1e-10);
            double u_c = u[idx];
            double v_c = v[idx];
            double nt_c = fmax(nt[idx], 0.0);

            /* Closure functions */
            double chi = nt_c / nu;
            double chi3 = chi * chi * chi;
            double fv1 = chi3 / (chi3 + cv1_3);
            double fv2 = 1.0 - chi / (1.0 + chi * fv1);

            /* Vorticity magnitude (2D) */
            double du_dy = (u[idx + nx] - u[idx - nx]) * inv_2dy;
            double dv_dx = (v[idx + 1] - v[idx - 1]) * inv_2dx;
            double omega = fabs(dv_dx - du_dy);

            /* Wall distance and modified vorticity S_tilde */
            int has_wall = 0;
            double d = turb_wall_distance(grid, &params->turb_bc, (size_t)i, (size_t)j,
                                          &has_wall);
            double s_tilde = omega;
            double destr_rate = 0.0; /* cw1*fw*nt/d^2, the Patankar sink rate */
            if (has_wall && d > 1e-12) {
                double inv_kd2 = 1.0 / (SA_KAPPA * SA_KAPPA * d * d);
                s_tilde = fmax(omega + nt_c * fv2 * inv_kd2, 0.3 * omega);

                double r = fmin(nt_c / (s_tilde * SA_KAPPA * SA_KAPPA * d * d + 1e-16),
                                10.0);
                double r6 = pow(r, 6.0);
                double g = r + SA_CW2 * (r6 - r);
                double g6 = pow(g, 6.0);
                double cw3_6 = pow(SA_CW3, 6.0);
                double fw = g * pow((1.0 + cw3_6) / (g6 + cw3_6), 1.0 / 6.0);

                destr_rate = SA_CW1 * fw * nt_c / (d * d);
            }

            /* First-order upwind advection */
            double adv = (u_c > 0.0 ? u_c * (nt[idx] - nt[idx - 1]) * inv_dx
                                    : u_c * (nt[idx + 1] - nt[idx]) * inv_dx) +
                         (v_c > 0.0 ? v_c * (nt[idx] - nt[idx - nx]) * inv_dy
                                    : v_c * (nt[idx + nx] - nt[idx]) * inv_dy);

            /* Conservative face-averaged diffusion + cb2 gradient-squared term */
            double gam_e = (nu + 0.5 * (fmax(nt[idx], 0.0) + fmax(nt[idx + 1], 0.0))) / SA_SIGMA;
            double gam_w = (nu + 0.5 * (fmax(nt[idx], 0.0) + fmax(nt[idx - 1], 0.0))) / SA_SIGMA;
            double gam_n = (nu + 0.5 * (fmax(nt[idx], 0.0) + fmax(nt[idx + nx], 0.0))) / SA_SIGMA;
            double gam_s = (nu + 0.5 * (fmax(nt[idx], 0.0) + fmax(nt[idx - nx], 0.0))) / SA_SIGMA;
            double diff = (gam_e * (nt[idx + 1] - nt[idx]) -
                           gam_w * (nt[idx] - nt[idx - 1])) * inv_dx2 +
                          (gam_n * (nt[idx + nx] - nt[idx]) -
                           gam_s * (nt[idx] - nt[idx - nx])) * inv_dy2;

            double dnt_dx = (nt[idx + 1] - nt[idx - 1]) * inv_2dx;
            double dnt_dy = (nt[idx + nx] - nt[idx - nx]) * inv_2dy;
            double grad2 = (SA_CB2 / SA_SIGMA) * (dnt_dx * dnt_dx + dnt_dy * dnt_dy);

            /* Production */
            double prod = SA_CB1 * s_tilde * nt_c;

            /* Patankar semi-implicit update: destruction cannot cross zero */
            double nt_upd = (nt[idx] + dt * (-adv + diff + grad2 + prod)) /
                            (1.0 + dt * destr_rate);

            nt_new[idx] = fmax(nt_upd, 0.0);
        }
    }
}

cfd_status_t turbulence_step_explicit_omp_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* workspace, size_t workspace_size) {
    (void)time;

    if (!params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "turbulence_solver_omp: params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (params->turb_model == TURB_MODEL_NONE) {
        return CFD_SUCCESS;
    }

    cfd_status_t status = turb_validate_step_args(field, grid, params);
    if (status != CFD_SUCCESS) {
        return status;
    }

    size_t total = field->nx * field->ny * field->nz;

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

    int has_nan = 0;
    ptrdiff_t total_int = (ptrdiff_t)total;
    ptrdiff_t n;

    if (params->turb_model == TURB_MODEL_K_EPSILON) {
        double* k_new = buf;
        double* eps_new = buf + total;
        memcpy(k_new, field->turb_k, total * sizeof(double));
        memcpy(eps_new, field->turb_eps, total * sizeof(double));

        turb_kepsilon_step_omp(field, grid, params, dt, k_new, eps_new);

#pragma omp parallel for reduction(| : has_nan) schedule(static)
        for (n = 0; n < total_int; n++) {
            if (!isfinite(k_new[n]) || !isfinite(eps_new[n])) {
                has_nan = 1;
            }
        }
        if (has_nan) {
            cfd_set_error(CFD_ERROR_DIVERGED,
                          "NaN/Inf detected in turbulence_step_explicit_omp (k-epsilon)");
            if (owns_buffer) cfd_free(buf);
            return CFD_ERROR_DIVERGED;
        }
        memcpy(field->turb_k, k_new, total * sizeof(double));
        memcpy(field->turb_eps, eps_new, total * sizeof(double));
    } else {
        double* nt_new = buf;
        memcpy(nt_new, field->turb_nu_tilde, total * sizeof(double));

        turb_sa_step_omp(field, grid, params, dt, nt_new);

#pragma omp parallel for reduction(| : has_nan) schedule(static)
        for (n = 0; n < total_int; n++) {
            if (!isfinite(nt_new[n])) {
                has_nan = 1;
            }
        }
        if (has_nan) {
            cfd_set_error(CFD_ERROR_DIVERGED,
                          "NaN/Inf detected in turbulence_step_explicit_omp (SA)");
            if (owns_buffer) cfd_free(buf);
            return CFD_ERROR_DIVERGED;
        }
        memcpy(field->turb_nu_tilde, nt_new, total * sizeof(double));
    }

    turb_update_nu_t(field, params);

    if (owns_buffer) cfd_free(buf);
    return CFD_SUCCESS;
}
