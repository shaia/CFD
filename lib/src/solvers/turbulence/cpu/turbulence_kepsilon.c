/**
 * @file turbulence_kepsilon.c
 * @brief Scalar CPU kernel for the standard k-epsilon transport equations
 *
 * dk/dt   + u*grad(k)   = div((nu + nu_t/sigma_k) grad k)   + P_k - eps
 * deps/dt + u*grad(eps) = div((nu + nu_t/sigma_e) grad eps) + (eps/k)(C1*P_k - C2*eps)
 *
 * First-order upwind advection (positivity), conservative face-averaged
 * diffusion, production from central velocity gradients with the limiter
 * P_k <= 10*eps, and semi-implicit (Patankar) sink treatment: destruction is
 * written as (eps/k)*k_new resp. C2*(eps/k)*eps_new so the update divides by
 * (1 + dt*sink_rate) and can never cross zero. Floors keep k and eps positive.
 */

#include "../turbulence_solver_internal.h"

#include "cfd/core/indexing.h"

#include <math.h>

void turb_kepsilon_step_scalar(const flow_field* field, const grid* grid,
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

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, j, nx);

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
