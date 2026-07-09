/**
 * @file turbulence_sa.c
 * @brief Scalar CPU kernel for the Spalart-Allmaras transport equation
 *
 * No-ft2 (fully turbulent) variant — the trip terms only matter for laminar-
 * turbulent transition prediction and are omitted by virtually all RANS
 * solvers by default:
 *
 * dnt/dt + u*grad(nt) = (1/sigma)[div((nu+nt) grad nt) + cb2*|grad nt|^2]
 *                       + cb1*S_tilde*nt - cw1*fw*(nt/d)^2
 *
 * First-order upwind advection, conservative face-averaged diffusion, and
 * semi-implicit (Patankar) destruction so nt can never cross zero. Cells with
 * no wall-function wall configured (has_wall == 0) drop the wall-destruction
 * and fv2 near-wall terms entirely (S_tilde = Omega).
 */

#include "../turbulence_solver_internal.h"

#include "cfd/core/indexing.h"

#include <math.h>

void turb_sa_step_scalar(const flow_field* field, const grid* grid,
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

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, j, nx);

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
            double d = turb_wall_distance(grid, &params->turb_bc, i, j, &has_wall);
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
