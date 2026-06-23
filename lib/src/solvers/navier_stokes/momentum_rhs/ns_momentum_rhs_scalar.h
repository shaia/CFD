/**
 * @file ns_momentum_rhs_scalar.h
 * @brief Shared scalar momentum RHS kernel for the explicit RK time integrators.
 *
 * Defines the semi-discrete Navier-Stokes right-hand side used identically by
 * the scalar RK2 (`solver_rk2.c`) and RK4 (`solver_rk4.c`) solvers. Extracted to
 * a single source of truth so the two integrators cannot drift apart.
 *
 * Include requirements (the includer must satisfy these BEFORE #include-ing):
 *   - <math.h> (fabs/fmax/fmin)
 *   - "cfd/core/grid.h", "cfd/core/indexing.h" (grid, IDX_2D)
 *   - "cfd/solvers/navier_stokes_solver.h" (ns_solver_params_t)
 *   - "../../energy/energy_solver_internal.h" (compute_source_terms,
 *     energy_compute_buoyancy)
 *
 * This header is #include-only and is never compiled as a standalone TU (see the
 * SIMD template pattern in lib/src/solvers/linear/simd_template/).
 */
#ifndef CFD_NS_MOMENTUM_RHS_SCALAR_H
#define CFD_NS_MOMENTUM_RHS_SCALAR_H

/* Physical stability limits (shared by the RK2/RK4 scalar RHS kernel) */
#ifndef MAX_DERIVATIVE_LIMIT
#define MAX_DERIVATIVE_LIMIT        100.0
#endif
#ifndef MAX_SECOND_DERIVATIVE_LIMIT
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#endif
#ifndef MAX_DIVERGENCE_LIMIT
#define MAX_DIVERGENCE_LIMIT        10.0
#endif
#ifndef PRESSURE_UPDATE_FACTOR
#define PRESSURE_UPDATE_FACTOR      0.1
#endif

/**
 * Compute the right-hand side of the semi-discrete Navier-Stokes equations.
 *
 * For each interior point, computes:
 *   du/dt = -u·∇u - (1/ρ)∂p/∂x + ν∇²u + source_u
 *   dv/dt = -u·∇v - (1/ρ)∂p/∂y + ν∇²v + source_v
 *   dw/dt = -u·∇w - (1/ρ)∂p/∂z + ν∇²w + source_w
 *   dp/dt = -0.1 * ρ * (∂u/∂x + ∂v/∂y + ∂w/∂z)
 *
 * Uses periodic stencil indices to avoid relying on ghost cells, which is
 * critical for preserving RK temporal order (ghost cell values may be stale
 * during intermediate RK stages).
 */
static void compute_rhs(const double* u, const double* v, const double* w,
                         const double* p, const double* rho, const double* T,
                         double* rhs_u, double* rhs_v, double* rhs_w, double* rhs_p,
                         const grid* grid, const ns_solver_params_t* params,
                         size_t nx, size_t ny, size_t nz,
                         size_t stride_z, size_t k_start, size_t k_end,
                         double inv_2dz, double inv_dz2,
                         int iter, double dt) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

                /* Safety checks */
                if (rho[idx] <= 1e-10) {
                    rhs_u[idx] = 0.0;
                    rhs_v[idx] = 0.0;
                    rhs_w[idx] = 0.0;
                    rhs_p[idx] = 0.0;
                    continue;
                }
                if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) {
                    rhs_u[idx] = 0.0;
                    rhs_v[idx] = 0.0;
                    rhs_w[idx] = 0.0;
                    rhs_p[idx] = 0.0;
                    continue;
                }

                /* Periodic stencil indices in x and y — avoids relying on ghost cells,
                 * which is critical for preserving RK temporal order. */
                size_t il = (i > 1)      ? idx - 1  : k * stride_z + IDX_2D(nx - 2, j, nx);
                size_t ir = (i < nx - 2) ? idx + 1  : k * stride_z + IDX_2D(1, j, nx);
                size_t jd = (j > 1)      ? idx - nx : k * stride_z + IDX_2D(i, ny - 2, nx);
                size_t ju = (j < ny - 2) ? idx + nx : k * stride_z + IDX_2D(i, 1, nx);

                /* Periodic stencil indices in z.
                 * When nz==1: k=0, stride_z=0, so kd=ku=idx → z-terms vanish. */
                size_t kd = (k > 1)      ? idx - stride_z
                                         : (nz - 2) * stride_z + IDX_2D(i, j, nx);
                size_t ku = (k < nz - 2) ? idx + stride_z
                                         : 1 * stride_z + IDX_2D(i, j, nx);

                /* First derivatives (central differences) */
                double du_dx = (u[ir] - u[il]) / (2.0 * grid->dx[i]);
                double du_dy = (u[ju] - u[jd]) / (2.0 * grid->dy[j]);
                double du_dz = (u[ku] - u[kd]) * inv_2dz;

                double dv_dx = (v[ir] - v[il]) / (2.0 * grid->dx[i]);
                double dv_dy = (v[ju] - v[jd]) / (2.0 * grid->dy[j]);
                double dv_dz = (v[ku] - v[kd]) * inv_2dz;

                double dw_dx = (w[ir] - w[il]) / (2.0 * grid->dx[i]);
                double dw_dy = (w[ju] - w[jd]) / (2.0 * grid->dy[j]);
                double dw_dz = (w[ku] - w[kd]) * inv_2dz;

                /* Pressure gradients */
                double dp_dx = (p[ir] - p[il]) / (2.0 * grid->dx[i]);
                double dp_dy = (p[ju] - p[jd]) / (2.0 * grid->dy[j]);
                double dp_dz = (p[ku] - p[kd]) * inv_2dz;

                /* Second derivatives (viscous terms) */
                double d2u_dx2 = (u[ir] - 2.0 * u[idx] + u[il]) / (grid->dx[i] * grid->dx[i]);
                double d2u_dy2 = (u[ju] - 2.0 * u[idx] + u[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2u_dz2 = (u[ku] - 2.0 * u[idx] + u[kd]) * inv_dz2;

                double d2v_dx2 = (v[ir] - 2.0 * v[idx] + v[il]) / (grid->dx[i] * grid->dx[i]);
                double d2v_dy2 = (v[ju] - 2.0 * v[idx] + v[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2v_dz2 = (v[ku] - 2.0 * v[idx] + v[kd]) * inv_dz2;

                double d2w_dx2 = (w[ir] - 2.0 * w[idx] + w[il]) / (grid->dx[i] * grid->dx[i]);
                double d2w_dy2 = (w[ju] - 2.0 * w[idx] + w[jd]) / (grid->dy[j] * grid->dy[j]);
                double d2w_dz2 = (w[ku] - 2.0 * w[idx] + w[kd]) * inv_dz2;

                /* Kinematic viscosity */
                double nu = params->mu / fmax(rho[idx], 1e-10);
                nu = fmin(nu, 1.0);

                /* Clamp first derivatives */
                du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
                du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
                du_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dz));
                dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
                dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
                dv_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dz));
                dw_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dx));
                dw_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dy));
                dw_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dz));
                dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
                dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
                dp_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dz));

                /* Clamp second derivatives */
                d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
                d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
                d2u_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dz2));
                d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
                d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));
                d2v_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dz2));
                d2w_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dx2));
                d2w_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dy2));
                d2w_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dz2));

                /* Source terms */
                double source_u = 0.0, source_v = 0.0, source_w = 0.0;
                double z_coord = (nz > 1 && grid->z) ? grid->z[k] : 0.0;
                compute_source_terms(grid->x[i], grid->y[j], z_coord, iter, dt,
                                     params, &source_u, &source_v, &source_w);

                /* Boussinesq buoyancy source */
                if (T) {
                    energy_compute_buoyancy(T[idx], params,
                                            &source_u, &source_v, &source_w);
                }

                /* RHS for u-momentum */
                rhs_u[idx] = -u[idx] * du_dx - v[idx] * du_dy - w[idx] * du_dz
                             - dp_dx / rho[idx]
                             + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
                             + source_u;

                /* RHS for v-momentum */
                rhs_v[idx] = -u[idx] * dv_dx - v[idx] * dv_dy - w[idx] * dv_dz
                             - dp_dy / rho[idx]
                             + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
                             + source_v;

                /* RHS for w-momentum */
                rhs_w[idx] = -u[idx] * dw_dx - v[idx] * dw_dy - w[idx] * dw_dz
                             - dp_dz / rho[idx]
                             + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2)
                             + source_w;

                /* Simplified pressure RHS (divergence-based) */
                double divergence = du_dx + dv_dy + dw_dz;
                divergence = fmax(-MAX_DIVERGENCE_LIMIT,
                                  fmin(MAX_DIVERGENCE_LIMIT, divergence));
                rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
            }
        }
    }
}

#endif /* CFD_NS_MOMENTUM_RHS_SCALAR_H */
