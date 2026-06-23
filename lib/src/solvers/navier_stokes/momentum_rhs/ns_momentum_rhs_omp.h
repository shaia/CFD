/**
 * @file ns_momentum_rhs_omp.h
 * @brief Shared OpenMP momentum RHS kernel for the explicit RK time integrators.
 *
 * OpenMP-parallelized counterpart of ns_momentum_rhs_scalar.h, used identically
 * by the OMP RK2 (`solver_rk2_omp.c`) and RK4 (`solver_rk4_omp.c`) solvers.
 *
 * Include requirements (the includer must satisfy these BEFORE #include-ing, and
 * must include this header inside its `#ifdef CFD_ENABLE_OPENMP` block):
 *   - <math.h>, <stddef.h>, <omp.h>
 *   - "cfd/core/grid.h", "cfd/core/indexing.h" (grid, IDX_2D)
 *   - "cfd/solvers/navier_stokes_solver.h" (ns_solver_params_t)
 *   - "../../energy/energy_solver_internal.h" (compute_source_terms,
 *     energy_compute_buoyancy)
 *
 * This header is #include-only and is never compiled as a standalone TU.
 */
#ifndef CFD_NS_MOMENTUM_RHS_OMP_H
#define CFD_NS_MOMENTUM_RHS_OMP_H

/* Physical stability limits (shared by the RK2/RK4 OMP RHS kernel) */
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
 * Compute the right-hand side of the semi-discrete Navier-Stokes equations
 * (OpenMP-parallelized over the j-loop). Identical physics to the scalar kernel.
 *
 * Uses periodic stencil indices to avoid relying on ghost cells, which is
 * critical for preserving RK temporal order.
 */
static void compute_rhs_omp(const double* u, const double* v, const double* w,
                             const double* p, const double* rho, const double* T,
                             double* rhs_u, double* rhs_v, double* rhs_w,
                             double* rhs_p,
                             const grid* grid, const ns_solver_params_t* params,
                             size_t nx, size_t ny, size_t nz,
                             size_t stride_z, size_t k_start, size_t k_end,
                             double inv_2dz, double inv_dz2,
                             int iter, double dt) {
    ptrdiff_t ny_int = (ptrdiff_t)ny;
    ptrdiff_t nx_int = (ptrdiff_t)nx;

    for (size_t k = k_start; k < k_end; k++) {
        ptrdiff_t j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (ptrdiff_t i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

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

                /* Periodic stencil indices in x and y — avoids relying on ghost
                 * cells, critical for preserving RK temporal order. */
                size_t il = ((size_t)i > 1)      ? idx - 1  : k * stride_z + IDX_2D(nx - 2, (size_t)j, nx);
                size_t ir = ((size_t)i < nx - 2) ? idx + 1  : k * stride_z + IDX_2D(1, (size_t)j, nx);
                size_t jd = ((size_t)j > 1)      ? idx - nx : k * stride_z + IDX_2D((size_t)i, ny - 2, nx);
                size_t ju = ((size_t)j < ny - 2) ? idx + nx : k * stride_z + IDX_2D((size_t)i, 1, nx);

                /* Periodic stencil indices in z.
                 * When nz==1: k=0, stride_z=0, so kd=ku=idx → z-terms vanish. */
                size_t kd = (k > 1)      ? idx - stride_z
                                         : (nz - 2) * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                size_t ku = (k < nz - 2) ? idx + stride_z
                                         : 1 * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

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

#endif /* CFD_NS_MOMENTUM_RHS_OMP_H */
