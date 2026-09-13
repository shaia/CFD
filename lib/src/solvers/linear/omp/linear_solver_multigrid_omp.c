/**
 * @file linear_solver_multigrid_omp.c
 * @brief Geometric multigrid Poisson solver - OpenMP parallelized implementation
 *
 * Supplies OpenMP-parallelized smoother, residual, boundary and grid-transfer
 * primitives to the shared multigrid algorithm template
 * (multigrid_template/linear_solver_multigrid_template.h); this file holds no
 * algorithm code.
 *
 * Every primitive is element-wise: each k-plane (or coarse/fine z index) opens
 * one `parallel for schedule(static)` over rows with an int loop variable
 * (MSVC OpenMP 2.0), every row writes a disjoint index range, and per-point
 * stencil sums keep the scalar term order. No loop accumulates across threads:
 * the Neumann interior mean (mg_interior_mean) and the convergence residual
 * (poisson_solver_compute_residual) stay serial. Red-Black Gauss-Seidel updates
 * within one color read only the other color, so they are order-independent.
 * The result is bit-identical to the scalar backend at any thread count under
 * the same floating-point contraction settings.
 *
 * A region uses the thread team only when its work reaches MG_OMP_MIN_POINTS
 * (mg_omp_parallel): the interior points of the plane it writes, or 2(nx+ny) for
 * the boundary pass. Smaller loops (the coarser levels, small grids) run on the
 * calling thread.
 *
 * Boundary conditions run the boundary core that bc_apply_scalar_omp and
 * bc_apply_scalar_cpu share (same face order, copies only), entered through
 * the OpenMP function only above the same threshold.
 */

#include "../linear_solver_internal.h"
#include "../multigrid_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/indexing.h"

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>
#include <string.h>

/* ============================================================================
 * PARALLEL THRESHOLD
 * ============================================================================ */

/**
 * Nonzero when a region covering `points` points should use the thread team
 * (see MG_OMP_MIN_POINTS). Smaller regions run serially through the OpenMP `if`
 * clause, so the loop body, and the result, are unchanged.
 */
static inline int mg_omp_parallel(size_t points) {
    return points >= MG_OMP_MIN_POINTS;
}

/* ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================ */

/** Zero-gradient BCs on one nx x ny plane; the pass copies 2(nx+ny) points */
static void mg_bc_neumann_plane_omp(double* plane, size_t nx, size_t ny) {
    if (mg_omp_parallel(2 * (nx + ny))) {
        bc_apply_scalar_omp(plane, nx, ny, BC_TYPE_NEUMANN);
    } else {
        bc_apply_scalar_cpu(plane, nx, ny, BC_TYPE_NEUMANN);
    }
}

/* ============================================================================
 * SMOOTHER SWEEPS (no BCs; the template applies them after each sweep)
 * ============================================================================ */

/**
 * One Red-Black Gauss-Seidel pass (omega = 1). A point of one color reads only
 * points of the other color (5/7-point stencil; the 2D z-term reads the point
 * itself with stride_z = 0), so rows of one color update independently.
 */
static void mg_rbgs_sweep_omp(const mg_level_t* L, double* x,
                              const double* rhs) {
    size_t nx = L->nx;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    double inv_factor = L->inv_factor;
    size_t stride_z = L->stride_z;
    int ny_int = poisson_solver_size_to_int(L->ny);
    int par = mg_omp_parallel((nx - 2) * (L->ny - 2));

    /* Red pass: (i+j+k) % 2 == 1, then black pass: (i+j+k) % 2 == 0 */
    for (int color = 0; color < 2; color++) {
        for (size_t k = L->k_start; k < L->k_end; k++) {
            int j;
#pragma omp parallel for schedule(static) if(par)
            for (j = 1; j < ny_int - 1; j++) {
                size_t i_start =
                    (((size_t)j + k) % 2 == (size_t)color) ? 1 : 2;
                for (size_t i = i_start; i < nx - 1; i += 2) {
                    size_t idx = k * stride_z + IDX_2D(i, (size_t)j, nx);

                    x[idx] = -(rhs[idx]
                        - (x[idx + 1] + x[idx - 1]) / dx2
                        - (x[idx + nx] + x[idx - nx]) / dy2
                        - (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                        ) * inv_factor;
                }
            }
        }
    }
}

/**
 * One weighted Jacobi pass (omega = MG_JACOBI_OMEGA) into x_temp, then an
 * interior-only copy back to x (boundary values preserved).
 */
static void mg_jacobi_sweep_omp(const mg_level_t* L, double* x,
                                double* x_temp, const double* rhs) {
    size_t nx = L->nx;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    double inv_factor = L->inv_factor;
    size_t stride_z = L->stride_z;
    int ny_int = poisson_solver_size_to_int(L->ny);
    int nx_int = poisson_solver_size_to_int(nx);
    int par = mg_omp_parallel((nx - 2) * (L->ny - 2));

    for (size_t k = L->k_start; k < L->k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

                double x_jacobi = -(rhs[idx]
                    - (x[idx + 1] + x[idx - 1]) / dx2
                    - (x[idx + nx] + x[idx - nx]) / dy2
                    - (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                    ) * inv_factor;

                x_temp[idx] = x[idx] + MG_JACOBI_OMEGA * (x_jacobi - x[idx]);
            }
        }
    }

    for (size_t k = L->k_start; k < L->k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < ny_int - 1; j++) {
            size_t row_start = k * stride_z + (size_t)j * nx;
            memcpy(&x[row_start + 1], &x_temp[row_start + 1], (nx - 2) * sizeof(double));
        }
    }
}

/* ============================================================================
 * RESIDUAL
 * ============================================================================ */

/**
 * r = rhs - Laplacian(x) at interior points, dividing by dx2/dy2 exactly as the
 * scalar multigrid residual does. The Krylov residual in
 * linear_solver_primitives_omp.h computes -rhs + Laplacian with inverse
 * spacings, which is not bit-identical, so it is not reused here.
 */
static void mg_residual_omp(const mg_level_t* L, const double* x,
                            const double* rhs, double* r) {
    size_t nx = L->nx;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    size_t stride_z = L->stride_z;
    int ny_int = poisson_solver_size_to_int(L->ny);
    int nx_int = poisson_solver_size_to_int(nx);
    int par = mg_omp_parallel((nx - 2) * (L->ny - 2));

    for (size_t k = L->k_start; k < L->k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

                double laplacian =
                    (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / dx2
                  + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / dy2
                  + (x[idx + stride_z] + x[idx - stride_z]
                     - 2.0 * x[idx]) * inv_dz2;

                r[idx] = rhs[idx] - laplacian;
            }
        }
    }
}

/* ============================================================================
 * RESTRICTION (fine -> coarse, full weighting)
 * ============================================================================ */

/** Full-weighting restriction, 2D; parallel over coarse rows */
static void mg_restrict_2d_omp(const double* fine, double* coarse,
                               size_t nxf, size_t nyf, size_t nxc, size_t nyc,
                               int fold_neumann) {
    (void)nyf;
    int nyc_int = poisson_solver_size_to_int(nyc);
    int par = mg_omp_parallel((nxc - 2) * (nyc - 2));

    int J;
#pragma omp parallel for schedule(static) if(par)
    for (J = 1; J < nyc_int - 1; J++) {
        double wy[3];
        mg_weights_1d((size_t)J, nyc, fold_neumann, wy);
        for (size_t I = 1; I < nxc - 1; I++) {
            double wx[3];
            mg_weights_1d(I, nxc, fold_neumann, wx);

            size_t i = 2 * I;
            size_t j = 2 * (size_t)J;
            double sum = 0.0;
            for (int oj = -1; oj <= 1; oj++) {
                for (int oi = -1; oi <= 1; oi++) {
                    double w = wx[oi + 1] * wy[oj + 1];
                    sum += w * fine[IDX_2D(i + oi, j + oj, nxf)];
                }
            }
            coarse[IDX_2D(I, (size_t)J, nxc)] = sum / 16.0;
        }
    }
}

/** Full-weighting restriction, 3D; serial over coarse planes, parallel over rows */
static void mg_restrict_3d_omp(const double* fine, double* coarse,
                               size_t nxf, size_t nyf, size_t nzf,
                               size_t nxc, size_t nyc, size_t nzc,
                               int fold_neumann) {
    (void)nzf;
    int nyc_int = poisson_solver_size_to_int(nyc);
    int par = mg_omp_parallel((nxc - 2) * (nyc - 2));

    for (size_t K = 1; K < nzc - 1; K++) {
        double wz[3];
        mg_weights_1d(K, nzc, fold_neumann, wz);

        int J;
#pragma omp parallel for schedule(static) if(par)
        for (J = 1; J < nyc_int - 1; J++) {
            double wy[3];
            mg_weights_1d((size_t)J, nyc, fold_neumann, wy);
            for (size_t I = 1; I < nxc - 1; I++) {
                double wx[3];
                mg_weights_1d(I, nxc, fold_neumann, wx);

                size_t i = 2 * I;
                size_t j = 2 * (size_t)J;
                size_t k = 2 * K;
                double sum = 0.0;
                for (int ok = -1; ok <= 1; ok++) {
                    for (int oj = -1; oj <= 1; oj++) {
                        for (int oi = -1; oi <= 1; oi++) {
                            double w = wx[oi + 1] * wy[oj + 1] * wz[ok + 1];
                            sum += w * fine[IDX_3D(i + oi, j + oj, k + ok,
                                                   nxf, nyf)];
                        }
                    }
                }
                coarse[IDX_3D(I, (size_t)J, K, nxc, nyc)] = sum / 64.0;
            }
        }
    }
}

/* ============================================================================
 * PROLONGATION (coarse -> fine, bilinear/trilinear, additive)
 * ============================================================================ */

/** Bilinear prolongation, adds into fine interior; parallel over fine rows */
static void mg_prolongate_add_2d_omp(const double* coarse, double* fine,
                                     size_t nxc, size_t nyc,
                                     size_t nxf, size_t nyf) {
    (void)nyc;
    int nyf_int = poisson_solver_size_to_int(nyf);
    int par = mg_omp_parallel((nxf - 2) * (nyf - 2));

    int j;
#pragma omp parallel for schedule(static) if(par)
    for (j = 1; j < nyf_int - 1; j++) {
        size_t J = (size_t)j >> 1;
        int jodd = (int)((size_t)j & 1);
        for (size_t i = 1; i < nxf - 1; i++) {
            size_t I = i >> 1;
            int iodd = (int)(i & 1);
            double e;
            if (!iodd && !jodd) {
                e = coarse[IDX_2D(I, J, nxc)];
            } else if (iodd && !jodd) {
                e = 0.5 * (coarse[IDX_2D(I, J, nxc)]
                         + coarse[IDX_2D(I + 1, J, nxc)]);
            } else if (!iodd && jodd) {
                e = 0.5 * (coarse[IDX_2D(I, J, nxc)]
                         + coarse[IDX_2D(I, J + 1, nxc)]);
            } else {
                e = 0.25 * (coarse[IDX_2D(I, J, nxc)]
                          + coarse[IDX_2D(I + 1, J, nxc)]
                          + coarse[IDX_2D(I, J + 1, nxc)]
                          + coarse[IDX_2D(I + 1, J + 1, nxc)]);
            }
            fine[IDX_2D(i, (size_t)j, nxf)] += e;
        }
    }
}

/** Trilinear prolongation, adds into fine interior; serial over fine planes */
static void mg_prolongate_add_3d_omp(const double* coarse, double* fine,
                                     size_t nxc, size_t nyc, size_t nzc,
                                     size_t nxf, size_t nyf, size_t nzf) {
    (void)nzc;
    int nyf_int = poisson_solver_size_to_int(nyf);
    int par = mg_omp_parallel((nxf - 2) * (nyf - 2));

    for (size_t k = 1; k < nzf - 1; k++) {
        size_t K = k >> 1;
        size_t nk = (k & 1) ? 2 : 1;
        double wk = (k & 1) ? 0.5 : 1.0;

        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < nyf_int - 1; j++) {
            size_t J = (size_t)j >> 1;
            size_t nj = ((size_t)j & 1) ? 2 : 1;
            double wj = ((size_t)j & 1) ? 0.5 : 1.0;
            for (size_t i = 1; i < nxf - 1; i++) {
                size_t I = i >> 1;
                size_t ni = (i & 1) ? 2 : 1;
                double wi = (i & 1) ? 0.5 : 1.0;
                double e = 0.0;
                for (size_t ck = 0; ck < nk; ck++) {
                    for (size_t cj = 0; cj < nj; cj++) {
                        for (size_t ci = 0; ci < ni; ci++) {
                            e += wi * wj * wk
                               * coarse[IDX_3D(I + ci, J + cj, K + ck,
                                               nxc, nyc)];
                        }
                    }
                }
                fine[IDX_3D(i, (size_t)j, k, nxf, nyf)] += e;
            }
        }
    }
}

/* ============================================================================
 * INTERIOR MEAN / INTERIOR ZERO
 * ============================================================================ */

/** Subtract the (serially computed) interior mean from interior points */
static void mg_subtract_interior_mean_omp(double* f, size_t nx, size_t ny,
                                          size_t nz) {
    double mean = mg_interior_mean(f, nx, ny, nz);

    size_t stride_z, k_start, k_end;
    poisson_solver_compute_3d_bounds(nz, nx, ny, &stride_z, &k_start, &k_end);
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);
    int par = mg_omp_parallel((nx - 2) * (ny - 2));

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                f[k * stride_z + IDX_2D((size_t)i, (size_t)j, nx)] -= mean;
            }
        }
    }
}

/** Zero the interior of x at one level (boundary values untouched) */
static void mg_zero_interior_omp(const mg_level_t* L, double* x) {
    size_t nx = L->nx;
    size_t stride_z = L->stride_z;
    int ny_int = poisson_solver_size_to_int(L->ny);
    int nx_int = poisson_solver_size_to_int(nx);
    int par = mg_omp_parallel((nx - 2) * (L->ny - 2));

    for (size_t k = L->k_start; k < L->k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) if(par)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                x[k * stride_z + IDX_2D((size_t)i, (size_t)j, nx)] = 0.0;
            }
        }
    }
}

/* ============================================================================
 * ALGORITHM (shared multigrid template)
 * ============================================================================ */

#define MGT_SUFFIX                  omp
#define MGT_SOLVER_NAME             POISSON_SOLVER_TYPE_MG_OMP
#define MGT_DESCRIPTION             "Geometric multigrid V/W/F-cycle (OpenMP)"
#define MGT_BACKEND                 POISSON_BACKEND_OMP

#define MGT_BC_NEUMANN_PLANE        mg_bc_neumann_plane_omp
#define MGT_RBGS_SWEEP              mg_rbgs_sweep_omp
#define MGT_JACOBI_SWEEP            mg_jacobi_sweep_omp
#define MGT_RESIDUAL                mg_residual_omp
#define MGT_RESTRICT_2D             mg_restrict_2d_omp
#define MGT_RESTRICT_3D             mg_restrict_3d_omp
#define MGT_PROLONGATE_ADD_2D       mg_prolongate_add_2d_omp
#define MGT_PROLONGATE_ADD_3D       mg_prolongate_add_3d_omp
#define MGT_SUBTRACT_INTERIOR_MEAN  mg_subtract_interior_mean_omp
#define MGT_ZERO_INTERIOR           mg_zero_interior_omp

#include "../multigrid_template/linear_solver_multigrid_template.h"

#endif /* CFD_ENABLE_OPENMP */
