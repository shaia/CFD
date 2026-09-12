/**
 * @file linear_solver_multigrid.c
 * @brief Geometric multigrid Poisson solver - scalar CPU implementation
 *
 * Reference backend. Supplies the scalar smoother, residual and boundary
 * primitives, plus the grid-transfer operators of cpu/multigrid_transfer.c, to
 * the shared multigrid algorithm template
 * (multigrid_template/linear_solver_multigrid_template.h), which documents the
 * algorithm and holds its only implementation.
 */

#include "../linear_solver_internal.h"
#include "../multigrid_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/indexing.h"

/* ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================ */

/** Zero-gradient BCs on one nx x ny plane */
static void mg_bc_neumann_plane_scalar(double* plane, size_t nx, size_t ny) {
    bc_apply_scalar_cpu(plane, nx, ny, BC_TYPE_NEUMANN);
}

/* ============================================================================
 * SMOOTHER SWEEPS (no BCs; the template applies them after each sweep)
 * ============================================================================ */

/**
 * One Red-Black Gauss-Seidel pass (omega = 1). Same stencil algebra as
 * linear_solver_redblack.c.
 */
static void mg_rbgs_sweep_scalar(const mg_level_t* L, double* x,
                                 const double* rhs) {
    size_t nx = L->nx;
    size_t ny = L->ny;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    double inv_factor = L->inv_factor;
    size_t stride_z = L->stride_z;

    /* Red pass: (i+j+k) % 2 == 0, then black pass: (i+j+k) % 2 == 1 */
    for (int color = 0; color < 2; color++) {
        for (size_t k = L->k_start; k < L->k_end; k++) {
            for (size_t j = 1; j < ny - 1; j++) {
                size_t i_start =
                    ((j + k) % 2 == (size_t)color) ? 1 : 2;
                for (size_t i = i_start; i < nx - 1; i += 2) {
                    size_t idx = k * stride_z + IDX_2D(i, j, nx);

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
static void mg_jacobi_sweep_scalar(const mg_level_t* L, double* x,
                                   double* x_temp, const double* rhs) {
    size_t nx = L->nx;
    size_t ny = L->ny;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    double inv_factor = L->inv_factor;
    size_t stride_z = L->stride_z;

    for (size_t k = L->k_start; k < L->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

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
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                x[idx] = x_temp[idx];
            }
        }
    }
}

/* ============================================================================
 * RESIDUAL
 * ============================================================================ */

/**
 * r = rhs - Laplacian(x) at interior points. The coarse problem is
 * Laplacian(e) = r, so that Laplacian(x + P e) ~ rhs. Boundary entries of r
 * are never written (they stay zero from allocation; restriction never
 * reads them).
 */
static void mg_residual_scalar(const mg_level_t* L, const double* x,
                               const double* rhs, double* r) {
    size_t nx = L->nx;
    size_t ny = L->ny;
    double dx2 = L->dx2;
    double dy2 = L->dy2;
    double inv_dz2 = L->inv_dz2;
    size_t stride_z = L->stride_z;

    for (size_t k = L->k_start; k < L->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

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

/** Zero the interior of x at one level (boundary values untouched) */
static void mg_zero_interior_scalar(const mg_level_t* L, double* x) {
    for (size_t k = L->k_start; k < L->k_end; k++) {
        for (size_t j = 1; j < L->ny - 1; j++) {
            for (size_t i = 1; i < L->nx - 1; i++) {
                x[k * L->stride_z + IDX_2D(i, j, L->nx)] = 0.0;
            }
        }
    }
}

/* ============================================================================
 * ALGORITHM (shared multigrid template)
 * ============================================================================ */

#define MGT_SUFFIX                  scalar
#define MGT_SOLVER_NAME             POISSON_SOLVER_TYPE_MG_SCALAR
#define MGT_DESCRIPTION             "Geometric multigrid V/W/F-cycle (scalar CPU)"
#define MGT_BACKEND                 POISSON_BACKEND_SCALAR

#define MGT_BC_NEUMANN_PLANE        mg_bc_neumann_plane_scalar
#define MGT_RBGS_SWEEP              mg_rbgs_sweep_scalar
#define MGT_JACOBI_SWEEP            mg_jacobi_sweep_scalar
#define MGT_RESIDUAL                mg_residual_scalar
#define MGT_RESTRICT_2D             mg_restrict_2d
#define MGT_RESTRICT_3D             mg_restrict_3d
#define MGT_PROLONGATE_ADD_2D       mg_prolongate_add_2d
#define MGT_PROLONGATE_ADD_3D       mg_prolongate_add_3d
#define MGT_SUBTRACT_INTERIOR_MEAN  mg_subtract_interior_mean
#define MGT_ZERO_INTERIOR           mg_zero_interior_scalar

#include "../multigrid_template/linear_solver_multigrid_template.h"
