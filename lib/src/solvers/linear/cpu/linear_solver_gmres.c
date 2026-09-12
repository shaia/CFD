/**
 * @file linear_solver_gmres.c
 * @brief Restarted GMRES(m) solver - scalar CPU implementation
 *
 * Reference backend. Supplies the scalar O(n) vector primitives to the shared
 * GMRES(m) algorithm template (gmres_template/linear_solver_gmres_template.h),
 * which documents the algorithm and holds its only implementation.
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"

/* ============================================================================
 * O(n) VECTOR PRIMITIVES (interior points only)
 *
 * dot_product / axpy / apply_laplacian / compute_residual / copy_vector /
 * apply_jacobi_precond mirror linear_solver_cg.c exactly. scale_vector is the
 * GMRES-specific addition.
 * ============================================================================ */

/** Dot product over interior points */
static double dot_product(const double* a, const double* b,
                          size_t nx, size_t ny,
                          size_t k_start, size_t k_end, size_t stride_z) {
    double sum = 0.0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                sum += a[idx] * b[idx];
            }
        }
    }
    return sum;
}

/** y = y + alpha * x (interior points only) */
static void axpy(double alpha, const double* x, double* y,
                 size_t nx, size_t ny,
                 size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                y[idx] += alpha * x[idx];
            }
        }
    }
}

/** x = alpha * x (interior points only) */
static void scale_vector(double alpha, double* x,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                x[idx] *= alpha;
            }
        }
    }
}

/**
 * Apply negative Laplacian operator: Ap = -nabla^2(p) (matches CG's A = -nabla^2).
 */
static void apply_laplacian(const double* p, double* Ap,
                            size_t nx, size_t ny,
                            double dx2, double dy2, double inv_dz2,
                            size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));

                double laplacian =
                    ((p[idx + 1] - (2.0 * p[idx]) + p[idx - 1]) * dx2_inv)
                  + ((p[idx + (nx)] - (2.0 * p[idx]) + p[idx - (nx)]) * dy2_inv)
                  + ((p[idx + (stride_z)] + p[idx - (stride_z)] - (2.0 * p[idx])) * inv_dz2);
                Ap[idx] = -laplacian;
            }
        }
    }
}

/**
 * Compute residual r = b - A x = -rhs + nabla^2 x (matches CG convention).
 */
static void compute_residual(const double* x, const double* rhs, double* r,
                             size_t nx, size_t ny,
                             double dx2, double dy2, double inv_dz2,
                             size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));

                double laplacian =
                    ((x[idx + 1] - (2.0 * x[idx]) + x[idx - 1]) * dx2_inv)
                  + ((x[idx + (nx)] - (2.0 * x[idx]) + x[idx - (nx)]) * dy2_inv)
                  + ((x[idx + (stride_z)] + x[idx - (stride_z)] - (2.0 * x[idx])) * inv_dz2);

                r[idx] = -rhs[idx] + laplacian;
            }
        }
    }
}

/** Copy dst = src (interior points only) */
static void copy_vector(const double* src, double* dst,
                        size_t nx, size_t ny,
                        size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                dst[idx] = src[idx];
            }
        }
    }
}

/** Apply Jacobi preconditioner: z = M^{-1} r = diag_inv * r (interior points) */
static void apply_jacobi_precond(const double* r, double* z,
                                 size_t nx, size_t ny,
                                 double diag_inv,
                                 size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                z[idx] = diag_inv * r[idx];
            }
        }
    }
}

/* ============================================================================
 * ALGORITHM (shared GMRES(m) template)
 * ============================================================================ */

#define GMRES_SUFFIX            scalar
#define GMRES_SOLVER_NAME       POISSON_SOLVER_TYPE_GMRES_SCALAR
#define GMRES_DESCRIPTION       "Restarted GMRES(m) (scalar CPU)"
#define GMRES_BACKEND           POISSON_BACKEND_SCALAR
#define GMRES_LOG_TAG           "GMRES"
#define GMRES_VEC_CALLOC(count) cfd_calloc((count), sizeof(double))
#define GMRES_VEC_FREE(ptr)     cfd_free(ptr)

#define GMRES_DOT               dot_product
#define GMRES_AXPY              axpy
#define GMRES_SCALE             scale_vector
#define GMRES_COPY              copy_vector
#define GMRES_APPLY_A           apply_laplacian
#define GMRES_RESIDUAL          compute_residual
#define GMRES_PRECOND           apply_jacobi_precond

#include "../gmres_template/linear_solver_gmres_template.h"
