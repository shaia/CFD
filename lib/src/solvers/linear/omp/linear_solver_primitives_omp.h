/**
 * @file linear_solver_primitives_omp.h
 * @brief OpenMP-parallelized O(n) vector primitives shared by the OMP Krylov solvers
 *
 * Included by linear_solver_cg_omp.c and linear_solver_gmres_omp.c inside their
 * CFD_ENABLE_OPENMP guards. Every primitive acts on interior points of a 2D
 * (nz == 1: stride_z = 0, one k-plane) or 3D grid, with bounds from
 * poisson_solver_compute_3d_bounds. Each k-plane opens one
 * `parallel for schedule(static)` over rows j with an int loop variable
 * (MSVC OpenMP 2.0). Rows write disjoint index ranges, so inputs and outputs
 * must not alias except where a primitive updates in place.
 *
 * dot_product_omp is the only cross-element accumulation (reduction(+:sum)): it
 * splits the sum into per-thread partial sums whose combination order OpenMP
 * leaves unspecified, so its last bits can change with the thread count and
 * between runs. The other primitives are element-wise and reproducible.
 */

#ifndef CFD_LINEAR_SOLVER_PRIMITIVES_OMP_H
#define CFD_LINEAR_SOLVER_PRIMITIVES_OMP_H

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"

#include <string.h>

/** Dot product over interior points */
static inline double dot_product_omp(const double* a, const double* b,
                                     size_t nx, size_t ny,
                                     size_t k_start, size_t k_end, size_t stride_z) {
    double sum = 0.0;
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                sum += a[idx] * b[idx];
            }
        }
    }
    return sum;
}

/** y = y + alpha * x (interior points only) */
static inline void axpy_omp(double alpha, const double* x, double* y,
                            size_t nx, size_t ny,
                            size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                y[idx] += alpha * x[idx];
            }
        }
    }
}

/** x = alpha * x (interior points only) */
static inline void scale_vector_omp(double alpha, double* x,
                                    size_t nx, size_t ny,
                                    size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                x[idx] *= alpha;
            }
        }
    }
}

/** Apply negative Laplacian operator: Ap = -nabla^2(p) (A = -nabla^2) */
static inline void apply_laplacian_omp(const double* p, double* Ap,
                                       size_t nx, size_t ny,
                                       double dx2, double dy2, double inv_dz2,
                                       size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                double laplacian =
                    (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv
                  + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv
                  + (p[idx + stride_z] + p[idx - stride_z] - 2.0 * p[idx]) * inv_dz2;
                Ap[idx] = -laplacian;
            }
        }
    }
}

/** Residual r = b - A x = -rhs + nabla^2 x */
static inline void compute_residual_omp(const double* x, const double* rhs, double* r,
                                        size_t nx, size_t ny,
                                        double dx2, double dy2, double inv_dz2,
                                        size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                double laplacian =
                    (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) * dx2_inv
                  + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) * dy2_inv
                  + (x[idx + stride_z] + x[idx - stride_z] - 2.0 * x[idx]) * inv_dz2;
                r[idx] = -rhs[idx] + laplacian;
            }
        }
    }
}

/** dst = src (interior points only) */
static inline void copy_vector_omp(const double* src, double* dst,
                                   size_t nx, size_t ny,
                                   size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            size_t row_start = k * stride_z + (size_t)j * nx;
            memcpy(&dst[row_start + 1], &src[row_start + 1], (nx - 2) * sizeof(double));
        }
    }
}

/** Jacobi preconditioner: z = M^{-1} r = diag_inv * r (interior points) */
static inline void apply_jacobi_precond_omp(const double* r, double* z,
                                            size_t nx, size_t ny,
                                            double diag_inv,
                                            size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                z[idx] = diag_inv * r[idx];
            }
        }
    }
}

#endif /* CFD_LINEAR_SOLVER_PRIMITIVES_OMP_H */
