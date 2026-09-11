/**
 * @file linear_solver_gmres_simd_template.h
 * @brief Restarted GMRES(m) SIMD primitives template (AVX2/NEON parameterized)
 *
 * Parameterized header that defines the vectorized O(n) GMRES primitives with
 * architecture-agnostic SIMD macros and then includes the shared GMRES(m)
 * algorithm template (gmres_template/linear_solver_gmres_template.h). Included
 * twice (AVX2, NEON) with different macro definitions. The dense
 * Givens/Hessenberg and back-substitution work in the algorithm template is plain
 * scalar, so SIMD and scalar results stay consistent to rounding.
 *
 * The operator/sign convention matches linear_solver_gmres.c and CG exactly:
 * A = -nabla^2, b = -rhs. Supports 2D (nz==1) and 3D via branch-free stencil
 * (stride_z=0, inv_dz2=0.0 for 2D).
 *
 * REQUIRED MACROS (see linear_solver_bicgstab_simd_template.h for the full list):
 * SIMD_SUFFIX, SIMD_VEC, SIMD_WIDTH, SIMD_LOAD, SIMD_STORE, SIMD_SET1,
 * SIMD_SETZERO, SIMD_ADD, SIMD_SUB, SIMD_MUL, SIMD_FMA, SIMD_HSUM.
 */

#include "cfd/core/indexing.h"

#ifndef SIMD_SUFFIX
#error "SIMD_SUFFIX must be defined before including linear_solver_gmres_simd_template.h"
#endif
#ifndef SIMD_VEC
#error "SIMD_VEC must be defined before including linear_solver_gmres_simd_template.h"
#endif
#ifndef SIMD_WIDTH
#error "SIMD_WIDTH must be defined before including linear_solver_gmres_simd_template.h"
#endif

//=============================================================================
// TOKEN PASTING MACROS
//=============================================================================

#define CONCAT_IMPL(a, b) a##_##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define SIMD_FUNC(name) CONCAT(name, SIMD_SUFFIX)

//=============================================================================
// SIMD O(n) PRIMITIVES (interior points only)
//=============================================================================

/** Dot product over interior points (SIMD + OpenMP reduction) */
static inline double SIMD_FUNC(gmres_dot)(const double* a, const double* b,
                                          size_t nx, size_t ny,
                                          size_t k_start, size_t k_end,
                                          size_t stride_z) {
    double sum = 0.0;
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return 0.0;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            double row_sum = 0.0;
            size_t i = 1;
            SIMD_VEC acc = SIMD_SETZERO();

            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_VEC va = SIMD_LOAD(&a[idx]);
                SIMD_VEC vb = SIMD_LOAD(&b[idx]);
                acc = SIMD_FMA(va, vb, acc);
            }
            row_sum += SIMD_HSUM(acc);

            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                row_sum += a[idx] * b[idx];
            }
            sum += row_sum;
        }
    }
    return sum;
}

/** y = y + alpha*x (SIMD FMA) */
static inline void SIMD_FUNC(gmres_axpy)(double alpha, const double* x, double* y,
                                         size_t nx, size_t ny,
                                         size_t k_start, size_t k_end,
                                         size_t stride_z) {
    SIMD_VEC alpha_vec = SIMD_SET1(alpha);
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_VEC vx = SIMD_LOAD(&x[idx]);
                SIMD_VEC vy = SIMD_LOAD(&y[idx]);
                vy = SIMD_FMA(alpha_vec, vx, vy);
                SIMD_STORE(&y[idx], vy);
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                y[idx] += alpha * x[idx];
            }
        }
    }
}

/** x = alpha*x (SIMD) */
static inline void SIMD_FUNC(gmres_scale)(double alpha, double* x,
                                          size_t nx, size_t ny,
                                          size_t k_start, size_t k_end,
                                          size_t stride_z) {
    SIMD_VEC alpha_vec = SIMD_SET1(alpha);
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_VEC vx = SIMD_LOAD(&x[idx]);
                SIMD_STORE(&x[idx], SIMD_MUL(alpha_vec, vx));
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                x[idx] *= alpha;
            }
        }
    }
}

/** dst = src (interior points) */
static inline void SIMD_FUNC(gmres_copy)(const double* src, double* dst,
                                         size_t nx, size_t ny,
                                         size_t k_start, size_t k_end,
                                         size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_STORE(&dst[idx], SIMD_LOAD(&src[idx]));
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                dst[idx] = src[idx];
            }
        }
    }
}

/** Apply negative Laplacian: Ap = -nabla^2(p) = A p (SIMD stencil) */
static inline void SIMD_FUNC(gmres_apply_A)(const double* p, double* Ap,
                                            size_t nx, size_t ny,
                                            double dx2, double dy2, double inv_dz2,
                                            size_t k_start, size_t k_end,
                                            size_t stride_z) {
    SIMD_VEC dx2_inv = SIMD_SET1(1.0 / dx2);
    SIMD_VEC dy2_inv = SIMD_SET1(1.0 / dy2);
    SIMD_VEC dz2_inv = SIMD_SET1(inv_dz2);
    SIMD_VEC two_vec = SIMD_SET1(2.0);
    SIMD_VEC zero = SIMD_SETZERO();
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_VEC p_c = SIMD_LOAD(&p[idx]);
                SIMD_VEC p_w = SIMD_LOAD(&p[idx - 1]);
                SIMD_VEC p_e = SIMD_LOAD(&p[idx + 1]);
                SIMD_VEC p_s = SIMD_LOAD(&p[idx - nx]);
                SIMD_VEC p_n = SIMD_LOAD(&p[idx + nx]);
                SIMD_VEC p_zm = SIMD_LOAD(&p[idx - stride_z]);
                SIMD_VEC p_zp = SIMD_LOAD(&p[idx + stride_z]);

                SIMD_VEC two_center = SIMD_MUL(two_vec, p_c);
                SIMD_VEC d2x = SIMD_MUL(SIMD_SUB(SIMD_ADD(p_e, p_w), two_center), dx2_inv);
                SIMD_VEC d2y = SIMD_MUL(SIMD_SUB(SIMD_ADD(p_n, p_s), two_center), dy2_inv);
                SIMD_VEC d2z = SIMD_MUL(SIMD_SUB(SIMD_ADD(p_zp, p_zm), two_center), dz2_inv);
                SIMD_VEC laplacian = SIMD_ADD(SIMD_ADD(d2x, d2y), d2z);
                /* Ap = -laplacian */
                SIMD_STORE(&Ap[idx], SIMD_SUB(zero, laplacian));
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                double lap = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2
                           + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2
                           + (p[idx + stride_z] + p[idx - stride_z] - 2.0 * p[idx]) * inv_dz2;
                Ap[idx] = -lap;
            }
        }
    }
}

/** Residual r = b - A x = -rhs + nabla^2 x (SIMD, fused) */
static inline void SIMD_FUNC(gmres_residual)(const double* x, const double* rhs,
                                             double* r, size_t nx, size_t ny,
                                             double dx2, double dy2, double inv_dz2,
                                             size_t k_start, size_t k_end,
                                             size_t stride_z) {
    SIMD_VEC dx2_inv = SIMD_SET1(1.0 / dx2);
    SIMD_VEC dy2_inv = SIMD_SET1(1.0 / dy2);
    SIMD_VEC dz2_inv = SIMD_SET1(inv_dz2);
    SIMD_VEC two_vec = SIMD_SET1(2.0);
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_VEC c = SIMD_LOAD(&x[idx]);
                SIMD_VEC l = SIMD_LOAD(&x[idx - 1]);
                SIMD_VEC rr = SIMD_LOAD(&x[idx + 1]);
                SIMD_VEC d = SIMD_LOAD(&x[idx - nx]);
                SIMD_VEC u = SIMD_LOAD(&x[idx + nx]);
                SIMD_VEC bk = SIMD_LOAD(&x[idx - stride_z]);
                SIMD_VEC fr = SIMD_LOAD(&x[idx + stride_z]);

                SIMD_VEC two_center = SIMD_MUL(two_vec, c);
                SIMD_VEC d2x = SIMD_MUL(SIMD_SUB(SIMD_ADD(l, rr), two_center), dx2_inv);
                SIMD_VEC d2y = SIMD_MUL(SIMD_SUB(SIMD_ADD(d, u), two_center), dy2_inv);
                SIMD_VEC d2z = SIMD_MUL(SIMD_SUB(SIMD_ADD(bk, fr), two_center), dz2_inv);
                SIMD_VEC lap = SIMD_ADD(SIMD_ADD(d2x, d2y), d2z);
                /* r = -rhs + lap */
                SIMD_VEC vrhs = SIMD_LOAD(&rhs[idx]);
                SIMD_STORE(&r[idx], SIMD_SUB(lap, vrhs));
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                double lap = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / dx2
                           + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / dy2
                           + (x[idx + stride_z] + x[idx - stride_z] - 2.0 * x[idx]) * inv_dz2;
                r[idx] = -rhs[idx] + lap;
            }
        }
    }
}

/** z = diag_inv * r (Jacobi preconditioner, SIMD) */
static inline void SIMD_FUNC(gmres_precond)(const double* r, double* z,
                                            size_t nx, size_t ny,
                                            double diag_inv,
                                            size_t k_start, size_t k_end,
                                            size_t stride_z) {
    SIMD_VEC diag_vec = SIMD_SET1(diag_inv);
    int ny_int = poisson_solver_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = k_start; k < k_end; k++) {
        int jj;
        #pragma omp parallel for schedule(static)
        for (jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            size_t i = 1;
            for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                SIMD_STORE(&z[idx], SIMD_MUL(diag_vec, SIMD_LOAD(&r[idx])));
            }
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                z[idx] = diag_inv * r[idx];
            }
        }
    }
}

//=============================================================================
// ALGORITHM (shared GMRES(m) template)
//=============================================================================

#define GMRES_SUFFIX            SIMD_SUFFIX
#define GMRES_SOLVER_NAME       POISSON_SOLVER_TYPE_GMRES_SIMD
#define GMRES_DESCRIPTION       "Restarted GMRES(m) (SIMD + OpenMP)"
#define GMRES_BACKEND           POISSON_BACKEND_SIMD
#define GMRES_LOG_TAG           "GMRES-SIMD"
#define GMRES_VEC_CALLOC(count) cfd_aligned_calloc((count), sizeof(double))
#define GMRES_VEC_FREE(ptr)     cfd_aligned_free(ptr)

#define GMRES_DOT               SIMD_FUNC(gmres_dot)
#define GMRES_AXPY              SIMD_FUNC(gmres_axpy)
#define GMRES_SCALE             SIMD_FUNC(gmres_scale)
#define GMRES_COPY              SIMD_FUNC(gmres_copy)
#define GMRES_APPLY_A           SIMD_FUNC(gmres_apply_A)
#define GMRES_RESIDUAL          SIMD_FUNC(gmres_residual)
#define GMRES_PRECOND           SIMD_FUNC(gmres_precond)

#include "../gmres_template/linear_solver_gmres_template.h"

//=============================================================================
// CLEANUP MACROS
//=============================================================================

#undef CONCAT_IMPL
#undef CONCAT
#undef SIMD_FUNC
