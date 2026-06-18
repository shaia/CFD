/**
 * @file energy_solver_avx2.c
 * @brief AVX2-vectorized energy equation solver (advection-diffusion step)
 *
 * Solves: dT/dt + u*nabla(T) = alpha * nabla^2(T) + Q
 * using explicit Euler time integration and central differences.
 *
 * Same numerics as the scalar reference (energy/cpu/energy_solver.c). The
 * interior stencil is vectorized along the unit-stride i direction (4 doubles
 * per AVX2 register) with a scalar remainder tail; the j loop is parallelized
 * with OpenMP. When AVX2 is unavailable at compile time the whole interior is
 * processed with the scalar path, so this file always provides a correct
 * implementation. The optional heat source Q is applied in a unified scalar
 * pass (only when a callback is configured), keeping the vector path branchless.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz/inv_dz2=0.0 cause all
 * z-terms to vanish, producing identical results to a 2D code path.
 */

#define _USE_MATH_DEFINES

#include "../energy_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(CFD_HAS_AVX2)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

cfd_status_t energy_step_explicit_avx2_with_workspace(
    flow_field* field, const grid* grid,
    const ns_solver_params_t* params,
    double dt, double time,
    double* T_workspace, size_t workspace_size) {
    if (!field || !grid || !params) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_avx2: field, grid, and params must be non-NULL");
        return CFD_ERROR_INVALID;
    }
    if (!field->T) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_avx2: missing temperature field");
        return CFD_ERROR_INVALID;
    }

    /* Skip when energy equation is disabled */
    if (params->alpha <= 0.0) {
        return CFD_SUCCESS;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;

    if (!grid->dx || !grid->dy || nx < 3 || ny < 3) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "energy_solver_avx2: grid too small or missing dx/dy");
        return CFD_ERROR_INVALID;
    }
    size_t plane = nx * ny;
    size_t total = plane * nz;
    double alpha = params->alpha;

    /* Validate uniform spacing (central-difference stencil assumes it) */
    const double dx0 = grid->dx[0];
    const double dy0 = grid->dy[0];
    {
        const double tol_x = 1e-12 * fmax(1.0, fabs(dx0));
        for (size_t i = 1; i < nx - 1; i++) {
            if (fabs(grid->dx[i] - dx0) > tol_x) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_avx2: non-uniform dx not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
        const double tol_y = 1e-12 * fmax(1.0, fabs(dy0));
        for (size_t j = 1; j < ny - 1; j++) {
            if (fabs(grid->dy[j] - dy0) > tol_y) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_avx2: non-uniform dy not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
    }
    if (nz > 1) {
        if (!grid->dz) {
            cfd_set_error(CFD_ERROR_INVALID,
                          "energy_solver_avx2: missing dz for 3D energy solve");
            return CFD_ERROR_INVALID;
        }
        const double dz0 = grid->dz[0];
        const double tol_z = 1e-12 * fmax(1.0, fabs(dz0));
        for (size_t k = 1; k < nz - 1; k++) {
            if (fabs(grid->dz[k] - dz0) > tol_z) {
                cfd_set_error(CFD_ERROR_UNSUPPORTED,
                              "energy_solver_avx2: non-uniform dz not supported");
                return CFD_ERROR_UNSUPPORTED;
            }
        }
    }

    /* Precomputed constants for uniform-grid stencil */
    double inv_2dx = 1.0 / (2.0 * dx0);
    double inv_2dy = 1.0 / (2.0 * dy0);
    double inv_dx2 = 1.0 / (dx0 * dx0);
    double inv_dy2 = 1.0 / (dy0 * dy0);

    /* Branch-free 3D constants */
    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start  = (nz > 1) ? 1 : 0;
    size_t k_end    = (nz > 1) ? (nz - 1) : 1;
    double inv_2dz  = (nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    double inv_dz2  = (nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    /* Use caller's workspace or allocate internally */
    int owns_buffer = 0;
    double* T_new;
    if (T_workspace && workspace_size >= total) {
        T_new = T_workspace;
    } else {
        T_new = (double*)cfd_calloc(total, sizeof(double));
        if (!T_new) {
            return CFD_ERROR_NOMEM;
        }
        owns_buffer = 1;
    }
    memcpy(T_new, field->T, total * sizeof(double));

    const double* T = field->T;
    const double* u = field->u;
    const double* v = field->v;
    const double* w = field->w;

    /* Number of interior columns processed in 4-wide vector groups (i=1..nx-2) */
    size_t n_interior = nx - 2;
    size_t i_tail = 1 + (n_interior / 4) * 4;

    for (size_t k = k_start; k < k_end; k++) {
        size_t k_offset = k * stride_z;
        int j;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (j = 1; j < (int)ny - 1; j++) {
            size_t row = k_offset + (size_t)j * nx;

#if USE_AVX
            const __m256d inv_2dx_v = _mm256_set1_pd(inv_2dx);
            const __m256d inv_2dy_v = _mm256_set1_pd(inv_2dy);
            const __m256d inv_2dz_v = _mm256_set1_pd(inv_2dz);
            const __m256d inv_dx2_v = _mm256_set1_pd(inv_dx2);
            const __m256d inv_dy2_v = _mm256_set1_pd(inv_dy2);
            const __m256d inv_dz2_v = _mm256_set1_pd(inv_dz2);
            const __m256d alpha_v = _mm256_set1_pd(alpha);
            const __m256d dt_v = _mm256_set1_pd(dt);
            const __m256d two_v = _mm256_set1_pd(2.0);

            for (size_t i = 1; i < i_tail; i += 4) {
                size_t idx = row + i;

                __m256d T_c = _mm256_loadu_pd(&T[idx]);
                __m256d uc = _mm256_loadu_pd(&u[idx]);
                __m256d vc = _mm256_loadu_pd(&v[idx]);
                __m256d wc = _mm256_loadu_pd(&w[idx]);

                __m256d T_xp = _mm256_loadu_pd(&T[idx + 1]);
                __m256d T_xm = _mm256_loadu_pd(&T[idx - 1]);
                __m256d T_yp = _mm256_loadu_pd(&T[idx + nx]);
                __m256d T_ym = _mm256_loadu_pd(&T[idx - nx]);
                __m256d T_zp = _mm256_loadu_pd(&T[idx + stride_z]);
                __m256d T_zm = _mm256_loadu_pd(&T[idx - stride_z]);

                /* Advection: u*dT/dx + v*dT/dy + w*dT/dz */
                __m256d dT_dx = _mm256_mul_pd(_mm256_sub_pd(T_xp, T_xm), inv_2dx_v);
                __m256d dT_dy = _mm256_mul_pd(_mm256_sub_pd(T_yp, T_ym), inv_2dy_v);
                __m256d dT_dz = _mm256_mul_pd(_mm256_sub_pd(T_zp, T_zm), inv_2dz_v);
                __m256d adv = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(uc, dT_dx), _mm256_mul_pd(vc, dT_dy)),
                    _mm256_mul_pd(wc, dT_dz));

                /* Diffusion: alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2) */
                __m256d d2x = _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(T_xp, T_xm), _mm256_mul_pd(two_v, T_c)), inv_dx2_v);
                __m256d d2y = _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(T_yp, T_ym), _mm256_mul_pd(two_v, T_c)), inv_dy2_v);
                __m256d d2z = _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_add_pd(T_zp, T_zm), _mm256_mul_pd(two_v, T_c)), inv_dz2_v);
                __m256d diff = _mm256_mul_pd(alpha_v,
                    _mm256_add_pd(_mm256_add_pd(d2x, d2y), d2z));

                /* Explicit Euler update (heat source added in the scalar pass) */
                __m256d dT = _mm256_mul_pd(dt_v, _mm256_sub_pd(diff, adv));
                _mm256_storeu_pd(&T_new[idx], _mm256_add_pd(T_c, dT));
            }
#endif
            /* Scalar remainder (and the full interior when AVX2 is disabled) */
            for (size_t i = (USE_AVX ? i_tail : 1); i < nx - 1; i++) {
                size_t idx = row + i;

                double T_cs = T[idx];
                double dT_dx = (T[idx + 1] - T[idx - 1]) * inv_2dx;
                double dT_dy = (T[idx + nx] - T[idx - nx]) * inv_2dy;
                double dT_dz = (T[idx + stride_z] - T[idx - stride_z]) * inv_2dz;
                double adv = u[idx] * dT_dx + v[idx] * dT_dy + w[idx] * dT_dz;

                double d2x = (T[idx + 1] - 2.0 * T_cs + T[idx - 1]) * inv_dx2;
                double d2y = (T[idx + nx] - 2.0 * T_cs + T[idx - nx]) * inv_dy2;
                double d2z = (T[idx + stride_z] - 2.0 * T_cs + T[idx - stride_z]) * inv_dz2;
                double diff = alpha * (d2x + d2y + d2z);

                T_new[idx] = T_cs + dt * (diff - adv);
            }

            /* Heat source term (scalar; only when a callback is configured) */
            if (params->heat_source_func) {
                double y = grid->y[j];
                double z = (nz > 1 && grid->z) ? grid->z[k] : 0.0;
                for (size_t i = 1; i < nx - 1; i++) {
                    size_t idx = row + i;
                    double Q = params->heat_source_func(grid->x[i], y, z, time,
                                                         params->heat_source_context);
                    T_new[idx] += dt * Q;
                }
            }
        }
    }

    /* Check for NaN/Inf */
    int has_nan = 0;
    ptrdiff_t total_int = (ptrdiff_t)total;
    ptrdiff_t n;
#ifdef _OPENMP
#pragma omp parallel for reduction(| : has_nan) schedule(static)
#endif
    for (n = 0; n < total_int; n++) {
        if (!isfinite(T_new[n])) {
            has_nan = 1;
        }
    }
    if (has_nan) {
        cfd_set_error(CFD_ERROR_DIVERGED,
                      "NaN/Inf detected in energy_step_explicit_avx2");
        if (owns_buffer) cfd_free(T_new);
        return CFD_ERROR_DIVERGED;
    }

    memcpy(field->T, T_new, total * sizeof(double));
    if (owns_buffer) cfd_free(T_new);

    return CFD_SUCCESS;
}
