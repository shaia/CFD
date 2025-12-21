/**
 * @file linear_solver_cg_neon_omp.c
 * @brief Conjugate Gradient solver - ARM NEON + OpenMP implementation
 *
 * Conjugate Gradient method optimized with:
 * - ARM NEON intrinsics for SIMD vectorization (2 doubles per operation)
 * - OpenMP for thread-level parallelism
 * - Runtime detection to ensure NEON is available
 *
 * Key optimizations:
 * - Vectorized dot products with horizontal add
 * - Vectorized axpy operations
 * - Vectorized Laplacian stencil
 * - OpenMP reduction for dot products
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ARM NEON + OpenMP detection */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define CG_HAS_NEON_OMP 1
#include <arm_neon.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(CG_HAS_NEON_OMP)

/* ============================================================================
 * CG NEON+OMP CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */

    /* Precomputed SIMD vectors */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t two_vec;

    /* CG working vectors (allocated during init) */
    double* r;         /* Residual vector */
    double* p;         /* Search direction */
    double* Ap;        /* A * p (Laplacian applied to p) */

    int initialized;
} cg_neon_omp_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * HELPER FUNCTIONS (NEON + OpenMP)
 * ============================================================================ */

/**
 * Compute dot product using NEON with OpenMP reduction
 */
static double dot_product_neon_omp(const double* a, const double* b,
                                    size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        double row_sum = 0.0;
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop: process 2 doubles at a time */
        float64x2_t acc = vdupq_n_f64(0.0);
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = row_start + i;
            float64x2_t va = vld1q_f64(&a[idx]);
            float64x2_t vb = vld1q_f64(&b[idx]);
            acc = vfmaq_f64(acc, va, vb);
        }

        /* Horizontal sum of acc */
        row_sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = row_start + i;
            row_sum += a[idx] * b[idx];
        }

        sum += row_sum;
    }

    return sum;
}

/**
 * Compute y = y + alpha * x using NEON with OpenMP
 */
static void axpy_neon_omp(double alpha, const double* x, double* y,
                          size_t nx, size_t ny) {
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop */
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = row_start + i;
            float64x2_t vx = vld1q_f64(&x[idx]);
            float64x2_t vy = vld1q_f64(&y[idx]);
            vy = vfmaq_f64(vy, alpha_vec, vx);
            vst1q_f64(&y[idx], vy);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = row_start + i;
            y[idx] += alpha * x[idx];
        }
    }
}

/**
 * Apply negative Laplacian using NEON with OpenMP
 */
static void apply_laplacian_neon_omp(const double* p, double* Ap,
                                      size_t nx, size_t ny,
                                      float64x2_t dx2_inv_vec, float64x2_t dy2_inv_vec,
                                      float64x2_t two_vec) {
    int ny_int = size_to_int(ny);
    int j;
    double dx2_inv_scalar = vgetq_lane_f64(dx2_inv_vec, 0);
    double dy2_inv_scalar = vgetq_lane_f64(dy2_inv_vec, 0);

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* SIMD loop */
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = (size_t)j * nx + i;

            /* Load neighbors */
            float64x2_t p_center = vld1q_f64(&p[idx]);
            float64x2_t p_xp = vld1q_f64(&p[idx + 1]);
            float64x2_t p_xm = vld1q_f64(&p[idx - 1]);
            float64x2_t p_yp = vld1q_f64(&p[idx + nx]);
            float64x2_t p_ym = vld1q_f64(&p[idx - nx]);

            /* Compute second derivatives */
            /* d2p_dx2 = (p[i+1] - 2*p[i] + p[i-1]) / dx2 */
            float64x2_t sum_x = vaddq_f64(p_xp, p_xm);
            float64x2_t two_center = vmulq_f64(two_vec, p_center);
            float64x2_t d2p_dx2 = vsubq_f64(sum_x, two_center);
            d2p_dx2 = vmulq_f64(d2p_dx2, dx2_inv_vec);

            /* d2p_dy2 = (p[j+1] - 2*p[j] + p[j-1]) / dy2 */
            float64x2_t sum_y = vaddq_f64(p_yp, p_ym);
            float64x2_t d2p_dy2 = vsubq_f64(sum_y, two_center);
            d2p_dy2 = vmulq_f64(d2p_dy2, dy2_inv_vec);

            /* -Laplacian = -(d2p_dx2 + d2p_dy2) */
            float64x2_t laplacian = vaddq_f64(d2p_dx2, d2p_dy2);
            float64x2_t neg_laplacian = vnegq_f64(laplacian);

            vst1q_f64(&Ap[idx], neg_laplacian);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = (size_t)j * nx + i;
            double laplacian = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv_scalar
                             + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv_scalar;
            Ap[idx] = -laplacian;
        }
    }
}

/**
 * Compute initial residual using NEON with OpenMP
 */
static void compute_residual_neon_omp(const double* x, const double* rhs, double* r,
                                       size_t nx, size_t ny,
                                       float64x2_t dx2_inv_vec, float64x2_t dy2_inv_vec,
                                       float64x2_t two_vec) {
    int ny_int = size_to_int(ny);
    int j;
    double dx2_inv_scalar = vgetq_lane_f64(dx2_inv_vec, 0);
    double dy2_inv_scalar = vgetq_lane_f64(dy2_inv_vec, 0);

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* SIMD loop */
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = (size_t)j * nx + i;

            /* Load neighbors */
            float64x2_t x_center = vld1q_f64(&x[idx]);
            float64x2_t x_xp = vld1q_f64(&x[idx + 1]);
            float64x2_t x_xm = vld1q_f64(&x[idx - 1]);
            float64x2_t x_yp = vld1q_f64(&x[idx + nx]);
            float64x2_t x_ym = vld1q_f64(&x[idx - nx]);
            float64x2_t rhs_vec = vld1q_f64(&rhs[idx]);

            /* Compute Laplacian */
            float64x2_t sum_x = vaddq_f64(x_xp, x_xm);
            float64x2_t two_center = vmulq_f64(two_vec, x_center);
            float64x2_t d2x_dx2 = vsubq_f64(sum_x, two_center);
            d2x_dx2 = vmulq_f64(d2x_dx2, dx2_inv_vec);

            float64x2_t sum_y = vaddq_f64(x_yp, x_ym);
            float64x2_t d2x_dy2 = vsubq_f64(sum_y, two_center);
            d2x_dy2 = vmulq_f64(d2x_dy2, dy2_inv_vec);

            float64x2_t laplacian = vaddq_f64(d2x_dx2, d2x_dy2);

            /* r = -rhs + laplacian */
            float64x2_t neg_rhs = vnegq_f64(rhs_vec);
            float64x2_t r_vec = vaddq_f64(neg_rhs, laplacian);

            vst1q_f64(&r[idx], r_vec);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = (size_t)j * nx + i;
            double laplacian = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) * dx2_inv_scalar
                             + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) * dy2_inv_scalar;
            r[idx] = -rhs[idx] + laplacian;
        }
    }
}

/**
 * Copy vector using OpenMP
 */
static void copy_vector_omp(const double* src, double* dst,
                            size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t row_start = (size_t)j * nx;
        memcpy(&dst[row_start + 1], &src[row_start + 1], (nx - 2) * sizeof(double));
    }
}

/**
 * Update p = r + beta * p using NEON with OpenMP
 */
static void update_search_direction_neon_omp(const double* r, double* p,
                                              double beta, size_t nx, size_t ny) {
    float64x2_t beta_vec = vdupq_n_f64(beta);
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop */
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = row_start + i;
            float64x2_t vr = vld1q_f64(&r[idx]);
            float64x2_t vp = vld1q_f64(&p[idx]);
            /* p = r + beta * p */
            vp = vfmaq_f64(vr, beta_vec, vp);
            vst1q_f64(&p[idx], vp);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = row_start + i;
            p[idx] = r[idx] + beta * p[idx];
        }
    }
}

/* ============================================================================
 * CG NEON+OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t cg_neon_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)params;

    cg_neon_omp_context_t* ctx = (cg_neon_omp_context_t*)cfd_aligned_calloc(
        1, sizeof(cg_neon_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Precompute SIMD vectors */
    ctx->dx2_inv_vec = vdupq_n_f64(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(1.0 / ctx->dy2);
    ctx->two_vec = vdupq_n_f64(2.0);

    /* Allocate working vectors (aligned for NEON SIMD access) */
    size_t n = nx * ny;
    ctx->r = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->Ap = (double*)cfd_aligned_calloc(n, sizeof(double));

    if (!ctx->r || !ctx->p || !ctx->Ap) {
        cfd_aligned_free(ctx->r);
        cfd_aligned_free(ctx->p);
        cfd_aligned_free(ctx->Ap);
        cfd_aligned_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void cg_neon_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cg_neon_omp_context_t* ctx = (cg_neon_omp_context_t*)solver->context;
        cfd_aligned_free(ctx->r);
        cfd_aligned_free(ctx->p);
        cfd_aligned_free(ctx->Ap);
        cfd_aligned_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t cg_neon_omp_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;

    cg_neon_omp_context_t* ctx = (cg_neon_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;

    double* r = ctx->r;
    double* p = ctx->p;
    double* Ap = ctx->Ap;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Apply initial boundary conditions (use SIMD+OMP BC) */
    bc_apply_scalar_simd_omp(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute initial residual */
    compute_residual_neon_omp(x, rhs, r, nx, ny,
                              ctx->dx2_inv_vec, ctx->dy2_inv_vec, ctx->two_vec);

    /* Initial search direction: p_0 = r_0 */
    copy_vector_omp(r, p, nx, ny);

    /* Compute initial r_dot_r */
    double r_dot_r = dot_product_neon_omp(r, r, nx, ny);
    double initial_res = sqrt(r_dot_r);

    if (stats) {
        stats->initial_residual = initial_res;
    }

    /* Check if already converged */
    double tolerance = params->tolerance * initial_res;
    if (tolerance < params->absolute_tolerance) {
        tolerance = params->absolute_tolerance;
    }

    if (initial_res < params->absolute_tolerance) {
        if (stats) {
            stats->status = POISSON_CONVERGED;
            stats->iterations = 0;
            stats->final_residual = initial_res;
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return CFD_SUCCESS;
    }

    int converged = 0;
    int iter;
    double res_norm = initial_res;

    for (iter = 0; iter < params->max_iterations; iter++) {
        /* Compute Ap = A * p */
        apply_laplacian_neon_omp(p, Ap, nx, ny,
                                  ctx->dx2_inv_vec, ctx->dy2_inv_vec, ctx->two_vec);

        /* alpha = (r, r) / (p, Ap) */
        double p_dot_Ap = dot_product_neon_omp(p, Ap, nx, ny);

        if (fabs(p_dot_Ap) < CG_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        double alpha = r_dot_r / p_dot_Ap;

        /* x_{k+1} = x_k + alpha * p */
        axpy_neon_omp(alpha, p, x, nx, ny);

        /* r_{k+1} = r_k - alpha * Ap */
        axpy_neon_omp(-alpha, Ap, r, nx, ny);

        /* Compute new r_dot_r */
        double r_dot_r_new = dot_product_neon_omp(r, r, nx, ny);
        res_norm = sqrt(r_dot_r_new);

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                printf("  CG NEON+OMP Iter %d: residual = %.6e\n", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        /* Check for breakdown in r_dot_r before computing beta */
        if (fabs(r_dot_r) < CG_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        /* beta = (r_{k+1}, r_{k+1}) / (r_k, r_k) */
        double beta = r_dot_r_new / r_dot_r;

        /* p_{k+1} = r_{k+1} + beta * p_k */
        update_search_direction_neon_omp(r, p, beta, nx, ny);

        r_dot_r = r_dot_r_new;
    }

    /* Final convergence check (in case we converged between check intervals) */
    if (!converged && (res_norm < tolerance || res_norm < params->absolute_tolerance)) {
        converged = 1;
    }

    /* Apply final boundary conditions */
    bc_apply_scalar_simd_omp(x, nx, ny, BC_TYPE_NEUMANN);

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        /* When loop breaks early (converged), iter is the last iteration index, so +1.
         * When loop completes naturally, iter == max_iterations, report as-is. */
        stats->iterations = (iter < params->max_iterations) ? (iter + 1) : iter;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

static cfd_status_t cg_neon_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }
    return CFD_SUCCESS;
}

#endif /* CG_HAS_NEON_OMP */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_cg_neon_omp_solver(void) {
#if defined(CG_HAS_NEON_OMP)
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_CG_SIMD_OMP;
    solver->description = "Conjugate Gradient (NEON + OpenMP)";
    solver->method = POISSON_METHOD_CG;
    solver->backend = POISSON_BACKEND_SIMD_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = cg_neon_omp_init;
    solver->destroy = cg_neon_omp_destroy;
    solver->solve = cg_neon_omp_solve;
    solver->iterate = cg_neon_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
