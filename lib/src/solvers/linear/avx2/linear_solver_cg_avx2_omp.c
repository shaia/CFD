/**
 * @file linear_solver_cg_avx2_omp.c
 * @brief Conjugate Gradient solver - AVX2 + OpenMP implementation
 *
 * Conjugate Gradient method optimized with:
 * - AVX2 intrinsics for SIMD vectorization (4 doubles per operation)
 * - OpenMP for thread-level parallelism
 * - Runtime detection to ensure AVX2 is available
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

/* AVX2 + OpenMP detection */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define CG_HAS_AVX2_OMP 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(CG_HAS_AVX2_OMP)

/* ============================================================================
 * CG AVX2+OMP CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */

    /* Precomputed SIMD vectors */
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d two_vec;

    /* CG working vectors (allocated during init) */
    double* r;         /* Residual vector */
    double* p;         /* Search direction */
    double* Ap;        /* A * p (Laplacian applied to p) */

    int initialized;
} cg_avx2_omp_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * HELPER FUNCTIONS (AVX2 + OpenMP)
 * ============================================================================ */

/**
 * Compute dot product using AVX2 with OpenMP reduction
 */
static double dot_product_avx2_omp(const double* a, const double* b,
                                    size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        double row_sum = 0.0;
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop: process 4 doubles at a time */
        __m256d acc = _mm256_setzero_pd();
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = row_start + i;
            __m256d va = _mm256_loadu_pd(&a[idx]);
            __m256d vb = _mm256_loadu_pd(&b[idx]);
            acc = _mm256_fmadd_pd(va, vb, acc);
        }

        /* Horizontal sum of acc */
        __m128d low = _mm256_castpd256_pd128(acc);
        __m128d high = _mm256_extractf128_pd(acc, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        sum128 = _mm_hadd_pd(sum128, sum128);
        row_sum = _mm_cvtsd_f64(sum128);

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
 * Compute y = y + alpha * x using AVX2 with OpenMP
 */
static void axpy_avx2_omp(double alpha, const double* x, double* y,
                          size_t nx, size_t ny) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop */
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = row_start + i;
            __m256d vx = _mm256_loadu_pd(&x[idx]);
            __m256d vy = _mm256_loadu_pd(&y[idx]);
            vy = _mm256_fmadd_pd(alpha_vec, vx, vy);
            _mm256_storeu_pd(&y[idx], vy);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = row_start + i;
            y[idx] += alpha * x[idx];
        }
    }
}

/**
 * Apply negative Laplacian using AVX2 with OpenMP
 */
static void apply_laplacian_avx2_omp(const double* p, double* Ap,
                                      size_t nx, size_t ny,
                                      __m256d dx2_inv_vec, __m256d dy2_inv_vec,
                                      __m256d two_vec) {
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* SIMD loop */
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = (size_t)j * nx + i;

            /* Load neighbors */
            __m256d p_center = _mm256_loadu_pd(&p[idx]);
            __m256d p_xp = _mm256_loadu_pd(&p[idx + 1]);
            __m256d p_xm = _mm256_loadu_pd(&p[idx - 1]);
            __m256d p_yp = _mm256_loadu_pd(&p[idx + nx]);
            __m256d p_ym = _mm256_loadu_pd(&p[idx - nx]);

            /* Compute second derivatives */
            /* d2p_dx2 = (p[i+1] - 2*p[i] + p[i-1]) / dx2 */
            __m256d sum_x = _mm256_add_pd(p_xp, p_xm);
            __m256d two_center = _mm256_mul_pd(two_vec, p_center);
            __m256d d2p_dx2 = _mm256_sub_pd(sum_x, two_center);
            d2p_dx2 = _mm256_mul_pd(d2p_dx2, dx2_inv_vec);

            /* d2p_dy2 = (p[j+1] - 2*p[j] + p[j-1]) / dy2 */
            __m256d sum_y = _mm256_add_pd(p_yp, p_ym);
            __m256d d2p_dy2 = _mm256_sub_pd(sum_y, two_center);
            d2p_dy2 = _mm256_mul_pd(d2p_dy2, dy2_inv_vec);

            /* -Laplacian = -(d2p_dx2 + d2p_dy2) */
            __m256d laplacian = _mm256_add_pd(d2p_dx2, d2p_dy2);
            __m256d neg_laplacian = _mm256_sub_pd(_mm256_setzero_pd(), laplacian);

            _mm256_storeu_pd(&Ap[idx], neg_laplacian);
        }

        /* Scalar remainder */
        double dx2_inv_scalar = _mm256_cvtsd_f64(dx2_inv_vec);
        double dy2_inv_scalar = _mm256_cvtsd_f64(dy2_inv_vec);
        for (; i < nx - 1; i++) {
            size_t idx = (size_t)j * nx + i;
            double laplacian = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv_scalar
                             + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv_scalar;
            Ap[idx] = -laplacian;
        }
    }
}

/**
 * Compute initial residual using AVX2 with OpenMP
 */
static void compute_residual_avx2_omp(const double* x, const double* rhs, double* r,
                                       size_t nx, size_t ny,
                                       __m256d dx2_inv_vec, __m256d dy2_inv_vec,
                                       __m256d two_vec) {
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* SIMD loop */
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = (size_t)j * nx + i;

            /* Load neighbors */
            __m256d x_center = _mm256_loadu_pd(&x[idx]);
            __m256d x_xp = _mm256_loadu_pd(&x[idx + 1]);
            __m256d x_xm = _mm256_loadu_pd(&x[idx - 1]);
            __m256d x_yp = _mm256_loadu_pd(&x[idx + nx]);
            __m256d x_ym = _mm256_loadu_pd(&x[idx - nx]);
            __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

            /* Compute Laplacian */
            __m256d sum_x = _mm256_add_pd(x_xp, x_xm);
            __m256d two_center = _mm256_mul_pd(two_vec, x_center);
            __m256d d2x_dx2 = _mm256_sub_pd(sum_x, two_center);
            d2x_dx2 = _mm256_mul_pd(d2x_dx2, dx2_inv_vec);

            __m256d sum_y = _mm256_add_pd(x_yp, x_ym);
            __m256d d2x_dy2 = _mm256_sub_pd(sum_y, two_center);
            d2x_dy2 = _mm256_mul_pd(d2x_dy2, dy2_inv_vec);

            __m256d laplacian = _mm256_add_pd(d2x_dx2, d2x_dy2);

            /* r = -rhs + laplacian */
            __m256d neg_rhs = _mm256_sub_pd(_mm256_setzero_pd(), rhs_vec);
            __m256d r_vec = _mm256_add_pd(neg_rhs, laplacian);

            _mm256_storeu_pd(&r[idx], r_vec);
        }

        /* Scalar remainder */
        double dx2_inv_scalar = _mm256_cvtsd_f64(dx2_inv_vec);
        double dy2_inv_scalar = _mm256_cvtsd_f64(dy2_inv_vec);
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
 * Update p = r + beta * p using AVX2 with OpenMP
 */
static void update_search_direction_avx2_omp(const double* r, double* p,
                                              double beta, size_t nx, size_t ny) {
    __m256d beta_vec = _mm256_set1_pd(beta);
    int ny_int = size_to_int(ny);
    int j;

    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;
        size_t row_start = (size_t)j * nx;

        /* SIMD loop */
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = row_start + i;
            __m256d vr = _mm256_loadu_pd(&r[idx]);
            __m256d vp = _mm256_loadu_pd(&p[idx]);
            /* p = r + beta * p */
            vp = _mm256_fmadd_pd(beta_vec, vp, vr);
            _mm256_storeu_pd(&p[idx], vp);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = row_start + i;
            p[idx] = r[idx] + beta * p[idx];
        }
    }
}

/* ============================================================================
 * CG AVX2+OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t cg_avx2_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)params;

    /* Use aligned allocation for SIMD context */
    cg_avx2_omp_context_t* ctx = (cg_avx2_omp_context_t*)cfd_aligned_calloc(
        1, sizeof(cg_avx2_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Precompute SIMD vectors */
    ctx->dx2_inv_vec = _mm256_set1_pd(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = _mm256_set1_pd(1.0 / ctx->dy2);
    ctx->two_vec = _mm256_set1_pd(2.0);

    /* Allocate working vectors (aligned for SIMD) */
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

static void cg_avx2_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cg_avx2_omp_context_t* ctx = (cg_avx2_omp_context_t*)solver->context;
        cfd_aligned_free(ctx->r);
        cfd_aligned_free(ctx->p);
        cfd_aligned_free(ctx->Ap);
        cfd_aligned_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t cg_avx2_omp_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;

    cg_avx2_omp_context_t* ctx = (cg_avx2_omp_context_t*)solver->context;
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
    compute_residual_avx2_omp(x, rhs, r, nx, ny,
                              ctx->dx2_inv_vec, ctx->dy2_inv_vec, ctx->two_vec);

    /* Initial search direction: p_0 = r_0 */
    copy_vector_omp(r, p, nx, ny);

    /* Compute initial r_dot_r */
    double r_dot_r = dot_product_avx2_omp(r, r, nx, ny);
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
        apply_laplacian_avx2_omp(p, Ap, nx, ny,
                                  ctx->dx2_inv_vec, ctx->dy2_inv_vec, ctx->two_vec);

        /* alpha = (r, r) / (p, Ap) */
        double p_dot_Ap = dot_product_avx2_omp(p, Ap, nx, ny);

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
        axpy_avx2_omp(alpha, p, x, nx, ny);

        /* r_{k+1} = r_k - alpha * Ap */
        axpy_avx2_omp(-alpha, Ap, r, nx, ny);

        /* Compute new r_dot_r */
        double r_dot_r_new = dot_product_avx2_omp(r, r, nx, ny);
        res_norm = sqrt(r_dot_r_new);

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                printf("  CG AVX2+OMP Iter %d: residual = %.6e\n", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        /* beta = (r_{k+1}, r_{k+1}) / (r_k, r_k) */
        double beta = r_dot_r_new / r_dot_r;

        /* p_{k+1} = r_{k+1} + beta * p_k */
        update_search_direction_avx2_omp(r, p, beta, nx, ny);

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
        stats->iterations = iter + 1;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

static cfd_status_t cg_avx2_omp_iterate(
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

#endif /* CG_HAS_AVX2_OMP */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_cg_avx2_omp_solver(void) {
#if defined(CG_HAS_AVX2_OMP)
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_CG_SIMD_OMP;
    solver->description = "Conjugate Gradient (AVX2 + OpenMP)";
    solver->method = POISSON_METHOD_CG;
    solver->backend = POISSON_BACKEND_SIMD_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = cg_avx2_omp_init;
    solver->destroy = cg_avx2_omp_destroy;
    solver->solve = cg_avx2_omp_solve;
    solver->iterate = cg_avx2_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
