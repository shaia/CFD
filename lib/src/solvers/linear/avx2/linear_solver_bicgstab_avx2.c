/**
 * @file linear_solver_bicgstab_avx2.c
 * @brief BiCGSTAB linear solver with AVX2 SIMD optimizations and OpenMP parallelization
 *
 * Implements the van der Vorst (1992) BiCGSTAB algorithm for solving Ax=b
 * using AVX2 vector instructions (4 doubles per vector) and OpenMP threading.
 *
 * ALGORITHM STRUCTURE (same as scalar, vectorized primitives):
 * - Initialization: r_0 = b - A*x_0, r_hat = r_0, rho=alpha=omega=1, v=p=0
 * - Per iteration:
 *   1. rho_new = (r_hat, r)                 [dot product]
 *   2. beta = (rho_new/rho) * (alpha/omega)
 *   3. p = r + beta*(p - omega*v)           [vector update]
 *   4. v = A*p                               [matvec #1]
 *   5. alpha = rho_new / (r_hat, v)         [dot product + breakdown check]
 *   6. s = r - alpha*v                      [vector update]
 *   7. Check early convergence on ||s||
 *   8. t = A*s                               [matvec #2]
 *   9. omega = (t,s) / (t,t)                [dot products + breakdown check]
 *   10. x = x + alpha*p + omega*s           [solution update]
 *   11. r = s - omega*t                     [residual update]
 *   12. Check convergence on ||r||
 *
 * SIMD OPTIMIZATION:
 * - dot_product: AVX2 horizontal sum with FMA
 * - axpy: Vectorized alpha*x + y
 * - apply_laplacian: Vectorized 5-point stencil
 * - compute_residual: Vectorized r = b - A*x
 */

#include "../linear_solver_internal.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <string.h>
#include <limits.h>

/* Platform detection for AVX2 + OpenMP */
#if defined(__AVX2__) && defined(_OPENMP)
#define BICGSTAB_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>
#endif

#if defined(BICGSTAB_HAS_AVX2)

//=============================================================================
// AVX2 CONTEXT AND CONSTANTS
//=============================================================================

typedef struct {
    double dx2;      /* dx^2 */
    double dy2;      /* dy^2 */

    /* Precomputed AVX2 vectors (4 doubles each) */
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d two_vec;

    /* Working vectors (6 total) */
    double* r;       /* Residual */
    double* r_hat;   /* Shadow residual */
    double* p;       /* Search direction */
    double* v;       /* A*p */
    double* s;       /* Intermediate residual */
    double* t;       /* A*s */

    int initialized;
} bicgstab_avx2_context_t;

/* Breakdown threshold */
static const double BICGSTAB_BREAKDOWN_TOL = 1.0e-30;

//=============================================================================
// HELPER: SIZE_T TO INT CONVERSION FOR OPENMP
//=============================================================================

static inline int size_to_int(size_t val) {
    if (val > (size_t)INT_MAX) {
        cfd_set_error("Grid size exceeds INT_MAX for OpenMP loop");
        return 0;
    }
    return (int)val;
}

//=============================================================================
// AVX2 SIMD PRIMITIVES
//=============================================================================

/**
 * Dot product: sum(a[i] * b[i]) over interior points
 * AVX2 vectorized with horizontal sum and OpenMP reduction
 */
static inline double dot_product_avx2(const double* a, const double* b,
                                       size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        double row_sum = 0.0;

        size_t i = 1;
        __m256d acc = _mm256_setzero_pd();

        /* SIMD loop (4 doubles per iteration) */
        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;
            __m256d va = _mm256_loadu_pd(&a[idx]);
            __m256d vb = _mm256_loadu_pd(&b[idx]);
            acc = _mm256_fmadd_pd(va, vb, acc);  /* acc += va * vb */
        }

        /* Horizontal sum of SIMD accumulator */
        __m128d low = _mm256_castpd256_pd128(acc);
        __m128d high = _mm256_extractf128_pd(acc, 1);
        __m128d sum128 = _mm_add_pd(low, high);
        sum128 = _mm_hadd_pd(sum128, sum128);
        row_sum += _mm_cvtsd_f64(sum128);

        /* Scalar remainder loop */
        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            row_sum += a[idx] * b[idx];
        }

        sum += row_sum;
    }

    return sum;
}

/**
 * AXPY: y = y + alpha*x
 * AVX2 vectorized with FMA
 */
static inline void axpy_avx2(double alpha, const double* x, double* y,
                              size_t nx, size_t ny) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;
            __m256d vx = _mm256_loadu_pd(&x[idx]);
            __m256d vy = _mm256_loadu_pd(&y[idx]);
            vy = _mm256_fmadd_pd(alpha_vec, vx, vy);  /* y += alpha * x */
            _mm256_storeu_pd(&y[idx], vy);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            y[idx] += alpha * x[idx];
        }
    }
}

/**
 * Apply Laplacian operator: Ap = ∇²p (5-point stencil)
 * AVX2 vectorized stencil computation
 */
static inline void apply_laplacian_avx2(const double* p, double* Ap,
                                         size_t nx, size_t ny,
                                         const bicgstab_avx2_context_t* ctx) {
    __m256d dx2_inv = ctx->dx2_inv_vec;
    __m256d dy2_inv = ctx->dy2_inv_vec;
    __m256d two_vec = ctx->two_vec;
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;

            /* Load 5-point stencil */
            __m256d p_c = _mm256_loadu_pd(&p[idx]);
            __m256d p_w = _mm256_loadu_pd(&p[idx - 1]);
            __m256d p_e = _mm256_loadu_pd(&p[idx + 1]);
            __m256d p_s = _mm256_loadu_pd(&p[idx - nx]);
            __m256d p_n = _mm256_loadu_pd(&p[idx + nx]);

            /* d²/dx² = (p_e - 2*p_c + p_w) / dx² */
            __m256d d2pdx2 = _mm256_sub_pd(p_e, _mm256_mul_pd(two_vec, p_c));
            d2pdx2 = _mm256_add_pd(d2pdx2, p_w);
            d2pdx2 = _mm256_mul_pd(d2pdx2, dx2_inv);

            /* d²/dy² = (p_n - 2*p_c + p_s) / dy² */
            __m256d d2pdy2 = _mm256_sub_pd(p_n, _mm256_mul_pd(two_vec, p_c));
            d2pdy2 = _mm256_add_pd(d2pdy2, p_s);
            d2pdy2 = _mm256_mul_pd(d2pdy2, dy2_inv);

            /* Laplacian = d²/dx² + d²/dy² */
            __m256d laplacian = _mm256_add_pd(d2pdx2, d2pdy2);
            _mm256_storeu_pd(&Ap[idx], laplacian);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double d2pdx2 = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / ctx->dx2;
            double d2pdy2 = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / ctx->dy2;
            Ap[idx] = d2pdx2 + d2pdy2;
        }
    }
}

/**
 * Compute residual: r = rhs - A*x
 * Uses apply_laplacian_avx2 internally
 */
static inline void compute_residual_avx2(const double* x, const double* rhs,
                                          double* r, size_t nx, size_t ny,
                                          const bicgstab_avx2_context_t* ctx) {
    /* Temporary storage for A*x */
    double* Ax = (double*)cfd_aligned_calloc(nx * ny, sizeof(double));
    if (!Ax) {
        cfd_set_error("Failed to allocate temporary Ax vector");
        return;
    }

    apply_laplacian_avx2(x, Ax, nx, ny, ctx);

    int ny_int = size_to_int(ny);
    if (ny_int == 0) {
        cfd_aligned_free(Ax);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;
            __m256d vrhs = _mm256_loadu_pd(&rhs[idx]);
            __m256d vAx = _mm256_loadu_pd(&Ax[idx]);
            __m256d vr = _mm256_sub_pd(vrhs, vAx);
            _mm256_storeu_pd(&r[idx], vr);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            r[idx] = rhs[idx] - Ax[idx];
        }
    }

    cfd_aligned_free(Ax);
}

/**
 * Vector copy: dst = src
 */
static inline void copy_vector_avx2(const double* src, double* dst,
                                     size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;
            __m256d v = _mm256_loadu_pd(&src[idx]);
            _mm256_storeu_pd(&dst[idx], v);
        }

        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            dst[idx] = src[idx];
        }
    }
}

/**
 * Zero interior points of vector
 */
static inline void zero_vector_avx2(double* vec, size_t nx, size_t ny) {
    __m256d zero = _mm256_setzero_pd();
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + 3 < nx - 1; i += 4) {
            size_t idx = j * nx + i;
            _mm256_storeu_pd(&vec[idx], zero);
        }

        for (; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            vec[idx] = 0.0;
        }
    }
}

//=============================================================================
// INITIALIZATION AND CLEANUP
//=============================================================================

static cfd_status_t bicgstab_avx2_init(void** context, size_t nx, size_t ny,
                                        double dx, double dy,
                                        const poisson_solver_params_t* params) {
    (void)params;  /* Not used for BiCGSTAB */

    bicgstab_avx2_context_t* ctx = (bicgstab_avx2_context_t*)cfd_aligned_calloc(
        1, sizeof(bicgstab_avx2_context_t));
    if (!ctx) {
        return CFD_ERROR_MEMORY;
    }

    size_t n = nx * ny;

    /* Allocate 6 working vectors with aligned memory for SIMD */
    ctx->r = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->r_hat = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->v = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->s = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->t = (double*)cfd_aligned_calloc(n, sizeof(double));

    if (!ctx->r || !ctx->r_hat || !ctx->p || !ctx->v || !ctx->s || !ctx->t) {
        cfd_aligned_free(ctx->r);
        cfd_aligned_free(ctx->r_hat);
        cfd_aligned_free(ctx->p);
        cfd_aligned_free(ctx->v);
        cfd_aligned_free(ctx->s);
        cfd_aligned_free(ctx->t);
        cfd_aligned_free(ctx);
        return CFD_ERROR_MEMORY;
    }

    /* Store grid spacing */
    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Precompute SIMD vectors for constants */
    ctx->dx2_inv_vec = _mm256_set1_pd(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = _mm256_set1_pd(1.0 / ctx->dy2);
    ctx->two_vec = _mm256_set1_pd(2.0);

    ctx->initialized = 1;
    *context = ctx;

    return CFD_SUCCESS;
}

static void bicgstab_avx2_destroy(void** context) {
    if (!context || !*context) return;

    bicgstab_avx2_context_t* ctx = (bicgstab_avx2_context_t*)(*context);

    cfd_aligned_free(ctx->r);
    cfd_aligned_free(ctx->r_hat);
    cfd_aligned_free(ctx->p);
    cfd_aligned_free(ctx->v);
    cfd_aligned_free(ctx->s);
    cfd_aligned_free(ctx->t);
    cfd_aligned_free(ctx);

    *context = NULL;
}

//=============================================================================
// BICGSTAB SOLVER
//=============================================================================

static int bicgstab_avx2_solve(void* context, double* x, const double* rhs,
                                size_t nx, size_t ny,
                                ns_solver_stats_t* stats) {
    bicgstab_avx2_context_t* ctx = (bicgstab_avx2_context_t*)context;
    if (!ctx || !ctx->initialized) {
        cfd_set_error("BiCGSTAB AVX2 solver not initialized");
        return -1;
    }

    /* Extract working vectors */
    double* r = ctx->r;
    double* r_hat = ctx->r_hat;
    double* p = ctx->p;
    double* v = ctx->v;
    double* s = ctx->s;
    double* t = ctx->t;

    /* INITIALIZATION */
    compute_residual_avx2(x, rhs, r, nx, ny, ctx);  /* r_0 = b - A*x_0 */
    copy_vector_avx2(r, r_hat, nx, ny);             /* r_hat = r_0 (shadow residual) */
    zero_vector_avx2(p, nx, ny);                    /* p_0 = 0 */
    zero_vector_avx2(v, nx, ny);                    /* v_0 = 0 */

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    /* Compute initial residual norm */
    double r_norm_init = sqrt(dot_product_avx2(r, r, nx, ny));
    if (r_norm_init == 0.0) {
        if (stats) stats->final_residual = 0.0;
        return 0;  /* Already converged */
    }

    const int max_iter = 5000;
    const double tol = 1.0e-6;

    /* BICGSTAB ITERATION LOOP */
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* 1. rho_new = (r_hat, r) */
        double rho_new = dot_product_avx2(r_hat, r, nx, ny);

        /* Check for breakdown */
        if (fabs(rho_new) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: rho = 0");
            if (stats) stats->final_residual = sqrt(dot_product_avx2(r, r, nx, ny)) / r_norm_init;
            return iter;
        }

        /* 2. beta = (rho_new / rho) * (alpha / omega) */
        double beta = (rho_new / rho) * (alpha / omega);

        /* 3. p = r + beta*(p - omega*v) */
        /* First: p = p - omega*v */
        axpy_avx2(-omega, v, p, nx, ny);
        /* Then: p = r + beta*p (reuse p as storage) */
        int ny_int = size_to_int(ny);
        #pragma omp parallel for schedule(static)
        for (int jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                p[idx] = r[idx] + beta * p[idx];
            }
        }

        /* 4. v = A*p */
        apply_laplacian_avx2(p, v, nx, ny, ctx);

        /* 5. alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = dot_product_avx2(r_hat, v, nx, ny);
        if (fabs(r_hat_dot_v) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: (r_hat, v) = 0");
            if (stats) stats->final_residual = sqrt(dot_product_avx2(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        alpha = rho_new / r_hat_dot_v;

        /* 6. s = r - alpha*v */
        copy_vector_avx2(r, s, nx, ny);
        axpy_avx2(-alpha, v, s, nx, ny);

        /* 7. Check early convergence on ||s|| */
        double s_norm = sqrt(dot_product_avx2(s, s, nx, ny));
        if (s_norm / r_norm_init < tol) {
            /* Early termination: x = x + alpha*p */
            axpy_avx2(alpha, p, x, nx, ny);
            if (stats) stats->final_residual = s_norm / r_norm_init;
            return iter + 1;
        }

        /* 8. t = A*s */
        apply_laplacian_avx2(s, t, nx, ny, ctx);

        /* 9. omega = (t, s) / (t, t) */
        double t_dot_s = dot_product_avx2(t, s, nx, ny);
        double t_dot_t = dot_product_avx2(t, t, nx, ny);
        if (fabs(t_dot_t) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: (t, t) = 0");
            if (stats) stats->final_residual = sqrt(dot_product_avx2(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        omega = t_dot_s / t_dot_t;

        /* 10. x = x + alpha*p + omega*s */
        axpy_avx2(alpha, p, x, nx, ny);
        axpy_avx2(omega, s, x, nx, ny);

        /* 11. r = s - omega*t */
        copy_vector_avx2(s, r, nx, ny);
        axpy_avx2(-omega, t, r, nx, ny);

        /* 12. Check convergence on ||r|| */
        double r_norm = sqrt(dot_product_avx2(r, r, nx, ny));
        if (r_norm / r_norm_init < tol) {
            if (stats) stats->final_residual = r_norm / r_norm_init;
            return iter + 1;
        }

        /* Update rho for next iteration */
        rho = rho_new;
    }

    /* Max iterations reached */
    double r_norm_final = sqrt(dot_product_avx2(r, r, nx, ny));
    if (stats) stats->final_residual = r_norm_final / r_norm_init;

    return max_iter;
}

static int bicgstab_avx2_iterate(void* context, double* x, const double* rhs,
                                  size_t nx, size_t ny) {
    /* Single iteration not commonly used for BiCGSTAB */
    /* Delegate to full solve with max_iter=1 */
    ns_solver_stats_t stats = ns_solver_stats_default();
    return bicgstab_avx2_solve(context, x, rhs, nx, ny, &stats);
}

//=============================================================================
// FACTORY FUNCTION
//=============================================================================

poisson_solver_t* create_bicgstab_avx2_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) return NULL;

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_SIMD;
    solver->description = "BiCGSTAB (AVX2 + OpenMP)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();

    solver->init = bicgstab_avx2_init;
    solver->destroy = bicgstab_avx2_destroy;
    solver->solve = bicgstab_avx2_solve;
    solver->iterate = bicgstab_avx2_iterate;
    solver->apply_bc = NULL;

    return solver;
}

#else  /* !BICGSTAB_HAS_AVX2 */

/* Stub for platforms without AVX2 */
poisson_solver_t* create_bicgstab_avx2_solver(void) {
    return NULL;
}

#endif  /* BICGSTAB_HAS_AVX2 */
