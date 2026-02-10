/**
 * @file linear_solver_bicgstab_neon.c
 * @brief BiCGSTAB linear solver with ARM NEON SIMD optimizations and OpenMP parallelization
 *
 * Implements the van der Vorst (1992) BiCGSTAB algorithm for solving Ax=b
 * using ARM NEON vector instructions (2 doubles per vector) and OpenMP threading.
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
 * - dot_product: NEON horizontal sum with FMA
 * - axpy: Vectorized alpha*x + y
 * - apply_laplacian: Vectorized 5-point stencil
 * - compute_residual: Vectorized r = b - A*x
 */

#include "../linear_solver_internal.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <string.h>
#include <limits.h>

/* Platform detection for NEON + OpenMP */
#if defined(__aarch64__) && defined(_OPENMP)
#define BICGSTAB_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#endif

#if defined(BICGSTAB_HAS_NEON)

//=============================================================================
// NEON CONTEXT AND CONSTANTS
//=============================================================================

typedef struct {
    double dx2;      /* dx^2 */
    double dy2;      /* dy^2 */

    /* Precomputed NEON vectors (2 doubles each) */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t two_vec;

    /* Working vectors (6 total) */
    double* r;       /* Residual */
    double* r_hat;   /* Shadow residual */
    double* p;       /* Search direction */
    double* v;       /* A*p */
    double* s;       /* Intermediate residual */
    double* t;       /* A*s */

    int initialized;
} bicgstab_neon_context_t;

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
// NEON SIMD PRIMITIVES
//=============================================================================

/**
 * Dot product: sum(a[i] * b[i]) over interior points
 * NEON vectorized with horizontal sum and OpenMP reduction
 */
static inline double dot_product_neon(const double* a, const double* b,
                                       size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        double row_sum = 0.0;

        size_t i = 1;
        float64x2_t acc = vdupq_n_f64(0.0);

        /* SIMD loop (2 doubles per iteration) */
        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;
            float64x2_t va = vld1q_f64(&a[idx]);
            float64x2_t vb = vld1q_f64(&b[idx]);
            acc = vfmaq_f64(acc, va, vb);  /* acc += va * vb */
        }

        /* Horizontal sum of NEON accumulator */
        row_sum += vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

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
 * NEON vectorized with FMA
 */
static inline void axpy_neon(double alpha, const double* x, double* y,
                              size_t nx, size_t ny) {
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;
            float64x2_t vx = vld1q_f64(&x[idx]);
            float64x2_t vy = vld1q_f64(&y[idx]);
            vy = vfmaq_f64(vy, alpha_vec, vx);  /* y += alpha * x */
            vst1q_f64(&y[idx], vy);
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
 * NEON vectorized stencil computation
 */
static inline void apply_laplacian_neon(const double* p, double* Ap,
                                         size_t nx, size_t ny,
                                         const bicgstab_neon_context_t* ctx) {
    float64x2_t dx2_inv = ctx->dx2_inv_vec;
    float64x2_t dy2_inv = ctx->dy2_inv_vec;
    float64x2_t two_vec = ctx->two_vec;
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;

            /* Load 5-point stencil */
            float64x2_t p_c = vld1q_f64(&p[idx]);
            float64x2_t p_w = vld1q_f64(&p[idx - 1]);
            float64x2_t p_e = vld1q_f64(&p[idx + 1]);
            float64x2_t p_s = vld1q_f64(&p[idx - nx]);
            float64x2_t p_n = vld1q_f64(&p[idx + nx]);

            /* d²/dx² = (p_e - 2*p_c + p_w) / dx² */
            float64x2_t d2pdx2 = vsubq_f64(p_e, vmulq_f64(two_vec, p_c));
            d2pdx2 = vaddq_f64(d2pdx2, p_w);
            d2pdx2 = vmulq_f64(d2pdx2, dx2_inv);

            /* d²/dy² = (p_n - 2*p_c + p_s) / dy² */
            float64x2_t d2pdy2 = vsubq_f64(p_n, vmulq_f64(two_vec, p_c));
            d2pdy2 = vaddq_f64(d2pdy2, p_s);
            d2pdy2 = vmulq_f64(d2pdy2, dy2_inv);

            /* Laplacian = d²/dx² + d²/dy² */
            float64x2_t laplacian = vaddq_f64(d2pdx2, d2pdy2);
            vst1q_f64(&Ap[idx], laplacian);
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
 * Uses apply_laplacian_neon internally
 */
static inline void compute_residual_neon(const double* x, const double* rhs,
                                          double* r, size_t nx, size_t ny,
                                          const bicgstab_neon_context_t* ctx) {
    /* Temporary storage for A*x */
    double* Ax = (double*)cfd_aligned_calloc(nx * ny, sizeof(double));
    if (!Ax) {
        cfd_set_error("Failed to allocate temporary Ax vector");
        return;
    }

    apply_laplacian_neon(x, Ax, nx, ny, ctx);

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
        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;
            float64x2_t vrhs = vld1q_f64(&rhs[idx]);
            float64x2_t vAx = vld1q_f64(&Ax[idx]);
            float64x2_t vr = vsubq_f64(vrhs, vAx);
            vst1q_f64(&r[idx], vr);
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
static inline void copy_vector_neon(const double* src, double* dst,
                                     size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;
            float64x2_t v = vld1q_f64(&src[idx]);
            vst1q_f64(&dst[idx], v);
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
static inline void zero_vector_neon(double* vec, size_t nx, size_t ny) {
    float64x2_t zero = vdupq_n_f64(0.0);
    int ny_int = size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + 1 < nx - 1; i += 2) {
            size_t idx = j * nx + i;
            vst1q_f64(&vec[idx], zero);
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

static cfd_status_t bicgstab_neon_init(void** context, size_t nx, size_t ny,
                                        double dx, double dy,
                                        const poisson_solver_params_t* params) {
    (void)params;  /* Not used for BiCGSTAB */

    bicgstab_neon_context_t* ctx = (bicgstab_neon_context_t*)cfd_aligned_calloc(
        1, sizeof(bicgstab_neon_context_t));
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

    /* Precompute NEON vectors for constants */
    ctx->dx2_inv_vec = vdupq_n_f64(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(1.0 / ctx->dy2);
    ctx->two_vec = vdupq_n_f64(2.0);

    ctx->initialized = 1;
    *context = ctx;

    return CFD_SUCCESS;
}

static void bicgstab_neon_destroy(void** context) {
    if (!context || !*context) return;

    bicgstab_neon_context_t* ctx = (bicgstab_neon_context_t*)(*context);

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

static int bicgstab_neon_solve(void* context, double* x, const double* rhs,
                                size_t nx, size_t ny,
                                ns_solver_stats_t* stats) {
    bicgstab_neon_context_t* ctx = (bicgstab_neon_context_t*)context;
    if (!ctx || !ctx->initialized) {
        cfd_set_error("BiCGSTAB NEON solver not initialized");
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
    compute_residual_neon(x, rhs, r, nx, ny, ctx);  /* r_0 = b - A*x_0 */
    copy_vector_neon(r, r_hat, nx, ny);             /* r_hat = r_0 (shadow residual) */
    zero_vector_neon(p, nx, ny);                    /* p_0 = 0 */
    zero_vector_neon(v, nx, ny);                    /* v_0 = 0 */

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    /* Compute initial residual norm */
    double r_norm_init = sqrt(dot_product_neon(r, r, nx, ny));
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
        double rho_new = dot_product_neon(r_hat, r, nx, ny);

        /* Check for breakdown */
        if (fabs(rho_new) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: rho = 0");
            if (stats) stats->final_residual = sqrt(dot_product_neon(r, r, nx, ny)) / r_norm_init;
            return iter;
        }

        /* 2. beta = (rho_new / rho) * (alpha / omega) */
        double beta = (rho_new / rho) * (alpha / omega);

        /* 3. p = r + beta*(p - omega*v) */
        /* First: p = p - omega*v */
        axpy_neon(-omega, v, p, nx, ny);
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
        apply_laplacian_neon(p, v, nx, ny, ctx);

        /* 5. alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = dot_product_neon(r_hat, v, nx, ny);
        if (fabs(r_hat_dot_v) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: (r_hat, v) = 0");
            if (stats) stats->final_residual = sqrt(dot_product_neon(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        alpha = rho_new / r_hat_dot_v;

        /* 6. s = r - alpha*v */
        copy_vector_neon(r, s, nx, ny);
        axpy_neon(-alpha, v, s, nx, ny);

        /* 7. Check early convergence on ||s|| */
        double s_norm = sqrt(dot_product_neon(s, s, nx, ny));
        if (s_norm / r_norm_init < tol) {
            /* Early termination: x = x + alpha*p */
            axpy_neon(alpha, p, x, nx, ny);
            if (stats) stats->final_residual = s_norm / r_norm_init;
            return iter + 1;
        }

        /* 8. t = A*s */
        apply_laplacian_neon(s, t, nx, ny, ctx);

        /* 9. omega = (t, s) / (t, t) */
        double t_dot_s = dot_product_neon(t, s, nx, ny);
        double t_dot_t = dot_product_neon(t, t, nx, ny);
        if (fabs(t_dot_t) < BICGSTAB_BREAKDOWN_TOL) {
            cfd_set_error("BiCGSTAB breakdown: (t, t) = 0");
            if (stats) stats->final_residual = sqrt(dot_product_neon(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        omega = t_dot_s / t_dot_t;

        /* 10. x = x + alpha*p + omega*s */
        axpy_neon(alpha, p, x, nx, ny);
        axpy_neon(omega, s, x, nx, ny);

        /* 11. r = s - omega*t */
        copy_vector_neon(s, r, nx, ny);
        axpy_neon(-omega, t, r, nx, ny);

        /* 12. Check convergence on ||r|| */
        double r_norm = sqrt(dot_product_neon(r, r, nx, ny));
        if (r_norm / r_norm_init < tol) {
            if (stats) stats->final_residual = r_norm / r_norm_init;
            return iter + 1;
        }

        /* Update rho for next iteration */
        rho = rho_new;
    }

    /* Max iterations reached */
    double r_norm_final = sqrt(dot_product_neon(r, r, nx, ny));
    if (stats) stats->final_residual = r_norm_final / r_norm_init;

    return max_iter;
}

static int bicgstab_neon_iterate(void* context, double* x, const double* rhs,
                                  size_t nx, size_t ny) {
    /* Single iteration not commonly used for BiCGSTAB */
    /* Delegate to full solve with max_iter=1 */
    ns_solver_stats_t stats = ns_solver_stats_default();
    return bicgstab_neon_solve(context, x, rhs, nx, ny, &stats);
}

//=============================================================================
// FACTORY FUNCTION
//=============================================================================

poisson_solver_t* create_bicgstab_neon_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) return NULL;

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_SIMD;
    solver->description = "BiCGSTAB (NEON + OpenMP)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();

    solver->init = bicgstab_neon_init;
    solver->destroy = bicgstab_neon_destroy;
    solver->solve = bicgstab_neon_solve;
    solver->iterate = bicgstab_neon_iterate;
    solver->apply_bc = NULL;

    return solver;
}

#else  /* !BICGSTAB_HAS_NEON */

/* Stub for platforms without NEON */
poisson_solver_t* create_bicgstab_neon_solver(void) {
    return NULL;
}

#endif  /* BICGSTAB_HAS_NEON */
