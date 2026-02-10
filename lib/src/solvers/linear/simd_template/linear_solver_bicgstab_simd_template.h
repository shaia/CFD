/**
 * @file linear_solver_bicgstab_simd_template.h
 * @brief BiCGSTAB SIMD implementation template (AVX2/NEON parameterized)
 *
 * This parameterized header implements BiCGSTAB using architecture-agnostic
 * SIMD macros. It is included twice (once for AVX2, once for NEON) with
 * different macro definitions.
 *
 * REQUIRED MACROS (must be #defined before including):
 * - SIMD_SUFFIX: avx2 or neon
 * - SIMD_VEC: vector type (__m256d or float64x2_t)
 * - SIMD_WIDTH: vector width (4 or 2)
 * - SIMD_LOAD(ptr): load vector from memory
 * - SIMD_STORE(ptr, vec): store vector to memory
 * - SIMD_SET1(val): broadcast scalar to vector
 * - SIMD_SETZERO(): create zero vector
 * - SIMD_ADD(a, b): vector addition
 * - SIMD_SUB(a, b): vector subtraction
 * - SIMD_MUL(a, b): vector multiplication
 * - SIMD_FMA(a, b, c): fused multiply-add (a*b + c)
 * - SIMD_HSUM(vec): horizontal sum of vector elements
 *
 * PATTERN: Follows boundary conditions SIMD template pattern (Phase 1.1.1)
 */

#ifndef SIMD_SUFFIX
#error "SIMD_SUFFIX must be defined before including linear_solver_bicgstab_simd_template.h"
#endif

#ifndef SIMD_VEC
#error "SIMD_VEC must be defined before including linear_solver_bicgstab_simd_template.h"
#endif

#ifndef SIMD_WIDTH
#error "SIMD_WIDTH must be defined before including linear_solver_bicgstab_simd_template.h"
#endif

//=============================================================================
// TOKEN PASTING MACROS
//=============================================================================

#define CONCAT_IMPL(a, b) a##_##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define SIMD_FUNC(name) CONCAT(name, SIMD_SUFFIX)

//=============================================================================
// CONTEXT STRUCTURE
//=============================================================================

#define bicgstab_simd_context_t SIMD_FUNC(bicgstab_context_t)

typedef struct {
    double dx2;      /* dx^2 */
    double dy2;      /* dy^2 */

    /* Precomputed SIMD vectors */
    SIMD_VEC dx2_inv_vec;
    SIMD_VEC dy2_inv_vec;
    SIMD_VEC two_vec;

    /* Working vectors (6 total) */
    double* r;       /* Residual */
    double* r_hat;   /* Shadow residual */
    double* p;       /* Search direction */
    double* v;       /* A*p */
    double* s;       /* Intermediate residual */
    double* t;       /* A*s */

    int initialized;
} bicgstab_simd_context_t;

//=============================================================================
// SIMD PRIMITIVES
//=============================================================================

/**
 * Dot product: sum(a[i] * b[i]) over interior points
 * SIMD vectorized with horizontal sum and OpenMP reduction
 */
static inline double SIMD_FUNC(dot_product)(const double* a, const double* b,
                                              size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return 0.0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        double row_sum = 0.0;

        size_t i = 1;
        SIMD_VEC acc = SIMD_SETZERO();

        /* SIMD loop */
        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;
            SIMD_VEC va = SIMD_LOAD(&a[idx]);
            SIMD_VEC vb = SIMD_LOAD(&b[idx]);
            acc = SIMD_FMA(va, vb, acc);  /* acc += va * vb */
        }

        /* Horizontal sum of SIMD accumulator */
        row_sum += SIMD_HSUM(acc);

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
 * SIMD vectorized with FMA
 */
static inline void SIMD_FUNC(axpy)(double alpha, const double* x, double* y,
                                    size_t nx, size_t ny) {
    SIMD_VEC alpha_vec = SIMD_SET1(alpha);
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;
            SIMD_VEC vx = SIMD_LOAD(&x[idx]);
            SIMD_VEC vy = SIMD_LOAD(&y[idx]);
            vy = SIMD_FMA(alpha_vec, vx, vy);  /* y += alpha * x */
            SIMD_STORE(&y[idx], vy);
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
 * SIMD vectorized stencil computation
 */
static inline void SIMD_FUNC(apply_laplacian)(const double* p, double* Ap,
                                                size_t nx, size_t ny,
                                                const bicgstab_simd_context_t* ctx) {
    SIMD_VEC dx2_inv = ctx->dx2_inv_vec;
    SIMD_VEC dy2_inv = ctx->dy2_inv_vec;
    SIMD_VEC two_vec = ctx->two_vec;
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;

            /* Load 5-point stencil */
            SIMD_VEC p_c = SIMD_LOAD(&p[idx]);
            SIMD_VEC p_w = SIMD_LOAD(&p[idx - 1]);
            SIMD_VEC p_e = SIMD_LOAD(&p[idx + 1]);
            SIMD_VEC p_s = SIMD_LOAD(&p[idx - nx]);
            SIMD_VEC p_n = SIMD_LOAD(&p[idx + nx]);

            /* d²/dx² = (p_e - 2*p_c + p_w) / dx² */
            SIMD_VEC d2pdx2 = SIMD_SUB(p_e, SIMD_MUL(two_vec, p_c));
            d2pdx2 = SIMD_ADD(d2pdx2, p_w);
            d2pdx2 = SIMD_MUL(d2pdx2, dx2_inv);

            /* d²/dy² = (p_n - 2*p_c + p_s) / dy² */
            SIMD_VEC d2pdy2 = SIMD_SUB(p_n, SIMD_MUL(two_vec, p_c));
            d2pdy2 = SIMD_ADD(d2pdy2, p_s);
            d2pdy2 = SIMD_MUL(d2pdy2, dy2_inv);

            /* Laplacian = d²/dx² + d²/dy² */
            SIMD_VEC laplacian = SIMD_ADD(d2pdx2, d2pdy2);
            SIMD_STORE(&Ap[idx], laplacian);
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
 * Uses apply_laplacian internally
 */
static inline void SIMD_FUNC(compute_residual)(const double* x, const double* rhs,
                                                 double* r, size_t nx, size_t ny,
                                                 const bicgstab_simd_context_t* ctx) {
    /* Temporary storage for A*x */
    double* Ax = (double*)cfd_aligned_calloc(nx * ny, sizeof(double));
    if (!Ax) {
        cfd_set_error(CFD_ERROR_NOMEM, "Failed to allocate temporary Ax vector");
        return;
    }

    SIMD_FUNC(apply_laplacian)(x, Ax, nx, ny, ctx);

    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) {
        cfd_aligned_free(Ax);
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        /* SIMD loop */
        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;
            SIMD_VEC vrhs = SIMD_LOAD(&rhs[idx]);
            SIMD_VEC vAx = SIMD_LOAD(&Ax[idx]);
            SIMD_VEC vr = SIMD_SUB(vrhs, vAx);
            SIMD_STORE(&r[idx], vr);
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
static inline void SIMD_FUNC(copy_vector)(const double* src, double* dst,
                                           size_t nx, size_t ny) {
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;
            SIMD_VEC v = SIMD_LOAD(&src[idx]);
            SIMD_STORE(&dst[idx], v);
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
static inline void SIMD_FUNC(zero_vector)(double* vec, size_t nx, size_t ny) {
    SIMD_VEC zero = SIMD_SETZERO();
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    #pragma omp parallel for schedule(static)
    for (int jj = 1; jj < ny_int - 1; jj++) {
        size_t j = (size_t)jj;
        size_t i = 1;

        for (; i + SIMD_WIDTH - 1 < nx - 1; i += SIMD_WIDTH) {
            size_t idx = j * nx + i;
            SIMD_STORE(&vec[idx], zero);
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

static cfd_status_t SIMD_FUNC(bicgstab_init)(void** context, size_t nx, size_t ny,
                                               double dx, double dy,
                                               const poisson_solver_params_t* params) {
    (void)params;  /* Not used for BiCGSTAB */

    bicgstab_simd_context_t* ctx = (bicgstab_simd_context_t*)cfd_aligned_calloc(
        1, sizeof(bicgstab_simd_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
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
        return CFD_ERROR_NOMEM;
    }

    /* Store grid spacing */
    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Precompute SIMD vectors for constants */
    ctx->dx2_inv_vec = SIMD_SET1(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = SIMD_SET1(1.0 / ctx->dy2);
    ctx->two_vec = SIMD_SET1(2.0);

    ctx->initialized = 1;
    *context = ctx;

    return CFD_SUCCESS;
}

static void SIMD_FUNC(bicgstab_destroy)(void** context) {
    if (!context || !*context) return;

    bicgstab_simd_context_t* ctx = (bicgstab_simd_context_t*)(*context);

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

static int SIMD_FUNC(bicgstab_solve)(void* context, double* x, const double* rhs,
                                      size_t nx, size_t ny,
                                      poisson_solver_stats_t* stats) {
    bicgstab_simd_context_t* ctx = (bicgstab_simd_context_t*)context;
    if (!ctx || !ctx->initialized) {
        cfd_set_error(CFD_ERROR, "BiCGSTAB SIMD solver not initialized");
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
    SIMD_FUNC(compute_residual)(x, rhs, r, nx, ny, ctx);  /* r_0 = b - A*x_0 */
    SIMD_FUNC(copy_vector)(r, r_hat, nx, ny);             /* r_hat = r_0 (shadow residual) */
    SIMD_FUNC(zero_vector)(p, nx, ny);                    /* p_0 = 0 */
    SIMD_FUNC(zero_vector)(v, nx, ny);                    /* v_0 = 0 */

    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    /* Compute initial residual norm */
    double r_norm_init = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny));
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
        double rho_new = SIMD_FUNC(dot_product)(r_hat, r, nx, ny);

        /* Check for breakdown */
        if (fabs(rho_new) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            cfd_set_error(CFD_ERROR_DIVERGED, "BiCGSTAB breakdown: rho = 0");
            if (stats) stats->final_residual = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny)) / r_norm_init;
            return iter;
        }

        /* 2. beta = (rho_new / rho) * (alpha / omega) */
        double beta = (rho_new / rho) * (alpha / omega);

        /* 3. p = r + beta*(p - omega*v) */
        /* First: p = p - omega*v */
        SIMD_FUNC(axpy)(-omega, v, p, nx, ny);
        /* Then: p = r + beta*p (reuse p as storage) */
        int ny_int = bicgstab_size_to_int(ny);
        #pragma omp parallel for schedule(static)
        for (int jj = 1; jj < ny_int - 1; jj++) {
            size_t j = (size_t)jj;
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                p[idx] = r[idx] + beta * p[idx];
            }
        }

        /* 4. v = A*p */
        SIMD_FUNC(apply_laplacian)(p, v, nx, ny, ctx);

        /* 5. alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = SIMD_FUNC(dot_product)(r_hat, v, nx, ny);
        if (fabs(r_hat_dot_v) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            cfd_set_error(CFD_ERROR_DIVERGED, "BiCGSTAB breakdown: (r_hat, v) = 0");
            if (stats) stats->final_residual = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        alpha = rho_new / r_hat_dot_v;

        /* 6. s = r - alpha*v */
        SIMD_FUNC(copy_vector)(r, s, nx, ny);
        SIMD_FUNC(axpy)(-alpha, v, s, nx, ny);

        /* 7. Check early convergence on ||s|| */
        double s_norm = sqrt(SIMD_FUNC(dot_product)(s, s, nx, ny));
        if (s_norm / r_norm_init < tol) {
            /* Early termination: x = x + alpha*p */
            SIMD_FUNC(axpy)(alpha, p, x, nx, ny);
            if (stats) stats->final_residual = s_norm / r_norm_init;
            return iter + 1;
        }

        /* 8. t = A*s */
        SIMD_FUNC(apply_laplacian)(s, t, nx, ny, ctx);

        /* 9. omega = (t, s) / (t, t) */
        double t_dot_s = SIMD_FUNC(dot_product)(t, s, nx, ny);
        double t_dot_t = SIMD_FUNC(dot_product)(t, t, nx, ny);
        if (fabs(t_dot_t) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            cfd_set_error(CFD_ERROR_DIVERGED, "BiCGSTAB breakdown: (t, t) = 0");
            if (stats) stats->final_residual = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny)) / r_norm_init;
            return iter;
        }
        omega = t_dot_s / t_dot_t;

        /* 10. x = x + alpha*p + omega*s */
        SIMD_FUNC(axpy)(alpha, p, x, nx, ny);
        SIMD_FUNC(axpy)(omega, s, x, nx, ny);

        /* 11. r = s - omega*t */
        SIMD_FUNC(copy_vector)(s, r, nx, ny);
        SIMD_FUNC(axpy)(-omega, t, r, nx, ny);

        /* 12. Check convergence on ||r|| */
        double r_norm = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny));
        if (r_norm / r_norm_init < tol) {
            if (stats) stats->final_residual = r_norm / r_norm_init;
            return iter + 1;
        }

        /* Update rho for next iteration */
        rho = rho_new;
    }

    /* Max iterations reached */
    double r_norm_final = sqrt(SIMD_FUNC(dot_product)(r, r, nx, ny));
    if (stats) stats->final_residual = r_norm_final / r_norm_init;

    return max_iter;
}

static int SIMD_FUNC(bicgstab_iterate)(void* context, double* x, const double* rhs,
                                        size_t nx, size_t ny) {
    /* Single iteration not commonly used for BiCGSTAB */
    /* Delegate to full solve with max_iter=1 */
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    return SIMD_FUNC(bicgstab_solve)(context, x, rhs, nx, ny, &stats);
}

//=============================================================================
// FACTORY FUNCTION
//=============================================================================

poisson_solver_t* SIMD_FUNC(create_bicgstab_solver)(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) return NULL;

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_SIMD;
    solver->description = "BiCGSTAB (SIMD + OpenMP)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();

    solver->init = SIMD_FUNC(bicgstab_init);
    solver->destroy = SIMD_FUNC(bicgstab_destroy);
    solver->solve = SIMD_FUNC(bicgstab_solve);
    solver->iterate = SIMD_FUNC(bicgstab_iterate);
    solver->apply_bc = NULL;

    return solver;
}

//=============================================================================
// CLEANUP MACROS
//=============================================================================

#undef bicgstab_simd_context_t
#undef CONCAT_IMPL
#undef CONCAT
#undef SIMD_FUNC
