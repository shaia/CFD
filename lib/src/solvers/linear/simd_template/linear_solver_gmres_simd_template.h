/**
 * @file linear_solver_gmres_simd_template.h
 * @brief Restarted GMRES(m) SIMD implementation template (AVX2/NEON parameterized)
 *
 * Parameterized header implementing GMRES(m) with architecture-agnostic SIMD
 * macros. Included twice (AVX2, NEON) with different macro definitions. Only the
 * O(n) vector primitives are vectorized; the small dense Givens/Hessenberg and
 * back-substitution work is plain scalar (identical to the scalar backend), so
 * SIMD and scalar results stay consistent to rounding.
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

#include <math.h>
#include <string.h>

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

/* Factory function name: create_gmres_<suffix>_solver */
#define FACTORY_NAME_IMPL(suffix) create_gmres_##suffix##_solver
#define FACTORY_NAME(suffix) FACTORY_NAME_IMPL(suffix)

//=============================================================================
// CONTEXT STRUCTURE
//=============================================================================

#define gmres_simd_context_t SIMD_FUNC(gmres_context_t)

typedef struct {
    double dx2;
    double dy2;
    double inv_dz2;
    double diag_inv;

    size_t stride_z;
    size_t k_start;
    size_t k_end;

    int    m;   /* restart length */
    size_t n;   /* total field size */

    /* Precomputed SIMD constant vectors */
    SIMD_VEC dx2_inv_vec;
    SIMD_VEC dy2_inv_vec;
    SIMD_VEC dz2_inv_vec;
    SIMD_VEC two_vec;

    /* Large O(n) working arrays */
    double* Vblock;  /* (m+1) Krylov basis vectors */
    double* w;
    double* Mv;      /* NULL if no precond */

    /* Small dense O(m^2)/O(m) arrays (scalar) */
    double* H;
    double* cs;
    double* sn;
    double* g;
    double* y;

    int use_precond;
    int initialized;
} gmres_simd_context_t;

//=============================================================================
// SIMD O(n) PRIMITIVES (interior points only)
//=============================================================================

/** Dot product over interior points (SIMD + OpenMP reduction) */
static inline double SIMD_FUNC(gmres_dot)(const double* a, const double* b,
                                          size_t nx, size_t ny,
                                          size_t k_start, size_t k_end,
                                          size_t stride_z) {
    double sum = 0.0;
    int ny_int = bicgstab_size_to_int(ny);
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

/** Euclidean norm over interior points */
static inline double SIMD_FUNC(gmres_norm)(const double* x,
                                           size_t nx, size_t ny,
                                           size_t k_start, size_t k_end,
                                           size_t stride_z) {
    return sqrt(SIMD_FUNC(gmres_dot)(x, x, nx, ny, k_start, k_end, stride_z));
}

/** y = y + alpha*x (SIMD FMA) */
static inline void SIMD_FUNC(gmres_axpy)(double alpha, const double* x, double* y,
                                         size_t nx, size_t ny,
                                         size_t k_start, size_t k_end,
                                         size_t stride_z) {
    SIMD_VEC alpha_vec = SIMD_SET1(alpha);
    int ny_int = bicgstab_size_to_int(ny);
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
    int ny_int = bicgstab_size_to_int(ny);
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
    int ny_int = bicgstab_size_to_int(ny);
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
                                            const gmres_simd_context_t* ctx) {
    SIMD_VEC dx2_inv = ctx->dx2_inv_vec;
    SIMD_VEC dy2_inv = ctx->dy2_inv_vec;
    SIMD_VEC dz2_inv = ctx->dz2_inv_vec;
    SIMD_VEC two_vec = ctx->two_vec;
    SIMD_VEC zero = SIMD_SETZERO();
    size_t stride_z = ctx->stride_z;
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
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
                double lap = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / ctx->dx2
                           + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / ctx->dy2
                           + (p[idx + stride_z] + p[idx - stride_z] - 2.0 * p[idx]) * ctx->inv_dz2;
                Ap[idx] = -lap;
            }
        }
    }
}

/** Residual r = b - A x = -rhs + nabla^2 x (SIMD, fused) */
static inline void SIMD_FUNC(gmres_residual)(const double* x, const double* rhs,
                                             double* r, size_t nx, size_t ny,
                                             const gmres_simd_context_t* ctx) {
    SIMD_VEC dx2_inv = ctx->dx2_inv_vec;
    SIMD_VEC dy2_inv = ctx->dy2_inv_vec;
    SIMD_VEC dz2_inv = ctx->dz2_inv_vec;
    SIMD_VEC two_vec = ctx->two_vec;
    size_t stride_z = ctx->stride_z;
    int ny_int = bicgstab_size_to_int(ny);
    if (ny_int == 0) return;

    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
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
                double lap = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / ctx->dx2
                           + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / ctx->dy2
                           + (x[idx + stride_z] + x[idx - stride_z] - 2.0 * x[idx]) * ctx->inv_dz2;
                r[idx] = -rhs[idx] + lap;
            }
        }
    }
}

/** z = diag_inv * r (Jacobi preconditioner, SIMD) */
static inline void SIMD_FUNC(gmres_precond)(const double* r, double* z,
                                            double diag_inv,
                                            size_t nx, size_t ny,
                                            size_t k_start, size_t k_end,
                                            size_t stride_z) {
    SIMD_VEC diag_vec = SIMD_SET1(diag_inv);
    int ny_int = bicgstab_size_to_int(ny);
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
// DENSE SCALAR HELPERS (identical to scalar backend; not vectorized)
//=============================================================================

static void SIMD_FUNC(gmres_givens)(double* H, double* cs, double* sn,
                                    double* g, int j, int m) {
    const int lead = m + 1;
    for (int i = 0; i < j; i++) {
        double temp     =  cs[i] * H[i + j * lead]     + sn[i] * H[(i + 1) + j * lead];
        H[(i + 1) + j * lead] = -sn[i] * H[i + j * lead] + cs[i] * H[(i + 1) + j * lead];
        H[i + j * lead] = temp;
    }

    double h_jj  = H[j + j * lead];
    double h_j1j = H[(j + 1) + j * lead];
    double denom = sqrt(h_jj * h_jj + h_j1j * h_j1j);
    if (denom < GMRES_BREAKDOWN_THRESHOLD) {
        cs[j] = 1.0;
        sn[j] = 0.0;
    } else {
        cs[j] = h_jj / denom;
        sn[j] = h_j1j / denom;
    }

    H[j + j * lead]       = cs[j] * h_jj + sn[j] * h_j1j;
    H[(j + 1) + j * lead] = 0.0;

    double g_j = g[j];
    g[j]     =  cs[j] * g_j;
    g[j + 1] = -sn[j] * g_j;
}

static void SIMD_FUNC(gmres_backsub)(const double* H, const double* g, double* y,
                                     int k, int m) {
    const int lead = m + 1;
    for (int i = k - 1; i >= 0; i--) {
        double sum = g[i];
        for (int l = i + 1; l < k; l++) {
            sum -= H[i + l * lead] * y[l];
        }
        double diag = H[i + i * lead];
        y[i] = (fabs(diag) > GMRES_BREAKDOWN_THRESHOLD) ? (sum / diag) : 0.0;
    }
}

//=============================================================================
// INIT / DESTROY
//=============================================================================

static void SIMD_FUNC(gmres_destroy)(poisson_solver_t* solver) {
    if (!solver || !solver->context) return;
    gmres_simd_context_t* ctx = (gmres_simd_context_t*)(solver->context);
    cfd_aligned_free(ctx->Vblock);
    cfd_aligned_free(ctx->w);
    cfd_aligned_free(ctx->Mv);
    cfd_free(ctx->H);
    cfd_free(ctx->cs);
    cfd_free(ctx->sn);
    cfd_free(ctx->g);
    cfd_free(ctx->y);
    cfd_aligned_free(ctx);
    solver->context = NULL;
}

static cfd_status_t SIMD_FUNC(gmres_init)(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params) {

    cfd_status_t precond_status = poisson_solver_reject_mg_precond(params);
    if (precond_status != CFD_SUCCESS) {
        return precond_status;
    }

    gmres_simd_context_t* ctx = (gmres_simd_context_t*)cfd_aligned_calloc(
        1, sizeof(gmres_simd_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);
    ctx->diag_inv = 1.0 / (2.0 / ctx->dx2 + 2.0 / ctx->dy2 + 2.0 * ctx->inv_dz2);
    ctx->use_precond = (params && params->preconditioner == POISSON_PRECOND_JACOBI);

    int m = (params && params->restart > 0) ? params->restart : GMRES_DEFAULT_RESTART;
    if (m < 1) m = 1;
    ctx->m = m;

    size_t n = nx * ny * nz;
    ctx->n = n;

    ctx->Vblock = (double*)cfd_aligned_calloc((size_t)(m + 1) * n, sizeof(double));
    ctx->w = (double*)cfd_aligned_calloc(n, sizeof(double));
    ctx->Mv = ctx->use_precond ? (double*)cfd_aligned_calloc(n, sizeof(double)) : NULL;

    ctx->H = (double*)cfd_calloc((size_t)(m + 1) * m, sizeof(double));
    ctx->cs = (double*)cfd_calloc(m, sizeof(double));
    ctx->sn = (double*)cfd_calloc(m, sizeof(double));
    ctx->g = (double*)cfd_calloc((size_t)m + 1, sizeof(double));
    ctx->y = (double*)cfd_calloc(m, sizeof(double));

    int precond_ok = (!ctx->use_precond) || (ctx->Mv != NULL);
    if (!ctx->Vblock || !ctx->w || !ctx->H || !ctx->cs || !ctx->sn ||
        !ctx->g || !ctx->y || !precond_ok) {
        solver->context = ctx;
        SIMD_FUNC(gmres_destroy)(solver);
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2_inv_vec = SIMD_SET1(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = SIMD_SET1(1.0 / ctx->dy2);
    ctx->dz2_inv_vec = SIMD_SET1(ctx->inv_dz2);
    ctx->two_vec = SIMD_SET1(2.0);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

//=============================================================================
// SOLVE
//=============================================================================

static cfd_status_t SIMD_FUNC(gmres_solve)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats) {
    (void)x_temp;

    gmres_simd_context_t* ctx = (gmres_simd_context_t*)(solver->context);
    if (!ctx || !ctx->initialized) {
        return CFD_ERROR;
    }

    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t stride_z = ctx->stride_z;
    size_t k_start = ctx->k_start;
    size_t k_end = ctx->k_end;

    int m = ctx->m;
    size_t n = ctx->n;
    const int lead = m + 1;

    double* Vblock = ctx->Vblock;
    double* w = ctx->w;
    double* Mv = ctx->Mv;
    double* H = ctx->H;
    double* cs = ctx->cs;
    double* sn = ctx->sn;
    double* g = ctx->g;
    double* y = ctx->y;
    int use_precond = ctx->use_precond;
    double diag_inv = ctx->diag_inv;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    poisson_solver_apply_bc(solver, x);

    SIMD_FUNC(gmres_residual)(x, rhs, Vblock, nx, ny, ctx);
    double beta = SIMD_FUNC(gmres_norm)(Vblock, nx, ny, k_start, k_end, stride_z);
    double initial_res = beta;

    if (stats) {
        stats->initial_residual = initial_res;
    }

    double tolerance = params->tolerance * initial_res;
    if (tolerance < params->absolute_tolerance) {
        tolerance = params->absolute_tolerance;
    }

    if (initial_res < params->absolute_tolerance) {
        poisson_solver_apply_bc(solver, x);
        if (stats) {
            stats->status = POISSON_CONVERGED;
            stats->iterations = 0;
            stats->final_residual = initial_res;
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return CFD_SUCCESS;
    }

    int total_inner = 0;
    int converged = 0;
    double final_res = initial_res;

    while (total_inner < params->max_iterations && !converged) {
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
            final_res = beta;
            break;
        }

        SIMD_FUNC(gmres_scale)(1.0 / beta, Vblock, nx, ny, k_start, k_end, stride_z);
        memset(g, 0, (size_t)(m + 1) * sizeof(double));
        g[0] = beta;

        int k = 0;
        for (int j = 0; j < m; j++) {
            double* v_j = Vblock + (size_t)j * n;

            if (use_precond) {
                SIMD_FUNC(gmres_precond)(v_j, Mv, diag_inv, nx, ny, k_start, k_end, stride_z);
                SIMD_FUNC(gmres_apply_A)(Mv, w, nx, ny, ctx);
            } else {
                SIMD_FUNC(gmres_apply_A)(v_j, w, nx, ny, ctx);
            }

            for (int i = 0; i <= j; i++) {
                double* v_i = Vblock + (size_t)i * n;
                double hij = SIMD_FUNC(gmres_dot)(w, v_i, nx, ny, k_start, k_end, stride_z);
                H[i + j * lead] = hij;
                SIMD_FUNC(gmres_axpy)(-hij, v_i, w, nx, ny, k_start, k_end, stride_z);
            }

            double hnorm = SIMD_FUNC(gmres_norm)(w, nx, ny, k_start, k_end, stride_z);
            H[(j + 1) + j * lead] = hnorm;

            int happy = (hnorm < GMRES_BREAKDOWN_THRESHOLD);
            if (!happy) {
                SIMD_FUNC(gmres_scale)(1.0 / hnorm, w, nx, ny, k_start, k_end, stride_z);
                SIMD_FUNC(gmres_copy)(w, Vblock + (size_t)(j + 1) * n, nx, ny, k_start, k_end, stride_z);
            }

            SIMD_FUNC(gmres_givens)(H, cs, sn, g, j, m);
            total_inner++;
            k = j + 1;

            double resid_est = fabs(g[j + 1]);
            if (resid_est < tolerance || happy || total_inner >= params->max_iterations) {
                break;
            }
        }

        SIMD_FUNC(gmres_backsub)(H, g, y, k, m);
        if (use_precond) {
            memset(w, 0, n * sizeof(double));
            for (int jj = 0; jj < k; jj++) {
                SIMD_FUNC(gmres_axpy)(y[jj], Vblock + (size_t)jj * n, w,
                                      nx, ny, k_start, k_end, stride_z);
            }
            SIMD_FUNC(gmres_precond)(w, Mv, diag_inv, nx, ny, k_start, k_end, stride_z);
            SIMD_FUNC(gmres_axpy)(1.0, Mv, x, nx, ny, k_start, k_end, stride_z);
        } else {
            for (int jj = 0; jj < k; jj++) {
                SIMD_FUNC(gmres_axpy)(y[jj], Vblock + (size_t)jj * n, x,
                                      nx, ny, k_start, k_end, stride_z);
            }
        }

        SIMD_FUNC(gmres_residual)(x, rhs, Vblock, nx, ny, ctx);
        beta = SIMD_FUNC(gmres_norm)(Vblock, nx, ny, k_start, k_end, stride_z);
        final_res = beta;
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
        }
    }

    poisson_solver_apply_bc(solver, x);

    if (stats) {
        stats->iterations = total_inner;
        stats->final_residual = final_res;
        stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

static cfd_status_t SIMD_FUNC(gmres_iterate)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual) {
    (void)x_temp;
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }
    return CFD_SUCCESS;
}

//=============================================================================
// FACTORY
//=============================================================================

poisson_solver_t* FACTORY_NAME(SIMD_SUFFIX)(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) return NULL;

    solver->name = POISSON_SOLVER_TYPE_GMRES_SIMD;
    solver->description = "Restarted GMRES(m) (SIMD + OpenMP)";
    solver->method = POISSON_METHOD_GMRES;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();

    solver->init = SIMD_FUNC(gmres_init);
    solver->destroy = SIMD_FUNC(gmres_destroy);
    solver->solve = SIMD_FUNC(gmres_solve);
    solver->iterate = SIMD_FUNC(gmres_iterate);
    solver->apply_bc = NULL;

    return solver;
}

//=============================================================================
// CLEANUP MACROS
//=============================================================================

#undef gmres_simd_context_t
#undef CONCAT_IMPL
#undef CONCAT
#undef SIMD_FUNC
#undef FACTORY_NAME_IMPL
#undef FACTORY_NAME
