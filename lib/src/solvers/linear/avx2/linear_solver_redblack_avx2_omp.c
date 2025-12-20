/**
 * @file linear_solver_redblack_avx2_omp.c
 * @brief Red-Black SOR solver - AVX2 + OpenMP implementation
 *
 * Red-Black SOR characteristics:
 * - Two-color sweep (red then black) allows parallelization
 * - In-place updates (no temporary buffer needed)
 * - SOR acceleration (omega > 1) for faster convergence
 * - Best balance of convergence speed and parallelism
 *
 * This implementation uses:
 * - AVX2 intrinsics for SIMD vectorization (4 doubles per operation)
 * - OpenMP for thread-level parallelism across rows
 * - Runtime detection to ensure AVX2 is available
 *
 * Note: Red-Black has stride-2 access pattern which limits SIMD efficiency.
 * We manually gather 4 same-color cells, compute, and scatter back.
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>

/* AVX2 + OpenMP detection */
#if defined(__AVX2__) && defined(CFD_ENABLE_OPENMP)
#define REDBLACK_HAS_AVX2_OMP 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(REDBLACK_HAS_AVX2_OMP)

/* ============================================================================
 * RED-BLACK AVX2+OMP CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    double omega;      /* SOR relaxation parameter */
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d inv_factor_vec;
    __m256d omega_vec;
    __m256d one_minus_omega_vec;
    int initialized;
} redblack_avx2_omp_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * RED-BLACK AVX2+OMP IMPLEMENTATION
 * ============================================================================ */

/**
 * Process a single row for one color (red or black) using AVX2 SIMD.
 *
 * @param j       Row index
 * @param i_start Starting column index for this color
 * @param x       Solution vector (in/out)
 * @param rhs     Right-hand side vector
 * @param nx      Grid width
 * @param ctx     Solver context with precomputed SIMD vectors
 */
static inline void redblack_avx2_process_row(
    int j,
    size_t i_start,
    double* x,
    const double* rhs,
    size_t nx,
    const redblack_avx2_omp_context_t* ctx)
{
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;
    size_t i = i_start;

    /* SIMD loop: gather 4 same-color cells (stride 2) */
    for (; i + 8 <= nx - 1; i += 8) {
        /* Gather 4 values with stride 2 */
        double vals[4];
        double p_xp[4], p_xm[4], p_yp[4], p_ym[4], rhs_vals[4];

        for (int k = 0; k < 4; k++) {
            size_t idx = (size_t)j * nx + i + (size_t)k * 2;
            vals[k] = x[idx];
            p_xp[k] = x[idx + 1];
            p_xm[k] = x[idx - 1];
            p_yp[k] = x[idx + nx];
            p_ym[k] = x[idx - nx];
            rhs_vals[k] = rhs[idx];
        }

        /* Load into SIMD registers */
        __m256d v_vals = _mm256_loadu_pd(vals);
        __m256d v_xp = _mm256_loadu_pd(p_xp);
        __m256d v_xm = _mm256_loadu_pd(p_xm);
        __m256d v_yp = _mm256_loadu_pd(p_yp);
        __m256d v_ym = _mm256_loadu_pd(p_ym);
        __m256d v_rhs = _mm256_loadu_pd(rhs_vals);

        /* Compute: p_new = -(rhs - (xp+xm)/dx2 - (yp+ym)/dy2) * inv_factor */
        __m256d sum_x = _mm256_add_pd(v_xp, v_xm);
        __m256d sum_y = _mm256_add_pd(v_yp, v_ym);
        __m256d term_x = _mm256_mul_pd(sum_x, ctx->dx2_inv_vec);
        __m256d term_y = _mm256_mul_pd(sum_y, ctx->dy2_inv_vec);
        __m256d sum_terms = _mm256_add_pd(term_x, term_y);
        __m256d diff = _mm256_sub_pd(v_rhs, sum_terms);
        __m256d v_p_new = _mm256_mul_pd(diff, ctx->inv_factor_vec);

        /* SOR update: x = x + omega * (p_new - x) */
        __m256d v_diff = _mm256_sub_pd(v_p_new, v_vals);
        __m256d v_update = _mm256_mul_pd(ctx->omega_vec, v_diff);
        __m256d v_result = _mm256_add_pd(v_vals, v_update);

        /* Scatter back */
        double result[4];
        _mm256_storeu_pd(result, v_result);
        for (int k = 0; k < 4; k++) {
            size_t idx = (size_t)j * nx + i + (size_t)k * 2;
            x[idx] = result[k];
        }
    }

    /* Scalar remainder */
    for (; i < nx - 1; i += 2) {
        size_t idx = (size_t)j * nx + i;
        double p_new = -(rhs[idx]
            - (x[idx + 1] + x[idx - 1]) / dx2
            - (x[idx + nx] + x[idx - nx]) / dy2) * inv_factor;
        x[idx] = x[idx] + omega * (p_new - x[idx]);
    }
}

static cfd_status_t redblack_avx2_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny;

    /* Use aligned allocation for struct containing __m256d members */
    redblack_avx2_omp_context_t* ctx = (redblack_avx2_omp_context_t*)cfd_aligned_calloc(1, sizeof(redblack_avx2_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;
    ctx->omega = params ? params->omega : 1.5;

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = _mm256_set1_pd(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = _mm256_set1_pd(1.0 / ctx->dy2);
    ctx->inv_factor_vec = _mm256_set1_pd(-ctx->inv_factor);
    ctx->omega_vec = _mm256_set1_pd(ctx->omega);
    ctx->one_minus_omega_vec = _mm256_set1_pd(1.0 - ctx->omega);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void redblack_avx2_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_aligned_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t redblack_avx2_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    redblack_avx2_omp_context_t* ctx = (redblack_avx2_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    int ny_int = size_to_int(ny);
    int j;

    /* Red sweep with SIMD + OpenMP */
    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i_start = (j % 2 == 0) ? 1 : 2;
        redblack_avx2_process_row(j, i_start, x, rhs, nx, ctx);
    }

    /* Black sweep with SIMD + OpenMP */
    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i_start = (j % 2 == 0) ? 2 : 1;
        redblack_avx2_process_row(j, i_start, x, rhs, nx, ctx);
    }

    /* Apply boundary conditions (use SIMD+OMP BC if available) */
    bc_apply_scalar_simd_omp(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

#endif /* REDBLACK_HAS_AVX2_OMP */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_redblack_avx2_omp_solver(void) {
#if defined(REDBLACK_HAS_AVX2_OMP)
    /* Runtime check - ensure AVX2 is actually available */
    if (cfd_detect_simd_arch() != CFD_SIMD_AVX2) {
        return NULL;
    }

    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_REDBLACK_SIMD_OMP;
    solver->description = "Red-Black SOR iteration (AVX2 + OpenMP)";
    solver->method = POISSON_METHOD_REDBLACK_SOR;
    solver->backend = POISSON_BACKEND_SIMD_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = redblack_avx2_omp_init;
    solver->destroy = redblack_avx2_omp_destroy;
    solver->solve = NULL;
    solver->iterate = redblack_avx2_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
