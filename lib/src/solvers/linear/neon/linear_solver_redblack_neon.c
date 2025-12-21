/**
 * @file linear_solver_redblack_neon.c
 * @brief Red-Black SOR solver - ARM NEON + OpenMP implementation
 *
 * Red-Black SOR characteristics:
 * - Two-color sweep (red then black) allows parallelization
 * - In-place updates (no temporary buffer needed)
 * - SOR acceleration (omega > 1) for faster convergence
 * - Best balance of convergence speed and parallelism
 *
 * This implementation uses:
 * - ARM NEON intrinsics for SIMD vectorization (2 doubles per operation)
 * - OpenMP for thread-level parallelism across rows
 * - Runtime detection to ensure NEON is available
 *
 * Note: Red-Black has stride-2 access pattern which limits SIMD efficiency.
 * We manually gather 2 same-color cells, compute, and scatter back.
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>

/* ARM NEON + OpenMP detection */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define REDBLACK_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(REDBLACK_HAS_NEON)

/* ============================================================================
 * RED-BLACK NEON CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    double omega;      /* SOR relaxation parameter */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t neg_inv_factor_vec;
    float64x2_t omega_vec;
    int initialized;
} redblack_neon_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * RED-BLACK NEON IMPLEMENTATION
 * ============================================================================ */

/**
 * Process a single row for one color (red or black) using NEON SIMD.
 *
 * @param j       Row index
 * @param i_start Starting column index for this color
 * @param x       Solution vector (in/out)
 * @param rhs     Right-hand side vector
 * @param nx      Grid width
 * @param ctx     Solver context with precomputed SIMD vectors
 */
static inline void redblack_neon_process_row(
    int j,
    size_t i_start,
    double* x,
    const double* rhs,
    size_t nx,
    const redblack_neon_context_t* ctx)
{
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;
    size_t i = i_start;

    /* NEON loop: gather 2 same-color cells (stride 2) */
    for (; i + 4 <= nx - 1; i += 4) {
        /* Gather 2 values with stride 2 */
        double vals[2];
        double p_xp[2], p_xm[2], p_yp[2], p_ym[2], rhs_vals[2];

        for (int k = 0; k < 2; k++) {
            size_t idx = (size_t)j * nx + i + (size_t)k * 2;
            vals[k] = x[idx];
            p_xp[k] = x[idx + 1];
            p_xm[k] = x[idx - 1];
            p_yp[k] = x[idx + nx];
            p_ym[k] = x[idx - nx];
            rhs_vals[k] = rhs[idx];
        }

        /* Load into SIMD registers */
        float64x2_t v_vals = vld1q_f64(vals);
        float64x2_t v_xp = vld1q_f64(p_xp);
        float64x2_t v_xm = vld1q_f64(p_xm);
        float64x2_t v_yp = vld1q_f64(p_yp);
        float64x2_t v_ym = vld1q_f64(p_ym);
        float64x2_t v_rhs = vld1q_f64(rhs_vals);

        /* Compute: p_new = -(rhs - (xp+xm)/dx2 - (yp+ym)/dy2) * inv_factor */
        float64x2_t sum_x = vaddq_f64(v_xp, v_xm);
        float64x2_t sum_y = vaddq_f64(v_yp, v_ym);
        float64x2_t term_x = vmulq_f64(sum_x, ctx->dx2_inv_vec);
        float64x2_t term_y = vmulq_f64(sum_y, ctx->dy2_inv_vec);
        float64x2_t sum_terms = vaddq_f64(term_x, term_y);
        float64x2_t diff = vsubq_f64(v_rhs, sum_terms);
        float64x2_t v_p_new = vmulq_f64(diff, ctx->neg_inv_factor_vec);

        /* SOR update: x = x + omega * (p_new - x) */
        float64x2_t v_diff = vsubq_f64(v_p_new, v_vals);
        float64x2_t v_update = vmulq_f64(ctx->omega_vec, v_diff);
        float64x2_t v_result = vaddq_f64(v_vals, v_update);

        /* Scatter back */
        double result[2];
        vst1q_f64(result, v_result);
        for (int k = 0; k < 2; k++) {
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

static cfd_status_t redblack_neon_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny;

    redblack_neon_context_t* ctx = (redblack_neon_context_t*)cfd_calloc(1, sizeof(redblack_neon_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;
    ctx->omega = params ? params->omega : 1.5;

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = vdupq_n_f64(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(1.0 / ctx->dy2);
    ctx->neg_inv_factor_vec = vdupq_n_f64(-ctx->inv_factor);
    ctx->omega_vec = vdupq_n_f64(ctx->omega);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void redblack_neon_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t redblack_neon_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    redblack_neon_context_t* ctx = (redblack_neon_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    int ny_int = size_to_int(ny);
    int j;

    /* Red sweep with NEON + OpenMP */
    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i_start = (j % 2 == 0) ? 1 : 2;
        redblack_neon_process_row(j, i_start, x, rhs, nx, ctx);
    }

    /* Black sweep with NEON + OpenMP */
    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i_start = (j % 2 == 0) ? 2 : 1;
        redblack_neon_process_row(j, i_start, x, rhs, nx, ctx);
    }

    /* Apply boundary conditions (use SIMD BC if available) */
    bc_apply_scalar_simd_omp(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

#endif /* REDBLACK_HAS_NEON */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_redblack_neon_solver(void) {
#if defined(REDBLACK_HAS_NEON)
    /* Note: Runtime SIMD check is done by the dispatcher (linear_solver_simd_dispatch.c)
     * before calling this function. No need to check again here. */
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_REDBLACK_SIMD_OMP;
    solver->description = "Red-Black SOR iteration (NEON + OpenMP)";
    solver->method = POISSON_METHOD_REDBLACK_SOR;
    solver->backend = POISSON_BACKEND_SIMD_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = redblack_neon_init;
    solver->destroy = redblack_neon_destroy;
    solver->solve = NULL;
    solver->iterate = redblack_neon_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
