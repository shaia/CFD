/**
 * @file linear_solver_jacobi_neon.c
 * @brief Jacobi iteration solver - ARM NEON implementation
 *
 * Jacobi method characteristics:
 * - Fully parallelizable (reads from old array, writes to new array)
 * - Requires temporary buffer for double-buffering
 * - Slower convergence than SOR (~2x iterations)
 * - Well-suited for SIMD + multithreading
 *
 * This implementation uses:
 * - ARM NEON intrinsics for SIMD vectorization (2 doubles per operation)
 * - OpenMP for thread-level parallelism across rows
 * - Runtime detection to ensure NEON is available
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ARM NEON detection */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define JACOBI_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(JACOBI_HAS_NEON)

/* ============================================================================
 * JACOBI NEON CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t neg_inv_factor_vec;
    int initialized;
} jacobi_neon_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * JACOBI NEON IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t jacobi_neon_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny; (void)params;

    jacobi_neon_context_t* ctx = (jacobi_neon_context_t*)cfd_calloc(1, sizeof(jacobi_neon_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = vdupq_n_f64(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(1.0 / ctx->dy2);
    ctx->neg_inv_factor_vec = vdupq_n_f64(-ctx->inv_factor);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void jacobi_neon_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t jacobi_neon_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    if (!x_temp) {
        return CFD_ERROR_INVALID;
    }

    jacobi_neon_context_t* ctx = (jacobi_neon_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;

    double* p_old = x;
    double* p_new = x_temp;

    int ny_int = size_to_int(ny);

    /* Process interior rows with OpenMP parallelization */
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* NEON loop: process 2 doubles at a time */
        for (; i + 2 <= nx - 1; i += 2) {
            size_t idx = IDX_2D(i, (size_t)j, nx);

            /* Load neighbors */
            float64x2_t p_xp = vld1q_f64(&p_old[idx + 1]);   /* x+1 */
            float64x2_t p_xm = vld1q_f64(&p_old[idx - 1]);   /* x-1 */
            float64x2_t p_yp = vld1q_f64(&p_old[idx + nx]);  /* y+1 */
            float64x2_t p_ym = vld1q_f64(&p_old[idx - nx]);  /* y-1 */
            float64x2_t rhs_vec = vld1q_f64(&rhs[idx]);

            /* Sum neighbors in each direction */
            float64x2_t sum_x = vaddq_f64(p_xp, p_xm);
            float64x2_t sum_y = vaddq_f64(p_yp, p_ym);

            /* Divide by dx^2 and dy^2 */
            float64x2_t term_x = vmulq_f64(sum_x, ctx->dx2_inv_vec);
            float64x2_t term_y = vmulq_f64(sum_y, ctx->dy2_inv_vec);

            /* Compute: -(rhs - term_x - term_y) * inv_factor */
            float64x2_t sum_terms = vaddq_f64(term_x, term_y);
            float64x2_t diff = vsubq_f64(rhs_vec, sum_terms);
            float64x2_t p_result = vmulq_f64(diff, ctx->neg_inv_factor_vec);

            /* Store result */
            vst1q_f64(&p_new[idx], p_result);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = IDX_2D(i, (size_t)j, nx);
            double p_result = -(rhs[idx]
                - (p_old[idx + 1] + p_old[idx - 1]) / dx2
                - (p_old[idx + nx] + p_old[idx - nx]) / dy2) * inv_factor;
            p_new[idx] = p_result;
        }
    }

    /* Copy result back to x (parallel) */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < size_to_int(ny); j++) {
        memcpy(&x[(size_t)j * nx], &x_temp[(size_t)j * nx], nx * sizeof(double));
    }

    /* Apply boundary conditions (use SIMD BC if available) */
    bc_apply_scalar_simd(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

#endif /* JACOBI_HAS_NEON */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_jacobi_neon_solver(void) {
#if defined(JACOBI_HAS_NEON)
    /* Note: Runtime SIMD check is done by the dispatcher (linear_solver_simd_dispatch.c)
     * before calling this function. No need to check again here. */
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_JACOBI_SIMD;
    solver->description = "Jacobi iteration (NEON)";
    solver->method = POISSON_METHOD_JACOBI;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();
    solver->params.max_iterations = 2000;
    solver->params.check_interval = 10;

    solver->init = jacobi_neon_init;
    solver->destroy = jacobi_neon_destroy;
    solver->solve = NULL;
    solver->iterate = jacobi_neon_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
