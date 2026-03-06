/**
 * @file linear_solver_sor_neon.c
 * @brief Block SOR (Successive Over-Relaxation) solver - ARM NEON implementation
 *
 * Block SOR characteristics:
 * - In-place updates (reads from and writes to the same array)
 * - SOR relaxation (omega > 1) for faster convergence than Jacobi
 * - Sequential row sweeps (row j depends on j-1, no row-level parallelism)
 * - Block processing: 2 consecutive cells per NEON operation
 *
 * This implementation uses:
 * - ARM NEON intrinsics for SIMD vectorization (2 doubles per operation)
 * - Block SOR: within each row, consecutive pairs of cells are processed
 *   together; left-neighbor dependency is satisfied at block boundaries
 * - Runtime detection to ensure NEON is available
 *
 * For technical details on the Block SOR SIMD strategy, see:
 * docs/technical-notes/block-sor-simd.md
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>

/* ARM NEON detection */
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define SOR_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(SOR_HAS_NEON)

/* ============================================================================
 * SOR NEON CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_dz2;    /* 1/dz^2 (0.0 for 2D) */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2 + inv_dz2)) */
    double omega;      /* SOR relaxation parameter */
    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t dz2_inv_vec;
    float64x2_t neg_inv_factor_vec;
    float64x2_t omega_vec;
    int initialized;
} sor_neon_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * SOR NEON IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t sor_neon_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny;

    sor_neon_context_t* ctx = (sor_neon_context_t*)cfd_calloc(1, sizeof(sor_neon_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2 + ctx->inv_dz2);
    ctx->inv_factor = 1.0 / factor;
    ctx->omega = poisson_solver_resolve_omega(
        params ? params->omega : 0.0, nx, ny, dx, dy);

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = vdupq_n_f64(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(1.0 / ctx->dy2);
    ctx->dz2_inv_vec = vdupq_n_f64(ctx->inv_dz2);
    ctx->neg_inv_factor_vec = vdupq_n_f64(-ctx->inv_factor);
    ctx->omega_vec = vdupq_n_f64(ctx->omega);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_neon_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

/**
 * Block SOR iteration using NEON SIMD.
 *
 * Rows are swept sequentially (j=1..ny-2) because row j depends on the
 * already-updated row j-1 (Gauss-Seidel behavior). Within each row,
 * consecutive pairs of cells are processed with NEON 2-wide operations.
 * The left-neighbor dependency (idx-1) is satisfied at block entry because
 * cell i-1 was updated in the previous block or scalar step.
 */
static cfd_status_t sor_neon_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;  /* Not needed for in-place SOR */

    sor_neon_context_t* ctx = (sor_neon_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;
    size_t stride_z = ctx->stride_z;
    double inv_dz2 = ctx->inv_dz2;

    /* Single sweep: sequential row-major order (no OpenMP on j-loop) */
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            size_t i = 1;

            /* NEON loop: process 2 consecutive cells at a time */
            for (; i + 2 <= nx - 1; i += 2) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

                /* Load current values */
                float64x2_t v_x = vld1q_f64(&x[idx]);

                /* Load neighbors */
                float64x2_t p_xp = vld1q_f64(&x[idx + 1]);        /* x+1 */
                float64x2_t p_xm = vld1q_f64(&x[idx - 1]);        /* x-1 */
                float64x2_t p_yp = vld1q_f64(&x[idx + nx]);       /* y+1 */
                float64x2_t p_ym = vld1q_f64(&x[idx - nx]);       /* y-1 */
                float64x2_t p_zp = vld1q_f64(&x[idx + stride_z]); /* z+1 */
                float64x2_t p_zm = vld1q_f64(&x[idx - stride_z]); /* z-1 */
                float64x2_t rhs_vec = vld1q_f64(&rhs[idx]);

                /* Sum neighbors in each direction */
                float64x2_t sum_x = vaddq_f64(p_xp, p_xm);
                float64x2_t sum_y = vaddq_f64(p_yp, p_ym);
                float64x2_t sum_z = vaddq_f64(p_zp, p_zm);

                /* Divide by dx^2, dy^2, and dz^2 */
                float64x2_t term_x = vmulq_f64(sum_x, ctx->dx2_inv_vec);
                float64x2_t term_y = vmulq_f64(sum_y, ctx->dy2_inv_vec);
                float64x2_t term_z = vmulq_f64(sum_z, ctx->dz2_inv_vec);

                /* Compute p_new = -(rhs - term_x - term_y - term_z) * inv_factor */
                float64x2_t sum_terms = vaddq_f64(vaddq_f64(term_x, term_y), term_z);
                float64x2_t diff = vsubq_f64(rhs_vec, sum_terms);
                float64x2_t p_new = vmulq_f64(diff, ctx->neg_inv_factor_vec);

                /* SOR update: x = x + omega * (p_new - x) */
                float64x2_t v_diff = vsubq_f64(p_new, v_x);
                /* vfmaq_f64(c, a, b) = c + a*b */
                float64x2_t v_result = vfmaq_f64(v_x, ctx->omega_vec, v_diff);

                /* Store result in-place */
                vst1q_f64(&x[idx], v_result);
            }

            /* Scalar remainder */
            for (; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                double p_new = -(rhs[idx]
                    - (x[idx + 1] + x[idx - 1]) / dx2
                    - (x[idx + nx] + x[idx - nx]) / dy2
                    - (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                    ) * inv_factor;
                x[idx] = x[idx] + omega * (p_new - x[idx]);
            }
        }
    }

    /* Apply boundary conditions */
    poisson_solver_apply_bc(solver, x);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

#endif /* SOR_HAS_NEON */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_sor_neon_solver(void) {
#if defined(SOR_HAS_NEON)
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) return NULL;

    solver->name = POISSON_SOLVER_TYPE_SOR_SIMD;
    solver->description = "SOR iteration (NEON, Block SOR)";
    solver->method = POISSON_METHOD_SOR;
    solver->backend = POISSON_BACKEND_SIMD;
    solver->params = poisson_solver_params_default();

    solver->init = sor_neon_init;
    solver->destroy = sor_neon_destroy;
    solver->solve = NULL;
    solver->iterate = sor_neon_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
