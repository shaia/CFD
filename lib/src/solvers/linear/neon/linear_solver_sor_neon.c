/**
 * @file linear_solver_sor_neon.c
 * @brief SOR (Successive Over-Relaxation) solver - ARM NEON implementation
 *
 * SOR characteristics:
 * - In-place updates (reads from and writes to the same array)
 * - Sequential row sweeps (row j depends on j-1, no row-level parallelism)
 * - Lexicographic order, the same iteration as the scalar solver, so the
 *   automatic omega applies to it
 *
 * Each row is swept in two passes:
 * - NEON, two cells at a time: every stencil term except the left neighbour.
 *   The right neighbour, the rows above and below and the planes either side
 *   are not written while this row is swept, so these reads are the scalar
 *   sweep's own.
 * - In order: add the left neighbour, which this sweep has just updated, and
 *   relax.
 *
 * The Block SOR this replaces read the second cell's left neighbour in each pair
 * from the previous sweep, which is a different iteration from SOR, and not one
 * the automatic omega is the optimum of. See docs/technical-notes/block-sor-simd.md.
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
    double inv_dx2;    /* 1/dx^2 */
    double inv_dy2;    /* 1/dy^2 */
    double inv_dz2;    /* 1/dz^2 (0.0 for 2D) */
    double factor;     /* 2 * (1/dx^2 + 1/dy^2 + inv_dz2) */
    double inv_factor; /* 1 / factor */
    double omega;      /* SOR relaxation parameter */
    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */
    double* partial;   /* nx doubles: a row's stencil terms without the left neighbour */
    float64x2_t dx2_inv_vec;
    float64x2_t dy2_inv_vec;
    float64x2_t dz2_inv_vec;
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
    (void)ny;

    sor_neon_context_t* ctx = (sor_neon_context_t*)cfd_calloc(1, sizeof(sor_neon_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }
    ctx->partial = (double*)cfd_calloc(nx, sizeof(double));
    if (!ctx->partial) {
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dx2 = 1.0 / ctx->dx2;
    ctx->inv_dy2 = 1.0 / ctx->dy2;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    ctx->factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2 + ctx->inv_dz2);
    ctx->inv_factor = 1.0 / ctx->factor;
    ctx->omega = poisson_solver_resolve_omega(solver, params ? params->omega : 0.0);

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = vdupq_n_f64(ctx->inv_dx2);
    ctx->dy2_inv_vec = vdupq_n_f64(ctx->inv_dy2);
    ctx->dz2_inv_vec = vdupq_n_f64(ctx->inv_dz2);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_neon_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        sor_neon_context_t* ctx = (sor_neon_context_t*)solver->context;
        cfd_free(ctx->partial);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

/**
 * SOR iteration using NEON for the stencil terms.
 *
 * Rows are swept sequentially (j=1..ny-2) because row j depends on the
 * already-updated row j-1 (Gauss-Seidel behavior). Within each row, the terms
 * that do not depend on this sweep are computed two cells at a time, then the
 * cells are relaxed in order against the updated left neighbour.
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
    double inv_dx2 = ctx->inv_dx2;
    double inv_dy2 = ctx->inv_dy2;
    double inv_dz2 = ctx->inv_dz2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;
    size_t stride_z = ctx->stride_z;
    double* partial = ctx->partial;
    int walls = poisson_solver_uses_default_walls(solver);

    /* Single sweep: sequential row-major order (no OpenMP on j-loop) */
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            double w_row, w_edge;
            poisson_solver_row_omegas(walls, omega, ctx->factor, nx, ny, solver->nz, j, k,
                                      inv_dx2, inv_dy2, inv_dz2, &w_row, &w_edge);
            size_t row = k * stride_z + IDX_2D(0, j, nx);
            size_t i = 1;

            /* Pass 1, two cells at a time: the right neighbour, the rows above and
             * below and the planes either side, less the right-hand side. Nothing
             * this row's sweep writes is read here. */
            for (; i + 2 <= nx - 1; i += 2) {
                size_t idx = row + i;
                float64x2_t x_xp = vld1q_f64(&x[idx + 1]);
                float64x2_t x_yp = vld1q_f64(&x[idx + nx]);
                float64x2_t x_ym = vld1q_f64(&x[idx - nx]);
                float64x2_t x_zp = vld1q_f64(&x[idx + stride_z]);
                float64x2_t x_zm = vld1q_f64(&x[idx - stride_z]);
                float64x2_t rhs_vec = vld1q_f64(&rhs[idx]);

                float64x2_t terms = vaddq_f64(vmulq_f64(x_xp, ctx->dx2_inv_vec),
                                              vmulq_f64(vaddq_f64(x_yp, x_ym), ctx->dy2_inv_vec));
                terms = vaddq_f64(terms, vmulq_f64(vaddq_f64(x_zp, x_zm), ctx->dz2_inv_vec));
                vst1q_f64(&partial[i], vsubq_f64(terms, rhs_vec));
            }
            for (; i < nx - 1; i++) {
                size_t idx = row + i;
                partial[i] = x[idx + 1] * inv_dx2
                           + (x[idx + nx] + x[idx - nx]) * inv_dy2
                           + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                           - rhs[idx];
            }

            /* Pass 2, in order: add the left neighbour this sweep has just updated
             * and relax, with the wall factor at the row's first and last point */
            for (i = 1; i < nx - 1; i++) {
                size_t idx = row + i;
                double p_new = (partial[i] + x[idx - 1] * inv_dx2) * inv_factor;
                double w = (i == 1 || i == nx - 2) ? w_edge : w_row;
                x[idx] = x[idx] + w * (p_new - x[idx]);
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
    solver->description = "SOR iteration (NEON stencil terms, sequential relaxation)";
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
