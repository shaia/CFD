/**
 * @file linear_solver_sor_avx2.c
 * @brief SOR (Successive Over-Relaxation) solver - AVX2 implementation
 *
 * SOR method characteristics:
 * - In-place update: reads from and writes to the same array (no double-buffer)
 * - Sequential row dependency: row j depends on row j-1 (no OpenMP on j-loop)
 * - Lexicographic order, the same iteration as the scalar solver, so the
 *   automatic omega applies to it
 *
 * Each row is swept in two passes:
 * - AVX2, four cells at a time: every stencil term except the left neighbour.
 *   The right neighbour, the rows above and below and the planes either side
 *   are not written while this row is swept, so these reads are the scalar
 *   sweep's own.
 * - In order: add the left neighbour, which this sweep has just updated, and
 *   relax.
 *
 * The Block SOR this replaces read the left neighbour inside each block of four
 * from the previous sweep. That is a different iteration: on every grid measured,
 * 17x17 to 65x65, it diverged for omega between 1.40 and 1.50, below those grids'
 * automatic omega. See docs/technical-notes/block-sor-simd.md.
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>

/* AVX2 + OpenMP detection
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define SOR_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(SOR_HAS_AVX2)

/* ============================================================================
 * SOR AVX2 CONTEXT
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
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d dz2_inv_vec;
    int initialized;
} sor_avx2_context_t;

/* ============================================================================
 * SOR AVX2 IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t sor_avx2_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    (void)ny;

    /* Use aligned allocation for struct containing __m256d members */
    sor_avx2_context_t* ctx = (sor_avx2_context_t*)cfd_aligned_calloc(1, sizeof(sor_avx2_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }
    ctx->partial = (double*)cfd_calloc(nx, sizeof(double));
    if (!ctx->partial) {
        cfd_aligned_free(ctx);
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
    ctx->dx2_inv_vec = _mm256_set1_pd(ctx->inv_dx2);
    ctx->dy2_inv_vec = _mm256_set1_pd(ctx->inv_dy2);
    ctx->dz2_inv_vec = _mm256_set1_pd(ctx->inv_dz2);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_avx2_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        sor_avx2_context_t* ctx = (sor_avx2_context_t*)solver->context;
        cfd_free(ctx->partial);
        cfd_aligned_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t sor_avx2_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,    /* unused for SOR — in-place update */
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    sor_avx2_context_t* ctx = (sor_avx2_context_t*)solver->context;
    size_t nx         = solver->nx;
    size_t ny         = solver->ny;
    double inv_dx2    = ctx->inv_dx2;
    double inv_dy2    = ctx->inv_dy2;
    double inv_dz2    = ctx->inv_dz2;
    double inv_factor = ctx->inv_factor;
    double omega      = ctx->omega;
    size_t stride_z   = ctx->stride_z;
    double* partial   = ctx->partial;
    int walls         = poisson_solver_uses_default_walls(solver);

    __m256d dx2_inv = ctx->dx2_inv_vec;
    __m256d dy2_inv = ctx->dy2_inv_vec;
    __m256d dz2_inv = ctx->dz2_inv_vec;

    /* Sequential k→j loop.
     * SOR rows are sequential: row j uses the updated row j-1, so no OpenMP
     * on the j-loop. */
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            double w_row, w_edge;
            poisson_solver_row_omegas(walls, omega, ctx->factor, nx, ny, solver->nz, j, k,
                                      inv_dx2, inv_dy2, inv_dz2, &w_row, &w_edge);
            size_t row = k * stride_z + IDX_2D(0, j, nx);
            size_t i = 1;

            /* Pass 1, four cells at a time: the right neighbour, the rows above and
             * below and the planes either side, less the right-hand side. Nothing
             * this row's sweep writes is read here. */
            for (; i + 4 <= nx - 1; i += 4) {
                size_t idx = row + i;
                __m256d x_xp    = _mm256_loadu_pd(&x[idx + 1]);
                __m256d x_yp    = _mm256_loadu_pd(&x[idx + nx]);
                __m256d x_ym    = _mm256_loadu_pd(&x[idx - nx]);
                __m256d x_zp    = _mm256_loadu_pd(&x[idx + stride_z]);
                __m256d x_zm    = _mm256_loadu_pd(&x[idx - stride_z]);
                __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

                __m256d terms = _mm256_add_pd(_mm256_mul_pd(x_xp, dx2_inv),
                                              _mm256_mul_pd(_mm256_add_pd(x_yp, x_ym), dy2_inv));
                terms = _mm256_add_pd(terms, _mm256_mul_pd(_mm256_add_pd(x_zp, x_zm), dz2_inv));
                _mm256_storeu_pd(&partial[i], _mm256_sub_pd(terms, rhs_vec));
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

#endif /* SOR_HAS_AVX2 */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_sor_avx2_solver(void) {
#if defined(SOR_HAS_AVX2)
    /* Note: Runtime SIMD check is done by the dispatcher (linear_solver_simd_dispatch.c)
     * before calling this function. No need to check again here. */
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name        = POISSON_SOLVER_TYPE_SOR_SIMD;
    solver->description = "SOR iteration (AVX2 stencil terms, sequential relaxation)";
    solver->method      = POISSON_METHOD_SOR;
    solver->backend     = POISSON_BACKEND_SIMD;
    solver->params      = poisson_solver_params_default();

    solver->init     = sor_avx2_init;
    solver->destroy  = sor_avx2_destroy;
    solver->solve    = NULL;
    solver->iterate  = sor_avx2_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
