/**
 * @file linear_solver_sor_avx2.c
 * @brief Block SOR (Successive Over-Relaxation) solver - AVX2 implementation
 *
 * SOR method characteristics:
 * - In-place update: reads from and writes to the same array (no double-buffer)
 * - Sequential row dependency: row j depends on row j-1 (no OpenMP on j-loop)
 * - Faster convergence than Jacobi (~1/2 iterations for optimal omega)
 * - SIMD speedup comes from vectorizing within each row only
 *
 * This implementation uses:
 * - AVX2 intrinsics for SIMD vectorization (4 doubles per operation)
 * - Block SOR: process 4 consecutive cells per SIMD iteration; intra-block
 *   left-neighbor dependency uses stale values (well-known HPC technique)
 * - Sequential j-loop (rows are not parallelizable due to data dependency)
 *
 * See docs/technical-notes/block-sor-simd.md for algorithm details and
 * convergence analysis.
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
    double inv_dz2;    /* 1/dz^2 (0.0 for 2D) */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2 + inv_dz2)) */
    double omega;      /* SOR relaxation parameter (default: 1.5) */
    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d dz2_inv_vec;
    __m256d neg_inv_factor_vec;
    __m256d omega_vec;
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
    (void)nx; (void)ny;

    /* Use aligned allocation for struct containing __m256d members */
    sor_avx2_context_t* ctx = (sor_avx2_context_t*)cfd_aligned_calloc(1, sizeof(sor_avx2_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2 + ctx->inv_dz2);
    ctx->inv_factor = 1.0 / factor;

    ctx->omega = (params && params->omega > 0.0) ? params->omega : 1.5;

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec        = _mm256_set1_pd(1.0 / ctx->dx2);
    ctx->dy2_inv_vec        = _mm256_set1_pd(1.0 / ctx->dy2);
    ctx->dz2_inv_vec        = _mm256_set1_pd(ctx->inv_dz2);
    ctx->neg_inv_factor_vec = _mm256_set1_pd(-ctx->inv_factor);
    ctx->omega_vec          = _mm256_set1_pd(ctx->omega);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_avx2_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_aligned_free(solver->context);
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
    size_t nx       = solver->nx;
    size_t ny       = solver->ny;
    double dx2      = ctx->dx2;
    double dy2      = ctx->dy2;
    double inv_dz2  = ctx->inv_dz2;
    double inv_factor = ctx->inv_factor;
    double omega    = ctx->omega;
    size_t stride_z = ctx->stride_z;

    __m256d dx2_inv      = ctx->dx2_inv_vec;
    __m256d dy2_inv      = ctx->dy2_inv_vec;
    __m256d dz2_inv      = ctx->dz2_inv_vec;
    __m256d neg_inv_factor = ctx->neg_inv_factor_vec;
    __m256d omega_vec    = ctx->omega_vec;

    /* Sequential k→j loop.
     * SOR rows are sequential: row j uses the updated row j-1, so no OpenMP
     * on the j-loop.  SIMD vectorization is applied within each row only. */
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            size_t i = 1;

            /* SIMD loop: process 4 doubles at a time (Block SOR).
             * Between blocks the left-neighbor dependency is satisfied.
             * Within a block, the intra-block left-neighbor uses stale values
             * from the beginning of the block — this is the accepted Block SOR
             * approximation and does not affect convergence in practice. */
            for (; i + 4 <= nx - 1; i += 4) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

                /* Load center and all six neighbors */
                __m256d x_c  = _mm256_loadu_pd(&x[idx]);
                __m256d x_xp = _mm256_loadu_pd(&x[idx + 1]);
                __m256d x_xm = _mm256_loadu_pd(&x[idx - 1]);
                __m256d x_yp = _mm256_loadu_pd(&x[idx + nx]);
                __m256d x_ym = _mm256_loadu_pd(&x[idx - nx]);
                __m256d x_zp = _mm256_loadu_pd(&x[idx + stride_z]);
                __m256d x_zm = _mm256_loadu_pd(&x[idx - stride_z]);
                __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

                /* Compute stencil: sum each axis pair, scale by 1/h^2 */
                __m256d sum_x = _mm256_mul_pd(_mm256_add_pd(x_xp, x_xm), dx2_inv);
                __m256d sum_y = _mm256_mul_pd(_mm256_add_pd(x_yp, x_ym), dy2_inv);
                __m256d sum_z = _mm256_mul_pd(_mm256_add_pd(x_zp, x_zm), dz2_inv);
                __m256d sum_terms = _mm256_add_pd(_mm256_add_pd(sum_x, sum_y), sum_z);

                /* p_new = -(rhs - sum_terms) * inv_factor */
                __m256d diff  = _mm256_sub_pd(rhs_vec, sum_terms);
                __m256d p_new = _mm256_mul_pd(diff, neg_inv_factor);

                /* SOR relaxation: x = x + omega * (p_new - x) */
                __m256d delta  = _mm256_sub_pd(p_new, x_c);
                __m256d update = _mm256_fmadd_pd(omega_vec, delta, x_c);

                _mm256_storeu_pd(&x[idx], update);
            }

            /* Scalar remainder for cells that don't fill a full SIMD block */
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
    solver->description = "SOR iteration (AVX2, Block SOR)";
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
