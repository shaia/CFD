/**
 * @file linear_solver_jacobi_avx2_omp.c
 * @brief Jacobi iteration solver - AVX2 + OpenMP implementation
 *
 * Jacobi method characteristics:
 * - Fully parallelizable (reads from old array, writes to new array)
 * - Requires temporary buffer for double-buffering
 * - Slower convergence than SOR (~2x iterations)
 * - Well-suited for SIMD + multithreading
 *
 * This implementation uses:
 * - AVX2 intrinsics for SIMD vectorization (4 doubles per operation)
 * - OpenMP for thread-level parallelism across rows
 * - Runtime detection to ensure AVX2 is available
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* AVX2 + OpenMP detection
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define JACOBI_HAS_AVX2_OMP 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(JACOBI_HAS_AVX2_OMP)

/* ============================================================================
 * JACOBI AVX2+OMP CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    __m256d dx2_inv_vec;
    __m256d dy2_inv_vec;
    __m256d inv_factor_vec;
    __m256d neg_inv_factor_vec;
    int initialized;
} jacobi_avx2_omp_context_t;

/**
 * Safe conversion from size_t to int for OpenMP loop variables.
 */
static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

/* ============================================================================
 * JACOBI AVX2+OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t jacobi_avx2_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny; (void)params;

    /* Use aligned allocation for struct containing __m256d members */
    jacobi_avx2_omp_context_t* ctx = (jacobi_avx2_omp_context_t*)cfd_aligned_calloc(1, sizeof(jacobi_avx2_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;

    /* Pre-compute SIMD vectors */
    ctx->dx2_inv_vec = _mm256_set1_pd(1.0 / ctx->dx2);
    ctx->dy2_inv_vec = _mm256_set1_pd(1.0 / ctx->dy2);
    ctx->inv_factor_vec = _mm256_set1_pd(ctx->inv_factor);
    ctx->neg_inv_factor_vec = _mm256_set1_pd(-ctx->inv_factor);

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void jacobi_avx2_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_aligned_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t jacobi_avx2_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    if (!x_temp) {
        return CFD_ERROR_INVALID;
    }

    jacobi_avx2_omp_context_t* ctx = (jacobi_avx2_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;

    double* p_old = x;
    double* p_new = x_temp;

    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
    (void)nx_int;  /* suppress unused variable warning */

    /* Process interior rows with OpenMP parallelization */
    #pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        size_t i = 1;

        /* SIMD loop: process 4 doubles at a time */
        for (; i + 4 <= nx - 1; i += 4) {
            size_t idx = (size_t)j * nx + i;

            /* Load neighbors */
            __m256d p_xp = _mm256_loadu_pd(&p_old[idx + 1]);   /* x+1 */
            __m256d p_xm = _mm256_loadu_pd(&p_old[idx - 1]);   /* x-1 */
            __m256d p_yp = _mm256_loadu_pd(&p_old[idx + nx]);  /* y+1 */
            __m256d p_ym = _mm256_loadu_pd(&p_old[idx - nx]);  /* y-1 */
            __m256d rhs_vec = _mm256_loadu_pd(&rhs[idx]);

            /* Sum neighbors in each direction */
            __m256d sum_x = _mm256_add_pd(p_xp, p_xm);
            __m256d sum_y = _mm256_add_pd(p_yp, p_ym);

            /* Divide by dx^2 and dy^2 */
            __m256d term_x = _mm256_mul_pd(sum_x, ctx->dx2_inv_vec);
            __m256d term_y = _mm256_mul_pd(sum_y, ctx->dy2_inv_vec);

            /* Compute: -(rhs - term_x - term_y) * inv_factor */
            __m256d sum_terms = _mm256_add_pd(term_x, term_y);
            __m256d diff = _mm256_sub_pd(rhs_vec, sum_terms);
            __m256d p_result = _mm256_mul_pd(diff, ctx->neg_inv_factor_vec);

            /* Store result */
            _mm256_storeu_pd(&p_new[idx], p_result);
        }

        /* Scalar remainder */
        for (; i < nx - 1; i++) {
            size_t idx = (size_t)j * nx + i;
            double p_result = -(rhs[idx]
                - (p_old[idx + 1] + p_old[idx - 1]) / dx2
                - (p_old[idx + nx] + p_old[idx - nx]) / dy2) * inv_factor;
            p_new[idx] = p_result;
        }
    }

    /* Copy result back to x (parallel) */
    #pragma omp parallel for schedule(static)
    for (j = 0; j < ny_int; j++) {
        memcpy(&x[(size_t)j * nx], &x_temp[(size_t)j * nx], nx * sizeof(double));
    }

    /* Apply boundary conditions (use SIMD+OMP BC if available) */
    bc_apply_scalar_simd_omp(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

#endif /* JACOBI_HAS_AVX2_OMP */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_jacobi_avx2_omp_solver(void) {
#if defined(JACOBI_HAS_AVX2_OMP)
    /* Note: Runtime SIMD check is done by the dispatcher (linear_solver_simd_omp_dispatch.c)
     * before calling this function. No need to check again here. */
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_JACOBI_SIMD_OMP;
    solver->description = "Jacobi iteration (AVX2 + OpenMP)";
    solver->method = POISSON_METHOD_JACOBI;
    solver->backend = POISSON_BACKEND_SIMD_OMP;
    solver->params = poisson_solver_params_default();
    solver->params.max_iterations = 2000;
    solver->params.check_interval = 10;

    solver->init = jacobi_avx2_omp_init;
    solver->destroy = jacobi_avx2_omp_destroy;
    solver->solve = NULL;
    solver->iterate = jacobi_avx2_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
#else
    return NULL;
#endif
}
