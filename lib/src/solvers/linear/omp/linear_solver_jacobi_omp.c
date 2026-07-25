/**
 * @file linear_solver_jacobi_omp.c
 * @brief Jacobi iteration solver - OpenMP parallelized implementation
 *
 * Same algorithm as scalar Jacobi (double-buffered: reads p_old, writes p_new),
 * with the interior update loop parallelized across OpenMP threads. Jacobi is
 * embarrassingly parallel — each interior update reads only the previous
 * iterate, so there is no cross-cell dependency within a sweep.
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <string.h>

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

/* ============================================================================
 * JACOBI CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_dz2;    /* 1/dz^2 (0 for 2D) */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2 + inv_dz2)) */
    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */
    size_t nz;
    int initialized;
} jacobi_omp_context_t;

/* ============================================================================
 * JACOBI OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t jacobi_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    (void)params;

    jacobi_omp_context_t* ctx = (jacobi_omp_context_t*)cfd_calloc(1, sizeof(jacobi_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    ctx->nz = nz;
    poisson_solver_compute_3d_bounds(nz, nx, ny,
        &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2 + ctx->inv_dz2);
    ctx->inv_factor = 1.0 / factor;
    ctx->initialized = 1;

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void jacobi_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t jacobi_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    if (!x_temp) {
        return CFD_ERROR_INVALID;  /* Jacobi requires temp buffer */
    }

    jacobi_omp_context_t* ctx = (jacobi_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_dz2 = ctx->inv_dz2;
    double inv_factor = ctx->inv_factor;
    size_t stride_z = ctx->stride_z;

    const double* p_old = x;
    double* p_new = x_temp;

    /* Jacobi update: reads from p_old, writes to p_new (no in-place dependency) */
    for (size_t k = ctx->k_start; k < ctx->k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (int i = 1; i < (int)nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);

                double p_result = -(rhs[idx]
                    - (p_old[idx + 1] + p_old[idx - 1]) / dx2
                    - (p_old[idx + nx] + p_old[idx - nx]) / dy2
                    - (p_old[idx + stride_z] + p_old[idx - stride_z]) * inv_dz2
                    ) * inv_factor;

                p_new[idx] = p_result;
            }
        }
    }

    /* Copy result back to x */
    memcpy(x, x_temp, nx * ny * ctx->nz * sizeof(double));

    /* Apply boundary conditions */
    poisson_solver_apply_bc(solver, x);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_jacobi_omp_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_JACOBI_OMP;
    solver->description = "Jacobi iteration (OpenMP)";
    solver->method = POISSON_METHOD_JACOBI;
    solver->backend = POISSON_BACKEND_OMP;
    solver->params = poisson_solver_params_default();
    solver->params.max_iterations = 2000;  /* Jacobi needs more iterations */
    solver->params.check_interval = 10;    /* Check less frequently for speed */

    solver->init = jacobi_omp_init;
    solver->destroy = jacobi_omp_destroy;
    solver->solve = NULL;  /* Use common solve loop */
    solver->iterate = jacobi_omp_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}

#endif /* CFD_ENABLE_OPENMP */
