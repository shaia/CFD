/**
 * @file linear_solver_redblack_omp.c
 * @brief Red-Black SOR solver - OpenMP parallel implementation
 *
 * Red-Black SOR characteristics:
 * - Two-color sweep (red then black) allows parallelization
 * - In-place updates (no temporary buffer needed)
 * - SOR acceleration (omega > 1) for faster convergence
 * - Best balance of convergence speed and parallelism
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"

#include <math.h>

#ifdef CFD_ENABLE_OPENMP
    #include <omp.h>
#endif

/* ============================================================================
 * RED-BLACK CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_dx2;    /* 1/dx^2 */
    double inv_dy2;    /* 1/dy^2 */
    double inv_dz2;    /* 1/dz^2 (0 for 2D) */
    double factor;     /* 2 * (1/dx^2 + 1/dy^2 + inv_dz2) */
    double inv_factor; /* 1 / factor */
    double omega;      /* SOR relaxation parameter */
    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */
    int initialized;
} redblack_omp_context_t;

/* ============================================================================
 * RED-BLACK OPENMP IMPLEMENTATION
 * ============================================================================ */

#ifdef CFD_ENABLE_OPENMP

static cfd_status_t redblack_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    redblack_omp_context_t* ctx = (redblack_omp_context_t*)cfd_calloc(1, sizeof(redblack_omp_context_t));
    if (!ctx) {
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
    ctx->initialized = 1;

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void redblack_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t redblack_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    redblack_omp_context_t* ctx = (redblack_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t nz = solver->nz;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_dz2 = ctx->inv_dz2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;
    size_t stride_z = ctx->stride_z;
    size_t k_start = ctx->k_start;
    size_t k_end = ctx->k_end;
    int walls = poisson_solver_uses_default_walls(solver);

    /* Red sweep (parallel) */
    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            double w_row, w_edge;
            poisson_solver_row_omegas(walls, omega, ctx->factor, nx, ny, nz, (size_t)j, k,
                                      ctx->inv_dx2, ctx->inv_dy2, inv_dz2, &w_row, &w_edge);
            size_t i_start = ((j + k) % 2 == 0) ? 1 : 2;
            size_t i;
            for (i = i_start; i < nx - 1; i += 2) {
                size_t idx = k * stride_z + IDX_2D(i, (size_t)j, nx);

                double p_new = -(rhs[idx]
                    - (x[idx + 1] + x[idx - 1]) / dx2
                    - (x[idx + nx] + x[idx - nx]) / dy2
                    - (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                    ) * inv_factor;

                double w = (i == 1 || i == nx - 2) ? w_edge : w_row;
                x[idx] = x[idx] + w * (p_new - x[idx]);
            }
        }
    }

    /* Black sweep (parallel) */
    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            double w_row, w_edge;
            poisson_solver_row_omegas(walls, omega, ctx->factor, nx, ny, nz, (size_t)j, k,
                                      ctx->inv_dx2, ctx->inv_dy2, inv_dz2, &w_row, &w_edge);
            size_t i_start = ((j + k) % 2 == 0) ? 2 : 1;
            size_t i;
            for (i = i_start; i < nx - 1; i += 2) {
                size_t idx = k * stride_z + IDX_2D(i, (size_t)j, nx);

                double p_new = -(rhs[idx]
                    - (x[idx + 1] + x[idx - 1]) / dx2
                    - (x[idx + nx] + x[idx - nx]) / dy2
                    - (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
                    ) * inv_factor;

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

#endif /* CFD_ENABLE_OPENMP */

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_redblack_omp_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_REDBLACK_OMP;
    solver->description = "Red-Black SOR iteration (OpenMP parallel)";
    solver->method = POISSON_METHOD_REDBLACK_SOR;
    solver->backend = POISSON_BACKEND_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = redblack_omp_init;
    solver->destroy = redblack_omp_destroy;
    solver->solve = NULL;
    solver->iterate = redblack_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
}
#endif
