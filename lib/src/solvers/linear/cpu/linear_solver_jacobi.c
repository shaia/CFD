/**
 * @file linear_solver_jacobi.c
 * @brief Jacobi iteration solver - scalar CPU implementation
 *
 * Jacobi method characteristics:
 * - Fully parallelizable (reads from old array, writes to new array)
 * - Requires temporary buffer for double-buffering
 * - Slower convergence than SOR (~2x iterations)
 * - Better suited for GPU/massively parallel execution
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

/* ============================================================================
 * JACOBI CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    int initialized;
} jacobi_context_t;

/* ============================================================================
 * JACOBI SCALAR IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t jacobi_scalar_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny; (void)params;

    jacobi_context_t* ctx = (jacobi_context_t*)cfd_calloc(1, sizeof(jacobi_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;
    ctx->initialized = 1;

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void jacobi_scalar_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t jacobi_scalar_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    if (!x_temp) {
        return CFD_ERROR_INVALID;  /* Jacobi requires temp buffer */
    }

    jacobi_context_t* ctx = (jacobi_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;

    double* p_old = x;
    double* p_new = x_temp;

    /* Jacobi update: reads from p_old, writes to p_new */
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            double p_result = -(rhs[idx]
                - (p_old[idx + 1] + p_old[idx - 1]) / dx2
                - (p_old[idx + nx] + p_old[idx - nx]) / dy2) * inv_factor;

            p_new[idx] = p_result;
        }
    }

    /* Copy result back to x */
    memcpy(x, x_temp, nx * ny * sizeof(double));

    /* Apply boundary conditions */
    bc_apply_scalar(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute residual if requested */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_jacobi_scalar_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_JACOBI_SCALAR;
    solver->description = "Jacobi iteration (scalar CPU)";
    solver->method = POISSON_METHOD_JACOBI;
    solver->backend = POISSON_BACKEND_SCALAR;
    solver->params = poisson_solver_params_default();
    solver->params.max_iterations = 2000;  /* Jacobi needs more iterations */
    solver->params.check_interval = 10;    /* Check less frequently for speed */

    solver->init = jacobi_scalar_init;
    solver->destroy = jacobi_scalar_destroy;
    solver->solve = NULL;  /* Use common solve loop */
    solver->iterate = jacobi_scalar_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}
