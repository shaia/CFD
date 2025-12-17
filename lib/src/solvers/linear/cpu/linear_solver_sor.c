/**
 * @file linear_solver_sor.c
 * @brief Sequential SOR (Successive Over-Relaxation) solver - scalar CPU implementation
 *
 * SOR characteristics:
 * - Inherently sequential (Gauss-Seidel + relaxation)
 * - In-place updates (no temporary buffer needed)
 * - Fastest convergence per iteration
 * - Cannot be parallelized (unlike Red-Black SOR)
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"

#include <math.h>

/* ============================================================================
 * SOR CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_factor; /* 1 / (2 * (1/dx^2 + 1/dy^2)) */
    double omega;      /* SOR relaxation parameter */
    int initialized;
} sor_context_t;

/* ============================================================================
 * SOR SCALAR IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t sor_scalar_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)nx; (void)ny;

    sor_context_t* ctx = (sor_context_t*)cfd_calloc(1, sizeof(sor_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    double factor = 2.0 * (1.0 / ctx->dx2 + 1.0 / ctx->dy2);
    ctx->inv_factor = 1.0 / factor;
    ctx->omega = params ? params->omega : 1.5;
    ctx->initialized = 1;

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void sor_scalar_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

/**
 * Sequential SOR iteration
 *
 * Updates cells in row-major order, using already-updated values
 * from the current iteration (Gauss-Seidel behavior).
 */
static cfd_status_t sor_scalar_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;  /* Not needed for in-place SOR */

    sor_context_t* ctx = (sor_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_factor = ctx->inv_factor;
    double omega = ctx->omega;

    /* Single sweep: row-major order */
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            /* Compute Gauss-Seidel update
             * Note: x[idx-1] and x[idx-nx] are already updated this iteration
             */
            double p_new = -(rhs[idx]
                - (x[idx + 1] + x[idx - 1]) / dx2
                - (x[idx + nx] + x[idx - nx]) / dy2) * inv_factor;

            /* SOR relaxation */
            x[idx] = x[idx] + omega * (p_new - x[idx]);
        }
    }

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

poisson_solver_t* create_sor_scalar_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_SOR_SCALAR;
    solver->description = "SOR iteration (scalar CPU, sequential)";
    solver->method = POISSON_METHOD_SOR;
    solver->backend = POISSON_BACKEND_SCALAR;
    solver->params = poisson_solver_params_default();

    solver->init = sor_scalar_init;
    solver->destroy = sor_scalar_destroy;
    solver->solve = NULL;  /* Use common solve loop */
    solver->iterate = sor_scalar_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}
