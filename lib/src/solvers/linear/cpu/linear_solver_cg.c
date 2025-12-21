/**
 * @file linear_solver_cg.c
 * @brief Conjugate Gradient solver - scalar CPU implementation
 *
 * Conjugate Gradient method for solving Ax = b where A is symmetric positive
 * definite (SPD). For the Poisson equation with Neumann BCs, the Laplacian
 * operator is SPD on the space orthogonal to constants.
 *
 * CG characteristics:
 * - Optimal for SPD systems (Poisson with appropriate BCs)
 * - Converges in at most N iterations for N unknowns (in exact arithmetic)
 * - Typically converges in O(sqrt(cond(A))) iterations
 * - Requires extra storage for r, p, Ap vectors
 * - Each iteration: 1 matrix-vector product, 2 dot products, 2 axpy operations
 *
 * Algorithm (standard CG):
 *   r_0 = b - A*x_0
 *   p_0 = r_0
 *   for k = 0, 1, 2, ...
 *     alpha_k = (r_k, r_k) / (p_k, A*p_k)
 *     x_{k+1} = x_k + alpha_k * p_k
 *     r_{k+1} = r_k - alpha_k * A*p_k
 *     beta_k = (r_{k+1}, r_{k+1}) / (r_k, r_k)
 *     p_{k+1} = r_{k+1} + beta_k * p_k
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * CG CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */

    /* CG working vectors (allocated during init) */
    double* r;         /* Residual vector */
    double* p;         /* Search direction */
    double* Ap;        /* A * p (Laplacian applied to p) */

    int initialized;
} cg_context_t;

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Compute dot product of two vectors (interior points only)
 */
static double dot_product(const double* a, const double* b,
                          size_t nx, size_t ny) {
    double sum = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            sum += a[idx] * b[idx];
        }
    }
    return sum;
}

/**
 * Compute y = y + alpha * x (interior points only)
 */
static void axpy(double alpha, const double* x, double* y,
                 size_t nx, size_t ny) {
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            y[idx] += alpha * x[idx];
        }
    }
}

/**
 * Apply negative Laplacian operator: Ap = -nabla^2(p)
 * For Poisson equation: nabla^2(x) = rhs, we solve -nabla^2(x) = -rhs
 * which is SPD with positive eigenvalues.
 */
static void apply_laplacian(const double* p, double* Ap,
                            size_t nx, size_t ny,
                            double dx2, double dy2) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            /* -Laplacian = -(d^2p/dx^2 + d^2p/dy^2) */
            double laplacian = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv
                             + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv;
            Ap[idx] = -laplacian;
        }
    }
}

/**
 * Compute initial residual: r = b - A*x
 * For our formulation: r = -rhs - (-nabla^2 x) = -rhs + nabla^2 x
 * which simplifies to: r = -(rhs - nabla^2 x)
 *
 * To maintain consistency with the standard CG formulation where we solve
 * -nabla^2 x = -rhs (i.e., A = -nabla^2, b = -rhs), we compute:
 * r = b - Ax = -rhs - (-nabla^2 x) = -rhs + nabla^2 x
 */
static void compute_residual(const double* x, const double* rhs, double* r,
                             size_t nx, size_t ny,
                             double dx2, double dy2) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            /* Laplacian(x) = d^2x/dx^2 + d^2x/dy^2 */
            double laplacian = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) * dx2_inv
                             + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) * dy2_inv;

            /* r = -rhs + laplacian = -(rhs - laplacian)
             * This is b - Ax where A = -nabla^2 and b = -rhs */
            r[idx] = -rhs[idx] + laplacian;
        }
    }
}

/**
 * Copy vector p = r (interior points only)
 */
static void copy_vector(const double* src, double* dst,
                        size_t nx, size_t ny) {
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            dst[idx] = src[idx];
        }
    }
}

/* ============================================================================
 * CG SCALAR IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t cg_scalar_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)params;

    cg_context_t* ctx = (cg_context_t*)cfd_calloc(1, sizeof(cg_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Allocate working vectors */
    size_t n = nx * ny;
    ctx->r = (double*)cfd_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_calloc(n, sizeof(double));
    ctx->Ap = (double*)cfd_calloc(n, sizeof(double));

    if (!ctx->r || !ctx->p || !ctx->Ap) {
        cfd_free(ctx->r);
        cfd_free(ctx->p);
        cfd_free(ctx->Ap);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void cg_scalar_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cg_context_t* ctx = (cg_context_t*)solver->context;
        cfd_free(ctx->r);
        cfd_free(ctx->p);
        cfd_free(ctx->Ap);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

/**
 * CG solve function
 *
 * CG uses its own solve loop (not the common one) because:
 * 1. The iterate function doesn't fit the standard pattern
 * 2. CG maintains state across iterations (r_dot_r)
 * 3. Convergence is measured by residual norm, not separate residual computation
 */
static cfd_status_t cg_scalar_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;  /* CG doesn't use the temp buffer */

    cg_context_t* ctx = (cg_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;

    double* r = ctx->r;
    double* p = ctx->p;
    double* Ap = ctx->Ap;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Apply initial boundary conditions */
    bc_apply_scalar(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute initial residual: r_0 = b - A*x_0 */
    compute_residual(x, rhs, r, nx, ny, dx2, dy2);

    /* Initial search direction: p_0 = r_0 */
    copy_vector(r, p, nx, ny);

    /* Compute initial r_dot_r = (r_0, r_0) */
    double r_dot_r = dot_product(r, r, nx, ny);
    double initial_res = sqrt(r_dot_r);

    if (stats) {
        stats->initial_residual = initial_res;
    }

    /* Check if already converged */
    double tolerance = params->tolerance * initial_res;
    if (tolerance < params->absolute_tolerance) {
        tolerance = params->absolute_tolerance;
    }

    if (initial_res < params->absolute_tolerance) {
        if (stats) {
            stats->status = POISSON_CONVERGED;
            stats->iterations = 0;
            stats->final_residual = initial_res;
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return CFD_SUCCESS;
    }

    int converged = 0;
    int iter;
    double res_norm = initial_res;

    for (iter = 0; iter < params->max_iterations; iter++) {
        /* Compute Ap = A * p */
        apply_laplacian(p, Ap, nx, ny, dx2, dy2);

        /* alpha = (r, r) / (p, Ap) */
        double p_dot_Ap = dot_product(p, Ap, nx, ny);

        /* Check for breakdown (p_dot_Ap should be positive for SPD) */
        if (fabs(p_dot_Ap) < CG_BREAKDOWN_THRESHOLD) {
            /* Stagnation or breakdown */
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        double alpha = r_dot_r / p_dot_Ap;

        /* x_{k+1} = x_k + alpha * p */
        axpy(alpha, p, x, nx, ny);

        /* r_{k+1} = r_k - alpha * Ap */
        axpy(-alpha, Ap, r, nx, ny);

        /* Compute new r_dot_r */
        double r_dot_r_new = dot_product(r, r, nx, ny);
        res_norm = sqrt(r_dot_r_new);

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                printf("  CG Iter %d: residual = %.6e\n", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        /* Check for breakdown in r_dot_r before computing beta */
        if (fabs(r_dot_r) < CG_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        /* beta = (r_{k+1}, r_{k+1}) / (r_k, r_k) */
        double beta = r_dot_r_new / r_dot_r;

        /* p_{k+1} = r_{k+1} + beta * p_k */
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                p[idx] = r[idx] + beta * p[idx];
            }
        }

        r_dot_r = r_dot_r_new;
    }

    /* Final convergence check (in case we converged between check intervals) */
    if (!converged && (res_norm < tolerance || res_norm < params->absolute_tolerance)) {
        converged = 1;
    }

    /* Apply final boundary conditions */
    bc_apply_scalar(x, nx, ny, BC_TYPE_NEUMANN);

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        /* When loop breaks early (converged), iter is the last iteration index, so +1.
         * When loop completes naturally, iter == max_iterations, report as-is. */
        stats->iterations = (iter < params->max_iterations) ? (iter + 1) : iter;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/**
 * Single iteration is not well-defined for CG as it maintains internal state.
 * We provide a minimal implementation that returns the current residual.
 */
static cfd_status_t cg_scalar_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    /* CG doesn't support single iteration mode well.
     * Return the current residual from the Laplacian. */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }
    return CFD_SUCCESS;
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_cg_scalar_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_CG_SCALAR;
    solver->description = "Conjugate Gradient (scalar CPU)";
    solver->method = POISSON_METHOD_CG;
    solver->backend = POISSON_BACKEND_SCALAR;
    solver->params = poisson_solver_params_default();

    solver->init = cg_scalar_init;
    solver->destroy = cg_scalar_destroy;
    solver->solve = cg_scalar_solve;  /* CG uses custom solve loop */
    solver->iterate = cg_scalar_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}
