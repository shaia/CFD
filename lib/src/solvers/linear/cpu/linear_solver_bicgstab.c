/**
 * @file linear_solver_bicgstab.c
 * @brief BiCGSTAB solver - scalar CPU implementation
 *
 * Biconjugate Gradient Stabilized (BiCGSTAB) method for solving Ax = b where A
 * may be non-symmetric. This is useful for advection-dominated problems where
 * the discretized operator is not symmetric.
 *
 * BiCGSTAB characteristics:
 * - Works for non-symmetric matrices (unlike CG which requires SPD)
 * - Smoother convergence than BiCG (avoids irregular convergence behavior)
 * - Requires 2 matrix-vector products per iteration
 * - Requires 4 inner products and 4 axpy operations per iteration
 * - Needs storage for 6 vectors: r, r_hat, p, v, s, t
 *
 * Algorithm (van der Vorst, 1992):
 *   r_0 = b - A*x_0
 *   r_hat = r_0 (arbitrary, typically r_0)
 *   rho_0 = alpha = omega = 1
 *   v_0 = p_0 = 0
 *
 *   for k = 1, 2, 3, ...
 *     rho_k = (r_hat, r_{k-1})
 *     beta = (rho_k / rho_{k-1}) * (alpha / omega)
 *     p_k = r_{k-1} + beta * (p_{k-1} - omega * v_{k-1})
 *     v_k = A * p_k
 *     alpha = rho_k / (r_hat, v_k)
 *     s = r_{k-1} - alpha * v_k
 *     t = A * s
 *     omega = (t, s) / (t, t)
 *     x_k = x_{k-1} + alpha * p_k + omega * s
 *     r_k = s - omega * t
 *
 * Note: For the Poisson equation with symmetric BCs, CG is preferred.
 * BiCGSTAB is provided for future use with non-symmetric operators.
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * BICGSTAB CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */

    /* BiCGSTAB working vectors (allocated during init) */
    double* r;         /* Residual vector */
    double* r_hat;     /* Shadow residual (typically r_0) */
    double* p;         /* Search direction */
    double* v;         /* A * p */
    double* s;         /* Intermediate residual */
    double* t;         /* A * s */

    int initialized;
} bicgstab_context_t;

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

            /* r = b - Ax = -rhs - (-laplacian) = -rhs + laplacian */
            r[idx] = -rhs[idx] + laplacian;
        }
    }
}

/**
 * Copy vector: dst = src (interior points only)
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

/**
 * Set vector to zero (interior points only)
 */
static void zero_vector(double* v, size_t nx, size_t ny) {
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            v[idx] = 0.0;
        }
    }
}

/* ============================================================================
 * BICGSTAB SCALAR IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t bicgstab_scalar_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    (void)params;

    bicgstab_context_t* ctx = (bicgstab_context_t*)cfd_calloc(1, sizeof(bicgstab_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;

    /* Allocate working vectors */
    size_t n = nx * ny;
    ctx->r = (double*)cfd_calloc(n, sizeof(double));
    ctx->r_hat = (double*)cfd_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_calloc(n, sizeof(double));
    ctx->v = (double*)cfd_calloc(n, sizeof(double));
    ctx->s = (double*)cfd_calloc(n, sizeof(double));
    ctx->t = (double*)cfd_calloc(n, sizeof(double));

    if (!ctx->r || !ctx->r_hat || !ctx->p || !ctx->v || !ctx->s || !ctx->t) {
        cfd_free(ctx->r);
        cfd_free(ctx->r_hat);
        cfd_free(ctx->p);
        cfd_free(ctx->v);
        cfd_free(ctx->s);
        cfd_free(ctx->t);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void bicgstab_scalar_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        bicgstab_context_t* ctx = (bicgstab_context_t*)solver->context;
        cfd_free(ctx->r);
        cfd_free(ctx->r_hat);
        cfd_free(ctx->p);
        cfd_free(ctx->v);
        cfd_free(ctx->s);
        cfd_free(ctx->t);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

/**
 * BiCGSTAB solve function
 *
 * BiCGSTAB uses its own solve loop because:
 * 1. It maintains complex state across iterations (rho, omega, alpha)
 * 2. Has two convergence checks per iteration (after s and after r)
 * 3. Requires two matrix-vector products per iteration
 */
static cfd_status_t bicgstab_scalar_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;  /* BiCGSTAB doesn't use the temp buffer */

    bicgstab_context_t* ctx = (bicgstab_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;

    double* r = ctx->r;
    double* r_hat = ctx->r_hat;
    double* p = ctx->p;
    double* v = ctx->v;
    double* s = ctx->s;
    double* t = ctx->t;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Apply initial boundary conditions */
    bc_apply_scalar(x, nx, ny, BC_TYPE_NEUMANN);

    /* Compute initial residual: r_0 = b - A*x_0 */
    compute_residual(x, rhs, r, nx, ny, dx2, dy2);

    /* r_hat = r_0 (shadow residual) */
    copy_vector(r, r_hat, nx, ny);

    /* Initialize: rho = alpha = omega = 1, v = p = 0 */
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    zero_vector(v, nx, ny);
    zero_vector(p, nx, ny);

    /* Compute initial residual norm */
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
        /* rho_new = (r_hat, r) */
        double rho_new = dot_product(r_hat, r, nx, ny);

        /* Check for breakdown */
        if (fabs(rho_new) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        /* beta = (rho_new / rho) * (alpha / omega) */
        double beta = (rho_new / rho) * (alpha / omega);

        /* p = r + beta * (p - omega * v) */
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                p[idx] = r[idx] + beta * (p[idx] - omega * v[idx]);
            }
        }

        /* v = A * p */
        apply_laplacian(p, v, nx, ny, dx2, dy2);

        /* alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = dot_product(r_hat, v, nx, ny);

        /* Check for breakdown */
        if (fabs(r_hat_dot_v) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        alpha = rho_new / r_hat_dot_v;

        /* s = r - alpha * v */
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                s[idx] = r[idx] - alpha * v[idx];
            }
        }

        /* Check for early convergence on s */
        double s_norm = sqrt(dot_product(s, s, nx, ny));
        if (s_norm < tolerance || s_norm < params->absolute_tolerance) {
            /* Update x and return */
            axpy(alpha, p, x, nx, ny);
            res_norm = s_norm;
            converged = 1;
            break;
        }

        /* t = A * s */
        apply_laplacian(s, t, nx, ny, dx2, dy2);

        /* omega = (t, s) / (t, t) */
        double t_dot_s = dot_product(t, s, nx, ny);
        double t_dot_t = dot_product(t, t, nx, ny);

        /* Check for breakdown */
        if (fabs(t_dot_t) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            /* Update x with available progress */
            axpy(alpha, p, x, nx, ny);
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = s_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }

        omega = t_dot_s / t_dot_t;

        /* x = x + alpha * p + omega * s */
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                x[idx] += alpha * p[idx] + omega * s[idx];
            }
        }

        /* r = s - omega * t */
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = j * nx + i;
                r[idx] = s[idx] - omega * t[idx];
            }
        }

        /* Update rho for next iteration */
        rho = rho_new;

        /* Compute residual norm */
        res_norm = sqrt(dot_product(r, r, nx, ny));

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                printf("  BiCGSTAB Iter %d: residual = %.6e\n", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        /* Check for omega breakdown (would cause division by zero next iteration) */
        if (fabs(omega) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            if (stats) {
                stats->status = POISSON_STAGNATED;
                stats->iterations = iter + 1;
                stats->final_residual = res_norm;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return CFD_ERROR_MAX_ITER;
        }
    }

    /* Final convergence check */
    if (!converged && (res_norm < tolerance || res_norm < params->absolute_tolerance)) {
        converged = 1;
    }

    /* Apply final boundary conditions */
    bc_apply_scalar(x, nx, ny, BC_TYPE_NEUMANN);

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        stats->iterations = (iter < params->max_iterations) ? (iter + 1) : iter;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/**
 * Single iteration is not well-defined for BiCGSTAB as it maintains internal state.
 * We provide a minimal implementation that returns the current residual.
 */
static cfd_status_t bicgstab_scalar_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    /* BiCGSTAB doesn't support single iteration mode well.
     * Return the current residual from the Laplacian. */
    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }
    return CFD_SUCCESS;
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_bicgstab_scalar_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_SCALAR;
    solver->description = "BiCGSTAB (scalar CPU)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_SCALAR;
    solver->params = poisson_solver_params_default();

    solver->init = bicgstab_scalar_init;
    solver->destroy = bicgstab_scalar_destroy;
    solver->solve = bicgstab_scalar_solve;  /* BiCGSTAB uses custom solve loop */
    solver->iterate = bicgstab_scalar_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}
