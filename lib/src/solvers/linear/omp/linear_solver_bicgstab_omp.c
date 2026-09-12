/**
 * @file linear_solver_bicgstab_omp.c
 * @brief BiCGSTAB solver - OpenMP parallelized implementation
 *
 * Same algorithm as scalar BiCGSTAB (van der Vorst, 1992) but with the vector
 * primitives (dot products, axpy, Laplacian, and the per-iteration vector
 * updates) parallelized across OpenMP threads. The high-level solve loop and
 * all breakdown checks are identical to the scalar reference — only the
 * primitives differ, per the algorithm/primitive separation rule.
 *
 * Each interior element is computed independently and identically regardless of
 * thread count; only the dot-product reductions accumulate in a different order,
 * so OMP results agree with scalar to within reduction rounding.
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"

#include <math.h>

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

#include "linear_solver_primitives_omp.h"

/* ============================================================================
 * BICGSTAB CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_dz2;    /* 1/dz^2 (0 for 2D) */

    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */

    /* BiCGSTAB working vectors (allocated during init) */
    double* r;         /* Residual vector */
    double* r_hat;     /* Shadow residual (typically r_0) */
    double* p;         /* Search direction */
    double* v;         /* A * p */
    double* s;         /* Intermediate residual */
    double* t;         /* A * s */

    int initialized;
} bicgstab_omp_context_t;

/* ============================================================================
 * OMP-PARALLELIZED PRIMITIVES
 *
 * dot_product_omp, axpy_omp, apply_laplacian_omp, compute_residual_omp and
 * copy_vector_omp come from linear_solver_primitives_omp.h (shared with CG and
 * GMRES OMP).
 * ============================================================================ */

/* v = 0 (interior points only) */
static void zero_vector_omp(double* v, size_t nx, size_t ny,
                            size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                v[idx] = 0.0;
            }
        }
    }
}

/*
 * Fused per-iteration vector updates. Each preserves the scalar reference's
 * exact per-element floating-point operation order (only the dot-product
 * reductions differ across threads), so OMP and scalar agree tightly.
 */

/* p = r + beta * (p - omega * v) */
static void update_p_omp(double* p, const double* r, const double* v,
                         double beta, double omega,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                p[idx] = r[idx] + beta * (p[idx] - omega * v[idx]);
            }
        }
    }
}

/* s = r - alpha * v */
static void update_s_omp(double* s, const double* r, const double* v,
                         double alpha,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                s[idx] = r[idx] - alpha * v[idx];
            }
        }
    }
}

/* x = x + alpha * p + omega * s */
static void update_x_omp(double* x, const double* p, const double* s,
                         double alpha, double omega,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                x[idx] += alpha * p[idx] + omega * s[idx];
            }
        }
    }
}

/* r = s - omega * t */
static void update_r_omp(double* r, const double* s, const double* t,
                         double omega,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                r[idx] = s[idx] - omega * t[idx];
            }
        }
    }
}

/* ============================================================================
 * BICGSTAB OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t bicgstab_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    (void)params;

    /* The shared primitives loop over int bounds, which collapse to empty loops
     * for oversized dimensions; reject those grids before allocating. */
    size_t n = 0;
    cfd_status_t size_status = poisson_solver_validate_grid_size(nx, ny, nz, 1, &n);
    if (size_status != CFD_SUCCESS) {
        return size_status;
    }

    bicgstab_omp_context_t* ctx = (bicgstab_omp_context_t*)cfd_calloc(1, sizeof(bicgstab_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    /* Allocate working vectors */
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

static void bicgstab_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        bicgstab_omp_context_t* ctx = (bicgstab_omp_context_t*)solver->context;
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

/*
 * BiCGSTAB solve loop — structurally identical to the scalar reference. It uses
 * its own solve function because it maintains complex state across iterations
 * (rho, omega, alpha), has two convergence checks per iteration, and requires
 * two matrix-vector products per iteration.
 */
static cfd_status_t bicgstab_omp_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;  /* BiCGSTAB doesn't use the temp buffer */

    bicgstab_omp_context_t* ctx = (bicgstab_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_dz2 = ctx->inv_dz2;
    size_t stride_z = ctx->stride_z;
    size_t k_start = ctx->k_start;
    size_t k_end = ctx->k_end;

    double* r = ctx->r;
    double* r_hat = ctx->r_hat;
    double* p = ctx->p;
    double* v = ctx->v;
    double* s = ctx->s;
    double* t = ctx->t;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Apply initial boundary conditions */
    poisson_solver_apply_bc(solver, x);

    /* Compute initial residual: r_0 = b - A*x_0 */
    compute_residual_omp(x, rhs, r, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);

    /* r_hat = r_0 (shadow residual) */
    copy_vector_omp(r, r_hat, nx, ny, k_start, k_end, stride_z);

    /* Initialize: rho = alpha = omega = 1, v = p = 0 */
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    zero_vector_omp(v, nx, ny, k_start, k_end, stride_z);
    zero_vector_omp(p, nx, ny, k_start, k_end, stride_z);

    /* Compute initial residual norm */
    double r_dot_r = dot_product_omp(r, r, nx, ny, k_start, k_end, stride_z);
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
        double rho_new = dot_product_omp(r_hat, r, nx, ny, k_start, k_end, stride_z);

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
        update_p_omp(p, r, v, beta, omega, nx, ny, k_start, k_end, stride_z);

        /* v = A * p */
        apply_laplacian_omp(p, v, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);

        /* alpha = rho_new / (r_hat, v) */
        double r_hat_dot_v = dot_product_omp(r_hat, v, nx, ny, k_start, k_end, stride_z);

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
        update_s_omp(s, r, v, alpha, nx, ny, k_start, k_end, stride_z);

        /* Check for early convergence on s */
        double s_norm = sqrt(dot_product_omp(s, s, nx, ny, k_start, k_end, stride_z));
        if (s_norm < tolerance || s_norm < params->absolute_tolerance) {
            /* Update x and return */
            axpy_omp(alpha, p, x, nx, ny, k_start, k_end, stride_z);
            res_norm = s_norm;
            converged = 1;
            break;
        }

        /* t = A * s */
        apply_laplacian_omp(s, t, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);

        /* omega = (t, s) / (t, t) */
        double t_dot_s = dot_product_omp(t, s, nx, ny, k_start, k_end, stride_z);
        double t_dot_t = dot_product_omp(t, t, nx, ny, k_start, k_end, stride_z);

        /* Check for breakdown */
        if (fabs(t_dot_t) < BICGSTAB_BREAKDOWN_THRESHOLD) {
            /* Update x with available progress */
            axpy_omp(alpha, p, x, nx, ny, k_start, k_end, stride_z);
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
        update_x_omp(x, p, s, alpha, omega, nx, ny, k_start, k_end, stride_z);

        /* r = s - omega * t */
        update_r_omp(r, s, t, omega, nx, ny, k_start, k_end, stride_z);

        /* Update rho for next iteration */
        rho = rho_new;

        /* Compute residual norm */
        res_norm = sqrt(dot_product_omp(r, r, nx, ny, k_start, k_end, stride_z));

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                CFD_LOG_DEBUG("poisson", "BiCGSTAB-OMP Iter %d: residual = %.6e", iter, res_norm);
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
    poisson_solver_apply_bc(solver, x);

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        stats->iterations = (iter < params->max_iterations) ? (iter + 1) : iter;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/*
 * Single iteration is not well-defined for BiCGSTAB (it maintains internal
 * state). Provide a minimal implementation that returns the current residual.
 */
static cfd_status_t bicgstab_omp_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;

    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }
    return CFD_SUCCESS;
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* create_bicgstab_omp_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_BICGSTAB_OMP;
    solver->description = "BiCGSTAB (OpenMP)";
    solver->method = POISSON_METHOD_BICGSTAB;
    solver->backend = POISSON_BACKEND_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = bicgstab_omp_init;
    solver->destroy = bicgstab_omp_destroy;
    solver->solve = bicgstab_omp_solve;  /* BiCGSTAB uses custom solve loop */
    solver->iterate = bicgstab_omp_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}

#endif /* CFD_ENABLE_OPENMP */
