/**
 * @file linear_solver_cg_omp.c
 * @brief Conjugate Gradient solver - OpenMP parallelized implementation
 *
 * Same algorithm as scalar CG but with OpenMP-parallelized primitives
 * (dot_product, axpy, apply_laplacian, etc.).
 */

#include "../linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

/* ============================================================================
 * CG CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;
    double dy2;
    double diag_inv;

    double* r;
    double* z;
    double* p;
    double* Ap;

    int use_precond;
    int initialized;
} cg_omp_context_t;

/* ============================================================================
 * OMP-PARALLELIZED PRIMITIVES
 * ============================================================================ */

static inline int size_to_int(size_t val) {
    if (val > (size_t)INT_MAX) {
        return 0;
    }
    return (int)val;
}

static double dot_product_omp(const double* a, const double* b,
                              size_t nx, size_t ny) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static) reduction(+:sum)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            sum += a[idx] * b[idx];
        }
    }
    return sum;
}

static void axpy_omp(double alpha, const double* x, double* y,
                     size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            y[idx] += alpha * x[idx];
        }
    }
}

static void apply_laplacian_omp(const double* p, double* Ap,
                                size_t nx, size_t ny,
                                double dx2, double dy2) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            double laplacian = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv
                             + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv;
            Ap[idx] = -laplacian;
        }
    }
}

static void compute_residual_omp(const double* x, const double* rhs, double* r,
                                 size_t nx, size_t ny,
                                 double dx2, double dy2) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            double laplacian = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) * dx2_inv
                             + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) * dy2_inv;
            r[idx] = -rhs[idx] + laplacian;
        }
    }
}

static void copy_vector_omp(const double* src, double* dst,
                            size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        memcpy(&dst[j * nx + 1], &src[j * nx + 1], (nx - 2) * sizeof(double));
    }
}

static void apply_jacobi_precond_omp(const double* r, double* z,
                                     size_t nx, size_t ny,
                                     double diag_inv) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            z[idx] = diag_inv * r[idx];
        }
    }
}

static void update_search_direction_omp(const double* src, double* p,
                                        double beta, size_t nx, size_t ny) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);
    int j;
#pragma omp parallel for schedule(static)
    for (j = 1; j < ny_int - 1; j++) {
        for (int i = 1; i < nx_int - 1; i++) {
            size_t idx = (size_t)j * nx + (size_t)i;
            p[idx] = src[idx] + beta * p[idx];
        }
    }
}

/* ============================================================================
 * CG OMP IMPLEMENTATION
 * ============================================================================ */

static cfd_status_t cg_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    cg_omp_context_t* ctx = (cg_omp_context_t*)cfd_calloc(1, sizeof(cg_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->diag_inv = 1.0 / (2.0 / ctx->dx2 + 2.0 / ctx->dy2);
    ctx->use_precond = (params && params->preconditioner == POISSON_PRECOND_JACOBI);

    size_t n = nx * ny;
    ctx->r = (double*)cfd_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_calloc(n, sizeof(double));
    ctx->Ap = (double*)cfd_calloc(n, sizeof(double));
    ctx->z = NULL;

    if (!ctx->r || !ctx->p || !ctx->Ap) {
        cfd_free(ctx->r);
        cfd_free(ctx->p);
        cfd_free(ctx->Ap);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    if (ctx->use_precond) {
        ctx->z = (double*)cfd_calloc(n, sizeof(double));
        if (!ctx->z) {
            cfd_free(ctx->r);
            cfd_free(ctx->p);
            cfd_free(ctx->Ap);
            cfd_free(ctx);
            return CFD_ERROR_NOMEM;
        }
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void cg_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cg_omp_context_t* ctx = (cg_omp_context_t*)solver->context;
        cfd_free(ctx->r);
        cfd_free(ctx->z);
        cfd_free(ctx->p);
        cfd_free(ctx->Ap);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t cg_omp_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;

    cg_omp_context_t* ctx = (cg_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;

    double* r = ctx->r;
    double* z = ctx->z;
    double* p = ctx->p;
    double* Ap = ctx->Ap;
    int use_precond = ctx->use_precond;
    double diag_inv = ctx->diag_inv;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    bc_apply_scalar_omp(x, nx, ny, BC_TYPE_NEUMANN);

    compute_residual_omp(x, rhs, r, nx, ny, dx2, dy2);

    double rho;
    if (use_precond) {
        apply_jacobi_precond_omp(r, z, nx, ny, diag_inv);
        copy_vector_omp(z, p, nx, ny);
        rho = dot_product_omp(r, z, nx, ny);
    } else {
        copy_vector_omp(r, p, nx, ny);
        rho = dot_product_omp(r, r, nx, ny);
    }

    double initial_res = sqrt(dot_product_omp(r, r, nx, ny));

    if (stats) {
        stats->initial_residual = initial_res;
    }

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
        apply_laplacian_omp(p, Ap, nx, ny, dx2, dy2);

        double p_dot_Ap = dot_product_omp(p, Ap, nx, ny);
        CG_CHECK_BREAKDOWN(p_dot_Ap, stats, iter, res_norm, start_time);

        double alpha = rho / p_dot_Ap;

        axpy_omp(alpha, p, x, nx, ny);
        axpy_omp(-alpha, Ap, r, nx, ny);

        double rho_new;
        if (use_precond) {
            apply_jacobi_precond_omp(r, z, nx, ny, diag_inv);
            rho_new = dot_product_omp(r, z, nx, ny);
        } else {
            rho_new = dot_product_omp(r, r, nx, ny);
        }

        res_norm = sqrt(dot_product_omp(r, r, nx, ny));

        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                printf("  CG-OMP Iter %d: residual = %.6e\n", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        CG_CHECK_BREAKDOWN(rho, stats, iter, res_norm, start_time);

        double beta = rho_new / rho;
        update_search_direction_omp(use_precond ? z : r, p, beta, nx, ny);

        rho = rho_new;
    }

    if (!converged && (res_norm < tolerance || res_norm < params->absolute_tolerance)) {
        converged = 1;
    }

    bc_apply_scalar_omp(x, nx, ny, BC_TYPE_NEUMANN);

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        stats->iterations = (iter < params->max_iterations) ? (iter + 1) : iter;
        stats->final_residual = res_norm;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

static cfd_status_t cg_omp_iterate(
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

poisson_solver_t* create_cg_omp_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_CG_OMP;
    solver->description = "Conjugate Gradient (OpenMP)";
    solver->method = POISSON_METHOD_CG;
    solver->backend = POISSON_BACKEND_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = cg_omp_init;
    solver->destroy = cg_omp_destroy;
    solver->solve = cg_omp_solve;
    solver->iterate = cg_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
}

#endif /* CFD_ENABLE_OPENMP */
