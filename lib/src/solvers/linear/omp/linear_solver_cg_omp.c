/**
 * @file linear_solver_cg_omp.c
 * @brief Conjugate Gradient solver - OpenMP parallelized implementation
 *
 * Same algorithm as scalar CG but with OpenMP-parallelized primitives
 * (dot_product, axpy, apply_laplacian, etc.). The multigrid preconditioner
 * (POISSON_PRECOND_MULTIGRID) runs its V-cycle on the OpenMP multigrid backend.
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
 * CG CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;
    double dy2;
    double inv_dz2;    /* 1/dz^2 (0 for 2D) */
    double diag_inv;

    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */

    double* r;
    double* z;         /* Preconditioned residual (NULL if no precond) */
    double* p;
    double* Ap;

    poisson_solver_t* mg_precond;        /* Inner OpenMP MG V-cycle solver (NULL unless
                                            POISSON_PRECOND_MULTIGRID) */
    poisson_precond_type_t precond_type; /* Which preconditioner use_precond refers to */
    int use_precond;
    int initialized;
} cg_omp_context_t;

/* ============================================================================
 * OMP-PARALLELIZED PRIMITIVES
 *
 * dot_product_omp, axpy_omp, scale_vector_omp, apply_laplacian_omp,
 * compute_residual_omp, copy_vector_omp and apply_jacobi_precond_omp come from
 * linear_solver_primitives_omp.h (shared with GMRES OMP).
 * ============================================================================ */

static void update_search_direction_omp(const double* src, double* p,
                                        double beta, size_t nx, size_t ny,
                                        size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = poisson_solver_size_to_int(ny);
    int nx_int = poisson_solver_size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                p[idx] = src[idx] + beta * p[idx];
            }
        }
    }
}

/**
 * Apply multigrid preconditioner: z = M^{-1} * r for A = -nabla^2.
 *
 * Same steps as apply_mg_precond in the scalar CG solver: one V-cycle on
 * nabla^2 z' = r from a zero guess, then z = -z'. The OpenMP multigrid cycle is
 * bit-identical to the scalar one and the sign flip is exact, so both backends
 * apply the same M. The NULL residual argument skips the serial residual
 * evaluation after the cycle.
 */
static cfd_status_t apply_mg_precond_omp(poisson_solver_t* mg, const double* r, double* z,
                                         size_t n_total, size_t nx, size_t ny,
                                         size_t k_start, size_t k_end, size_t stride_z) {
    memset(z, 0, n_total * sizeof(double));

    cfd_status_t status = poisson_solver_iterate(mg, z, NULL, r, NULL);
    if (status != CFD_SUCCESS) {
        return status;
    }

    scale_vector_omp(-1.0, z, nx, ny, k_start, k_end, stride_z);
    return CFD_SUCCESS;
}

/* ============================================================================
 * CG OMP IMPLEMENTATION
 * ============================================================================ */

static void cg_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        cg_omp_context_t* ctx = (cg_omp_context_t*)solver->context;
        poisson_solver_destroy(ctx->mg_precond);
        cfd_free(ctx->r);
        cfd_free(ctx->z);
        cfd_free(ctx->p);
        cfd_free(ctx->Ap);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t cg_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    /* The shared primitives loop over int bounds, which collapse to empty loops
     * for oversized dimensions; reject those grids before allocating. */
    size_t n = 0;
    cfd_status_t size_status = poisson_solver_validate_grid_size(nx, ny, nz, 1, &n);
    if (size_status != CFD_SUCCESS) {
        return size_status;
    }

    cg_omp_context_t* ctx = (cg_omp_context_t*)cfd_calloc(1, sizeof(cg_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }
    /* Attached before anything else is allocated, so every failure below
     * releases what was built through cg_omp_destroy */
    solver->context = ctx;

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);
    ctx->diag_inv = 1.0 / (2.0 / ctx->dx2 + 2.0 / ctx->dy2 + 2.0 * ctx->inv_dz2);
    ctx->precond_type = params ? params->preconditioner : POISSON_PRECOND_NONE;
    ctx->use_precond = (ctx->precond_type == POISSON_PRECOND_JACOBI ||
                        ctx->precond_type == POISSON_PRECOND_MULTIGRID);

    ctx->r = (double*)cfd_calloc(n, sizeof(double));
    ctx->p = (double*)cfd_calloc(n, sizeof(double));
    ctx->Ap = (double*)cfd_calloc(n, sizeof(double));
    if (ctx->use_precond) {
        ctx->z = (double*)cfd_calloc(n, sizeof(double));
    }

    if (!ctx->r || !ctx->p || !ctx->Ap || (ctx->use_precond && !ctx->z)) {
        cg_omp_destroy(solver);
        return CFD_ERROR_NOMEM;
    }

    /* OpenMP multigrid only: a scalar inner cycle would serialize every apply */
    if (ctx->precond_type == POISSON_PRECOND_MULTIGRID) {
        cfd_status_t mg_status = poisson_solver_create_mg_precond(
            create_multigrid_omp_solver, nx, ny, nz, dx, dy, dz, &ctx->mg_precond);
        if (mg_status != CFD_SUCCESS) {
            cg_omp_destroy(solver);
            return mg_status;  /* CFD_ERROR_INVALID for non-2^k+1 dims */
        }
    }

    ctx->initialized = 1;
    return CFD_SUCCESS;
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
    size_t n_total = nx * ny * solver->nz;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_dz2 = ctx->inv_dz2;
    size_t k_start = ctx->k_start;
    size_t k_end = ctx->k_end;
    size_t stride_z = ctx->stride_z;

    double* r = ctx->r;
    double* z = ctx->z;
    double* p = ctx->p;
    double* Ap = ctx->Ap;
    int use_precond = ctx->use_precond;
    double diag_inv = ctx->diag_inv;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Not poisson_solver_apply_bc: the initial residual has to see the same walls
     * the iteration below inverts. See poisson_solver_krylov_apply_bc. */
    poisson_solver_krylov_apply_bc(solver, x);

    compute_residual_omp(x, rhs, r, nx, ny, dx2, dy2, inv_dz2,
                         k_start, k_end, stride_z);

    double initial_res = sqrt(dot_product_omp(r, r, nx, ny,
                                              k_start, k_end, stride_z));

    double rho;
    if (use_precond) {
        if (ctx->precond_type == POISSON_PRECOND_MULTIGRID) {
            cfd_status_t precond_status = apply_mg_precond_omp(
                ctx->mg_precond, r, z, n_total, nx, ny, k_start, k_end, stride_z);
            if (precond_status != CFD_SUCCESS) {
                if (stats) {
                    stats->status = POISSON_ERROR;
                    stats->iterations = 0;
                    stats->final_residual = initial_res;
                    stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
                }
                return precond_status;
            }
        } else {
            apply_jacobi_precond_omp(r, z, nx, ny, diag_inv,
                                     k_start, k_end, stride_z);
        }
        copy_vector_omp(z, p, nx, ny, k_start, k_end, stride_z);
        rho = dot_product_omp(r, z, nx, ny, k_start, k_end, stride_z);
    } else {
        copy_vector_omp(r, p, nx, ny, k_start, k_end, stride_z);
        rho = dot_product_omp(r, r, nx, ny, k_start, k_end, stride_z);
    }

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
        /* The halo carries the homogeneous boundary condition, which is what makes
         * this the operator the walls describe. The direction is rebuilt from
         * interior-only updates, so it has to be reapplied every iteration. */
        poisson_solver_krylov_apply_bc_homogeneous(solver, p);
        apply_laplacian_omp(p, Ap, nx, ny, dx2, dy2, inv_dz2,
                            k_start, k_end, stride_z);

        double p_dot_Ap = dot_product_omp(p, Ap, nx, ny,
                                          k_start, k_end, stride_z);
        CG_CHECK_BREAKDOWN(p_dot_Ap, stats, iter, res_norm, start_time);

        double alpha = rho / p_dot_Ap;

        axpy_omp(alpha, p, x, nx, ny, k_start, k_end, stride_z);
        axpy_omp(-alpha, Ap, r, nx, ny, k_start, k_end, stride_z);

        double rho_new;
        if (use_precond) {
            if (ctx->precond_type == POISSON_PRECOND_MULTIGRID) {
                cfd_status_t precond_status = apply_mg_precond_omp(
                    ctx->mg_precond, r, z, n_total, nx, ny, k_start, k_end, stride_z);
                if (precond_status != CFD_SUCCESS) {
                    if (stats) {
                        stats->status = POISSON_ERROR;
                        stats->iterations = iter + 1;
                        stats->final_residual = res_norm;
                        stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
                    }
                    return precond_status;
                }
            } else {
                apply_jacobi_precond_omp(r, z, nx, ny, diag_inv,
                                         k_start, k_end, stride_z);
            }
            rho_new = dot_product_omp(r, z, nx, ny,
                                      k_start, k_end, stride_z);
        } else {
            rho_new = dot_product_omp(r, r, nx, ny,
                                      k_start, k_end, stride_z);
        }

        res_norm = sqrt(dot_product_omp(r, r, nx, ny,
                                        k_start, k_end, stride_z));

        if (iter % params->check_interval == 0) {
            if (params->verbose) {
                CFD_LOG_DEBUG("poisson", "CG-OMP Iter %d: residual = %.6e", iter, res_norm);
            }

            if (res_norm < tolerance || res_norm < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }

        CG_CHECK_BREAKDOWN(rho, stats, iter, res_norm, start_time);

        double beta = rho_new / rho;
        update_search_direction_omp(use_precond ? z : r, p, beta, nx, ny,
                                    k_start, k_end, stride_z);

        rho = rho_new;
    }

    if (!converged && (res_norm < tolerance || res_norm < params->absolute_tolerance)) {
        converged = 1;
    }

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
