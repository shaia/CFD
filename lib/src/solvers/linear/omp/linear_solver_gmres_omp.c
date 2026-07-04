/**
 * @file linear_solver_gmres_omp.c
 * @brief Restarted GMRES(m) solver - OpenMP parallelized implementation
 *
 * Same algorithm as scalar GMRES (linear_solver_gmres.c) but with the O(n)
 * vector primitives OpenMP-parallelized. The dense Givens/Hessenberg and
 * back-substitution work is O(m^2) and stays scalar, identical to the scalar
 * and SIMD backends, so results are consistent across backends.
 *
 * Operator/sign convention matches CG and scalar GMRES: A = -nabla^2, b = -rhs.
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

/* ============================================================================
 * GMRES OMP CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;
    double dy2;
    double inv_dz2;
    double diag_inv;

    size_t stride_z;
    size_t k_start;
    size_t k_end;

    int    m;
    size_t n;

    double* Vblock;
    double* w;
    double* Mv;

    double* H;
    double* cs;
    double* sn;
    double* g;
    double* y;

    int use_precond;
    int initialized;
} gmres_omp_context_t;

/* ============================================================================
 * OMP-PARALLELIZED O(n) PRIMITIVES (interior points only)
 * ============================================================================ */

static inline int size_to_int(size_t val) {
    if (val > (size_t)INT_MAX) {
        return INT_MAX;
    }
    return (int)val;
}

static double dot_product_omp(const double* a, const double* b,
                              size_t nx, size_t ny,
                              size_t k_start, size_t k_end, size_t stride_z) {
    double sum = 0.0;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                sum += a[idx] * b[idx];
            }
        }
    }
    return sum;
}

static double vector_norm_omp(const double* x,
                              size_t nx, size_t ny,
                              size_t k_start, size_t k_end, size_t stride_z) {
    return sqrt(dot_product_omp(x, x, nx, ny, k_start, k_end, stride_z));
}

static void axpy_omp(double alpha, const double* x, double* y,
                     size_t nx, size_t ny,
                     size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                y[idx] += alpha * x[idx];
            }
        }
    }
}

static void scale_vector_omp(double alpha, double* x,
                             size_t nx, size_t ny,
                             size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                x[idx] *= alpha;
            }
        }
    }
}

static void apply_laplacian_omp(const double* p, double* Ap,
                                size_t nx, size_t ny,
                                double dx2, double dy2, double inv_dz2,
                                size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                double laplacian =
                    (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) * dx2_inv
                  + (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) * dy2_inv
                  + (p[idx + stride_z] + p[idx - stride_z] - 2.0 * p[idx]) * inv_dz2;
                Ap[idx] = -laplacian;
            }
        }
    }
}

static void compute_residual_omp(const double* x, const double* rhs, double* r,
                                 size_t nx, size_t ny,
                                 double dx2, double dy2, double inv_dz2,
                                 size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                double laplacian =
                    (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) * dx2_inv
                  + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) * dy2_inv
                  + (x[idx + stride_z] + x[idx - stride_z] - 2.0 * x[idx]) * inv_dz2;
                r[idx] = -rhs[idx] + laplacian;
            }
        }
    }
}

static void copy_vector_omp(const double* src, double* dst,
                            size_t nx, size_t ny,
                            size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = size_to_int(ny);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            size_t row_start = k * stride_z + (size_t)j * nx;
            memcpy(&dst[row_start + 1], &src[row_start + 1], (nx - 2) * sizeof(double));
        }
    }
}

static void apply_jacobi_precond_omp(const double* r, double* z,
                                     size_t nx, size_t ny,
                                     double diag_inv,
                                     size_t k_start, size_t k_end, size_t stride_z) {
    int ny_int = size_to_int(ny);
    int nx_int = size_to_int(nx);

    for (size_t k = k_start; k < k_end; k++) {
        int j;
#pragma omp parallel for schedule(static)
        for (j = 1; j < ny_int - 1; j++) {
            for (int i = 1; i < nx_int - 1; i++) {
                size_t idx = k * stride_z + IDX_2D((size_t)i, (size_t)j, nx);
                z[idx] = diag_inv * r[idx];
            }
        }
    }
}

/* ============================================================================
 * DENSE SCALAR HELPERS (identical to scalar backend; not parallelized)
 * ============================================================================ */

static void gmres_givens_omp(double* H, double* cs, double* sn,
                             double* g, int j, int m) {
    const int lead = m + 1;
    for (int i = 0; i < j; i++) {
        double temp     =  cs[i] * H[i + j * lead]     + sn[i] * H[(i + 1) + j * lead];
        H[(i + 1) + j * lead] = -sn[i] * H[i + j * lead] + cs[i] * H[(i + 1) + j * lead];
        H[i + j * lead] = temp;
    }

    double h_jj  = H[j + j * lead];
    double h_j1j = H[(j + 1) + j * lead];
    double denom = sqrt(h_jj * h_jj + h_j1j * h_j1j);
    if (denom < GMRES_BREAKDOWN_THRESHOLD) {
        cs[j] = 1.0;
        sn[j] = 0.0;
    } else {
        cs[j] = h_jj / denom;
        sn[j] = h_j1j / denom;
    }

    H[j + j * lead]       = cs[j] * h_jj + sn[j] * h_j1j;
    H[(j + 1) + j * lead] = 0.0;

    double g_j = g[j];
    g[j]     =  cs[j] * g_j;
    g[j + 1] = -sn[j] * g_j;
}

static void gmres_backsub_omp(const double* H, const double* g, double* y,
                              int k, int m) {
    const int lead = m + 1;
    for (int i = k - 1; i >= 0; i--) {
        double sum = g[i];
        for (int l = i + 1; l < k; l++) {
            sum -= H[i + l * lead] * y[l];
        }
        double diag = H[i + i * lead];
        y[i] = (fabs(diag) > GMRES_BREAKDOWN_THRESHOLD) ? (sum / diag) : 0.0;
    }
}

/* ============================================================================
 * GMRES OMP IMPLEMENTATION
 * ============================================================================ */

static void gmres_omp_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        gmres_omp_context_t* ctx = (gmres_omp_context_t*)solver->context;
        cfd_free(ctx->Vblock);
        cfd_free(ctx->w);
        cfd_free(ctx->Mv);
        cfd_free(ctx->H);
        cfd_free(ctx->cs);
        cfd_free(ctx->sn);
        cfd_free(ctx->g);
        cfd_free(ctx->y);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t gmres_omp_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    gmres_omp_context_t* ctx = (gmres_omp_context_t*)cfd_calloc(1, sizeof(gmres_omp_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);
    ctx->diag_inv = 1.0 / (2.0 / ctx->dx2 + 2.0 / ctx->dy2 + 2.0 * ctx->inv_dz2);
    ctx->use_precond = (params && params->preconditioner == POISSON_PRECOND_JACOBI);

    int m = (params && params->restart > 0) ? params->restart : GMRES_DEFAULT_RESTART;
    if (m < 1) m = 1;
    ctx->m = m;

    size_t n = nx * ny * nz;
    ctx->n = n;

    ctx->Vblock = (double*)cfd_calloc((size_t)(m + 1) * n, sizeof(double));
    ctx->w = (double*)cfd_calloc(n, sizeof(double));
    ctx->Mv = ctx->use_precond ? (double*)cfd_calloc(n, sizeof(double)) : NULL;

    ctx->H = (double*)cfd_calloc((size_t)(m + 1) * m, sizeof(double));
    ctx->cs = (double*)cfd_calloc(m, sizeof(double));
    ctx->sn = (double*)cfd_calloc(m, sizeof(double));
    ctx->g = (double*)cfd_calloc((size_t)m + 1, sizeof(double));
    ctx->y = (double*)cfd_calloc(m, sizeof(double));

    int precond_ok = (!ctx->use_precond) || (ctx->Mv != NULL);
    if (!ctx->Vblock || !ctx->w || !ctx->H || !ctx->cs || !ctx->sn ||
        !ctx->g || !ctx->y || !precond_ok) {
        solver->context = ctx;
        gmres_omp_destroy(solver);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static cfd_status_t gmres_omp_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;

    gmres_omp_context_t* ctx = (gmres_omp_context_t*)solver->context;
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = ctx->dx2;
    double dy2 = ctx->dy2;
    double inv_dz2 = ctx->inv_dz2;
    size_t stride_z = ctx->stride_z;
    size_t k_start = ctx->k_start;
    size_t k_end = ctx->k_end;

    int m = ctx->m;
    size_t n = ctx->n;
    const int lead = m + 1;

    double* Vblock = ctx->Vblock;
    double* w = ctx->w;
    double* Mv = ctx->Mv;
    double* H = ctx->H;
    double* cs = ctx->cs;
    double* sn = ctx->sn;
    double* g = ctx->g;
    double* y = ctx->y;
    int use_precond = ctx->use_precond;
    double diag_inv = ctx->diag_inv;

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    poisson_solver_apply_bc(solver, x);

    compute_residual_omp(x, rhs, Vblock, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
    double beta = vector_norm_omp(Vblock, nx, ny, k_start, k_end, stride_z);
    double initial_res = beta;

    if (stats) {
        stats->initial_residual = initial_res;
    }

    double tolerance = params->tolerance * initial_res;
    if (tolerance < params->absolute_tolerance) {
        tolerance = params->absolute_tolerance;
    }

    if (initial_res < params->absolute_tolerance) {
        poisson_solver_apply_bc(solver, x);
        if (stats) {
            stats->status = POISSON_CONVERGED;
            stats->iterations = 0;
            stats->final_residual = initial_res;
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return CFD_SUCCESS;
    }

    int total_inner = 0;
    int converged = 0;
    double final_res = initial_res;

    while (total_inner < params->max_iterations && !converged) {
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
            final_res = beta;
            break;
        }

        scale_vector_omp(1.0 / beta, Vblock, nx, ny, k_start, k_end, stride_z);
        memset(g, 0, (size_t)(m + 1) * sizeof(double));
        g[0] = beta;

        int k = 0;
        for (int j = 0; j < m; j++) {
            double* v_j = Vblock + (size_t)j * n;

            if (use_precond) {
                apply_jacobi_precond_omp(v_j, Mv, nx, ny, diag_inv, k_start, k_end, stride_z);
                apply_laplacian_omp(Mv, w, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
            } else {
                apply_laplacian_omp(v_j, w, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
            }

            for (int i = 0; i <= j; i++) {
                double* v_i = Vblock + (size_t)i * n;
                double hij = dot_product_omp(w, v_i, nx, ny, k_start, k_end, stride_z);
                H[i + j * lead] = hij;
                axpy_omp(-hij, v_i, w, nx, ny, k_start, k_end, stride_z);
            }

            double hnorm = vector_norm_omp(w, nx, ny, k_start, k_end, stride_z);
            H[(j + 1) + j * lead] = hnorm;

            int happy = (hnorm < GMRES_BREAKDOWN_THRESHOLD);
            if (!happy) {
                scale_vector_omp(1.0 / hnorm, w, nx, ny, k_start, k_end, stride_z);
                copy_vector_omp(w, Vblock + (size_t)(j + 1) * n, nx, ny, k_start, k_end, stride_z);
            }

            gmres_givens_omp(H, cs, sn, g, j, m);
            total_inner++;
            k = j + 1;

            double resid_est = fabs(g[j + 1]);
            if (params->verbose) {
                CFD_LOG_DEBUG("poisson", "GMRES-OMP inner %d (total %d): residual est = %.6e",
                              j, total_inner, resid_est);
            }
            if (resid_est < tolerance || happy || total_inner >= params->max_iterations) {
                break;
            }
        }

        gmres_backsub_omp(H, g, y, k, m);
        if (use_precond) {
            memset(w, 0, n * sizeof(double));
            for (int jj = 0; jj < k; jj++) {
                axpy_omp(y[jj], Vblock + (size_t)jj * n, w, nx, ny, k_start, k_end, stride_z);
            }
            apply_jacobi_precond_omp(w, Mv, nx, ny, diag_inv, k_start, k_end, stride_z);
            axpy_omp(1.0, Mv, x, nx, ny, k_start, k_end, stride_z);
        } else {
            for (int jj = 0; jj < k; jj++) {
                axpy_omp(y[jj], Vblock + (size_t)jj * n, x, nx, ny, k_start, k_end, stride_z);
            }
        }

        compute_residual_omp(x, rhs, Vblock, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
        beta = vector_norm_omp(Vblock, nx, ny, k_start, k_end, stride_z);
        final_res = beta;
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
        }
    }

    poisson_solver_apply_bc(solver, x);

    if (stats) {
        stats->iterations = total_inner;
        stats->final_residual = final_res;
        stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

static cfd_status_t gmres_omp_iterate(
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

poisson_solver_t* create_gmres_omp_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_GMRES_OMP;
    solver->description = "Restarted GMRES(m) (OpenMP)";
    solver->method = POISSON_METHOD_GMRES;
    solver->backend = POISSON_BACKEND_OMP;
    solver->params = poisson_solver_params_default();

    solver->init = gmres_omp_init;
    solver->destroy = gmres_omp_destroy;
    solver->solve = gmres_omp_solve;
    solver->iterate = gmres_omp_iterate;
    solver->apply_bc = NULL;

    return solver;
}

#endif /* CFD_ENABLE_OPENMP */
