/**
 * @file linear_solver_gmres.c
 * @brief Restarted GMRES(m) solver - scalar CPU implementation
 *
 * Generalized Minimal Residual method for solving Ax = b. Unlike CG, GMRES does
 * not require A to be symmetric positive definite, so it is the Krylov method of
 * choice for non-symmetric systems (e.g. implicit advection-diffusion). For the
 * pressure-Poisson operator used here (A = -nabla^2, SPD) GMRES converges to the
 * same solution as CG and serves as an independent cross-check.
 *
 * GMRES characteristics:
 * - Minimizes the residual norm over the Krylov subspace at every inner step.
 * - Restarted GMRES(m) caps the basis at m+1 vectors to bound memory; the outer
 *   loop restarts from the current iterate until convergence or max_iterations.
 * - Each inner iteration: 1 matrix-vector product, (j+1) dot products + axpys for
 *   modified Gram-Schmidt, plus O(m) scalar work for Givens rotations.
 *
 * Algorithm (restarted GMRES(m), right-preconditioned when enabled):
 *   outer restart:
 *     r = b - A x;  beta = ||r||;  v_0 = r / beta;  g = (beta, 0, ..., 0)
 *     inner j = 0..m-1 (Arnoldi + modified Gram-Schmidt):
 *       w = A M^{-1} v_j          (w = A v_j when unpreconditioned)
 *       for i <= j: H[i,j] = (w, v_i);  w -= H[i,j] v_i
 *       H[j+1,j] = ||w||;  v_{j+1} = w / H[j+1,j]
 *       apply stored Givens rotations to column j; build a new one to zero
 *       H[j+1,j]; rotate g.  Residual estimate = |g[j+1]|.
 *     solve upper-triangular H y = g;  x += M^{-1}(V y)  (x += V y unpreconditioned)
 *
 * Right (not left) preconditioning is used so the cheap Givens estimate |g[k]|
 * equals the true unpreconditioned residual ||b - A x||, keeping the convergence
 * check, reported statistics, and the CG cross-check consistent. See the plan and
 * the PCG notes: on a uniform grid the Laplacian diagonal is constant, so Jacobi
 * preconditioning is a scalar multiply and does not reduce iteration count.
 */

#include "../linear_solver_internal.h"

#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <string.h>

/* ============================================================================
 * GMRES CONTEXT
 * ============================================================================ */

typedef struct {
    double dx2;        /* dx^2 */
    double dy2;        /* dy^2 */
    double inv_dz2;    /* 1/dz^2 (0 for 2D) */
    double diag_inv;   /* Jacobi preconditioner: 1/(2/dx2 + 2/dy2 + 2*inv_dz2) */

    size_t stride_z;   /* nx*ny for 3D, 0 for 2D */
    size_t k_start;    /* first interior k index */
    size_t k_end;      /* one-past-last interior k index */

    int    m;          /* restart length (Krylov subspace dimension) */
    size_t n;          /* total field size nx*ny*nz */

    /* Large O(n) working vectors (allocated during init) */
    double* Vblock;    /* (m+1) Krylov basis vectors, V[j] = Vblock + j*n */
    double* w;         /* A*v_j scratch */
    double* Mv;        /* M^{-1}*v scratch (NULL if no precond) */

    /* Small dense O(m^2)/O(m) arrays (scalar in every backend) */
    double* H;         /* upper-Hessenberg (m+1) x m, column-major H[i + j*(m+1)] */
    double* cs;        /* Givens cosines, length m */
    double* sn;        /* Givens sines, length m */
    double* g;         /* rotated least-squares RHS, length m+1 */
    double* y;         /* back-substitution solution, length m */

    int use_precond;   /* Flag: is preconditioner enabled? */
    int initialized;
} gmres_context_t;

/* ============================================================================
 * O(n) VECTOR PRIMITIVES (interior points only)
 *
 * dot_product / axpy / apply_laplacian / compute_residual / copy_vector /
 * apply_jacobi_precond mirror linear_solver_cg.c exactly. scale_vector,
 * vector_norm, and linear_combination are the GMRES-specific additions.
 * ============================================================================ */

/** Dot product over interior points */
static double dot_product(const double* a, const double* b,
                          size_t nx, size_t ny,
                          size_t k_start, size_t k_end, size_t stride_z) {
    double sum = 0.0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                sum += a[idx] * b[idx];
            }
        }
    }
    return sum;
}

/** y = y + alpha * x (interior points only) */
static void axpy(double alpha, const double* x, double* y,
                 size_t nx, size_t ny,
                 size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                y[idx] += alpha * x[idx];
            }
        }
    }
}

/** x = alpha * x (interior points only) */
static void scale_vector(double alpha, double* x,
                         size_t nx, size_t ny,
                         size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                x[idx] *= alpha;
            }
        }
    }
}

/** Euclidean norm over interior points: sqrt(dot(x,x)) */
static double vector_norm(const double* x,
                          size_t nx, size_t ny,
                          size_t k_start, size_t k_end, size_t stride_z) {
    return sqrt(dot_product(x, x, nx, ny, k_start, k_end, stride_z));
}

/**
 * Apply negative Laplacian operator: Ap = -nabla^2(p) (matches CG's A = -nabla^2).
 */
static void apply_laplacian(const double* p, double* Ap,
                            size_t nx, size_t ny,
                            double dx2, double dy2, double inv_dz2,
                            size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));

                double laplacian =
                    ((p[idx + 1] - (2.0 * p[idx]) + p[idx - 1]) * dx2_inv)
                  + ((p[idx + (nx)] - (2.0 * p[idx]) + p[idx - (nx)]) * dy2_inv)
                  + ((p[idx + (stride_z)] + p[idx - (stride_z)] - (2.0 * p[idx])) * inv_dz2);
                Ap[idx] = -laplacian;
            }
        }
    }
}

/**
 * Compute residual r = b - A x = -rhs + nabla^2 x (matches CG convention).
 */
static void compute_residual(const double* x, const double* rhs, double* r,
                             size_t nx, size_t ny,
                             double dx2, double dy2, double inv_dz2,
                             size_t k_start, size_t k_end, size_t stride_z) {
    double dx2_inv = 1.0 / dx2;
    double dy2_inv = 1.0 / dy2;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));

                double laplacian =
                    ((x[idx + 1] - (2.0 * x[idx]) + x[idx - 1]) * dx2_inv)
                  + ((x[idx + (nx)] - (2.0 * x[idx]) + x[idx - (nx)]) * dy2_inv)
                  + ((x[idx + (stride_z)] + x[idx - (stride_z)] - (2.0 * x[idx])) * inv_dz2);

                r[idx] = -rhs[idx] + laplacian;
            }
        }
    }
}

/** Copy dst = src (interior points only) */
static void copy_vector(const double* src, double* dst,
                        size_t nx, size_t ny,
                        size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                dst[idx] = src[idx];
            }
        }
    }
}

/** Apply Jacobi preconditioner: z = M^{-1} r = diag_inv * r (interior points) */
static void apply_jacobi_precond(const double* r, double* z,
                                 size_t nx, size_t ny,
                                 double diag_inv,
                                 size_t k_start, size_t k_end, size_t stride_z) {
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + (IDX_2D(i, j, nx));
                z[idx] = diag_inv * r[idx];
            }
        }
    }
}

/**
 * Accumulate x += sum_{j=0..k-1} y[j] * V[j] over interior points.
 * Simplest correct form is k axpy calls; V holds the Krylov basis.
 */
static void linear_combination(double* x, const double* Vblock, size_t n,
                               const double* y, int k,
                               size_t nx, size_t ny,
                               size_t k_start, size_t k_end, size_t stride_z) {
    for (int jj = 0; jj < k; jj++) {
        axpy(y[jj], Vblock + (size_t)jj * n, x, nx, ny, k_start, k_end, stride_z);
    }
}

/* ============================================================================
 * DENSE SCALAR HELPERS (small O(m^2)/O(m) work — scalar in every backend)
 * ============================================================================ */

/**
 * Apply previously stored Givens rotations to column j of H, then build and apply
 * a new rotation that zeros H[j+1,j], and rotate the RHS vector g accordingly.
 * H is column-major, (m+1) x m: element (row i, col j) is H[i + j*(m+1)].
 * After the call, |g[j+1]| is the GMRES residual estimate for the current basis.
 */
static void apply_givens_and_update_hessenberg(double* H, double* cs, double* sn,
                                               double* g, int j, int m) {
    const int lead = m + 1;  /* column stride */

    /* Apply rotations 0..j-1 to the new column j */
    for (int i = 0; i < j; i++) {
        double temp     =  cs[i] * H[i + j * lead]     + sn[i] * H[(i + 1) + j * lead];
        H[(i + 1) + j * lead] = -sn[i] * H[i + j * lead] + cs[i] * H[(i + 1) + j * lead];
        H[i + j * lead] = temp;
    }

    /* Compute new rotation to zero H[j+1,j] */
    double h_jj  = H[j + j * lead];
    double h_j1j = H[(j + 1) + j * lead];
    double denom = sqrt(h_jj * h_jj + h_j1j * h_j1j);
    if (denom < GMRES_BREAKDOWN_THRESHOLD) {
        /* Degenerate column (both entries ~0): identity rotation. */
        cs[j] = 1.0;
        sn[j] = 0.0;
    } else {
        cs[j] = h_jj / denom;
        sn[j] = h_j1j / denom;
    }

    /* Apply the new rotation to H (H[j,j] becomes denom, H[j+1,j] becomes 0) */
    H[j + j * lead]       = cs[j] * h_jj + sn[j] * h_j1j;
    H[(j + 1) + j * lead] = 0.0;

    /* Rotate g */
    double g_j = g[j];
    g[j]     =  cs[j] * g_j;
    g[j + 1] = -sn[j] * g_j;
}

/**
 * Solve the k x k upper-triangular top block of H for y: H(0:k,0:k) y = g(0:k).
 * H column-major, (m+1) x m.
 */
static void back_substitution(const double* H, const double* g, double* y,
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
 * GMRES SCALAR IMPLEMENTATION
 * ============================================================================ */

static void gmres_scalar_destroy(poisson_solver_t* solver) {
    if (solver && solver->context) {
        gmres_context_t* ctx = (gmres_context_t*)solver->context;
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

static cfd_status_t gmres_scalar_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    cfd_status_t precond_status = poisson_solver_reject_mg_precond(params);
    if (precond_status != CFD_SUCCESS) {
        return precond_status;
    }

    gmres_context_t* ctx = (gmres_context_t*)cfd_calloc(1, sizeof(gmres_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->dx2 = dx * dx;
    ctx->dy2 = dy * dy;
    ctx->inv_dz2 = poisson_solver_compute_inv_dz2(dz);
    poisson_solver_compute_3d_bounds(nz, nx, ny, &ctx->stride_z, &ctx->k_start, &ctx->k_end);

    ctx->diag_inv = 1.0 / (2.0 / ctx->dx2 + 2.0 / ctx->dy2 + 2.0 * ctx->inv_dz2);
    ctx->use_precond = (params && params->preconditioner == POISSON_PRECOND_JACOBI);

    /* Resolve restart length m (0 = auto), clamp to >= 1 */
    int m = (params && params->restart > 0) ? params->restart : GMRES_DEFAULT_RESTART;
    if (m < 1) {
        m = 1;
    }
    ctx->m = m;

    size_t n = nx * ny * nz;
    ctx->n = n;

    /* Large O(n) arrays */
    ctx->Vblock = (double*)cfd_calloc((size_t)(m + 1) * n, sizeof(double));
    ctx->w = (double*)cfd_calloc(n, sizeof(double));
    ctx->Mv = ctx->use_precond ? (double*)cfd_calloc(n, sizeof(double)) : NULL;

    /* Small dense arrays */
    ctx->H = (double*)cfd_calloc((size_t)(m + 1) * m, sizeof(double));
    ctx->cs = (double*)cfd_calloc(m, sizeof(double));
    ctx->sn = (double*)cfd_calloc(m, sizeof(double));
    ctx->g = (double*)cfd_calloc((size_t)m + 1, sizeof(double));
    ctx->y = (double*)cfd_calloc(m, sizeof(double));

    int precond_ok = (!ctx->use_precond) || (ctx->Mv != NULL);
    if (!ctx->Vblock || !ctx->w || !ctx->H || !ctx->cs || !ctx->sn ||
        !ctx->g || !ctx->y || !precond_ok) {
        solver->context = ctx;
        gmres_scalar_destroy(solver);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static cfd_status_t gmres_scalar_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    (void)x_temp;  /* GMRES uses its own basis storage */

    gmres_context_t* ctx = (gmres_context_t*)solver->context;
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

    /* Apply initial boundary conditions */
    poisson_solver_apply_bc(solver, x);

    /* Initial residual r_0 = b - A x_0 into V[0] */
    compute_residual(x, rhs, Vblock, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
    double beta = vector_norm(Vblock, nx, ny, k_start, k_end, stride_z);
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
        /* V[0] and beta hold the current TRUE residual (fresh on first pass,
         * recomputed at the end of each restart below). */
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
            final_res = beta;
            break;
        }

        /* v_0 = r / beta */
        scale_vector(1.0 / beta, Vblock, nx, ny, k_start, k_end, stride_z);

        /* Reset least-squares RHS: g = (beta, 0, ..., 0) */
        memset(g, 0, (size_t)(m + 1) * sizeof(double));
        g[0] = beta;

        int k = 0;
        for (int j = 0; j < m; j++) {
            double* v_j = Vblock + (size_t)j * n;

            /* Arnoldi: w = A M^{-1} v_j  (w = A v_j unpreconditioned) */
            if (use_precond) {
                apply_jacobi_precond(v_j, Mv, nx, ny, diag_inv, k_start, k_end, stride_z);
                apply_laplacian(Mv, w, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
            } else {
                apply_laplacian(v_j, w, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
            }

            /* Modified Gram-Schmidt against v_0..v_j */
            for (int i = 0; i <= j; i++) {
                double* v_i = Vblock + (size_t)i * n;
                double hij = dot_product(w, v_i, nx, ny, k_start, k_end, stride_z);
                H[i + j * lead] = hij;
                axpy(-hij, v_i, w, nx, ny, k_start, k_end, stride_z);
            }

            double hnorm = vector_norm(w, nx, ny, k_start, k_end, stride_z);
            H[(j + 1) + j * lead] = hnorm;

            /* Happy breakdown: solution lies in the current Krylov subspace. */
            int happy = (hnorm < GMRES_BREAKDOWN_THRESHOLD);
            if (!happy) {
                scale_vector(1.0 / hnorm, w, nx, ny, k_start, k_end, stride_z);
                copy_vector(w, Vblock + (size_t)(j + 1) * n, nx, ny, k_start, k_end, stride_z);
            }

            apply_givens_and_update_hessenberg(H, cs, sn, g, j, m);
            total_inner++;
            k = j + 1;

            /* Cheap Givens residual estimate: only decides when to END the inner
             * cycle early (to update x and recheck the TRUE residual). It does NOT
             * decide convergence, so drift in the estimate cannot cause a false
             * positive. */
            double resid_est = fabs(g[j + 1]);
            if (params->verbose) {
                CFD_LOG_DEBUG("poisson", "GMRES inner %d (total %d): residual est = %.6e",
                              j, total_inner, resid_est);
            }
            if (resid_est < tolerance || happy || total_inner >= params->max_iterations) {
                break;
            }
        }

        /* Solve the least-squares problem and update x */
        back_substitution(H, g, y, k, m);
        if (use_precond) {
            memset(w, 0, n * sizeof(double));
            linear_combination(w, Vblock, n, y, k, nx, ny, k_start, k_end, stride_z);
            apply_jacobi_precond(w, Mv, nx, ny, diag_inv, k_start, k_end, stride_z);
            axpy(1.0, Mv, x, nx, ny, k_start, k_end, stride_z);
        } else {
            linear_combination(x, Vblock, n, y, k, nx, ny, k_start, k_end, stride_z);
        }

        /* Recompute the TRUE residual and decide convergence on it (prevents the
         * cheap Givens estimate from drifting below the real residual). This is
         * the residual reported to the caller, measured with the same fixed
         * boundary values used throughout the solve. */
        compute_residual(x, rhs, Vblock, nx, ny, dx2, dy2, inv_dz2, k_start, k_end, stride_z);
        beta = vector_norm(Vblock, nx, ny, k_start, k_end, stride_z);
        final_res = beta;
        if (beta < tolerance || beta < params->absolute_tolerance) {
            converged = 1;
        }
    }

    /* Apply final boundary conditions for output (mirrors CG). The reported
     * residual is the solve-consistent one measured above, NOT recomputed here. */
    poisson_solver_apply_bc(solver, x);

    if (stats) {
        stats->iterations = total_inner;
        stats->final_residual = final_res;
        stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
}

/**
 * Single iteration is not well-defined for GMRES (it maintains an internal
 * Krylov basis). Mirror CG: return the current residual only.
 */
static cfd_status_t gmres_scalar_iterate(
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

poisson_solver_t* create_gmres_scalar_solver(void) {
    poisson_solver_t* solver = (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = POISSON_SOLVER_TYPE_GMRES_SCALAR;
    solver->description = "Restarted GMRES(m) (scalar CPU)";
    solver->method = POISSON_METHOD_GMRES;
    solver->backend = POISSON_BACKEND_SCALAR;
    solver->params = poisson_solver_params_default();

    solver->init = gmres_scalar_init;
    solver->destroy = gmres_scalar_destroy;
    solver->solve = gmres_scalar_solve;  /* GMRES uses custom solve loop */
    solver->iterate = gmres_scalar_iterate;
    solver->apply_bc = NULL;  /* Use default Neumann */

    return solver;
}
