/**
 * @file linear_solver.c
 * @brief Core linear solver implementation
 *
 * Implements:
 * - Default parameter functions
 * - Backend selection
 * - Solver lifecycle (create, init, destroy)
 * - Common solve loop
 * - Legacy poisson_solve() wrapper
 */

#include "cfd/solvers/poisson_solver.h"
#include "linear_solver_internal.h"

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/indexing.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

/* After <windows.h>: this header includes it without WIN32_LEAN_AND_MEAN */
#include "../../core/cfd_threading_internal.h"

/* ============================================================================
 * DEFAULT PARAMETERS
 * ============================================================================ */

poisson_solver_params_t poisson_solver_params_default(void) {
    poisson_solver_params_t params;
    params.tolerance = 1e-6;
    params.absolute_tolerance = 1e-10;
    params.max_iterations = 5000;  /* Increased from 1000 for CG on fine grids */
    params.omega = 0.0;  /* Auto-compute optimal omega for grid dimensions */
    params.check_interval = 1;
    params.verbose = false;
    params.preconditioner = POISSON_PRECOND_NONE;
    params.restart = 0;  /* 0 = auto (GMRES_DEFAULT_RESTART); ignored by non-GMRES methods */
    params.mg_cycle = MG_CYCLE_V;
    params.mg_smoother = MG_SMOOTHER_REDBLACK_GS;
    params.mg_bc = MG_BC_NEUMANN;
    params.mg_pre_smooth = 0;      /* 0 = default (2) */
    params.mg_post_smooth = 0;     /* 0 = default (2) */
    params.mg_coarse_max_iter = 0; /* 0 = default (50) */
    params.mg_max_levels = 0;      /* 0 = auto */
    return params;
}

poisson_solver_stats_t poisson_solver_stats_default(void) {
    poisson_solver_stats_t stats;
    stats.status = POISSON_ERROR;
    stats.iterations = 0;
    stats.initial_residual = 0.0;
    stats.final_residual = 0.0;
    stats.elapsed_time_ms = 0.0;
    return stats;
}

/* ============================================================================
 * TIMING
 * ============================================================================ */

double poisson_solver_get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

/* ============================================================================
 * BACKEND SELECTION
 * ============================================================================ */

static poisson_solver_backend_t g_default_backend = POISSON_BACKEND_AUTO;

poisson_solver_backend_t poisson_solver_get_backend(void) {
    return g_default_backend;
}

const char* poisson_solver_get_backend_name(void) {
    switch (g_default_backend) {
        case POISSON_BACKEND_SCALAR:   return "scalar";
        case POISSON_BACKEND_OMP:      return "omp";
        case POISSON_BACKEND_SIMD: return "simd";
        case POISSON_BACKEND_GPU:      return "gpu";
        case POISSON_BACKEND_AUTO:
        default:                       return "auto";
    }
}

bool poisson_solver_set_backend(poisson_solver_backend_t backend) {
    if (!poisson_solver_backend_available(backend)) {
        return false;
    }
    g_default_backend = backend;
    return true;
}

bool poisson_solver_backend_available(poisson_solver_backend_t backend) {
    switch (backend) {
        case POISSON_BACKEND_AUTO:
        case POISSON_BACKEND_SCALAR:
            return true;

        case POISSON_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            return true;
#else
            return false;
#endif

        case POISSON_BACKEND_SIMD:
            return poisson_solver_simd_backend_available();

        case POISSON_BACKEND_GPU:
#ifdef CFD_HAS_CUDA
            return true;
#else
            return false;
#endif

        default:
            return false;
    }
}

/* ============================================================================
 * SOLVER LIFECYCLE
 * ============================================================================ */

/**
 * Auto-select best available backend
 *
 * Priority: SIMD (runtime detection) > Scalar
 */
static poisson_solver_backend_t select_best_backend(void) {
    /* Prefer SIMD with runtime detection */
    if (poisson_solver_simd_backend_available()) {
        return POISSON_BACKEND_SIMD;
    }
    return POISSON_BACKEND_SCALAR;
}

/* Factories must set a specific last-status on NULL (see cfd_solver_create),
 * so callers can distinguish an unavailable backend from allocation failure. */
static poisson_solver_t* backend_unavailable(const char* method_name) {
    char msg[128];
    snprintf(msg, sizeof(msg), "Requested backend not available for %s", method_name);
    cfd_set_error(CFD_ERROR_UNSUPPORTED, msg);
    return NULL;
}

poisson_solver_t* poisson_solver_create(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend)
{
    /* Auto-select backend if requested (keep the original request: methods
     * without a SIMD backend resolve AUTO differently) */
    poisson_solver_backend_t requested = backend;
    if (backend == POISSON_BACKEND_AUTO) {
        backend = select_best_backend();
    }

    /* Create appropriate solver - no silent fallbacks */
    switch (method) {
        case POISSON_METHOD_JACOBI:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_jacobi_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_jacobi_omp_solver();
#endif
#ifdef CFD_HAS_CUDA
                case POISSON_BACKEND_GPU:
                    return create_jacobi_gpu_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_jacobi_scalar_solver();
                default:
                    return backend_unavailable("Jacobi");
            }

        case POISSON_METHOD_SOR:
        case POISSON_METHOD_GAUSS_SEIDEL: {
            /* Gauss-Seidel is SOR at omega = 1: the same solvers, marked with the
             * requested method so that init resolves omega to 1 */
            poisson_solver_t* sor;
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    sor = create_sor_simd_solver();
                    break;
#ifdef CFD_HAS_CUDA
                case POISSON_BACKEND_GPU:
                    sor = create_sor_gpu_solver();
                    break;
#endif
                case POISSON_BACKEND_SCALAR:
                    sor = create_sor_scalar_solver();
                    break;
                default:
                    return backend_unavailable(method == POISSON_METHOD_GAUSS_SEIDEL ? "Gauss-Seidel" : "SOR");
            }
            if (sor) {
                sor->method = method;
            }
            return sor;
        }

        case POISSON_METHOD_REDBLACK_SOR:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_redblack_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_redblack_omp_solver();
#endif
#ifdef CFD_HAS_CUDA
                case POISSON_BACKEND_GPU:
                    return create_redblack_gpu_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_redblack_scalar_solver();
                default:
                    return backend_unavailable("Red-Black SOR");
            }

        case POISSON_METHOD_CG:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_cg_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_cg_omp_solver();
#endif
#ifdef CFD_HAS_CUDA
                case POISSON_BACKEND_GPU:
                    return create_cg_gpu_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_cg_scalar_solver();
                default:
                    return backend_unavailable("CG");
            }

        case POISSON_METHOD_BICGSTAB:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_bicgstab_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_bicgstab_omp_solver();
#endif
#ifdef CFD_HAS_CUDA
                case POISSON_BACKEND_GPU:
                    return create_bicgstab_gpu_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_bicgstab_scalar_solver();
                default:
                    return backend_unavailable("BiCGSTAB");
            }

        case POISSON_METHOD_GMRES:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_gmres_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_gmres_omp_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_gmres_scalar_solver();
                default:
                    return backend_unavailable("GMRES");
            }

        case POISSON_METHOD_MULTIGRID:
            /* No SIMD/GPU multigrid. AUTO means "best available", which
             * select_best_backend() resolves to SIMD — a backend multigrid
             * lacks — so AUTO keeps resolving to the scalar reference. OMP is
             * opt-in, as for every other method; explicit SIMD/GPU requests
             * return NULL (no silent fallbacks). */
            if (requested == POISSON_BACKEND_AUTO) {
                return create_multigrid_scalar_solver();
            }
            switch (backend) {
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_multigrid_omp_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_multigrid_scalar_solver();
                default:
                    return backend_unavailable("multigrid");
            }

        default:
            cfd_set_error(CFD_ERROR_INVALID, "Unknown Poisson solver method");
            return NULL;
    }
}

cfd_status_t poisson_solver_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    if (!solver) {
        return CFD_ERROR_INVALID;
    }

    /* Require at least one interior cell in each active dimension.
     * For 2D (nz <= 1): nx, ny >= 3.  For 3D (nz > 1): nx, ny, nz >= 3.
     * Rejects degenerate grids (e.g. nz==2) where k_start==k_end. */
    if (nx < 3 || ny < 3 || (nz > 1 && nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    solver->nx = nx;
    solver->ny = ny;
    solver->nz = nz;
    solver->dx = dx;
    solver->dy = dy;
    solver->dz = dz;

    if (params) {
        solver->params = *params;
    } else {
        solver->params = poisson_solver_params_default();
    }

    /* Adjust max_iterations for Jacobi (needs more iterations) */
    if (solver->method == POISSON_METHOD_JACOBI && params == NULL) {
        solver->params.max_iterations = 2000;
    }

    /* Call solver-specific init if provided */
    if (solver->init) {
        return solver->init(solver, nx, ny, nz, dx, dy, dz, &solver->params);
    }

    return CFD_SUCCESS;
}

void poisson_solver_destroy(poisson_solver_t* solver) {
    if (!solver) {
        return;
    }

    if (solver->destroy) {
        solver->destroy(solver);
    }

    cfd_free(solver);
}

/* ============================================================================
 * SOLVER OPERATIONS
 * ============================================================================ */

double poisson_solver_compute_residual(
    poisson_solver_t* solver,
    const double* x,
    const double* rhs)
{
    if (!solver || !x || !rhs) {
        return -1.0;
    }

    size_t nx = solver->nx;
    size_t ny = solver->ny;
    double dx2 = solver->dx * solver->dx;
    double dy2 = solver->dy * solver->dy;
    double inv_dz2 = poisson_solver_compute_inv_dz2(solver->dz);

    size_t stride_z, k_start, k_end;
    poisson_solver_compute_3d_bounds(solver->nz, nx, ny,
                                     &stride_z, &k_start, &k_end);

    double max_residual = 0.0;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);

                /* Compute Laplacian: d^2x/dx^2 + d^2x/dy^2 + d^2x/dz^2 */
                double laplacian =
                    (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / dx2
                  + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / dy2
                  + (x[idx + stride_z] + x[idx - stride_z]
                     - 2.0 * x[idx]) * inv_dz2;

                double residual = fabs(laplacian - rhs[idx]);
                /* A NaN compares false against everything, so without the
                 * isnan() a diverged field would read as a zero residual */
                if (residual > max_residual || isnan(residual)) {
                    max_residual = residual;
                }
            }
        }
    }

    return max_residual;
}

void poisson_solver_apply_bc(
    poisson_solver_t* solver,
    double* x)
{
    if (!solver || !x) {
        return;
    }

    if (solver->apply_bc) {
        solver->apply_bc(solver, x);
        return;
    }

    /* Default: Neumann BCs (zero gradient) on all faces */
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t nz = solver->nz;
    size_t plane_size = nx * ny;

    /* Z-face Neumann: copy adjacent interior plane to boundary planes */
    if (nz > 1) {
        memcpy(x, x + plane_size, plane_size * sizeof(double));
        memcpy(x + (nz - 1) * plane_size,
               x + (nz - 2) * plane_size,
               plane_size * sizeof(double));
    }

    /* Apply 2D Neumann BCs on each z-plane using solver's own backend.
     * This avoids BC_BACKEND_AUTO selecting a different backend (e.g. OMP)
     * than the solver itself, which could spawn unexpected threads. */
    for (size_t k = 0; k < nz; k++) {
        double* plane = x + k * plane_size;
        switch (solver->backend) {
            case POISSON_BACKEND_OMP:
                bc_apply_scalar_omp(plane, nx, ny, BC_TYPE_NEUMANN);
                break;
            case POISSON_BACKEND_SIMD:
                bc_apply_scalar_simd(plane, nx, ny, BC_TYPE_NEUMANN);
                break;
            default:
                bc_apply_scalar_cpu(plane, nx, ny, BC_TYPE_NEUMANN);
                break;
        }
    }
}

/**
 * Common solve loop used by all solvers
 */
cfd_status_t poisson_solver_solve_common(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    if (!solver || !x || !rhs) {
        return CFD_ERROR_INVALID;
    }

    if (!solver->iterate) {
        return CFD_ERROR_UNSUPPORTED;
    }

    poisson_solver_params_t* params = &solver->params;
    double start_time = poisson_solver_get_time_ms();

    /* Compute initial residual */
    double initial_res = poisson_solver_compute_residual(solver, x, rhs);
    double tolerance = params->tolerance * initial_res;

    /* Ensure minimum absolute tolerance */
    if (tolerance < params->absolute_tolerance) {
        tolerance = params->absolute_tolerance;
    }

    if (stats) {
        stats->initial_residual = initial_res;
    }

    /* Already converged? */
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
    int diverged = 0;
    int iter;
    double res = initial_res;

    for (iter = 0; iter < params->max_iterations; iter++) {
        double new_res = 0.0;

        /* Perform one iteration */
        cfd_status_t status = solver->iterate(solver, x, x_temp, rhs,
            (iter % params->check_interval == 0) ? &new_res : NULL);

        if (status != CFD_SUCCESS) {
            if (stats) {
                stats->status = POISSON_ERROR;
                stats->iterations = iter + 1;
                stats->final_residual = res;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return status;
        }

        /* Check convergence at intervals */
        if (iter % params->check_interval == 0) {
            res = new_res;

            /* A residual that is no longer finite will never meet the tolerance */
            if (!isfinite(res)) {
                diverged = 1;
                break;
            }

            if (params->verbose) {
                CFD_LOG_DEBUG("poisson", "Iter %d: residual = %.6e", iter, res);
            }

            if (res < tolerance || res < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }
    }

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        /* A break leaves iter at the index of the last iteration run;
         * an exhausted loop leaves it at max_iterations, the count run */
        stats->iterations = (converged || diverged) ? iter + 1 : iter;
        stats->final_residual = res;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED
                      : diverged  ? POISSON_DIVERGED
                                  : POISSON_MAX_ITER;
    }

    if (converged) {
        return CFD_SUCCESS;
    }
    return diverged ? CFD_ERROR_DIVERGED : CFD_ERROR_MAX_ITER;
}

cfd_status_t poisson_solver_solve(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    if (!solver) {
        return CFD_ERROR_INVALID;
    }

    /* Use solver-specific solve if provided, otherwise common loop */
    if (solver->solve) {
        double start_time = poisson_solver_get_time_ms();
        cfd_status_t status = solver->solve(solver, x, x_temp, rhs, stats);
        if (stats) {
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return status;
    }

    return poisson_solver_solve_common(solver, x, x_temp, rhs, stats);
}

cfd_status_t poisson_solver_iterate(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    if (!solver || !x || !rhs) {
        return CFD_ERROR_INVALID;
    }

    if (!solver->iterate) {
        return CFD_ERROR_UNSUPPORTED;
    }

    return solver->iterate(solver, x, x_temp, rhs, residual);
}

/* ============================================================================
 * CACHED SOLVER INSTANCES
 * ============================================================================ */

/*
 * Cached solver instances for the poisson_solve() convenience API, one slot per
 * preset, so repeated calls skip solver creation.
 *
 * A call takes the instance out of its slot and puts it back when it returns.
 * A concurrent call for the same preset finds the slot empty and builds its own
 * instance, so two threads never share one, and an instance left in a slot is
 * always idle.
 */
static cfd_atomic_ptr g_cached_jacobi_simd;
static cfd_atomic_ptr g_cached_sor;
static cfd_atomic_ptr g_cached_sor_simd;
static cfd_atomic_ptr g_cached_redblack_simd;
static cfd_atomic_ptr g_cached_redblack_omp;
static cfd_atomic_ptr g_cached_redblack_scalar;
static cfd_atomic_ptr g_cached_cg_scalar;
static cfd_atomic_ptr g_cached_cg_omp;
static cfd_atomic_ptr g_cached_cg_simd;
static cfd_atomic_ptr g_cached_mg_scalar;
static cfd_atomic_ptr g_cached_mg_omp;
static cfd_atomic_ptr g_cached_pcg_mg_scalar;

/* Set to 1 once cleanup_cached_solvers is registered with atexit */
static cfd_atomic_int g_cleanup_registered = 0;

/** Empty a cache slot, returning the instance it held (NULL if none) */
static poisson_solver_t* take_cached_solver(cfd_atomic_ptr* slot) {
    return (poisson_solver_t*)cfd_atomic_ptr_exchange(slot, NULL);
}

/**
 * Cleanup cached solvers (called at program exit)
 */
static void cleanup_cached_solvers(void) {
    poisson_solver_destroy(take_cached_solver(&g_cached_jacobi_simd));
    poisson_solver_destroy(take_cached_solver(&g_cached_sor));
    poisson_solver_destroy(take_cached_solver(&g_cached_sor_simd));
    poisson_solver_destroy(take_cached_solver(&g_cached_redblack_simd));
    poisson_solver_destroy(take_cached_solver(&g_cached_redblack_omp));
    poisson_solver_destroy(take_cached_solver(&g_cached_redblack_scalar));
    poisson_solver_destroy(take_cached_solver(&g_cached_cg_scalar));
    poisson_solver_destroy(take_cached_solver(&g_cached_cg_omp));
    poisson_solver_destroy(take_cached_solver(&g_cached_cg_simd));
    poisson_solver_destroy(take_cached_solver(&g_cached_mg_scalar));
    poisson_solver_destroy(take_cached_solver(&g_cached_mg_omp));
    poisson_solver_destroy(take_cached_solver(&g_cached_pcg_mg_scalar));
}

int poisson_solve_3d(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    poisson_solver_type solver_type)
{
    cfd_atomic_ptr* slot;
    poisson_solver_method_t method;
    poisson_solver_backend_t backend;

    switch (solver_type) {
        case POISSON_SOLVER_JACOBI_SIMD:
            slot = &g_cached_jacobi_simd;
            method = POISSON_METHOD_JACOBI;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_REDBLACK_SIMD:
            slot = &g_cached_redblack_simd;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_REDBLACK_OMP:
            slot = &g_cached_redblack_omp;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_OMP;
            break;

        case POISSON_SOLVER_SOR_SCALAR:
            slot = &g_cached_sor;
            method = POISSON_METHOD_SOR;
            backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_SOLVER_REDBLACK_SCALAR:
            slot = &g_cached_redblack_scalar;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_SOLVER_CG_SCALAR:
            slot = &g_cached_cg_scalar;
            method = POISSON_METHOD_CG;
            backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_SOLVER_CG_SIMD:
            slot = &g_cached_cg_simd;
            method = POISSON_METHOD_CG;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_CG_OMP:
            slot = &g_cached_cg_omp;
            method = POISSON_METHOD_CG;
            backend = POISSON_BACKEND_OMP;
            break;

        case POISSON_SOLVER_SOR_SIMD:
            slot = &g_cached_sor_simd;
            method = POISSON_METHOD_SOR;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_MG_SCALAR:
            slot = &g_cached_mg_scalar;
            method = POISSON_METHOD_MULTIGRID;
            backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_SOLVER_MG_OMP:
            slot = &g_cached_mg_omp;
            method = POISSON_METHOD_MULTIGRID;
            backend = POISSON_BACKEND_OMP;
            break;

        case POISSON_SOLVER_PCG_MG_SCALAR:
            slot = &g_cached_pcg_mg_scalar;
            method = POISSON_METHOD_CG;
            backend = POISSON_BACKEND_SCALAR;
            break;

        default:
            CFD_LOG_ERROR("poisson", "poisson_solve_3d: Unknown solver type %d", solver_type);
            return -1;
    }

    /* This call owns the cached instance until it puts it back */
    poisson_solver_t* solver = take_cached_solver(slot);

    /* Recreate the solver if grid dimensions or spacing changed */
    if (solver
        && (solver->nx != nx || solver->ny != ny || solver->nz != nz
            || solver->dx != dx || solver->dy != dy || solver->dz != dz)) {
        poisson_solver_destroy(solver);
        solver = NULL;
    }

    if (!solver) {
        /* Register cleanup on first use */
        if (cfd_atomic_cas(&g_cleanup_registered, 0, 1)) {
            atexit(cleanup_cached_solvers);
        }

        solver = poisson_solver_create(method, backend);
        if (!solver) {
            return -1;
        }

        /* The convenience API has no params argument, so the PCG_MG
         * preset carries its preconditioner into the cached instance. */
        poisson_solver_params_t pcg_mg_params;
        const poisson_solver_params_t* init_params = NULL;
        if (solver_type == POISSON_SOLVER_PCG_MG_SCALAR) {
            pcg_mg_params = poisson_solver_params_default();
            pcg_mg_params.preconditioner = POISSON_PRECOND_MULTIGRID;
            init_params = &pcg_mg_params;
        }

        /* A failed init (e.g. multigrid on non-2^k+1 dims) must not leave a
         * broken solver in the cache. */
        if (poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, init_params) != CFD_SUCCESS) {
            poisson_solver_destroy(solver);
            return -1;
        }
    }

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    /* Put the instance back. Anything it displaces was put back by a concurrent
     * call and is idle. */
    poisson_solver_destroy((poisson_solver_t*)cfd_atomic_ptr_exchange(slot, solver));

    return (status == CFD_SUCCESS && stats.status == POISSON_CONVERGED)
        ? stats.iterations : -1;
}

int poisson_solve(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy,
    poisson_solver_type solver_type)
{
    return poisson_solve_3d(p, p_temp, rhs, nx, ny, 1, dx, dy, 0.0, solver_type);
}

/* Direct solver functions - delegate to unified interface */
int poisson_solve_sor_scalar(
    double* p, const double* rhs,
    size_t nx, size_t ny, double dx, double dy)
{
    /* SOR doesn't need temp buffer, pass NULL */
    return poisson_solve(p, NULL, rhs, nx, ny, dx, dy, POISSON_SOLVER_SOR_SCALAR);
}

/* SIMD functions with runtime CPU detection */
int poisson_solve_jacobi_simd(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy)
{
    return poisson_solve(p, p_temp, rhs, nx, ny, dx, dy, POISSON_SOLVER_JACOBI_SIMD);
}

int poisson_solve_redblack_simd(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy)
{
    return poisson_solve(p, p_temp, rhs, nx, ny, dx, dy, POISSON_SOLVER_REDBLACK_SIMD);
}
