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

/* ============================================================================
 * DEFAULT PARAMETERS
 * ============================================================================ */

poisson_solver_params_t poisson_solver_params_default(void) {
    poisson_solver_params_t params;
    params.tolerance = 1e-6;
    params.absolute_tolerance = 1e-10;
    params.max_iterations = 1000;
    params.omega = 1.5;
    params.check_interval = 1;
    params.verbose = false;
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

poisson_solver_t* poisson_solver_create(
    poisson_solver_method_t method,
    poisson_solver_backend_t backend)
{
    /* Auto-select backend if requested */
    if (backend == POISSON_BACKEND_AUTO) {
        backend = select_best_backend();
    }

    /* Create appropriate solver - no silent fallbacks */
    switch (method) {
        case POISSON_METHOD_JACOBI:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_jacobi_simd_solver();
                case POISSON_BACKEND_SCALAR:
                    return create_jacobi_scalar_solver();
                default:
                    return NULL;  /* Requested backend not available for Jacobi */
            }

        case POISSON_METHOD_SOR:
        case POISSON_METHOD_GAUSS_SEIDEL:
            /* SOR/GS are inherently sequential - only scalar backend */
            if (backend == POISSON_BACKEND_SCALAR) {
                return create_sor_scalar_solver();
            }
            return NULL;  /* SOR not available for SIMD/OMP/GPU */

        case POISSON_METHOD_REDBLACK_SOR:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_redblack_simd_solver();
#ifdef CFD_ENABLE_OPENMP
                case POISSON_BACKEND_OMP:
                    return create_redblack_omp_solver();
#endif
                case POISSON_BACKEND_SCALAR:
                    return create_redblack_scalar_solver();
                default:
                    return NULL;  /* Requested backend not available for Red-Black */
            }

        case POISSON_METHOD_CG:
            switch (backend) {
                case POISSON_BACKEND_SIMD:
                    return create_cg_simd_solver();
                case POISSON_BACKEND_SCALAR:
                    return create_cg_scalar_solver();
                default:
                    return NULL;  /* Requested backend not available for CG */
            }

        case POISSON_METHOD_BICGSTAB:
            /* BiCGSTAB for non-symmetric systems - currently only scalar backend */
            if (backend == POISSON_BACKEND_SCALAR) {
                return create_bicgstab_scalar_solver();
            }
            return NULL;  /* SIMD/OMP/GPU not yet implemented for BiCGSTAB */

        case POISSON_METHOD_MULTIGRID:
            /* Not yet implemented */
            return NULL;

        default:
            return NULL;
    }
}

cfd_status_t poisson_solver_init(
    poisson_solver_t* solver,
    size_t nx, size_t ny,
    double dx, double dy,
    const poisson_solver_params_t* params)
{
    if (!solver) {
        return CFD_ERROR_INVALID;
    }

    solver->nx = nx;
    solver->ny = ny;
    solver->dx = dx;
    solver->dy = dy;

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
        return solver->init(solver, nx, ny, dx, dy, &solver->params);
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

    double max_residual = 0.0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            /* Compute Laplacian: d^2x/dx^2 + d^2x/dy^2 */
            double laplacian = (x[idx + 1] - 2.0 * x[idx] + x[idx - 1]) / dx2
                             + (x[idx + nx] - 2.0 * x[idx] + x[idx - nx]) / dy2;

            double residual = fabs(laplacian - rhs[idx]);
            if (residual > max_residual) {
                max_residual = residual;
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
    } else {
        /* Default: Neumann BCs (zero gradient) */
        bc_apply_scalar(x, solver->nx, solver->ny, BC_TYPE_NEUMANN);
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

            if (params->verbose) {
                printf("  Iter %d: residual = %.6e\n", iter, res);
            }

            if (res < tolerance || res < params->absolute_tolerance) {
                converged = 1;
                break;
            }
        }
    }

    double end_time = poisson_solver_get_time_ms();

    if (stats) {
        stats->iterations = iter + 1;
        stats->final_residual = res;
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = converged ? POISSON_CONVERGED : POISSON_MAX_ITER;
    }

    return converged ? CFD_SUCCESS : CFD_ERROR_MAX_ITER;
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
 * Cached solver instances for poisson_solve() convenience API.
 * Avoids creating/destroying solvers on each call.
 */
static poisson_solver_t* g_cached_jacobi_simd = NULL;
static poisson_solver_t* g_cached_sor = NULL;
static poisson_solver_t* g_cached_redblack_simd = NULL;
static poisson_solver_t* g_cached_redblack_omp = NULL;
static poisson_solver_t* g_cached_redblack_scalar = NULL;
static size_t g_cached_nx = 0;
static size_t g_cached_ny = 0;

/**
 * Cleanup cached solvers (called at program exit)
 */
static void cleanup_cached_solvers(void) {
    if (g_cached_jacobi_simd) {
        poisson_solver_destroy(g_cached_jacobi_simd);
        g_cached_jacobi_simd = NULL;
    }
    if (g_cached_sor) {
        poisson_solver_destroy(g_cached_sor);
        g_cached_sor = NULL;
    }
    if (g_cached_redblack_simd) {
        poisson_solver_destroy(g_cached_redblack_simd);
        g_cached_redblack_simd = NULL;
    }
    if (g_cached_redblack_omp) {
        poisson_solver_destroy(g_cached_redblack_omp);
        g_cached_redblack_omp = NULL;
    }
    if (g_cached_redblack_scalar) {
        poisson_solver_destroy(g_cached_redblack_scalar);
        g_cached_redblack_scalar = NULL;
    }
    g_cached_nx = 0;
    g_cached_ny = 0;
}

int poisson_solve(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, double dx, double dy,
    poisson_solver_type solver_type)
{
    poisson_solver_t** solver_ptr;
    poisson_solver_method_t method;
    poisson_solver_backend_t backend;

    switch (solver_type) {
        case POISSON_SOLVER_JACOBI_SIMD:
            solver_ptr = &g_cached_jacobi_simd;
            method = POISSON_METHOD_JACOBI;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_REDBLACK_SIMD:
            solver_ptr = &g_cached_redblack_simd;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_SIMD;
            break;

        case POISSON_SOLVER_REDBLACK_OMP:
            solver_ptr = &g_cached_redblack_omp;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_OMP;
            break;

        case POISSON_SOLVER_SOR_SCALAR:
            solver_ptr = &g_cached_sor;
            method = POISSON_METHOD_SOR;
            backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_SOLVER_REDBLACK_SCALAR:
            solver_ptr = &g_cached_redblack_scalar;
            method = POISSON_METHOD_REDBLACK_SOR;
            backend = POISSON_BACKEND_SCALAR;
            break;

        default:
            fprintf(stderr, "poisson_solve: Unknown solver type %d\n", solver_type);
            return -1;
    }

    /* Recreate solver if grid size changed */
    if (*solver_ptr == NULL || g_cached_nx != nx || g_cached_ny != ny) {
        /* Register cleanup on first use */
        static int cleanup_registered = 0;
        if (!cleanup_registered) {
            atexit(cleanup_cached_solvers);
            cleanup_registered = 1;
        }

        /* Destroy old solver if exists */
        if (*solver_ptr) {
            poisson_solver_destroy(*solver_ptr);
            *solver_ptr = NULL;
        }

        /* Create new solver */
        *solver_ptr = poisson_solver_create(method, backend);

        if (*solver_ptr) {
            poisson_solver_init(*solver_ptr, nx, ny, dx, dy, NULL);
            g_cached_nx = nx;
            g_cached_ny = ny;
        }
    }

    if (!*solver_ptr) {
        return -1;
    }

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t status = poisson_solver_solve(*solver_ptr, p, p_temp, rhs, &stats);

    if (status == CFD_SUCCESS && stats.status == POISSON_CONVERGED) {
        return stats.iterations;
    }

    /* Return iterations even if not converged (legacy behavior) */
    if (stats.iterations > 0) {
        return -1;  /* Signal non-convergence */
    }

    return -1;
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
