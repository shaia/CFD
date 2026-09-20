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
#include "multigrid_internal.h"  /* mg_subtract_interior_mean */

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

poisson_walls_t poisson_walls_default(void) {
    poisson_walls_t walls;
    memset(&walls, 0, sizeof walls);  /* POISSON_WALL_ZERO_GRADIENT is 0 */
    return walls;
}

poisson_walls_t poisson_walls_uniform(poisson_wall_t type, double value) {
    poisson_walls_t walls = poisson_walls_default();
    walls.left = walls.right = walls.bottom = walls.top = walls.front = walls.back = type;
    if (type == POISSON_WALL_DIRICHLET) {
        walls.values.left = walls.values.right = value;
        walls.values.bottom = walls.values.top = value;
        walls.values.front = walls.values.back = value;
    }
    return walls;
}

/** Whether every face is zero-gradient, i.e. the operator the solvers default to. */
static int walls_are_all_zero_gradient(const poisson_walls_t* w)
{
    return w->left == POISSON_WALL_ZERO_GRADIENT
        && w->right == POISSON_WALL_ZERO_GRADIENT
        && w->bottom == POISSON_WALL_ZERO_GRADIENT
        && w->top == POISSON_WALL_ZERO_GRADIENT
        && w->front == POISSON_WALL_ZERO_GRADIENT
        && w->back == POISSON_WALL_ZERO_GRADIENT;
}

/** Whether this method's halo routines honour per-face walls (Krylov only). */
static int method_honours_walls(poisson_solver_method_t method) {
    return method == POISSON_METHOD_CG
        || method == POISSON_METHOD_BICGSTAB
        || method == POISSON_METHOD_GMRES;
}

cfd_status_t poisson_solver_check_walls(const poisson_solver_t* solver) {
    const poisson_walls_t* walls = &solver->params.walls;

    /* Checked over every face, the z-faces on a 2D grid included: a Dirichlet face
     * that does not exist changes no answer, but it does mean the caller believes
     * something about this solve that is not true, so say so rather than ignore it. */
    if (walls_are_all_zero_gradient(walls)) {
        return CFD_SUCCESS;  /* what every solver and backend already does */
    }

    /* Support first, then the hook conflict: an unsupported method should say so
     * rather than report a caller error. A solver's own walls live in
     * internal_apply_bc, so apply_bc below is the caller's and nothing else. */
    if (!method_honours_walls(solver->method)) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "per-face walls are implemented for the CG, BiCGSTAB and GMRES methods only; "
            "the stationary and multigrid solvers apply whole-domain walls inside their sweeps");
        return CFD_ERROR_UNSUPPORTED;
    }

    if (solver->backend == POISSON_BACKEND_GPU) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "per-face walls are not implemented on the GPU backend, which applies the "
            "zero-gradient walls on the device");
        return CFD_ERROR_UNSUPPORTED;
    }

    /* Past here the method is a Krylov one, which never installs a hook of its
     * own, so a hook here is the caller's. It and params.walls both prescribe wall
     * values; two sources for one thing is ambiguous, so take neither. */
    if (solver->apply_bc) {
        cfd_set_error(CFD_ERROR_INVALID,
            "params.walls and a custom apply_bc both prescribe wall values; use one or the other");
        return CFD_ERROR_INVALID;
    }

    if (!isfinite(walls->values.left) || !isfinite(walls->values.right)
        || !isfinite(walls->values.bottom) || !isfinite(walls->values.top)
        || !isfinite(walls->values.front) || !isfinite(walls->values.back)) {
        cfd_set_error(CFD_ERROR_INVALID, "params.walls.values must be finite");
        return CFD_ERROR_INVALID;
    }

    return CFD_SUCCESS;
}

bool poisson_walls_are_singular(const poisson_walls_t* walls, size_t nz) {
    if (!walls) {
        return true;  /* no walls given means the defaults, which are singular */
    }
    /* The z-faces do not exist on a 2D grid, so whatever they say is irrelevant. */
    if (nz > 1 && (walls->front != POISSON_WALL_ZERO_GRADIENT
                   || walls->back != POISSON_WALL_ZERO_GRADIENT)) {
        return false;
    }
    return walls->left == POISSON_WALL_ZERO_GRADIENT
        && walls->right == POISSON_WALL_ZERO_GRADIENT
        && walls->bottom == POISSON_WALL_ZERO_GRADIENT
        && walls->top == POISSON_WALL_ZERO_GRADIENT;
}

poisson_solver_params_t poisson_solver_params_default(void) {
    /* Zeroed first: the assignments below cover every named field, but the struct
     * is also compared with memcmp as the convenience-API cache key, so padding
     * has to be deterministic too. */
    poisson_solver_params_t params;
    memset(&params, 0, sizeof params);
    params.tolerance = 1e-6;
    params.absolute_tolerance = 1e-10;
    params.max_iterations = 5000;  /* Increased from 1000 for CG on fine grids */
    params.sor.omega = 0.0;  /* Auto-compute optimal omega for grid dimensions */
    params.check_interval = 1;
    params.verbose = false;
    params.krylov.preconditioner = POISSON_PRECOND_NONE;
    params.krylov.restart = 0;  /* 0 = auto (GMRES_DEFAULT_RESTART); ignored by non-GMRES methods */
    params.multigrid.cycle = MG_CYCLE_V;
    params.multigrid.smoother = MG_SMOOTHER_REDBLACK_GS;
    params.multigrid.bc = MG_BC_NEUMANN;
    params.multigrid.pre_smooth = 0;      /* 0 = default (2) */
    params.multigrid.post_smooth = 0;     /* 0 = default (2) */
    params.multigrid.coarse_max_iter = 0; /* 0 = default (50) */
    params.multigrid.max_levels = 0;      /* 0 = auto */
    params.walls = poisson_walls_default();
    return params;
}

poisson_solver_config_t poisson_solver_config_preset(poisson_preset_t preset) {
    poisson_solver_config_t cfg;
    cfg.method = POISSON_METHOD_CG;
    cfg.backend = POISSON_BACKEND_AUTO;  /* never named by a preset */
    cfg.params = poisson_solver_params_default();

    switch (preset) {
        case POISSON_PRESET_ACCURATE:
            cfg.params.tolerance = 1e-10;
            cfg.params.absolute_tolerance = 1e-14;
            cfg.params.max_iterations = 20000;
            break;

        case POISSON_PRESET_NONSYMMETRIC:
            cfg.method = POISSON_METHOD_BICGSTAB;
            break;

        case POISSON_PRESET_SMOOTHER:
            /* A fixed number of sweeps, deliberately not a convergence request:
             * the tolerances are 0 so the loop runs its budget out. This is the
             * one preset whose operator tolerates an incompatible rhs. */
            cfg.method = POISSON_METHOD_REDBLACK_SOR;
            cfg.params.tolerance = 0.0;
            cfg.params.absolute_tolerance = 0.0;
            cfg.params.max_iterations = 20;
            break;

        case POISSON_PRESET_MULTIGRID:
            cfg.method = POISSON_METHOD_MULTIGRID;
            break;

        case POISSON_PRESET_MULTIGRID_PCG:
            /* An ordinary CG config: the preconditioner is a parameter, not a
             * property of the preset, so nothing downstream needs a special case. */
            cfg.params.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
            break;

        case POISSON_PRESET_DEFAULT:
        default:
            break;
    }

    return cfg;
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

    {
        cfd_status_t wall_status = poisson_solver_check_walls(solver);
        if (wall_status != CFD_SUCCESS) {
            return wall_status;
        }
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

/**
 * Write one face's halo: copy the adjacent interior out for a zero-gradient face,
 * or write a constant for a Dirichlet one.
 *
 * `stride` steps inward from the face, `n_outer`/`n_inner` walk the face itself.
 */
static void write_face(double* x, size_t base, ptrdiff_t stride,
                       size_t n_outer, size_t outer_stride,
                       size_t n_inner, size_t inner_stride,
                       poisson_wall_t type, double value)
{
    for (size_t a = 0; a < n_outer; a++) {
        for (size_t b = 0; b < n_inner; b++) {
            size_t idx = base + (a * outer_stride) + (b * inner_stride);
            x[idx] = (type == POISSON_WALL_DIRICHLET)
                   ? value
                   : x[(size_t)((ptrdiff_t)idx + stride)];
        }
    }
}

/**
 * Apply per-face walls to x's halo.
 *
 * homogeneous != 0 writes 0 on every Dirichlet face: that is the homogeneous part
 * of the condition, which is what a Krylov search direction must carry. The
 * zero-gradient extension is already its own homogeneous form, so it is identical
 * either way.
 *
 * Faces run left, right, bottom, top, back, front so the order is deterministic;
 * a 5/7-point stencil never reads a corner halo cell, so which face owns a shared
 * corner does not change any answer.
 */
static void poisson_apply_walls(const poisson_solver_t* s, double* x, int homogeneous)
{
    const poisson_walls_t* w = &s->params.walls;
    size_t nx = s->nx;
    size_t ny = s->ny;
    size_t nz = s->nz;
    size_t plane = nx * ny;

    #define WALL_VALUE(face) (homogeneous ? 0.0 : w->values.face)

    /* x-faces: whole column of each plane */
    write_face(x, 0, +1, nz, plane, ny, nx, w->left, WALL_VALUE(left));
    write_face(x, nx - 1, -1, nz, plane, ny, nx, w->right, WALL_VALUE(right));

    /* y-faces: whole row of each plane */
    write_face(x, 0, (ptrdiff_t)nx, nz, plane, nx, 1, w->bottom, WALL_VALUE(bottom));
    write_face(x, (ny - 1) * nx, -(ptrdiff_t)nx, nz, plane, nx, 1, w->top, WALL_VALUE(top));

    /* z-faces: whole plane. Only on a 3D grid. */
    if (nz > 1) {
        write_face(x, 0, (ptrdiff_t)plane, 1, 0, plane, 1, w->back, WALL_VALUE(back));
        write_face(x, (nz - 1) * plane, -(ptrdiff_t)plane, 1, 0, plane, 1,
                   w->front, WALL_VALUE(front));
    }

    #undef WALL_VALUE
}

void poisson_solver_apply_bc(
    poisson_solver_t* solver,
    double* x)
{
    if (!solver || !x) {
        return;
    }

    /* The caller's function wins; a solver's own is the fallback. Nothing sets
     * both -- a solver that installs internal_apply_bc rejects a caller function
     * at init -- so the precedence never actually has to arbitrate. */
    if (solver->apply_bc) {
        solver->apply_bc(solver, x);
        return;
    }
    if (solver->internal_apply_bc) {
        solver->internal_apply_bc(solver, x);
        return;
    }

    /* Per-face walls, when the caller configured any. The all-zero-gradient case
     * falls through to the backend BC primitives below, which keeps the default
     * free and lets OMP/SIMD apply their walls in parallel. */
    if (!walls_are_all_zero_gradient(&solver->params.walls)) {
        poisson_apply_walls(solver, x, 0);
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

/** Write 0 into every boundary cell, leaving the interior alone. */
static void krylov_zero_halo(const poisson_solver_t* solver, double* x)
{
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t nz = solver->nz;
    size_t plane_size = nx * ny;

    size_t k_first = 0;
    size_t k_last = 1;
    if (nz > 1) {
        memset(x, 0, plane_size * sizeof(double));
        memset(x + (nz - 1) * plane_size, 0, plane_size * sizeof(double));
        k_first = 1;
        k_last = nz - 1;
    }

    for (size_t k = k_first; k < k_last; k++) {
        double* plane = x + k * plane_size;
        memset(plane, 0, nx * sizeof(double));                 /* j = 0 */
        memset(plane + (ny - 1) * nx, 0, nx * sizeof(double)); /* j = ny-1 */
        for (size_t j = 1; j < ny - 1; j++) {
            plane[j * nx] = 0.0;                               /* i = 0 */
            plane[(j * nx) + (nx - 1)] = 0.0;                  /* i = nx-1 */
        }
    }
}

int poisson_solver_krylov_is_singular(const poisson_solver_t* solver)
{
    /* A hook prescribes wall values, which pins the level and makes the operator
     * nonsingular, exactly as a Dirichlet face does. */
    return solver
        && solver->apply_bc == NULL
        && poisson_walls_are_singular(&solver->params.walls, solver->nz);
}

void poisson_solver_krylov_apply_bc_homogeneous(
    poisson_solver_t* solver,
    double* v)
{
    if (!solver || !v) {
        return;
    }

    /* The zero-gradient extension is already homogeneous -- it copies the interior
     * outwards and adds nothing -- so all-zero-gradient walls need it verbatim.
     * With a Dirichlet face configured, the prescribed value belongs to the iterate
     * and the direction gets 0 there instead. */
    if (!solver->apply_bc) {
        if (walls_are_all_zero_gradient(&solver->params.walls)) {
            poisson_solver_apply_bc(solver, v);
        } else {
            poisson_apply_walls(solver, v, 1);
        }
        return;
    }

    /* A custom hook prescribes wall values that do not depend on the interior, so
     * its homogeneous part is a zero halo. Running the hook on a search direction
     * would add its lift to every one of them, which is not a linear operator and
     * would leave the Krylov recurrences describing something other than A. */
    krylov_zero_halo(solver, v);
}

void poisson_solver_krylov_apply_bc(
    poisson_solver_t* solver,
    double* x)
{
    if (!solver || !x) {
        return;
    }

    /* The iterate, unlike a search direction, carries the lift as well, so the
     * residual formed from it is the residual of the real system. For the default
     * walls that is just the extension. */
    if (!solver->apply_bc) {
        poisson_solver_apply_bc(solver, x);
        return;
    }

    /* Zero first, then let the hook write over it: a hook that writes only some
     * walls, or that holds them at zero by writing nothing, then still leaves no
     * stale value behind for a warm start to trip on. */
    krylov_zero_halo(solver, x);
    solver->apply_bc(solver, x);
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

    /* A start whose residual is not finite has diverged before the first sweep */
    if (!isfinite(initial_res)) {
        if (stats) {
            stats->status = POISSON_DIVERGED;
            stats->iterations = 0;
            stats->final_residual = initial_res;
            stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
        }
        return CFD_ERROR_DIVERGED;
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

        /* Check at intervals and on the last iteration, so that a solve never ends
         * on a residual older than the field it returns */
        int check = (iter % params->check_interval == 0) || (iter + 1 == params->max_iterations);

        /* Perform one iteration */
        cfd_status_t status = solver->iterate(solver, x, x_temp, rhs, check ? &new_res : NULL);

        if (status != CFD_SUCCESS) {
            if (stats) {
                stats->status = POISSON_ERROR;
                stats->iterations = iter + 1;
                stats->final_residual = res;
                stats->elapsed_time_ms = poisson_solver_get_time_ms() - start_time;
            }
            return status;
        }

        if (check) {
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

void poisson_make_rhs_compatible(double* rhs, size_t nx, size_t ny, size_t nz) {
    if (rhs) {
        mg_subtract_interior_mean(rhs, nx, ny, nz);
    }
}

/**
 * Reject a right-hand side the singular operator has no solution for.
 *
 * With zero-gradient walls the constants are the operator's nullspace, so a rhs
 * with a nonzero interior mean lies partly outside its range: there is no x that
 * solves it, and the iteration either stalls on that component or, for Krylov
 * methods, drives the rest of the field away chasing it.
 *
 * The mean is compared against the residual the caller asked for, scaled by the
 * size of the rhs -- an incompatibility smaller than the target tolerance cannot
 * affect the answer, and the projection solvers leave one of that size behind
 * after mean-subtracting in floating point.
 */
static cfd_status_t check_rhs_compatible(const poisson_solver_t* solver, const double* rhs)
{
    /* Krylov methods only. On a system with no solution they do not merely fail to
     * converge: the minimisation chases the component in the nullspace and drives
     * the rest of the field away with it, which is how test_laplacian_accuracy
     * reached a residual of 1e21. The stationary and multigrid solvers relax
     * towards a solution modulo a drifting constant instead, which callers use
     * deliberately -- running a fixed sweep count on an arbitrary rhs is a
     * reasonable thing to ask of them, so it is left alone. */
    if (!method_honours_walls(solver->method)) {
        return CFD_SUCCESS;
    }

    if (!poisson_solver_krylov_is_singular(solver)) {
        return CFD_SUCCESS;  /* a prescribed face pins the level; any rhs is fine */
    }

    /* Solve before init leaves the dimensions at 0, where the interior loops below
     * would underflow. Let the solver itself report that; there is no rhs to judge
     * against a grid that does not exist yet. */
    if (solver->nx < 3 || solver->ny < 3) {
        return CFD_SUCCESS;
    }

    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t nz = solver->nz;
    size_t stride_z = (nz > 1) ? nx * ny : 0;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? nz - 1 : 1;

    double sum = 0.0;
    double sum_abs = 0.0;
    size_t count = 0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                double v = rhs[(k * stride_z) + IDX_2D(i, j, nx)];
                sum += v;
                sum_abs += fabs(v);
                count++;
            }
        }
    }
    if (count == 0 || sum_abs == 0.0) {
        return CFD_SUCCESS;
    }

    /* Relative to the mean magnitude, so the test is scale-free. */
    double tol = solver->params.tolerance;
    if (tol <= 0.0) {
        tol = 1e-6;
    }
    if (fabs(sum) > tol * sum_abs) {
        cfd_set_error(CFD_ERROR_INVALID,
            "zero-gradient walls make this operator singular, so the rhs must have zero "
            "interior mean; call poisson_make_rhs_compatible() first, or prescribe a "
            "wall value on one face via params.walls");
        return CFD_ERROR_INVALID;
    }
    return CFD_SUCCESS;
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

    if (rhs) {
        cfd_status_t compat = check_rhs_compatible(solver, rhs);
        if (compat != CFD_SUCCESS) {
            if (stats) {
                stats->status = POISSON_INCOMPATIBLE_RHS;
                stats->iterations = 0;
            }
            return compat;
        }
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
static cfd_atomic_ptr g_cached_pcg_mg_omp;

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
    poisson_solver_destroy(take_cached_solver(&g_cached_pcg_mg_omp));
}

int poisson_solve_3d(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    poisson_solver_type solver_type)
{
    return poisson_solve_3d_params(p, p_temp, rhs, nx, ny, nz, dx, dy, dz,
                                   solver_type, NULL);
}

int poisson_solve_3d_params(
    double* p, double* p_temp, const double* rhs,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    poisson_solver_type solver_type,
    const poisson_solver_params_t* params)
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

        case POISSON_SOLVER_PCG_MG_OMP:
            slot = &g_cached_pcg_mg_omp;
            method = POISSON_METHOD_CG;
            backend = POISSON_BACKEND_OMP;
            break;

        default:
            CFD_LOG_ERROR("poisson", "poisson_solve_3d: Unknown solver type %d", solver_type);
            return -1;
    }

    /* The parameters this call wants. The two PCG_MG presets ARE the choice of
     * preconditioner, so it is forced here rather than left to the caller. */
    poisson_solver_params_t effective =
        params ? *params : poisson_solver_params_default();
    if (solver_type == POISSON_SOLVER_PCG_MG_SCALAR ||
        solver_type == POISSON_SOLVER_PCG_MG_OMP) {
        effective.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
    }

    /* This call owns the cached instance until it puts it back */
    poisson_solver_t* solver = take_cached_solver(slot);

    /* Recreate the solver if the grid or the parameters changed. memcmp is safe
     * as a cache key in the only direction that matters: padding can make two
     * equal configurations compare different, costing a rebuild, but never makes
     * two different ones compare equal. poisson_solver_params_default() zeroes
     * the struct so well-formed callers stay on the fast path. A caller that
     * varies parameters per call rebuilds per call. */
    if (solver
        && (solver->nx != nx || solver->ny != ny || solver->nz != nz
            || solver->dx != dx || solver->dy != dy || solver->dz != dz
            || memcmp(&solver->params, &effective, sizeof effective) != 0)) {
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

        /* A failed init (e.g. multigrid on non-2^k+1 dims, or per-face walls on a
         * method that does not honour them) must not leave a broken solver in the
         * cache. */
        if (poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &effective) != CFD_SUCCESS) {
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
