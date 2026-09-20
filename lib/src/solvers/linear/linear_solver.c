/**
 * @file linear_solver.c
 * @brief Core linear solver implementation
 *
 * Implements:
 * - Default parameter functions
 * - Backend selection
 * - Solver lifecycle (create, init, destroy)
 * - Common solve loop
 * - Configuration validation
 * - The poisson_solve() convenience entry point
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

/* ============================================================================
 * CONFIGURATION VALIDATION
 *
 * One place decides what a given (method, backend) can honour, so a parameter it
 * cannot is refused at init rather than quietly ignored. The alternative -- a
 * check inside each of the twenty-odd solver inits -- is what let BiCGSTAB ignore
 * params.krylov.preconditioner on every backend, and three GPU solvers ignore a
 * caller's apply_bc, for as long as they have existed.
 * ============================================================================ */

/** Parameter groups, for the ownership table below. */
enum {
    PARAM_GROUP_SOR       = 1u << 0,
    PARAM_GROUP_KRYLOV    = 1u << 1,
    PARAM_GROUP_MULTIGRID = 1u << 2
};

/**
 * The groups this method reads. Anything else it is handed, it ignores -- which
 * is exactly what must be refused.
 */
static unsigned method_owned_groups(const poisson_solver_t* solver) {
    switch (solver->method) {
        case POISSON_METHOD_JACOBI:
            return 0;
        case POISSON_METHOD_SOR:
        case POISSON_METHOD_GAUSS_SEIDEL:
        case POISSON_METHOD_REDBLACK_SOR:
            return PARAM_GROUP_SOR;
        case POISSON_METHOD_CG:
            /* CG reads the multigrid group only when it is preconditioned by
             * one, where those parameters shape the inner V-cycle. */
            return PARAM_GROUP_KRYLOV
                 | (solver->params.krylov.preconditioner == POISSON_PRECOND_MULTIGRID
                        ? PARAM_GROUP_MULTIGRID : 0u);
        case POISSON_METHOD_BICGSTAB:
        case POISSON_METHOD_GMRES:
            return PARAM_GROUP_KRYLOV;
        case POISSON_METHOD_MULTIGRID:
            return PARAM_GROUP_MULTIGRID;
        default:
            return 0;
    }
}

/** Whether a parameter group is entirely zero, i.e. untouched by the caller. */
static int group_untouched(const void* group, size_t size) {
    const unsigned char* bytes = (const unsigned char*)group;
    for (size_t i = 0; i < size; i++) {
        if (bytes[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static cfd_status_t reject_unowned_group(const poisson_solver_t* solver,
                                         unsigned owned, unsigned group,
                                         const void* data, size_t size,
                                         const char* name) {
    if ((owned & group) || group_untouched(data, size)) {
        return CFD_SUCCESS;
    }
    char msg[192];
    snprintf(msg, sizeof(msg),
             "params.%s is set, but the %s method does not read it; "
             "it would have no effect on this solve",
             name, solver->name ? solver->name : "selected");
    cfd_set_error(CFD_ERROR_INVALID, msg);
    return CFD_ERROR_INVALID;
}

cfd_status_t poisson_solver_check_config(const poisson_solver_t* solver) {
    const poisson_solver_params_t* p = &solver->params;

    /* ---- walls -------------------------------------------------------- */
    if (!walls_are_all_zero_gradient(&p->walls)) {
        /* Support first, then the hook conflict: an unsupported method should say
         * so rather than report a caller error. */
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
        if (solver->apply_bc) {
            cfd_set_error(CFD_ERROR_INVALID,
                "params.walls and a custom apply_bc both prescribe wall values; use one or the other");
            return CFD_ERROR_INVALID;
        }
        if (!isfinite(p->walls.values.left) || !isfinite(p->walls.values.right)
            || !isfinite(p->walls.values.bottom) || !isfinite(p->walls.values.top)
            || !isfinite(p->walls.values.front) || !isfinite(p->walls.values.back)) {
            cfd_set_error(CFD_ERROR_INVALID, "params.walls.values must be finite");
            return CFD_ERROR_INVALID;
        }
        /* A 2D grid has no z-faces to prescribe. poisson_apply_walls skips them
         * and poisson_walls_are_singular ignores them, so without this the face
         * would be configured, accepted, and have no effect -- the silent ignore
         * this whole function exists to prevent. */
        if (solver->nz <= 1
            && (p->walls.front != POISSON_WALL_ZERO_GRADIENT
                || p->walls.back != POISSON_WALL_ZERO_GRADIENT)) {
            cfd_set_error(CFD_ERROR_INVALID,
                "params.walls.front and .back describe z-faces, which a 2D grid "
                "(nz == 1) does not have; they would have no effect on this solve");
            return CFD_ERROR_INVALID;
        }
    }

    /* ---- a caller's apply_bc ------------------------------------------ */
    /* solver->internal_apply_bc is a solver's own walls and is not a caller's
     * business; this is only about a function the caller installed. */
    if (solver->apply_bc) {
        if (solver->backend == POISSON_BACKEND_GPU) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                "the GPU solvers apply their walls on the device and never call apply_bc");
            return CFD_ERROR_UNSUPPORTED;
        }
        if (solver->method == POISSON_METHOD_MULTIGRID) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                "multigrid applies its walls inside the cycle and never calls apply_bc; "
                "use params.multigrid.bc to choose between zero-gradient and fixed walls");
            return CFD_ERROR_UNSUPPORTED;
        }
    }

    /* ---- parameter groups the method does not read --------------------- */
    {
        unsigned owned = method_owned_groups(solver);
        cfd_status_t status;

        status = reject_unowned_group(solver, owned, PARAM_GROUP_SOR,
                                      &p->sor, sizeof p->sor, "sor");
        if (status != CFD_SUCCESS) {
            return status;
        }
        status = reject_unowned_group(solver, owned, PARAM_GROUP_KRYLOV,
                                      &p->krylov, sizeof p->krylov, "krylov");
        if (status != CFD_SUCCESS) {
            return status;
        }
        status = reject_unowned_group(solver, owned, PARAM_GROUP_MULTIGRID,
                                      &p->multigrid, sizeof p->multigrid, "multigrid");
        if (status != CFD_SUCCESS) {
            return status;
        }
    }

    /* ---- exceptions within a group the method does own ----------------- */

    /* BiCGSTAB reads params.krylov, but has no preconditioner implementation on
     * any backend. It ignored one silently for as long as it has existed. */
    if (solver->method == POISSON_METHOD_BICGSTAB
        && p->krylov.preconditioner != POISSON_PRECOND_NONE) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "BiCGSTAB has no preconditioner implementation on any backend");
        return CFD_ERROR_UNSUPPORTED;
    }

    /* The multigrid preconditioner is implemented by the scalar and OpenMP CG
     * solvers only. Every other Krylov backend would quietly run unpreconditioned. */
    if (p->krylov.preconditioner == POISSON_PRECOND_MULTIGRID
        && !(solver->method == POISSON_METHOD_CG
             && (solver->backend == POISSON_BACKEND_SCALAR
                 || solver->backend == POISSON_BACKEND_OMP))) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "POISSON_PRECOND_MULTIGRID is implemented by the scalar and OpenMP CG solvers only");
        return CFD_ERROR_UNSUPPORTED;
    }

    /* restart is the GMRES(m) basis size and means nothing to the others. */
    if (p->krylov.restart != 0 && solver->method != POISSON_METHOD_GMRES) {
        cfd_set_error(CFD_ERROR_INVALID,
            "params.krylov.restart is the GMRES restart length; no other method reads it");
        return CFD_ERROR_INVALID;
    }

    /* Deliberately not rejected here: POISSON_METHOD_GAUSS_SEIDEL overriding
     * params.sor.omega to 1. That is what the method IS, it is documented on the
     * enum member, and test_gauss_seidel_is_sor_at_omega_one pins it. Refusing an
     * omega there would turn documented behaviour into an error, which is not the
     * class of problem this validator exists for.
     */

    /* The inner V-cycle of a preconditioner must stay symmetric and nonsingular,
     * or CG is no longer minimising over a Krylov space. These four are what
     * guarantee it, so they are not the caller's to change here; the rest of the
     * group is forwarded to the hierarchy by poisson_solver_create_mg_precond. */
    if (p->krylov.preconditioner == POISSON_PRECOND_MULTIGRID
        && (p->multigrid.cycle != 0 || p->multigrid.smoother != 0 || p->multigrid.bc != 0
            || p->multigrid.pre_smooth != p->multigrid.post_smooth)) {
        cfd_set_error(CFD_ERROR_INVALID,
            "a multigrid preconditioner fixes its cycle, smoother, boundary mode and an "
            "equal pre/post sweep count, which is what keeps it symmetric for CG; "
            "params.multigrid.{pre_smooth, post_smooth, coarse_max_iter, max_levels} "
            "are yours to set");
        return CFD_ERROR_INVALID;
    }

    /* iter % check_interval, with check_interval == 0, is undefined behaviour in
     * the common loop and in the CG and BiCGSTAB loops. */
    if (p->check_interval <= 0) {
        cfd_set_error(CFD_ERROR_INVALID, "params.check_interval must be at least 1");
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
    /* AUTO unless the preset needs a backend that AUTO would not choose; see
     * the two multigrid cases below. */
    cfg.backend = POISSON_BACKEND_AUTO;
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
            /* The stationary option, and the one operator that tolerates an
             * incompatible rhs. It keeps the ordinary tolerances so an ordinary
             * solve reports success; a caller who wants a fixed sweep count
             * instead sets tolerance = 0 and max_iterations themselves, and then
             * CFD_ERROR_MAX_ITER is their own explicit choice rather than
             * something this preset returns on every call. */
            cfg.method = POISSON_METHOD_REDBLACK_SOR;
            break;

        case POISSON_PRESET_MULTIGRID:
            cfg.method = POISSON_METHOD_MULTIGRID;
            cfg.backend = POISSON_BACKEND_SCALAR;
            break;

        case POISSON_PRESET_MULTIGRID_PCG:
            /* The preconditioner is an ordinary parameter, so nothing downstream
             * needs a special case for it -- but the backend does need naming.
             * AUTO resolves to SIMD wherever AVX2 or NEON is present, and neither
             * multigrid nor the multigrid preconditioner has a SIMD backend, so
             * this preset would otherwise be refused at init on the very machines
             * it is most likely to run on. Scalar is the one backend AUTO could
             * have reached that implements it; POISSON_BACKEND_OMP is a single
             * assignment away for a caller who wants the threaded hierarchy. */
            cfg.params.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
            cfg.backend = POISSON_BACKEND_SCALAR;
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

const char* poisson_solver_status_string(poisson_solver_status_t status) {
    switch (status) {
        case POISSON_CONVERGED:        return "converged";
        case POISSON_MAX_ITER:         return "max iterations";
        case POISSON_DIVERGED:         return "diverged";
        case POISSON_STAGNATED:        return "stagnated";
        case POISSON_INCOMPATIBLE_RHS: return "incompatible rhs";
        case POISSON_ERROR:            return "error";
    }
    return "unknown";
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
        cfd_status_t wall_status = poisson_solver_check_config(solver);
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

    /* Serial on every backend, deliberately.
     *
     * A Neumann face is a copy from the adjacent interior line, so all three
     * backends produce identical values -- but this runs once per Krylov
     * iteration, and an OpenMP region costs ~12 us to enter at 4 threads on
     * MSVC while the work here is O(nx + ny): about 130 writes on a 33x33
     * plane, less than the 31x31 Laplacian sweep it accompanies. Threading it
     * cost more than the solve. BiCGSTAB pays this twice per iteration. */
    for (size_t k = 0; k < nz; k++) {
        bc_apply_scalar_cpu(x + k * plane_size, nx, ny, BC_TYPE_NEUMANN);
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
                /* Reset first: a caller reusing one stats struct across solves
                 * would otherwise read the previous solve's residual and timing
                 * next to this refusal, which looks like a converged result. */
                *stats = poisson_solver_stats_default();
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
 * CONVENIENCE API
 * ============================================================================ */

cfd_status_t poisson_solve(
    double* x, double* x_temp, const double* rhs,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_config_t* config,
    poisson_solver_stats_t* stats)
{
    if (!x || !rhs) {
        cfd_set_error(CFD_ERROR_INVALID, "poisson_solve: x and rhs are required");
        return CFD_ERROR_INVALID;
    }

    /* Cleared so that the status read back below is this call's and not whatever
     * the thread failed at last. */
    cfd_clear_error();

    poisson_solver_config_t cfg =
        config ? *config : poisson_solver_config_preset(POISSON_PRESET_DEFAULT);

    poisson_solver_t* solver = poisson_solver_create(cfg.method, cfg.backend);
    if (!solver) {
        /* The factory named the reason -- an unavailable backend, an unknown
         * method -- so report that rather than overwriting it. A factory that
         * returned NULL without setting one (an allocation failure inside a
         * create_*_solver) must not read back as success. */
        cfd_status_t reason = cfd_get_last_status();
        if (reason == CFD_SUCCESS) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                "poisson_solve: the requested method and backend could not be created");
            reason = CFD_ERROR_UNSUPPORTED;
        }
        return reason;
    }

    cfd_status_t status =
        poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &cfg.params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return status;
    }

    poisson_solver_stats_t local = poisson_solver_stats_default();
    status = poisson_solver_solve(solver, x, x_temp, rhs, stats ? stats : &local);
    poisson_solver_destroy(solver);
    return status;
}
