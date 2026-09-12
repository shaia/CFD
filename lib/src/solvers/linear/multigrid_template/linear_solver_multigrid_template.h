/**
 * @file linear_solver_multigrid_template.h
 * @brief Geometric multigrid algorithm shared by the multigrid backends
 *
 * V/W/F(FMG) cycles with Red-Black Gauss-Seidel (default) or weighted Jacobi
 * smoothing, full-weighting restriction and bilinear/trilinear prolongation.
 * Grid dimensions must be 2^k+1 per active dimension.
 *
 * Boundary-condition modes (params.mg_bc):
 * - MG_BC_NEUMANN (default): zero-gradient BCs matching the other Poisson
 *   solvers. The system is singular (constant nullspace); restricted RHS
 *   vectors are projected to zero interior mean, coarse corrections are
 *   mean-subtracted, and the solution converges up to an additive constant.
 * - MG_BC_DIRICHLET: caller-supplied boundary values of x are held fixed on
 *   the finest grid; coarse-level corrections use homogeneous zero boundaries.
 *
 * Level 0 (finest) borrows the caller's x/rhs arrays; coarser levels own
 * their buffers. The public x_temp argument is unused (like CG); the Jacobi
 * smoother allocates its own per-level temporaries.
 *
 * This header is the only copy of the algorithm: per-level BCs, the smoother
 * sweep loop, level transfers, the V/W cycle, FMG, init/destroy, iterate,
 * solve, apply_bc and the factory. Each backend includes it once, after
 * defining its grid primitives, so backends differ only in those primitives:
 *   cpu/linear_solver_multigrid.c       scalar
 *   omp/linear_solver_multigrid_omp.c   OpenMP
 * Control flow, level bookkeeping, whole-buffer memsets, the Neumann interior
 * mean (mg_interior_mean) and the convergence residual are serial in every
 * backend, so element-wise primitives reproduce the scalar result exactly.
 *
 * REQUIRED MACROS (every MGT_* macro is #undef'd at the end of this header):
 *   MGT_SUFFIX         symbol suffix (scalar, ...); the factory is
 *                      create_multigrid_<MGT_SUFFIX>_solver
 *   MGT_SOLVER_NAME    POISSON_SOLVER_TYPE_MG_* name string
 *   MGT_DESCRIPTION    description string
 *   MGT_BACKEND        POISSON_BACKEND_* value
 *
 * REQUIRED PRIMITIVES (each macro names a function; interior points only
 * unless noted; transfer contracts in multigrid_internal.h):
 *   void MGT_BC_NEUMANN_PLANE(plane, nx, ny)          zero-gradient BCs on one
 *                                                     nx x ny plane (all edges)
 *   void MGT_RBGS_SWEEP(L, x, rhs)                    one red then one black
 *                                                     Gauss-Seidel pass, no BCs
 *   void MGT_JACOBI_SWEEP(L, x, x_temp, rhs)          one MG_JACOBI_OMEGA-weighted
 *                                                     Jacobi pass into x_temp, then
 *                                                     interior copy back, no BCs
 *   void MGT_RESIDUAL(L, x, rhs, r)                   r = rhs - nabla^2 x
 *   void MGT_RESTRICT_2D(fine, coarse, nxf, nyf, nxc, nyc, fold_neumann)
 *   void MGT_RESTRICT_3D(fine, coarse, nxf, nyf, nzf, nxc, nyc, nzc, fold_neumann)
 *   void MGT_PROLONGATE_ADD_2D(coarse, fine, nxc, nyc, nxf, nyf)
 *   void MGT_PROLONGATE_ADD_3D(coarse, fine, nxc, nyc, nzc, nxf, nyf, nzf)
 *   void MGT_SUBTRACT_INTERIOR_MEAN(f, nx, ny, nz)
 *   void MGT_ZERO_INTERIOR(L, x)                      interior of x = 0
 */

#include "../linear_solver_internal.h"
#include "../multigrid_internal.h"

#include "cfd/core/memory.h"

#include <string.h>

#if !defined(MGT_SUFFIX) || !defined(MGT_SOLVER_NAME) || \
    !defined(MGT_DESCRIPTION) || !defined(MGT_BACKEND)
#error "Multigrid configuration macros must be defined before including linear_solver_multigrid_template.h"
#endif

#if !defined(MGT_BC_NEUMANN_PLANE) || !defined(MGT_RBGS_SWEEP) ||          \
    !defined(MGT_JACOBI_SWEEP) || !defined(MGT_RESIDUAL) ||                \
    !defined(MGT_RESTRICT_2D) || !defined(MGT_RESTRICT_3D) ||              \
    !defined(MGT_PROLONGATE_ADD_2D) || !defined(MGT_PROLONGATE_ADD_3D) ||  \
    !defined(MGT_SUBTRACT_INTERIOR_MEAN) || !defined(MGT_ZERO_INTERIOR)
#error "Multigrid primitive macros must be defined before including linear_solver_multigrid_template.h"
#endif

/* ============================================================================
 * TOKEN PASTING
 * ============================================================================ */

#define MGT_CONCAT_IMPL(a, b) a##_##b
#define MGT_CONCAT(a, b) MGT_CONCAT_IMPL(a, b)
#define MGT_FUNC(name) MGT_CONCAT(name, MGT_SUFFIX)

#define MGT_FACTORY_IMPL(suffix) create_multigrid_##suffix##_solver
#define MGT_FACTORY(suffix) MGT_FACTORY_IMPL(suffix)

/* ============================================================================
 * BOUNDARY CONDITIONS
 * ============================================================================ */

/**
 * Apply the BC mode to a field at one level.
 *
 * Neumann: mirror of poisson_solver_apply_bc()'s default (z-plane copies plus
 * per-plane zero-gradient), but with this level's dimensions.
 * Dirichlet: no-op — boundary values are never touched anywhere in MG, so
 * finest-level data stays fixed and coarse corrections stay zero.
 */
static void MGT_FUNC(mg_apply_bc_level)(const mg_level_t* L, mg_bc_type_t bc,
                                        double* x) {
    if (bc == MG_BC_DIRICHLET) {
        return;
    }

    size_t nx = L->nx;
    size_t ny = L->ny;
    size_t nz = L->nz;
    size_t plane_size = nx * ny;

    if (nz > 1) {
        memcpy(x, x + plane_size, plane_size * sizeof(double));
        memcpy(x + (nz - 1) * plane_size,
               x + (nz - 2) * plane_size,
               plane_size * sizeof(double));
    }

    for (size_t k = 0; k < nz; k++) {
        MGT_BC_NEUMANN_PLANE(x + k * plane_size, nx, ny);
    }
}

/* ============================================================================
 * SMOOTHERS
 * ============================================================================ */

/**
 * Run `sweeps` smoother passes, applying BCs after each pass.
 *
 * Red-Black Gauss-Seidel uses omega = 1: over-relaxation degrades the
 * high-frequency smoothing factor, so plain GS is correct inside multigrid.
 * Weighted Jacobi uses omega = 2/3 (optimal high-frequency damping) and copies
 * interior points only back into x, preserving Dirichlet boundary values.
 */
static void MGT_FUNC(mg_smooth)(const mg_context_t* ctx, const mg_level_t* L,
                                double* x, const double* rhs, int sweeps) {
    for (int sweep = 0; sweep < sweeps; sweep++) {
        if (ctx->smoother_type == MG_SMOOTHER_JACOBI) {
            MGT_JACOBI_SWEEP(L, x, L->x_temp, rhs);
        } else {
            MGT_RBGS_SWEEP(L, x, rhs);
        }

        MGT_FUNC(mg_apply_bc_level)(L, ctx->bc_mode, x);
    }
}

/* ============================================================================
 * GRID TRANSFERS
 * ============================================================================ */

/**
 * Restrict a fine-level field into a coarse-level (MG-owned) RHS buffer.
 * In Neumann mode the coarse RHS is projected to zero interior mean — the
 * singular mirror-BC operator only admits solutions for compatible RHS.
 */
static void MGT_FUNC(mg_restrict_level)(const mg_context_t* ctx,
                                        const mg_level_t* Lf, const mg_level_t* Lc,
                                        const double* fine, double* coarse) {
    int fold = (ctx->bc_mode == MG_BC_NEUMANN);

    if (Lf->nz > 1) {
        MGT_RESTRICT_3D(fine, coarse, Lf->nx, Lf->ny, Lf->nz,
                        Lc->nx, Lc->ny, Lc->nz, fold);
    } else {
        MGT_RESTRICT_2D(fine, coarse, Lf->nx, Lf->ny, Lc->nx, Lc->ny, fold);
    }

    if (ctx->bc_mode == MG_BC_NEUMANN) {
        MGT_SUBTRACT_INTERIOR_MEAN(coarse, Lc->nx, Lc->ny, Lc->nz);
    }
}

static void MGT_FUNC(mg_prolongate_level)(const mg_level_t* Lc, const mg_level_t* Lf,
                                          const double* coarse, double* fine) {
    if (Lf->nz > 1) {
        MGT_PROLONGATE_ADD_3D(coarse, fine, Lc->nx, Lc->ny, Lc->nz,
                              Lf->nx, Lf->ny, Lf->nz);
    } else {
        MGT_PROLONGATE_ADD_2D(coarse, fine, Lc->nx, Lc->ny, Lf->nx, Lf->ny);
    }
}

/* ============================================================================
 * CYCLES
 * ============================================================================ */

/**
 * Coarsest-grid solve: over-smoothing. A 3x3 Dirichlet grid (one unknown) is
 * solved exactly by the first sweep; Neumann coarsest grids (>= 5x5, RHS
 * already mean-projected) converge geometrically in a few dozen sweeps at
 * negligible cost.
 */
static void MGT_FUNC(mg_coarse_solve)(const mg_context_t* ctx, const mg_level_t* L,
                                      double* x, const double* rhs) {
    MGT_FUNC(mg_smooth)(ctx, L, x, rhs, ctx->coarse_max_iter);
}

/** Recursive V/W cycle. Level 0 is called with the caller's arrays. */
static void MGT_FUNC(mg_cycle)(mg_context_t* ctx, int level, double* x,
                               const double* rhs) {
    mg_level_t* L = &ctx->levels[level];

    if (level == ctx->num_levels - 1) {
        MGT_FUNC(mg_coarse_solve)(ctx, L, x, rhs);
        return;
    }

    mg_level_t* Lc = &ctx->levels[level + 1];

    /* Pre-smooth, then form and restrict the residual */
    MGT_FUNC(mg_smooth)(ctx, L, x, rhs, ctx->nu1);
    MGT_RESIDUAL(L, x, rhs, L->residual);
    MGT_FUNC(mg_restrict_level)(ctx, L, Lc, L->residual, Lc->rhs);

    /* Coarse-grid correction with zero initial guess. W-cycle: the second
     * visit continues refining the same correction. */
    memset(Lc->x, 0, Lc->total * sizeof(double));
    int num_visits = (ctx->cycle_type == MG_CYCLE_W) ? 2 : 1;
    for (int v = 0; v < num_visits; v++) {
        MGT_FUNC(mg_cycle)(ctx, level + 1, Lc->x, Lc->rhs);
    }

    if (ctx->bc_mode == MG_BC_NEUMANN) {
        /* Constants are in the operator's nullspace: removing the mean keeps
         * the fine solution from drifting. Mirror the boundary so
         * prolongation interpolates consistent ghost values. */
        MGT_SUBTRACT_INTERIOR_MEAN(Lc->x, Lc->nx, Lc->ny, Lc->nz);
        MGT_FUNC(mg_apply_bc_level)(Lc, ctx->bc_mode, Lc->x);
    }

    MGT_FUNC(mg_prolongate_level)(Lc, L, Lc->x, x);
    MGT_FUNC(mg_apply_bc_level)(L, ctx->bc_mode, x);

    MGT_FUNC(mg_smooth)(ctx, L, x, rhs, ctx->nu2);
}

/**
 * FMG (full multigrid) nested iteration: restrict the original RHS through
 * all levels, solve on the coarsest, then work upward prolongating the
 * solution as the initial guess and running one V-cycle per level. Replaces
 * the caller's initial guess (interior only; boundary data preserved).
 *
 * Ascent ordering is load-bearing: levels[l+1].x must be consumed before
 * mg_cycle(l) overwrites levels[l+1].{x,rhs} with residual-correction data.
 */
static void MGT_FUNC(mg_fmg)(mg_context_t* ctx, double* x, const double* rhs) {
    int nl = ctx->num_levels;
    mg_level_t* levels = ctx->levels;

    if (nl == 1) {
        MGT_FUNC(mg_cycle)(ctx, 0, x, rhs);
        return;
    }

    /* Descent: levels[l].rhs = restricted original b for l >= 1 */
    MGT_FUNC(mg_restrict_level)(ctx, &levels[0], &levels[1], rhs, levels[1].rhs);
    for (int l = 1; l <= nl - 2; l++) {
        MGT_FUNC(mg_restrict_level)(ctx, &levels[l], &levels[l + 1],
                                    levels[l].rhs, levels[l + 1].rhs);
    }

    /* Coarsest solve */
    memset(levels[nl - 1].x, 0, levels[nl - 1].total * sizeof(double));
    MGT_FUNC(mg_coarse_solve)(ctx, &levels[nl - 1], levels[nl - 1].x,
                              levels[nl - 1].rhs);

    /* Ascent through MG-owned levels */
    for (int l = nl - 2; l >= 1; l--) {
        MGT_FUNC(mg_apply_bc_level)(&levels[l + 1], ctx->bc_mode, levels[l + 1].x);
        memset(levels[l].x, 0, levels[l].total * sizeof(double));
        MGT_FUNC(mg_prolongate_level)(&levels[l + 1], &levels[l],
                                      levels[l + 1].x, levels[l].x);
        MGT_FUNC(mg_apply_bc_level)(&levels[l], ctx->bc_mode, levels[l].x);
        MGT_FUNC(mg_cycle)(ctx, l, levels[l].x, levels[l].rhs);
    }

    /* Finest level: zero the interior of the caller's x (boundary data must
     * survive in Dirichlet mode), prolongate the level-1 solution in, and
     * finish with one V-cycle */
    MGT_FUNC(mg_apply_bc_level)(&levels[1], ctx->bc_mode, levels[1].x);
    MGT_ZERO_INTERIOR(&levels[0], x);
    MGT_FUNC(mg_prolongate_level)(&levels[1], &levels[0], levels[1].x, x);
    MGT_FUNC(mg_apply_bc_level)(&levels[0], ctx->bc_mode, x);
    MGT_FUNC(mg_cycle)(ctx, 0, x, rhs);
}

/* ============================================================================
 * LIFECYCLE
 * ============================================================================ */

static void MGT_FUNC(mg_free_levels)(mg_context_t* ctx) {
    if (!ctx->levels) {
        return;
    }
    for (int l = 0; l < ctx->num_levels; l++) {
        cfd_free(ctx->levels[l].x);
        cfd_free(ctx->levels[l].x_temp);
        cfd_free(ctx->levels[l].rhs);
        cfd_free(ctx->levels[l].residual);
    }
    cfd_free(ctx->levels);
    ctx->levels = NULL;
    ctx->num_levels = 0;
}

static void MGT_FUNC(mg_destroy)(poisson_solver_t* solver) {
    if (solver && solver->context) {
        mg_context_t* ctx = (mg_context_t*)solver->context;
        MGT_FUNC(mg_free_levels)(ctx);
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t MGT_FUNC(mg_init)(
    poisson_solver_t* solver,
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz,
    const poisson_solver_params_t* params)
{
    /* Geometric coarsening requires 2^k+1 points per active dimension */
    if (!mg_is_pow2_plus1(nx) || !mg_is_pow2_plus1(ny) ||
        (nz > 1 && !mg_is_pow2_plus1(nz))) {
        return CFD_ERROR_INVALID;
    }

    if ((int)params->mg_cycle < MG_CYCLE_V ||
        (int)params->mg_cycle > MG_CYCLE_F ||
        (int)params->mg_smoother < MG_SMOOTHER_REDBLACK_GS ||
        (int)params->mg_smoother > MG_SMOOTHER_JACOBI ||
        (int)params->mg_bc < MG_BC_NEUMANN ||
        (int)params->mg_bc > MG_BC_DIRICHLET ||
        params->mg_pre_smooth < 0 || params->mg_post_smooth < 0 ||
        params->mg_coarse_max_iter < 0 || params->mg_max_levels < 0) {
        return CFD_ERROR_INVALID;
    }

    /* Parallel backends loop over int row/column bounds, which
     * poisson_solver_size_to_int empties above INT_MAX (a zero residual would
     * then read as convergence), and level buffers must not wrap size_t:
     * reject oversized grids before allocating. */
    size_t n = 0;
    cfd_status_t size_status = poisson_solver_validate_grid_size(nx, ny, nz, 1, &n);
    if (size_status != CFD_SUCCESS) {
        return size_status;
    }

    /* Re-init support: drop any previous hierarchy */
    MGT_FUNC(mg_destroy)(solver);

    mg_context_t* ctx = (mg_context_t*)cfd_calloc(1, sizeof(mg_context_t));
    if (!ctx) {
        return CFD_ERROR_NOMEM;
    }

    ctx->cycle_type = params->mg_cycle;
    ctx->smoother_type = params->mg_smoother;
    ctx->bc_mode = params->mg_bc;
    ctx->nu1 = (params->mg_pre_smooth > 0)
        ? params->mg_pre_smooth : MG_DEFAULT_PRE_SMOOTH;
    ctx->nu2 = (params->mg_post_smooth > 0)
        ? params->mg_post_smooth : MG_DEFAULT_POST_SMOOTH;
    ctx->coarse_max_iter = (params->mg_coarse_max_iter > 0)
        ? params->mg_coarse_max_iter : MG_DEFAULT_COARSE_MAX_ITER;

    /* Count levels: coarsen all active dimensions simultaneously while every
     * next dimension stays at or above the BC-mode floor */
    size_t min_coarse = (ctx->bc_mode == MG_BC_NEUMANN)
        ? MG_MIN_COARSE_DIM_NEUMANN : MG_MIN_COARSE_DIM_DIRICHLET;
    int num_levels = 1;
    {
        size_t cx = nx, cy = ny, cz = nz;
        while (params->mg_max_levels == 0 ||
               num_levels < params->mg_max_levels) {
            size_t nx2 = (cx - 1) / 2 + 1;
            size_t ny2 = (cy - 1) / 2 + 1;
            size_t nz2 = (nz > 1) ? (cz - 1) / 2 + 1 : 1;
            if (nx2 < min_coarse || ny2 < min_coarse ||
                (nz > 1 && nz2 < min_coarse)) {
                break;
            }
            cx = nx2;
            cy = ny2;
            cz = nz2;
            num_levels++;
        }
    }
    ctx->num_levels = num_levels;

    ctx->levels = (mg_level_t*)cfd_calloc((size_t)num_levels,
                                          sizeof(mg_level_t));
    if (!ctx->levels) {
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    for (int l = 0; l < num_levels; l++) {
        mg_level_t* L = &ctx->levels[l];
        double scale = (double)((size_t)1 << l);

        L->nx = ((nx - 1) >> l) + 1;
        L->ny = ((ny - 1) >> l) + 1;
        L->nz = (nz > 1) ? ((nz - 1) >> l) + 1 : 1;
        L->total = L->nx * L->ny * L->nz;
        L->dx2 = (dx * scale) * (dx * scale);
        L->dy2 = (dy * scale) * (dy * scale);
        L->inv_dz2 = poisson_solver_compute_inv_dz2(dz * scale);
        poisson_solver_compute_3d_bounds(L->nz, L->nx, L->ny,
                                         &L->stride_z, &L->k_start, &L->k_end);
        L->inv_factor = 1.0 / (2.0 / L->dx2 + 2.0 / L->dy2 + 2.0 * L->inv_dz2);

        /* Level 0 borrows the caller's x/rhs; the coarsest level needs no
         * residual buffer */
        int need_owned = (l >= 1);
        int need_residual = (l <= num_levels - 2);
        int need_temp = (ctx->smoother_type == MG_SMOOTHER_JACOBI);

        if ((need_owned &&
             (!(L->x = (double*)cfd_calloc(L->total, sizeof(double))) ||
              !(L->rhs = (double*)cfd_calloc(L->total, sizeof(double))))) ||
            (need_residual &&
             !(L->residual = (double*)cfd_calloc(L->total, sizeof(double)))) ||
            (need_temp &&
             !(L->x_temp = (double*)cfd_calloc(L->total, sizeof(double))))) {
            MGT_FUNC(mg_free_levels)(ctx);
            cfd_free(ctx);
            return CFD_ERROR_NOMEM;
        }
    }

    ctx->fmg_pending = (ctx->cycle_type == MG_CYCLE_F);
    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

/* ============================================================================
 * SOLVE / ITERATE / APPLY_BC
 * ============================================================================ */

static cfd_status_t MGT_FUNC(mg_iterate)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    double* residual)
{
    (void)x_temp;  /* MG allocates its own temporaries */

    mg_context_t* ctx = (mg_context_t*)solver->context;
    if (!ctx || !ctx->initialized) {
        return CFD_ERROR_INVALID;
    }

    if (ctx->fmg_pending) {
        MGT_FUNC(mg_fmg)(ctx, x, rhs);
        ctx->fmg_pending = 0;
    } else {
        MGT_FUNC(mg_cycle)(ctx, 0, x, rhs);
    }

    if (residual) {
        *residual = poisson_solver_compute_residual(solver, x, rhs);
    }

    return CFD_SUCCESS;
}

/**
 * Custom solve wrapper: the FMG flag must reset at the start of every solve,
 * and iterate() has no solve-start hook. Delegates to the common loop (each
 * "iteration" is one cycle).
 */
static cfd_status_t MGT_FUNC(mg_solve)(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats)
{
    mg_context_t* ctx = (mg_context_t*)solver->context;
    if (!ctx || !ctx->initialized) {
        return CFD_ERROR_INVALID;
    }

    ctx->fmg_pending = (ctx->cycle_type == MG_CYCLE_F);
    return poisson_solver_solve_common(solver, x, x_temp, rhs, stats);
}

/**
 * Non-NULL apply_bc is required: the public poisson_solver_apply_bc() default
 * is Neumann, which would corrupt Dirichlet boundary data.
 */
static void MGT_FUNC(mg_apply_bc)(poisson_solver_t* solver, double* x) {
    mg_context_t* ctx = (mg_context_t*)solver->context;
    if (!ctx || !ctx->initialized) {
        return;
    }
    MGT_FUNC(mg_apply_bc_level)(&ctx->levels[0], ctx->bc_mode, x);
}

/* ============================================================================
 * FACTORY FUNCTION
 * ============================================================================ */

poisson_solver_t* MGT_FACTORY(MGT_SUFFIX)(void) {
    poisson_solver_t* solver =
        (poisson_solver_t*)cfd_calloc(1, sizeof(poisson_solver_t));
    if (!solver) {
        return NULL;
    }

    solver->name = MGT_SOLVER_NAME;
    solver->description = MGT_DESCRIPTION;
    solver->method = POISSON_METHOD_MULTIGRID;
    solver->backend = MGT_BACKEND;
    solver->params = poisson_solver_params_default();

    solver->init = MGT_FUNC(mg_init);
    solver->destroy = MGT_FUNC(mg_destroy);
    solver->solve = MGT_FUNC(mg_solve);      /* Resets FMG state, then common loop */
    solver->iterate = MGT_FUNC(mg_iterate);  /* One V/W cycle (or the FMG pass) */
    solver->apply_bc = MGT_FUNC(mg_apply_bc);

    return solver;
}

/* ============================================================================
 * CLEANUP MACROS
 * ============================================================================ */

#undef MGT_FACTORY
#undef MGT_FACTORY_IMPL
#undef MGT_FUNC
#undef MGT_CONCAT
#undef MGT_CONCAT_IMPL

#undef MGT_SUFFIX
#undef MGT_SOLVER_NAME
#undef MGT_DESCRIPTION
#undef MGT_BACKEND
#undef MGT_BC_NEUMANN_PLANE
#undef MGT_RBGS_SWEEP
#undef MGT_JACOBI_SWEEP
#undef MGT_RESIDUAL
#undef MGT_RESTRICT_2D
#undef MGT_RESTRICT_3D
#undef MGT_PROLONGATE_ADD_2D
#undef MGT_PROLONGATE_ADD_3D
#undef MGT_SUBTRACT_INTERIOR_MEAN
#undef MGT_ZERO_INTERIOR
