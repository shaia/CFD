/**
 * @file ns_pressure_internal.h
 * @brief The pressure Poisson solver a projection solver owns.
 *
 * A projection step solves the pressure Poisson equation once per inner
 * iteration. Building a solver for that each time would rebuild a whole
 * multigrid hierarchy per step, so each projection solver owns one across its
 * lifetime, created at init and destroyed with the solver.
 *
 * Both functions live here rather than in each backend because the mapping from
 * ns_solver_params_t to a Poisson configuration is the same question every
 * projection backend asks, and three copies of it drifted apart once already.
 */
#ifndef CFD_NS_PRESSURE_INTERNAL_H
#define CFD_NS_PRESSURE_INTERNAL_H

#include "cfd/core/cfd_status.h"
#include "cfd/core/logging.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include <string.h>

/**
 * The Poisson configuration a projection solver should use.
 *
 * `backend` is the backend that projection runs on: the OpenMP projection must
 * get OpenMP multigrid and never the scalar one, which is why the caller passes
 * it rather than letting AUTO decide.
 *
 * Returns CFD_ERROR_INVALID for an unknown pressure_solver value.
 */
static inline cfd_status_t ns_pressure_config(const ns_solver_params_t* params,
                                              poisson_solver_backend_t backend,
                                              poisson_solver_config_t* out) {
    ns_pressure_solver_t choice =
        params ? params->pressure_solver : NS_PRESSURE_SOLVER_DEFAULT;

    switch (choice) {
        case NS_PRESSURE_SOLVER_DEFAULT:
            *out = poisson_solver_config_preset(POISSON_PRESET_DEFAULT);
            break;
        case NS_PRESSURE_SOLVER_MULTIGRID:
            *out = poisson_solver_config_preset(POISSON_PRESET_MULTIGRID);
            break;
        case NS_PRESSURE_SOLVER_PCG_MG:
            *out = poisson_solver_config_preset(POISSON_PRESET_MULTIGRID_PCG);
            break;
        default:
            cfd_set_error(CFD_ERROR_INVALID, "Unknown ns_solver_params_t.pressure_solver");
            return CFD_ERROR_INVALID;
    }

    out->backend = backend;
    if (params) {
        out->params.walls = params->pressure_bc;
    }
    return CFD_SUCCESS;
}

/**
 * Make *slot a solver matching cfg on this grid, building or rebuilding it.
 *
 * Rebuilds when the configuration or the grid differs from what the existing
 * instance was built for, rather than assuming either is fixed: pressure_solver
 * is read from params on every step today, and a caller may resize between
 * steps. Both are rare, so a rebuild is the right cost, but silently solving
 * with the previous configuration would not be.
 *
 * Returns the init status, so an MG mode on a grid that is not 2^k+1 fails at
 * solver init -- where the caller can still choose something else -- rather than
 * on the first step. That particular rejection is reported as UNSUPPORTED rather
 * than INVALID, because the grid is fine and only this mode cannot run on it.
 */
static inline cfd_status_t ns_pressure_ensure(poisson_solver_t** slot,
                                              const poisson_solver_config_t* cfg,
                                              const grid* g) {
    if (!slot || !cfg || !g) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = g->nx;
    size_t ny = g->ny;
    size_t nz = g->nz;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double dz = (nz > 1 && g->dz) ? g->dz[0] : 0.0;

    /* Screened here so the remap below stays unambiguous: poisson_solver_init
     * reports a degenerate grid with CFD_ERROR_INVALID too, and that is a caller
     * error rather than an unsupported configuration. */
    if (nx < 3 || ny < 3 || (nz > 1 && nz < 3)) {
        cfd_set_error(CFD_ERROR_INVALID,
            "The pressure solve needs at least 3 points per active dimension");
        return CFD_ERROR_INVALID;
    }

    poisson_solver_t* existing = *slot;
    if (existing) {
        int same_grid = existing->nx == nx && existing->ny == ny && existing->nz == nz
                     && existing->dx == dx && existing->dy == dy && existing->dz == dz;
        int same_config = existing->method == cfg->method
                       && memcmp(&existing->params, &cfg->params, sizeof cfg->params) == 0;
        if (same_grid && same_config) {
            return CFD_SUCCESS;
        }
        CFD_LOG_DEBUG("projection", "pressure solver rebuilt: %s",
                      same_grid ? "configuration changed" : "grid changed");
        poisson_solver_destroy(existing);
        *slot = NULL;
    }

    poisson_solver_t* solver = poisson_solver_create(cfg->method, cfg->backend);
    if (!solver) {
        return CFD_ERROR_UNSUPPORTED;  /* factory set the specific last-status */
    }

    cfd_status_t status =
        poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &cfg->params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);

        /* A multigrid hierarchy needs 2^k+1 points per active dimension. The grid
         * was screened above, so INVALID from here can only be that rejection, and
         * it is reported as UNSUPPORTED: the caller's grid is fine, this pressure
         * mode just cannot run on it, and picking another one is the way out. */
        int uses_hierarchy = cfg->method == POISSON_METHOD_MULTIGRID
                          || cfg->params.krylov.preconditioner == POISSON_PRECOND_MULTIGRID;
        if (uses_hierarchy && status == CFD_ERROR_INVALID) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                "The multigrid pressure modes require 2^k+1 points per active dimension");
            return CFD_ERROR_UNSUPPORTED;
        }
        return status;
    }

    *slot = solver;
    return CFD_SUCCESS;
}

/** Destroy an owned pressure solver and clear the slot. NULL-safe. */
static inline void ns_pressure_release(poisson_solver_t** slot) {
    if (slot && *slot) {
        poisson_solver_destroy(*slot);
        *slot = NULL;
    }
}

#endif /* CFD_NS_PRESSURE_INTERNAL_H */
