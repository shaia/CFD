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

#include "../linear/multigrid_internal.h"  /* mg_is_pow2_plus1 */

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
 * Not specific to the pressure solve: the implicit viscous solver a projection
 * owns (ns_viscous_internal.h) is kept by the same rule, and its Helmholtz shift
 * depends on dt, so a caller that changes dt rebuilds it here.
 *
 * Rebuilds when the method, backend, parameters or grid differ from what the
 * existing instance was built for, rather than assuming any of them is fixed:
 * pressure_solver is read from params on every step today, and a caller may
 * resize between steps. All are rare, so a rebuild is the right cost, but
 * silently solving with the previous configuration would not be.
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
        /* The backend belongs in this test as much as the method does: reusing a
         * scalar solver for a request that named OpenMP would serialize the
         * pressure solve of a parallel projection, which is the cross-backend
         * fallback the error-handling rules forbid outright. AUTO matches
         * whatever is cached, because poisson_solver_create resolves it the same
         * way every time and so would hand back the backend already held. */
        int same_backend = cfg->backend == POISSON_BACKEND_AUTO
                        || existing->backend == cfg->backend;
        int same_config = existing->method == cfg->method
                       && same_backend
                       && memcmp(&existing->params, &cfg->params, sizeof cfg->params) == 0;
        if (same_grid && same_config) {
            return CFD_SUCCESS;
        }
        CFD_LOG_DEBUG("projection", "owned Poisson solver rebuilt: %s",
                      same_grid ? "configuration changed" : "grid changed");
        poisson_solver_destroy(existing);
        *slot = NULL;
    }

    /* Cleared so that the status read back below is this call's and not whatever
     * the thread failed at last -- cfd_get_last_status() is sticky and never
     * clears itself, so an earlier unrelated failure (a probe for an
     * unavailable backend, say) would otherwise be reported as this one's
     * reason. poisson_solve() clears for exactly this reason before the same
     * read. */
    cfd_clear_error();

    poisson_solver_t* solver = poisson_solver_create(cfg->method, cfg->backend);
    if (!solver) {
        /* Report what the factory reported. Collapsing everything to
         * UNSUPPORTED would tell a caller to try another backend when the real
         * problem was an allocation failure, and the next backend would fail
         * the same way. A factory that set nothing still must not read back as
         * success. */
        cfd_status_t reason = cfd_get_last_status();
        if (reason == CFD_SUCCESS) {
            cfd_set_error(CFD_ERROR_UNSUPPORTED,
                "The pressure solver could not be created for this method and backend");
            reason = CFD_ERROR_UNSUPPORTED;
        }
        return reason;
    }

    /* A multigrid hierarchy needs 2^k+1 points per active dimension. Tested here
     * rather than inferred from the init status: that grid shape is the caller's
     * to fix by choosing another pressure mode, so it is reported as UNSUPPORTED
     * rather than INVALID, but poisson_solver_init returns INVALID for a dozen
     * unrelated reasons now -- a prescribed z-face on a 2D grid, a non-finite
     * wall value, a parameter group the method does not read. Mapping all of
     * them to this message sent callers to resize a grid that was never the
     * problem. Everything else passes through with the message init set. */
    int uses_hierarchy = cfg->method == POISSON_METHOD_MULTIGRID
                      || cfg->params.krylov.preconditioner == POISSON_PRECOND_MULTIGRID;
    if (uses_hierarchy
        && (!mg_is_pow2_plus1(nx) || !mg_is_pow2_plus1(ny)
            || (nz > 1 && !mg_is_pow2_plus1(nz)))) {
        poisson_solver_destroy(solver);
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "The multigrid pressure modes require 2^k+1 points per active dimension");
        return CFD_ERROR_UNSUPPORTED;
    }

    cfd_status_t status =
        poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &cfg->params);
    if (status != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return status;
    }

    *slot = solver;
    return CFD_SUCCESS;
}

/**
 * Check that an owned pressure solver matches the buffers about to be swept.
 *
 * ns_pressure_ensure() sizes the solver from the grid, while a projection
 * backend's rhs, p_new and u_new are sized from whatever it was built for --
 * the field it is stepping, or the grid its context was created for. The two
 * are separate caller-owned objects and nothing upstream ties them together, so
 * a caller stepping with a mismatched pair would have the solver's kernels
 * sweep to solver->nx-1 and run off the end of all three buffers.
 *
 * Each backend passes the dimensions its own buffers were sized from, which is
 * the one part of this check that legitimately differs between them; the rule
 * and the diagnostic do not, and were copied into three files before this.
 */
static inline cfd_status_t ns_pressure_check_shape(const poisson_solver_t* pressure,
                                                   size_t nx, size_t ny, size_t nz) {
    if (!pressure) {
        /* Set rather than left to the caller's last failure: cfd_get_last_status()
         * is sticky, so a silent return here reports whatever the thread failed at
         * previously -- the same hazard ns_pressure_ensure() clears for above. */
        cfd_set_error(CFD_ERROR_INVALID,
            "The projection solver has no pressure solver; it was not initialized, "
            "or a previous init failed and its status was discarded.");
        return CFD_ERROR_INVALID;
    }
    if (pressure->nx != nx || pressure->ny != ny || pressure->nz != nz) {
        cfd_set_error(CFD_ERROR_INVALID,
            "The pressure solver was built for different dimensions than this solver's "
            "buffers were sized for; re-init the solver for the grid it is stepping.");
        return CFD_ERROR_INVALID;
    }
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
