/**
 * Inlet Boundary Conditions - OpenMP Implementation
 *
 * OpenMP parallelized inlet velocity boundary condition implementation.
 * Parallelizes over rows for left/right boundaries and
 * over columns for top/bottom boundaries.
 *
 * Supports:
 * - Uniform velocity profile
 * - Parabolic profile (fully-developed flow)
 * - Custom user-defined profiles via callback
 * - Velocity specification by components, magnitude+direction, or mass flow rate
 */

#include "../boundary_conditions_internal.h"
#include <math.h>

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>

/* ============================================================================
 * Table-driven edge configuration
 * ============================================================================ */

typedef struct {
    int u_sign;
    int v_sign;
} edge_mass_flow_t;

static const edge_mass_flow_t mass_flow_dir[] = {
    [BC_EDGE_LEFT]   = { .u_sign = +1, .v_sign =  0 },
    [BC_EDGE_RIGHT]  = { .u_sign = -1, .v_sign =  0 },
    [BC_EDGE_BOTTOM] = { .u_sign =  0, .v_sign = +1 },
    [BC_EDGE_TOP]    = { .u_sign =  0, .v_sign = -1 },
};

/* ============================================================================
 * Velocity computation
 * ============================================================================ */

static inline void inlet_get_base_velocity_omp(const bc_inlet_config_t* config,
                                                double* u_base, double* v_base) {
    switch (config->spec_type) {
        case BC_INLET_SPEC_VELOCITY:
            *u_base = config->spec.velocity.u;
            *v_base = config->spec.velocity.v;
            break;

        case BC_INLET_SPEC_MAGNITUDE_DIR:
            *u_base = config->spec.magnitude_dir.magnitude * cos(config->spec.magnitude_dir.direction);
            *v_base = config->spec.magnitude_dir.magnitude * sin(config->spec.magnitude_dir.direction);
            break;

        case BC_INLET_SPEC_MASS_FLOW: {
            double avg_velocity = config->spec.mass_flow.mass_flow_rate /
                                   (config->spec.mass_flow.density * config->spec.mass_flow.inlet_length);
            const edge_mass_flow_t* dir = &mass_flow_dir[config->edge];
            *u_base = avg_velocity * dir->u_sign;
            *v_base = avg_velocity * dir->v_sign;
            break;
        }

        default:
            *u_base = 0.0;
            *v_base = 0.0;
            break;
    }
}

static inline void inlet_apply_profile_omp(const bc_inlet_config_t* config,
                                            double u_base, double v_base,
                                            double position,
                                            double* u_out, double* v_out) {
    switch (config->profile) {
        case BC_INLET_PROFILE_UNIFORM:
            *u_out = u_base;
            *v_out = v_base;
            break;

        case BC_INLET_PROFILE_PARABOLIC: {
            double profile_factor = 4.0 * position * (1.0 - position);
            *u_out = u_base * profile_factor;
            *v_out = v_base * profile_factor;
            break;
        }

        case BC_INLET_PROFILE_CUSTOM:
            if (config->custom_profile != NULL) {
                config->custom_profile(position, u_out, v_out, config->custom_profile_user_data);
            } else {
                *u_out = u_base;
                *v_out = v_base;
            }
            break;

        default:
            *u_out = u_base;
            *v_out = v_base;
            break;
    }
}

static inline void inlet_compute_velocity_omp(const bc_inlet_config_t* config, double position,
                                               double* u_out, double* v_out) {
    double u_base, v_base;
    inlet_get_base_velocity_omp(config, &u_base, &v_base);
    inlet_apply_profile_omp(config, u_base, v_base, position, u_out, v_out);
}

/* ============================================================================
 * Boundary application with table-driven loop
 * ============================================================================ */

typedef size_t (*idx_func_t)(size_t i, size_t nx, size_t ny);

static inline size_t idx_left(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx;
}

static inline size_t idx_right(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx + (nx - 1);
}

static inline size_t idx_bottom(size_t i, size_t nx, size_t ny) {
    (void)nx; (void)ny;
    return i;
}

static inline size_t idx_top(size_t i, size_t nx, size_t ny) {
    return (ny - 1) * nx + i;
}

typedef struct {
    idx_func_t idx_fn;
    int use_ny_for_count;
} edge_loop_config_t;

static const edge_loop_config_t edge_loops[] = {
    [BC_EDGE_LEFT]   = { .idx_fn = idx_left,   .use_ny_for_count = 1 },
    [BC_EDGE_RIGHT]  = { .idx_fn = idx_right,  .use_ny_for_count = 1 },
    [BC_EDGE_BOTTOM] = { .idx_fn = idx_bottom, .use_ny_for_count = 0 },
    [BC_EDGE_TOP]    = { .idx_fn = idx_top,    .use_ny_for_count = 0 },
};

cfd_status_t bc_apply_inlet_omp_impl(double* u, double* v, size_t nx, size_t ny,
                                      const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (config->edge < 0 || config->edge > BC_EDGE_TOP) {
        return CFD_ERROR_INVALID;
    }

    const edge_loop_config_t* loop = &edge_loops[config->edge];
    size_t count = loop->use_ny_for_count ? ny : nx;
    double pos_denom = (count > 1) ? (double)(count - 1) : 1.0;
    idx_func_t idx_fn = loop->idx_fn;

    int idx;
#pragma omp parallel for schedule(static)
    for (idx = 0; idx < (int)count; idx++) {
        double position = (count > 1) ? (double)idx / pos_denom : 0.5;
        double u_val, v_val;
        inlet_compute_velocity_omp(config, position, &u_val, &v_val);

        size_t arr_idx = idx_fn((size_t)idx, nx, ny);
        u[arr_idx] = u_val;
        v[arr_idx] = v_val;
    }

    return CFD_SUCCESS;
}

#endif /* CFD_ENABLE_OPENMP */
