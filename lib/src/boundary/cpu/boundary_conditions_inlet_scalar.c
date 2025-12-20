/**
 * Inlet Boundary Conditions - Scalar (CPU) Implementation
 *
 * Baseline scalar implementation of inlet velocity boundary conditions.
 * No SIMD, no OpenMP - pure C loops.
 *
 * Supports:
 * - Uniform velocity profile
 * - Parabolic profile (fully-developed flow)
 * - Custom user-defined profiles via callback
 * - Velocity specification by components, magnitude+direction, or mass flow rate
 */

#include "../boundary_conditions_internal.h"
#include <math.h>

/* ============================================================================
 * Table-driven edge configuration
 * ============================================================================ */

/**
 * Edge configuration for boundary loop iteration.
 * Describes how to iterate over boundary points and compute indices.
 */
typedef struct {
    int u_sign;     /* Sign of u velocity for mass flow (+1 or -1, 0 if v-direction) */
    int v_sign;     /* Sign of v velocity for mass flow (+1 or -1, 0 if u-direction) */
} edge_mass_flow_t;

/* Mass flow velocity direction by edge (indexed by bc_edge_t) */
static const edge_mass_flow_t mass_flow_dir[] = {
    [BC_EDGE_LEFT]   = { .u_sign = +1, .v_sign =  0 },  /* Flow into domain (+x) */
    [BC_EDGE_RIGHT]  = { .u_sign = -1, .v_sign =  0 },  /* Flow into domain (-x) */
    [BC_EDGE_BOTTOM] = { .u_sign =  0, .v_sign = +1 },  /* Flow into domain (+y) */
    [BC_EDGE_TOP]    = { .u_sign =  0, .v_sign = -1 },  /* Flow into domain (-y) */
};

/* ============================================================================
 * Velocity computation
 * ============================================================================ */

/**
 * Get base velocity components from inlet specification.
 */
static inline void inlet_get_base_velocity(const bc_inlet_config_t* config,
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

/**
 * Apply velocity profile to base velocity.
 */
static inline void inlet_apply_profile(const bc_inlet_config_t* config,
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

/**
 * Compute velocity from inlet configuration at given normalized position.
 */
static inline void inlet_compute_velocity(const bc_inlet_config_t* config, double position,
                                           double* u_out, double* v_out) {
    double u_base, v_base;
    inlet_get_base_velocity(config, &u_base, &v_base);
    inlet_apply_profile(config, u_base, v_base, position, u_out, v_out);
}

/* ============================================================================
 * Boundary application with table-driven loop
 * ============================================================================ */

/**
 * Apply inlet BC along a boundary edge.
 *
 * @param u, v       Velocity field arrays
 * @param config     Inlet configuration
 * @param count      Number of boundary points
 * @param pos_denom  Denominator for position calculation (count - 1, or 1 if count == 1)
 * @param idx_fn     Function to compute array index from loop variable
 * @param nx         Grid x dimension (passed to idx_fn)
 */
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

/**
 * Edge loop configuration.
 */
typedef struct {
    idx_func_t idx_fn;      /* Index computation function */
    int use_ny_for_count;   /* 1 if loop count is ny, 0 if nx */
} edge_loop_config_t;

static const edge_loop_config_t edge_loops[] = {
    [BC_EDGE_LEFT]   = { .idx_fn = idx_left,   .use_ny_for_count = 1 },
    [BC_EDGE_RIGHT]  = { .idx_fn = idx_right,  .use_ny_for_count = 1 },
    [BC_EDGE_BOTTOM] = { .idx_fn = idx_bottom, .use_ny_for_count = 0 },
    [BC_EDGE_TOP]    = { .idx_fn = idx_top,    .use_ny_for_count = 0 },
};

cfd_status_t bc_apply_inlet_scalar_impl(double* u, double* v, size_t nx, size_t ny,
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

    for (size_t i = 0; i < count; i++) {
        double position = (count > 1) ? (double)i / pos_denom : 0.5;
        double u_val, v_val;
        inlet_compute_velocity(config, position, &u_val, &v_val);

        size_t idx = loop->idx_fn(i, nx, ny);
        u[idx] = u_val;
        v[idx] = v_val;
    }

    return CFD_SUCCESS;
}
