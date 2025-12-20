/**
 * Inlet Boundary Conditions - Common Definitions
 *
 * Shared helper functions and data structures used by all inlet BC
 * backend implementations (scalar, OMP, AVX2+OMP, NEON+OMP).
 *
 * This header is internal and should only be included by inlet BC
 * implementation files.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H
#define CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H

#include "boundary_conditions_internal.h"
#include <math.h>

/* ============================================================================
 * Edge validation and index conversion
 *
 * bc_edge_t uses bit flags (0x01, 0x02, 0x04, 0x08) for potential combining.
 * We convert these to sequential indices (0, 1, 2, 3) for array lookup.
 * ============================================================================ */

/**
 * Check if edge value is valid (exactly one of the four valid edges).
 */
static inline bool bc_inlet_is_valid_edge(bc_edge_t edge) {
    return edge == BC_EDGE_LEFT || edge == BC_EDGE_RIGHT ||
           edge == BC_EDGE_BOTTOM || edge == BC_EDGE_TOP;
}

/**
 * Convert bc_edge_t bit flag to sequential array index (0-3).
 * BC_EDGE_LEFT (0x01) -> 0, BC_EDGE_RIGHT (0x02) -> 1,
 * BC_EDGE_BOTTOM (0x04) -> 2, BC_EDGE_TOP (0x08) -> 3
 *
 * Uses bit manipulation: counts trailing zeros in the bit flag.
 * Returns 0 for invalid edges (caller must validate with bc_inlet_is_valid_edge first).
 */
static inline int bc_inlet_edge_to_index(bc_edge_t edge) {
    int idx = 0;
    unsigned int e = (unsigned int)edge;
    while ((e & 1) == 0 && idx < 3) {
        e >>= 1;
        idx++;
    }
    return idx;
}

/* ============================================================================
 * Table-driven edge configuration (indexed 0-3)
 * ============================================================================ */

/**
 * Mass flow velocity direction by edge.
 */
typedef struct {
    int u_sign;     /* Sign of u velocity for mass flow (+1 or -1, 0 if v-direction) */
    int v_sign;     /* Sign of v velocity for mass flow (+1 or -1, 0 if u-direction) */
} bc_inlet_mass_flow_dir_t;

/**
 * Mass flow velocity direction by edge index (0=left, 1=right, 2=bottom, 3=top).
 */
static const bc_inlet_mass_flow_dir_t bc_inlet_mass_flow_dir[4] = {
    { .u_sign = +1, .v_sign =  0 },  /* LEFT: Flow into domain (+x) */
    { .u_sign = -1, .v_sign =  0 },  /* RIGHT: Flow into domain (-x) */
    { .u_sign =  0, .v_sign = +1 },  /* BOTTOM: Flow into domain (+y) */
    { .u_sign =  0, .v_sign = -1 },  /* TOP: Flow into domain (-y) */
};

/* ============================================================================
 * Index computation functions for each edge
 * ============================================================================ */

typedef size_t (*bc_inlet_idx_func_t)(size_t i, size_t nx, size_t ny);

static inline size_t bc_inlet_idx_left(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx;
}

static inline size_t bc_inlet_idx_right(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx + (nx - 1);
}

static inline size_t bc_inlet_idx_bottom(size_t i, size_t nx, size_t ny) {
    (void)nx; (void)ny;
    return i;
}

static inline size_t bc_inlet_idx_top(size_t i, size_t nx, size_t ny) {
    return (ny - 1) * nx + i;
}

/**
 * Edge loop configuration for table-driven boundary application.
 */
typedef struct {
    bc_inlet_idx_func_t idx_fn;     /* Index computation function */
    int use_ny_for_count;           /* 1 if loop count is ny, 0 if nx */
} bc_inlet_edge_loop_t;

/**
 * Edge loop configuration indexed 0-3 (left, right, bottom, top).
 */
static const bc_inlet_edge_loop_t bc_inlet_edge_loops[4] = {
    { .idx_fn = bc_inlet_idx_left,   .use_ny_for_count = 1 },  /* LEFT */
    { .idx_fn = bc_inlet_idx_right,  .use_ny_for_count = 1 },  /* RIGHT */
    { .idx_fn = bc_inlet_idx_bottom, .use_ny_for_count = 0 },  /* BOTTOM */
    { .idx_fn = bc_inlet_idx_top,    .use_ny_for_count = 0 },  /* TOP */
};

/* ============================================================================
 * Velocity computation helpers
 * ============================================================================ */

/**
 * Get base velocity components from inlet configuration.
 */
static inline void bc_inlet_get_base_velocity(const bc_inlet_config_t* config,
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
            int idx = bc_inlet_edge_to_index(config->edge);
            *u_base = avg_velocity * bc_inlet_mass_flow_dir[idx].u_sign;
            *v_base = avg_velocity * bc_inlet_mass_flow_dir[idx].v_sign;
            break;
        }

        default:
            *u_base = 0.0;
            *v_base = 0.0;
            break;
    }
}

/**
 * Apply profile shape to base velocity.
 */
static inline void bc_inlet_apply_profile(const bc_inlet_config_t* config,
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
 * Compute inlet velocity at given position (combines base velocity + profile).
 */
static inline void bc_inlet_compute_velocity(const bc_inlet_config_t* config,
                                              double position,
                                              double* u_out, double* v_out) {
    double u_base, v_base;
    bc_inlet_get_base_velocity(config, &u_base, &v_base);
    bc_inlet_apply_profile(config, u_base, v_base, position, u_out, v_out);
}

#endif /* CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H */
