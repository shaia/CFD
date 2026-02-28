/**
 * Inlet Boundary Conditions - Common Definitions
 *
 * Shared helper functions and data structures used by all inlet BC
 * backend implementations (scalar, OMP, AVX2, NEON).
 *
 * This header is internal and should only be included by inlet BC
 * implementation files.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H
#define CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H

#include "boundary_conditions_internal.h"
#include "cfd/core/indexing.h"
#include <math.h>

/* ============================================================================
 * Edge validation and index conversion
 *
 * bc_edge_t uses bit flags (0x01..0x20) for potential combining.
 * We convert these to sequential indices (0-5) for array lookup.
 * ============================================================================ */

/**
 * Check if edge value is valid (exactly one of the six valid edges).
 */
static inline bool bc_inlet_is_valid_edge(bc_edge_t edge) {
    return edge == BC_EDGE_LEFT || edge == BC_EDGE_RIGHT ||
           edge == BC_EDGE_BOTTOM || edge == BC_EDGE_TOP ||
           edge == BC_EDGE_FRONT || edge == BC_EDGE_BACK;
}

/**
 * Convert bc_edge_t bit flag to sequential array index (0-5).
 * BC_EDGE_LEFT (0x01) -> 0, BC_EDGE_RIGHT (0x02) -> 1,
 * BC_EDGE_BOTTOM (0x04) -> 2, BC_EDGE_TOP (0x08) -> 3,
 * BC_EDGE_FRONT (0x10) -> 4, BC_EDGE_BACK (0x20) -> 5
 *
 * Uses bit manipulation to find the position of the least significant set bit
 * by counting trailing zeros in the bit flag (clamped to the range 0-5).
 * The behavior is only defined for valid, single-bit edges; callers must
 * ensure validity via bc_inlet_is_valid_edge() before calling.
 */
static inline int bc_inlet_edge_to_index(bc_edge_t edge) {
    int idx = 0;
    unsigned int e = (unsigned int)edge;
    while ((e & 1) == 0 && idx < 5) {
        e >>= 1;
        idx++;
    }
    return idx;
}

/* ============================================================================
 * Table-driven edge configuration (indexed 0-5)
 * ============================================================================ */

/**
 * Mass flow velocity direction by edge.
 */
typedef struct {
    int u_sign;     /* Sign of u velocity for mass flow (+1 or -1, 0 if not u-direction) */
    int v_sign;     /* Sign of v velocity for mass flow (+1 or -1, 0 if not v-direction) */
    int w_sign;     /* Sign of w velocity for mass flow (+1 or -1, 0 if not w-direction) */
} bc_inlet_mass_flow_dir_t;

/**
 * Mass flow velocity direction by edge index (0=left, 1=right, 2=bottom, 3=top, 4=front, 5=back).
 */
static const bc_inlet_mass_flow_dir_t bc_inlet_mass_flow_dir[6] = {
    { .u_sign = +1, .v_sign =  0, .w_sign =  0 },  /* LEFT: Flow into domain (+x) */
    { .u_sign = -1, .v_sign =  0, .w_sign =  0 },  /* RIGHT: Flow into domain (-x) */
    { .u_sign =  0, .v_sign = +1, .w_sign =  0 },  /* BOTTOM: Flow into domain (+y) */
    { .u_sign =  0, .v_sign = -1, .w_sign =  0 },  /* TOP: Flow into domain (-y) */
    { .u_sign =  0, .v_sign =  0, .w_sign = -1 },  /* FRONT: Flow into domain (-z) */
    { .u_sign =  0, .v_sign =  0, .w_sign = +1 },  /* BACK: Flow into domain (+z) */
};

/* ============================================================================
 * Index computation functions for each edge (2D: within a single xy-plane)
 * ============================================================================ */

typedef size_t (*bc_inlet_idx_func_t)(size_t i, size_t nx, size_t ny);

static inline size_t bc_inlet_idx_left(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return IDX_2D(0, j, nx);
}

static inline size_t bc_inlet_idx_right(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return IDX_2D(nx - 1, j, nx);
}

static inline size_t bc_inlet_idx_bottom(size_t i, size_t nx, size_t ny) {
    (void)nx; (void)ny;
    return i;
}

static inline size_t bc_inlet_idx_top(size_t i, size_t nx, size_t ny) {
    return IDX_2D(i, ny - 1, nx);
}

/**
 * Edge loop configuration for table-driven boundary application.
 * For 2D x/y edges: use_ny_for_count selects the loop count.
 * For 3D z-face edges: is_z_face is set and count is nx*ny (full plane).
 */
typedef struct {
    bc_inlet_idx_func_t idx_fn;     /* Index computation function (NULL for z-faces) */
    int use_ny_for_count;           /* 1 if loop count is ny, 0 if nx (ignored for z-faces) */
    int is_z_face;                  /* 1 for front/back z-face edges */
} bc_inlet_edge_loop_t;

/**
 * Edge loop configuration indexed 0-5 (left, right, bottom, top, front, back).
 */
static const bc_inlet_edge_loop_t bc_inlet_edge_loops[6] = {
    { .idx_fn = bc_inlet_idx_left,   .use_ny_for_count = 1, .is_z_face = 0 },  /* LEFT */
    { .idx_fn = bc_inlet_idx_right,  .use_ny_for_count = 1, .is_z_face = 0 },  /* RIGHT */
    { .idx_fn = bc_inlet_idx_bottom, .use_ny_for_count = 0, .is_z_face = 0 },  /* BOTTOM */
    { .idx_fn = bc_inlet_idx_top,    .use_ny_for_count = 0, .is_z_face = 0 },  /* TOP */
    { .idx_fn = NULL,                .use_ny_for_count = 0, .is_z_face = 1 },  /* FRONT */
    { .idx_fn = NULL,                .use_ny_for_count = 0, .is_z_face = 1 },  /* BACK */
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
            /* Validate edge before computing direction-dependent velocity */
            if (!bc_inlet_is_valid_edge(config->edge)) {
                *u_base = 0.0;
                *v_base = 0.0;
                return;
            }
            /* For 2D per unit depth: velocity = mass_flow / (density * inlet_length)
             * where mass_flow is kg/(s*m) and density*length gives kg/m^2 */
            double rho_L = config->spec.mass_flow.density * config->spec.mass_flow.inlet_length;
            if (rho_L <= 0.0) {
                /* Invalid density or inlet_length - return zero velocity */
                *u_base = 0.0;
                *v_base = 0.0;
                return;
            }
            double avg_velocity = config->spec.mass_flow.mass_flow_rate / rho_L;
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

/**
 * Compute w-velocity for z-face inlets from mass flow specification.
 * For non-mass-flow specs on z-faces, returns 0.0 (u,v are set by compute_velocity).
 */
static inline double bc_inlet_compute_w(const bc_inlet_config_t* config) {
    if (config->spec_type == BC_INLET_SPEC_MASS_FLOW) {
        double rho_L = config->spec.mass_flow.density * config->spec.mass_flow.inlet_length;
        if (rho_L <= 0.0) return 0.0;
        double avg_velocity = config->spec.mass_flow.mass_flow_rate / rho_L;
        int idx = bc_inlet_edge_to_index(config->edge);
        return avg_velocity * bc_inlet_mass_flow_dir[idx].w_sign;
    }
    return 0.0;
}

#endif /* CFD_BOUNDARY_CONDITIONS_INLET_COMMON_H */
