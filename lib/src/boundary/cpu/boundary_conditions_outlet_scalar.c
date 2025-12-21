/**
 * Outlet Boundary Conditions - Scalar (CPU) Implementation
 *
 * Baseline scalar implementation of outlet boundary conditions.
 * No SIMD, no OpenMP - pure C loops.
 *
 * Supports:
 * - Zero-gradient (Neumann) outlet
 * - Convective outlet (advection-based)
 */

#include "../boundary_conditions_outlet_common.h"

cfd_status_t bc_apply_outlet_scalar_impl(double* field, size_t nx, size_t ny,
                                          const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_outlet_is_valid_edge(config->edge)) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_outlet_is_valid_type(config->type)) {
        return CFD_ERROR_INVALID;
    }

    int edge_idx = bc_outlet_edge_to_index(config->edge);
    const bc_outlet_edge_loop_t* loop = &bc_outlet_edge_loops[edge_idx];
    size_t count = loop->use_ny_for_count ? ny : nx;

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
            /* Zero-gradient: boundary = adjacent interior value */
            for (size_t i = 0; i < count; i++) {
                size_t dst_idx = loop->dst_fn(i, nx, ny);
                size_t src_idx = loop->src_fn(i, nx, ny);
                field[dst_idx] = field[src_idx];
            }
            break;

        case BC_OUTLET_CONVECTIVE:
            /* Convective outlet: du/dt + U*du/dn = 0
             * For steady-state or when applied at each timestep without dt,
             * this reduces to zero-gradient as a first-order approximation.
             * Full convective BC requires temporal information (dt, previous values)
             * which would need to be added to the config structure.
             * For now, fall back to zero-gradient behavior. */
            for (size_t i = 0; i < count; i++) {
                size_t dst_idx = loop->dst_fn(i, nx, ny);
                size_t src_idx = loop->src_fn(i, nx, ny);
                field[dst_idx] = field[src_idx];
            }
            break;

        default:
            return CFD_ERROR_INVALID;
    }

    return CFD_SUCCESS;
}
