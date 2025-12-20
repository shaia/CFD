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

#include "../boundary_conditions_inlet_common.h"

cfd_status_t bc_apply_inlet_scalar_impl(double* u, double* v, size_t nx, size_t ny,
                                         const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_inlet_is_valid_edge(config->edge)) {
        return CFD_ERROR_INVALID;
    }

    int edge_idx = bc_inlet_edge_to_index(config->edge);
    const bc_inlet_edge_loop_t* loop = &bc_inlet_edge_loops[edge_idx];
    size_t count = loop->use_ny_for_count ? ny : nx;
    double pos_denom = (count > 1) ? (double)(count - 1) : 1.0;

    for (size_t i = 0; i < count; i++) {
        double position = (count > 1) ? (double)i / pos_denom : 0.5;
        double u_val, v_val;
        bc_inlet_compute_velocity(config, position, &u_val, &v_val);

        size_t idx = loop->idx_fn(i, nx, ny);
        u[idx] = u_val;
        v[idx] = v_val;
    }

    return CFD_SUCCESS;
}
