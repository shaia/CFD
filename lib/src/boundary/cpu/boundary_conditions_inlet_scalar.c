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
 * - 3D z-face inlets (FRONT/BACK edges)
 */

#include "../boundary_conditions_inlet_common.h"

cfd_status_t bc_apply_inlet_scalar_impl(double* u, double* v, double* w,
                                         size_t nx, size_t ny,
                                         size_t nz, size_t stride_z,
                                         const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_inlet_is_valid_edge(config->edge)) {
        return CFD_ERROR_INVALID;
    }

    int edge_idx = bc_inlet_edge_to_index(config->edge);
    const bc_inlet_edge_loop_t* loop = &bc_inlet_edge_loops[edge_idx];

    if (loop->is_z_face) {
        /* Z-face inlet (FRONT or BACK): loop over the full xy-plane */
        if (nz <= 1 || !w) {
            return CFD_ERROR_INVALID;
        }
        /* FRONT = k=nz-1, BACK = k=0 */
        size_t z_plane = (config->edge == BC_EDGE_FRONT)
                         ? (nz - 1) * stride_z
                         : 0;
        double w_val = bc_inlet_compute_w(config);

        /* For z-face inlets, 1D profiles are not meaningful.
         * Velocity is uniform across the plane — compute once. */
        double u_val, v_val;
        bc_inlet_compute_velocity(config, 0.5, &u_val, &v_val);

        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t idx = z_plane + IDX_2D(i, j, nx);
                u[idx] = u_val;
                v[idx] = v_val;
                w[idx] = w_val;
            }
        }
    } else {
        /* X/Y-face inlet: loop over each z-plane */
        size_t count = loop->use_ny_for_count ? ny : nx;
        double pos_denom = (count > 1) ? (double)(count - 1) : 1.0;

        for (size_t k = 0; k < nz; k++) {
            size_t base = k * stride_z;
            for (size_t i = 0; i < count; i++) {
                double position = (count > 1) ? (double)i / pos_denom : 0.5;
                double u_val, v_val;
                bc_inlet_compute_velocity(config, position, &u_val, &v_val);

                size_t idx = base + loop->idx_fn(i, nx, ny);
                u[idx] = u_val;
                v[idx] = v_val;
                if (w) {
                    w[idx] = 0.0;
                }
            }
        }
    }

    return CFD_SUCCESS;
}
