/**
 * Time-Varying Inlet Boundary Conditions - Scalar (CPU) Implementation
 *
 * Baseline scalar implementation of time-varying inlet velocity boundary
 * conditions. Extends the standard inlet BC with time modulation support.
 *
 * Supports:
 * - All standard inlet profiles (uniform, parabolic, custom)
 * - Time modulation: sinusoidal, ramp, step
 * - Custom time-varying profile callback
 * - 3D z-face inlets (FRONT/BACK edges)
 */

#include "../boundary_conditions_inlet_common.h"
#include "../boundary_conditions_time.h"

/**
 * Compute inlet velocity with time modulation at given position.
 *
 * This combines the spatial profile (uniform, parabolic, custom) with
 * time modulation (sinusoidal, ramp, step, custom).
 *
 * @param config    Inlet configuration with time settings
 * @param position  Normalized position along the edge [0, 1]
 * @param time      Current simulation time
 * @param dt        Current time step
 * @param u_out     Output: u velocity component
 * @param v_out     Output: v velocity component
 */
static void bc_inlet_compute_velocity_time(const bc_inlet_config_t* config,
                                            double position,
                                            double time, double dt,
                                            double* u_out, double* v_out) {
    /* Check for custom time-varying profile callback first */
    if (config->custom_profile_time != NULL) {
        config->custom_profile_time(position, time, dt, u_out, v_out,
                                    config->custom_profile_time_user_data);
        return;
    }

    /* Compute base velocity from spatial profile */
    double u_base, v_base;
    bc_inlet_compute_velocity(config, position, &u_base, &v_base);

    /* Apply time modulation */
    double modulator = bc_time_get_modulator(&config->time_config, time, dt);
    *u_out = u_base * modulator;
    *v_out = v_base * modulator;
}

cfd_status_t bc_apply_inlet_time_scalar_impl(double* u, double* v, double* w,
                                              size_t nx, size_t ny,
                                              size_t nz, size_t stride_z,
                                              const bc_inlet_config_t* config,
                                              const bc_time_context_t* time_ctx) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!time_ctx) {
        /* If no time context provided, use time=0, dt=0 */
        bc_time_context_t default_ctx = {0.0, 0.0};
        return bc_apply_inlet_time_scalar_impl(u, v, w, nx, ny, nz, stride_z,
                                                config, &default_ctx);
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
        size_t z_plane = (config->edge == BC_EDGE_FRONT)
                         ? (nz - 1) * stride_z
                         : 0;
        double w_val = bc_inlet_compute_w(config);
        /* Apply time modulation to w as well */
        double modulator = bc_time_get_modulator(&config->time_config,
                                                  time_ctx->time, time_ctx->dt);
        w_val *= modulator;

        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t idx = z_plane + IDX_2D(i, j, nx);
                double u_val, v_val;
                bc_inlet_compute_velocity_time(config, 0.5,
                                                time_ctx->time, time_ctx->dt,
                                                &u_val, &v_val);
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
                bc_inlet_compute_velocity_time(config, position,
                                                time_ctx->time, time_ctx->dt,
                                                &u_val, &v_val);

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
