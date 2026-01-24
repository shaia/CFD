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

cfd_status_t bc_apply_inlet_time_cpu(double* u, double* v, size_t nx, size_t ny,
                                      const bc_inlet_config_t* config,
                                      const bc_time_context_t* time_ctx) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!time_ctx) {
        /* If no time context provided, use time=0, dt=0 */
        bc_time_context_t default_ctx = {0.0, 0.0};
        return bc_apply_inlet_time_cpu(u, v, nx, ny, config, &default_ctx);
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
        bc_inlet_compute_velocity_time(config, position,
                                        time_ctx->time, time_ctx->dt,
                                        &u_val, &v_val);

        size_t idx = loop->idx_fn(i, nx, ny);
        u[idx] = u_val;
        v[idx] = v_val;
    }

    return CFD_SUCCESS;
}
