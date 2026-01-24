/**
 * @file boundary_conditions_time.h
 * @brief Time modulation helper functions for time-varying boundary conditions
 *
 * This internal header provides inline helper functions for computing
 * time-dependent modulation factors for boundary conditions.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_TIME_H
#define CFD_BOUNDARY_CONDITIONS_TIME_H

#include "cfd/boundary/boundary_conditions.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Compute sinusoidal time profile value.
 *
 * @param cfg  Sinusoidal configuration
 * @param t    Current time
 * @return     Modulation factor: offset + amplitude * sin(2*pi*frequency*t + phase)
 */
static inline double bc_time_sinusoidal_value(const bc_time_sinusoidal_t* cfg, double t) {
    return cfg->offset + cfg->amplitude * sin(2.0 * M_PI * cfg->frequency * t + cfg->phase);
}

/**
 * Compute ramp time profile value.
 *
 * @param cfg  Ramp configuration
 * @param t    Current time
 * @return     Modulation factor: linear interpolation between value_start and value_end
 *
 * @note If t_end <= t_start (invalid configuration), returns value_end to avoid
 *       division by zero. Callers should ensure t_start < t_end.
 */
static inline double bc_time_ramp_value(const bc_time_ramp_t* cfg, double t) {
    if (t <= cfg->t_start) {
        return cfg->value_start;
    }
    if (t >= cfg->t_end) {
        return cfg->value_end;
    }
    /* Guard against division by zero from invalid configuration */
    if (cfg->t_end <= cfg->t_start) {
        return cfg->value_end;
    }
    /* Linear interpolation */
    double frac = (t - cfg->t_start) / (cfg->t_end - cfg->t_start);
    return cfg->value_start + frac * (cfg->value_end - cfg->value_start);
}

/**
 * Compute step time profile value.
 *
 * @param cfg  Step configuration
 * @param t    Current time
 * @return     Modulation factor: value_before if t < t_step, else value_after
 */
static inline double bc_time_step_value(const bc_time_step_t* cfg, double t) {
    return (t < cfg->t_step) ? cfg->value_before : cfg->value_after;
}

/**
 * Compute time modulation factor from a time configuration.
 *
 * This is the main entry point for getting the time-dependent multiplier
 * that should be applied to base velocity values.
 *
 * @param cfg  Time configuration (profile type and parameters)
 * @param t    Current simulation time
 * @param dt   Current time step size
 * @return     Modulation factor (1.0 for CONSTANT profile or NULL config)
 */
static inline double bc_time_get_modulator(const bc_time_config_t* cfg, double t, double dt) {
    if (!cfg) {
        return 1.0;
    }

    switch (cfg->profile) {
        case BC_TIME_PROFILE_CONSTANT:
            return 1.0;

        case BC_TIME_PROFILE_SINUSOIDAL:
            return bc_time_sinusoidal_value(&cfg->params.sinusoidal, t);

        case BC_TIME_PROFILE_RAMP:
            return bc_time_ramp_value(&cfg->params.ramp, t);

        case BC_TIME_PROFILE_STEP:
            return bc_time_step_value(&cfg->params.step, t);

        case BC_TIME_PROFILE_CUSTOM:
            if (cfg->custom_fn) {
                return cfg->custom_fn(t, dt, cfg->custom_user_data);
            }
            return 1.0;

        default:
            return 1.0;
    }
}

/**
 * Check if an inlet configuration has time variation enabled.
 *
 * @param config  Inlet configuration
 * @return        true if time variation is configured, false otherwise
 */
static inline bool bc_inlet_has_time_variation(const bc_inlet_config_t* config) {
    if (!config) {
        return false;
    }
    /* Has time variation if profile is not CONSTANT or if time callback is set */
    return (config->time_config.profile != BC_TIME_PROFILE_CONSTANT) ||
           (config->custom_profile_time != NULL);
}

#endif /* CFD_BOUNDARY_CONDITIONS_TIME_H */
