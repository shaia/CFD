/**
 * @file cavity_validation_utils.h
 * @brief Shared utilities for Ghia validation tests
 *
 * This header provides common validation functions used across all
 * backend-specific Ghia validation tests.
 */

#ifndef CAVITY_VALIDATION_UTILS_H
#define CAVITY_VALIDATION_UTILS_H

#include "cavity_reference_data.h"
#include "lid_driven_cavity_common.h"

#include <string.h>

/* ============================================================================
 * GHIA VALIDATION RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    double rms_u_error;
    double rms_v_error;
    double u_at_center;
    double v_at_center;
    double u_min;
    double max_velocity;
    int success;
    char error_msg[256];
} ghia_result_t;

/* ============================================================================
 * INTERPOLATION HELPER
 * ============================================================================ */

static inline double ghia_interpolate_at(const double* coords, const double* vals,
                                          size_t n, double target) {
    for (size_t i = 0; i < n - 1; i++) {
        if (target >= coords[i] && target <= coords[i + 1]) {
            double t = (target - coords[i]) / (coords[i + 1] - coords[i]);
            return vals[i] + t * (vals[i + 1] - vals[i]);
        }
    }
    return vals[n - 1];
}

/* ============================================================================
 * RMS ERROR COMPUTATION
 * ============================================================================ */

static inline double ghia_compute_rms_error(const double* computed_coords,
                                             const double* computed_vals,
                                             size_t computed_n,
                                             const double* ref_coords,
                                             const double* ref_vals,
                                             size_t ref_n) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < ref_n; i++) {
        double computed = ghia_interpolate_at(computed_coords, computed_vals,
                                               computed_n, ref_coords[i]);
        double error = computed - ref_vals[i];
        sum_sq += error * error;
    }
    return sqrt(sum_sq / ref_n);
}

/* ============================================================================
 * RUN GHIA VALIDATION WITH SPECIFIC SOLVER
 * ============================================================================ */

static inline ghia_result_t run_ghia_validation(const char* solver_type,
                                                 size_t nx, size_t ny,
                                                 double reynolds, double lid_vel,
                                                 int max_steps, double dt) {
    ghia_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    /* Create context */
    cavity_context_t* ctx = cavity_context_create(nx, ny);
    if (!ctx) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create context");
        return result;
    }

    double L = ctx->g->xmax - ctx->g->xmin;
    double nu = lid_vel * L / reynolds;

    ns_solver_params_t params = {
        .dt = dt,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = nu,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Solver '%s' not available", solver_type);
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        return result;
    }

    solver_init(solver, ctx->g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_vel);
        solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (!check_field_finite(ctx->field)) {
            snprintf(result.error_msg, sizeof(result.error_msg), "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            return result;
        }
    }

    /* Extract results */
    double* y_coords = malloc(ny * sizeof(double));
    double* u_vals = malloc(ny * sizeof(double));
    double* x_coords = malloc(nx * sizeof(double));
    double* v_vals = malloc(nx * sizeof(double));

    size_t center_i = nx / 2;
    size_t center_j = ny / 2;
    result.u_min = 1e10;

    for (size_t j = 0; j < ny; j++) {
        y_coords[j] = ctx->g->y[j];
        u_vals[j] = ctx->field->u[j * nx + center_i];
        if (u_vals[j] < result.u_min) result.u_min = u_vals[j];
    }

    for (size_t i = 0; i < nx; i++) {
        x_coords[i] = ctx->g->x[i];
        v_vals[i] = ctx->field->v[center_j * nx + i];
    }

    size_t center_idx = center_j * nx + center_i;
    result.u_at_center = ctx->field->u[center_idx];
    result.v_at_center = ctx->field->v[center_idx];
    result.max_velocity = find_max_velocity(ctx->field);

    result.rms_u_error = ghia_compute_rms_error(
        y_coords, u_vals, ny,
        GHIA_Y_COORDS, GHIA_U_RE100, GHIA_NUM_POINTS
    );
    result.rms_v_error = ghia_compute_rms_error(
        x_coords, v_vals, nx,
        GHIA_X_COORDS, GHIA_V_RE100, GHIA_NUM_POINTS
    );

    result.success = 1;

    free(y_coords);
    free(u_vals);
    free(x_coords);
    free(v_vals);
    solver_destroy(solver);
    cfd_registry_destroy(registry);
    cavity_context_destroy(ctx);

    return result;
}

/* ============================================================================
 * PRINT VALIDATION RESULT
 * ============================================================================ */

static inline void print_ghia_result(const ghia_result_t* result, const char* solver_name) {
    printf("\n    %s Ghia Validation:\n", solver_name);
    printf("      RMS_u: %.4f (target: < %.2f)\n", result->rms_u_error, GHIA_TOLERANCE_MEDIUM);
    printf("      RMS_v: %.4f\n", result->rms_v_error);
    printf("      u_center: %.4f (Ghia: -0.20581)\n", result->u_at_center);
    printf("      u_min:    %.4f (Ghia: -0.21090)\n", result->u_min);

    if (result->rms_u_error > GHIA_TOLERANCE_MEDIUM) {
        printf("      [WARNING] RMS %.4f > target %.2f\n",
               result->rms_u_error, GHIA_TOLERANCE_MEDIUM);
    }
}

#endif /* CAVITY_VALIDATION_UTILS_H */
