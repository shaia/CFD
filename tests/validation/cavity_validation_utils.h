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
 * ============================================================================
 * Uses cavity_run_with_solver_ctx() for simulation, then extracts centerline
 * profiles for RMS error computation against Ghia reference data.
 */

static inline ghia_result_t run_ghia_validation(const char* solver_type,
                                                 size_t nx, size_t ny,
                                                 double reynolds, double lid_vel,
                                                 int max_steps, double dt) {
    ghia_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    /* Run simulation using shared base function */
    cavity_context_t* ctx = NULL;
    cavity_sim_result_t sim = cavity_run_with_solver_ctx(
        solver_type, nx, ny, reynolds, lid_vel, max_steps, dt, &ctx);

    if (!sim.success) {
        strncpy(result.error_msg, sim.error_msg, sizeof(result.error_msg) - 1);
        return result;
    }

    /* Copy basic results from simulation */
    result.u_at_center = sim.u_at_center;
    result.v_at_center = sim.v_at_center;
    result.u_min = sim.u_min;
    result.max_velocity = sim.max_velocity;

    /* Extract centerline profiles for RMS computation */
    double* y_coords = malloc(ny * sizeof(double));
    double* u_vals = malloc(ny * sizeof(double));
    double* x_coords = malloc(nx * sizeof(double));
    double* v_vals = malloc(nx * sizeof(double));

    size_t center_i = nx / 2;
    size_t center_j = ny / 2;

    for (size_t j = 0; j < ny; j++) {
        y_coords[j] = ctx->g->y[j];
        u_vals[j] = ctx->field->u[j * nx + center_i];
    }

    for (size_t i = 0; i < nx; i++) {
        x_coords[i] = ctx->g->x[i];
        v_vals[i] = ctx->field->v[center_j * nx + i];
    }

    /* Compute RMS errors against Ghia reference data */
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
    printf("      u_center: %.4f (Ghia: %.5f)\n", result->u_at_center, GHIA_U_CENTER_RE100);
    printf("      u_min:    %.4f (Ghia: %.5f)\n", result->u_min, GHIA_U_MIN_RE100);

    if (result->rms_u_error > GHIA_TOLERANCE_MEDIUM) {
        printf("      [WARNING] RMS %.4f > target %.2f\n",
               result->rms_u_error, GHIA_TOLERANCE_MEDIUM);
    }
}

#endif /* CAVITY_VALIDATION_UTILS_H */
