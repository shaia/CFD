/**
 * Shared Test Helper Functions for Solver Validation
 *
 * This header provides common utilities for mathematical validation of CFD solvers:
 * - Error norm computations (L2, L-infinity)
 * - Flow field validity checks
 * - Divergence computations
 * - Kinetic energy computations
 * - Standard test initializations (Taylor-Green, Poiseuille)
 * - Generic test runners that work with any solver type
 */

#ifndef TEST_SOLVER_HELPERS_H
#define TEST_SOLVER_HELPERS_H

#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test tolerances
#define TOLERANCE_STRICT   1e-6
#define TOLERANCE_MODERATE 1e-3
#define TOLERANCE_RELAXED  1e-2

//=============================================================================
// ERROR NORM COMPUTATIONS
//=============================================================================

/**
 * Compute L2 error norm between two arrays
 */
static inline double test_compute_l2_error(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = a[i] - b[i];
        sum += err * err;
    }
    return sqrt(sum / (double)n);
}

/**
 * Compute L2 norm of an array
 */
static inline double test_compute_l2_norm(const double* arr, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrt(sum / (double)n);
}

/**
 * Compute L-infinity (max) error norm
 */
static inline double test_compute_linf_error(const double* a, const double* b, size_t n) {
    double max_err = 0.0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

/**
 * Compute maximum absolute value in array
 */
static inline double test_compute_max_abs(const double* arr, size_t n) {
    double max_val = 0.0;
    for (size_t i = 0; i < n; i++) {
        double val = fabs(arr[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}

//=============================================================================
// FLOW FIELD VALIDITY CHECKS
//=============================================================================

/**
 * Check if all values in flow field are finite (no NaN or Inf)
 */
static inline int test_flow_field_is_valid(const flow_field* field) {
    size_t n = field->nx * field->ny;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(field->u[i]) || !isfinite(field->v[i]) ||
            !isfinite(field->p[i]) || !isfinite(field->rho[i])) {
            return 0;
        }
    }
    return 1;
}

//=============================================================================
// DIVERGENCE COMPUTATION
//=============================================================================

/**
 * Compute divergence of velocity field (should be ~0 for incompressible flow)
 * Returns L-infinity norm of divergence
 */
static inline double test_compute_divergence_linf(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double max_div = 0.0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
            double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
            double div = fabs(du_dx + dv_dy);

            if (div > max_div) {
                max_div = div;
            }
        }
    }
    return max_div;
}

/**
 * Compute L2 norm of divergence
 */
static inline double test_compute_divergence_l2(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double sum = 0.0;
    int count = 0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;

            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
            double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
            double div = du_dx + dv_dy;
            sum += div * div;
            count++;
        }
    }
    return sqrt(sum / count);
}

//=============================================================================
// ENERGY COMPUTATIONS
//=============================================================================

/**
 * Compute total kinetic energy of the flow field
 * KE = 0.5 * sum(rho * (u^2 + v^2)) * dx * dy
 */
static inline double test_compute_kinetic_energy(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double ke = 0.0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double rho = field->rho[idx];
            double u = field->u[idx];
            double v = field->v[idx];
            ke += 0.5 * rho * (u * u + v * v) * dx * dy;
        }
    }
    return ke;
}

/**
 * Compute kinetic energy without density weighting (simpler version)
 */
static inline double test_compute_kinetic_energy_simple(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double ke = 0.0;

    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double u = field->u[idx];
            double v = field->v[idx];
            ke += 0.5 * (u * u + v * v) * dx * dy;
        }
    }
    return ke;
}

//=============================================================================
// STANDARD TEST INITIALIZATIONS
//=============================================================================

/**
 * Initialize flow field with Taylor-Green vortex
 * u(x,y) =  U * cos(k*x) * sin(k*y)
 * v(x,y) = -U * sin(k*x) * cos(k*y)
 */
static inline void test_init_taylor_green_with_params(flow_field* field, const grid* g,
                                          double U, double k) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            size_t idx = j * nx + i;

            field->u[idx] = U * cos(k * x) * sin(k * y);
            field->v[idx] = -U * sin(k * x) * cos(k * y);
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

/**
 * Initialize flow field with Taylor-Green vortex using default parameters
 */
static inline void test_init_taylor_green(flow_field* field, const grid* g) {
    test_init_taylor_green_with_params(field, g, 0.1, 2.0 * M_PI);
}

/**
 * Initialize flow field with Poiseuille (parabolic) profile
 * u(y) = 4 * U_max * y * (1 - y)  for y in [0, 1]
 */
static inline void test_init_poiseuille(flow_field* field, const grid* g, double U_max) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dy = g->dy[0];
    double ymin = g->ymin;
    double ymax = g->ymax;
    double H = ymax - ymin;

    for (size_t j = 0; j < ny; j++) {
        double y = ymin + j * dy;
        double y_norm = (y - ymin) / H;  // Normalize to [0, 1]
        double u_analytical = 4.0 * U_max * y_norm * (1.0 - y_norm);

        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = u_analytical;
            field->v[idx] = 0.0;
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    // Apply no-slip at walls
    for (size_t i = 0; i < nx; i++) {
        field->u[i] = 0.0;                      // Bottom wall
        field->u[(ny - 1) * nx + i] = 0.0;      // Top wall
    }
}

/**
 * Initialize flow field with sinusoidal pattern
 */
static inline void test_init_sinusoidal(flow_field* field, const grid* g,
                                        double u_amp, double v_amp) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            size_t idx = j * nx + i;

            field->u[idx] = u_amp * sin(M_PI * x) * sin(M_PI * y);
            field->v[idx] = v_amp * cos(M_PI * x) * cos(M_PI * y);
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

/**
 * Initialize with pressure gradient (for pressure-velocity coupling tests)
 */
static inline void test_init_pressure_gradient(flow_field* field, const grid* g,
                                               double p_high, double p_low) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double xmin = g->xmin;
    double xmax = g->xmax;

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = xmin + i * dx;
            double x_norm = (x - xmin) / (xmax - xmin);
            size_t idx = j * nx + i;

            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->p[idx] = p_high + (p_low - p_high) * x_norm;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

//=============================================================================
// GENERIC TEST RESULT STRUCTURE
//=============================================================================

typedef struct {
    int passed;
    double error_l2;
    double error_linf;
    double relative_error;
    double initial_energy;
    double final_energy;
    double initial_divergence;
    double final_divergence;
    int steps_completed;
    char message[256];
} test_result;

static inline test_result test_result_init(void) {
    test_result r;
    r.passed = 1;
    r.error_l2 = 0.0;
    r.error_linf = 0.0;
    r.relative_error = 0.0;
    r.initial_energy = 0.0;
    r.final_energy = 0.0;
    r.initial_divergence = 0.0;
    r.final_divergence = 0.0;
    r.steps_completed = 0;
    r.message[0] = '\0';
    return r;
}

//=============================================================================
// GENERIC TEST RUNNERS (using solver type strings)
//=============================================================================

/**
 * Run stability test for a given solver type
 * Returns number of steps completed before instability (or total if stable)
 */
static inline test_result test_run_stability(
    const char* solver_type,
    size_t nx, size_t ny,
    const solver_params* params,
    int num_steps
) {
    test_result result = test_result_init();

    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);

    if (!g || !field) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Failed to create grid/field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }

    grid_initialize_uniform(g);
    test_init_taylor_green(field, g);

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, solver_type);
    if (!slv) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Solver not available");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    solver_init(slv, g, params);
    solver_stats stats = solver_stats_default();

    for (int step = 0; step < num_steps; step++) {
        solver_step(slv, field, g, params, &stats);
        result.steps_completed = step + 1;

        if (!test_flow_field_is_valid(field)) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Instability at step %d", step);
            break;
        }
    }

    if (result.passed) {
        snprintf(result.message, sizeof(result.message),
                 "Stable for %d steps", num_steps);
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

/**
 * Run energy decay test for a given solver type
 */
static inline test_result test_run_energy_decay(
    const char* solver_type,
    size_t nx, size_t ny,
    const solver_params* params,
    int num_steps
) {
    test_result result = test_result_init();

    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);

    if (!g || !field) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Failed to create grid/field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }

    grid_initialize_uniform(g);
    test_init_taylor_green(field, g);

    result.initial_energy = test_compute_kinetic_energy_simple(field, g);

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, solver_type);
    if (!slv) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Solver not available");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    solver_init(slv, g, params);
    solver_stats stats = solver_stats_default();

    for (int step = 0; step < num_steps; step++) {
        solver_step(slv, field, g, params, &stats);
        result.steps_completed = step + 1;

        if (!test_flow_field_is_valid(field)) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Field invalid at step %d", step);
            break;
        }
    }

    result.final_energy = test_compute_kinetic_energy_simple(field, g);

    if (result.passed) {
        if (result.final_energy >= result.initial_energy) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Energy did not decay (initial: %.6e, final: %.6e)",
                     result.initial_energy, result.final_energy);
        } else {
            double ratio = result.final_energy / result.initial_energy;
            snprintf(result.message, sizeof(result.message),
                     "Energy decayed to %.2f%% of initial", ratio * 100.0);
        }
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

/**
 * Run consistency test between two solver types
 */
static inline test_result test_run_consistency(
    const char* solver_type_a,
    const char* solver_type_b,
    size_t nx, size_t ny,
    const solver_params* params,
    int num_steps,
    double tolerance
) {
    test_result result = test_result_init();
    size_t n = nx * ny;

    // Create two grids and fields
    grid* g_a = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    grid* g_b = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field_a = flow_field_create(nx, ny);
    flow_field* field_b = flow_field_create(nx, ny);

    if (!g_a || !g_b || !field_a || !field_b) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Failed to create grids/fields");
        if (g_a) grid_destroy(g_a);
        if (g_b) grid_destroy(g_b);
        if (field_a) flow_field_destroy(field_a);
        if (field_b) flow_field_destroy(field_b);
        return result;
    }

    grid_initialize_uniform(g_a);
    grid_initialize_uniform(g_b);
    test_init_taylor_green(field_a, g_a);
    test_init_taylor_green(field_b, g_b);

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* slv_a = cfd_solver_create(registry, solver_type_a);
    solver* slv_b = cfd_solver_create(registry, solver_type_b);

    if (!slv_a || !slv_b) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message),
                 "One or both solvers not available");
        if (slv_a) solver_destroy(slv_a);
        if (slv_b) solver_destroy(slv_b);
        cfd_registry_destroy(registry);
        flow_field_destroy(field_a);
        flow_field_destroy(field_b);
        grid_destroy(g_a);
        grid_destroy(g_b);
        return result;
    }

    solver_init(slv_a, g_a, params);
    solver_init(slv_b, g_b, params);

    solver_stats stats_a = solver_stats_default();
    solver_stats stats_b = solver_stats_default();

    for (int step = 0; step < num_steps; step++) {
        solver_step(slv_a, field_a, g_a, params, &stats_a);
        solver_step(slv_b, field_b, g_b, params, &stats_b);
        result.steps_completed = step + 1;
    }

    if (!test_flow_field_is_valid(field_a) || !test_flow_field_is_valid(field_b)) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "One or both fields invalid");
    } else {
        double u_diff = test_compute_l2_error(field_a->u, field_b->u, n);
        double v_diff = test_compute_l2_error(field_a->v, field_b->v, n);
        double u_norm = test_compute_l2_norm(field_a->u, n);
        double v_norm = test_compute_l2_norm(field_a->v, n);

        result.error_l2 = u_diff;
        result.error_linf = v_diff;  // Store v_diff for reporting

        // Calculate relative errors separately for u and v components.
        // For very small norms (< 1e-15), use absolute error instead to avoid
        // division by near-zero. Take maximum of both relative errors.
        double u_rel_error = (u_norm > 1e-15) ? (u_diff / u_norm) : u_diff;
        double v_rel_error = (v_norm > 1e-15) ? (v_diff / v_norm) : v_diff;
        result.relative_error = fmax(u_rel_error, v_rel_error);

        if (result.relative_error > tolerance) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Relative error %.2e exceeds tolerance %.2e",
                     result.relative_error, tolerance);
        } else {
            snprintf(result.message, sizeof(result.message),
                     "Relative error %.2e within tolerance %.2e",
                     result.relative_error, tolerance);
        }
    }

    solver_destroy(slv_a);
    solver_destroy(slv_b);
    cfd_registry_destroy(registry);
    flow_field_destroy(field_a);
    flow_field_destroy(field_b);
    grid_destroy(g_a);
    grid_destroy(g_b);

    return result;
}

/**
 * Run divergence-free test for projection solvers
 */
static inline test_result test_run_divergence_free(
    const char* solver_type,
    size_t nx, size_t ny,
    const solver_params* params,
    int num_steps,
    double tolerance
) {
    test_result result = test_result_init();

    grid* g = grid_create(nx, ny, 0.0, 1.0, 0.0, 1.0);
    flow_field* field = flow_field_create(nx, ny);

    if (!g || !field) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Failed to create grid/field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }

    grid_initialize_uniform(g);

    // Initialize with non-divergence-free field
    double dx = g->dx[0];
    double dy = g->dy[0];
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            size_t idx = j * nx + i;

            field->u[idx] = 0.1 * sin(2.0 * M_PI * x);
            field->v[idx] = 0.1 * cos(2.0 * M_PI * y);
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    result.initial_divergence = test_compute_divergence_l2(field, g);

    solver_registry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    solver* slv = cfd_solver_create(registry, solver_type);
    if (!slv) {
        result.passed = 0;
        snprintf(result.message, sizeof(result.message), "Solver not available");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    solver_init(slv, g, params);
    solver_stats stats = solver_stats_default();

    for (int step = 0; step < num_steps; step++) {
        solver_step(slv, field, g, params, &stats);
        result.steps_completed = step + 1;

        if (!test_flow_field_is_valid(field)) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Field invalid at step %d", step);
            break;
        }
    }

    result.final_divergence = test_compute_divergence_l2(field, g);
    result.error_l2 = result.final_divergence;

    if (result.passed) {
        if (result.error_l2 > tolerance) {
            result.passed = 0;
            snprintf(result.message, sizeof(result.message),
                     "Final divergence %.6e exceeds tolerance %.6e",
                     result.error_l2, tolerance);
        } else {
            snprintf(result.message, sizeof(result.message),
                     "Divergence reduced from %.6e to %.6e",
                     result.initial_divergence, result.final_divergence);
        }
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

#endif // TEST_SOLVER_HELPERS_H
