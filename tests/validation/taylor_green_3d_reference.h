/**
 * @file taylor_green_3d_reference.h
 * @brief 3D Taylor-Green vortex analytical solution and validation utilities
 *
 * The 3D Taylor-Green vortex is an exact solution to the incompressible
 * Navier-Stokes equations on a triply-periodic domain [0, 2pi]^3.
 *
 * Analytical solution:
 *   u(x,y,z,t) =  cos(x) * sin(y) * cos(z) * exp(-3vt)
 *   v(x,y,z,t) = -sin(x) * cos(y) * cos(z) * exp(-3vt)
 *   w(x,y,z,t) = 0
 *
 * Key properties:
 *   - Velocity decays as exp(-3vt) (three eigenvalues contribute)
 *   - Kinetic energy decays as exp(-6vt)
 *   - Divergence-free: div(u) = 0
 *   - w remains zero for all time
 */

#ifndef TAYLOR_GREEN_3D_REFERENCE_H
#define TAYLOR_GREEN_3D_REFERENCE_H

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Domain: [0, 2pi]^3 */
#define TG3_DOMAIN_MIN  0.0
#define TG3_DOMAIN_MAX  (2.0 * M_PI)

/* Default test parameters — small grid for CI speed */
#define TG3_DEFAULT_NU       0.01
#define TG3_DEFAULT_N        16      /* Grid points per axis */
#define TG3_DEFAULT_DT       0.001
#define TG3_DEFAULT_STEPS    100

/* Tolerances — wider than 2D due to coarser grid (16^3 vs 32^2)
 * and 3D discretization errors */
#define TG3_VELOCITY_DECAY_TOL  0.15
#define TG3_KE_DECAY_TOL        0.25
#define TG3_DIVERGENCE_TOL      0.5
#define TG3_L2_ERROR_TOL        0.25

/* ============================================================================
 * ANALYTICAL SOLUTION
 * ============================================================================ */

static inline double tg3_analytical_u(double x, double y, double z, double t, double nu) {
    return cos(x) * sin(y) * cos(z) * exp(-3.0 * nu * t);
}

static inline double tg3_analytical_v(double x, double y, double z, double t, double nu) {
    return -sin(x) * cos(y) * cos(z) * exp(-3.0 * nu * t);
}

static inline double tg3_velocity_decay_factor(double t, double nu) {
    return exp(-3.0 * nu * t);
}

static inline double tg3_ke_decay_factor(double t, double nu) {
    return exp(-6.0 * nu * t);
}

/* ============================================================================
 * RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    int success;
    char error_msg[256];

    double l2_error_u;
    double l2_error_v;
    double max_divergence;

    double measured_velocity_decay;
    double expected_velocity_decay;
    double measured_ke_decay;
    double expected_ke_decay;

    double initial_ke;
    double final_ke;
    double initial_max_velocity;
    double final_max_velocity;
    double max_w;

    int steps_completed;
    double final_time;
} tg3_result_t;

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static inline double tg3_compute_max_velocity(const flow_field* field) {
    double max_vmag = 0.0;
    size_t total = field->nx * field->ny * field->nz;

    for (size_t i = 0; i < total; i++) {
        double w_val = field->w ? field->w[i] : 0.0;
        double vmag = sqrt(field->u[i] * field->u[i] +
                          field->v[i] * field->v[i] +
                          w_val * w_val);
        if (vmag > max_vmag) max_vmag = vmag;
    }
    return max_vmag;
}

static inline double tg3_compute_kinetic_energy(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double dz = g->dz[0];
    size_t stride_z = g->stride_z;
    double ke = 0.0;

    /* Integrate over physical interior only (exclude periodic ghost layers) */
    for (size_t k = g->k_start; k < g->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + IDX_2D(i, j, nx);
                double u = field->u[idx];
                double v = field->v[idx];
                double w = field->w ? field->w[idx] : 0.0;
                ke += 0.5 * (u * u + v * v + w * w) * dx * dy * dz;
            }
        }
    }
    return ke;
}

static inline double tg3_compute_max_divergence(const flow_field* field, const grid* g) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double dx = g->dx[0];
    double dy = g->dy[0];
    double dz = g->dz[0];
    size_t stride_z = g->stride_z;
    double max_div = 0.0;

    for (size_t k = g->k_start; k < g->k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (k * stride_z) + IDX_2D(i, j, nx);
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
                double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) / (2.0 * dz);
                double div = fabs(du_dx + dv_dy + dw_dz);
                if (div > max_div) max_div = div;
            }
        }
    }
    return max_div;
}

/* ============================================================================
 * SIMULATION RUNNER
 * ============================================================================ */

static inline tg3_result_t tg3_run_simulation(
    const char* solver_type, size_t n, double nu, double dt, int max_steps)
{
    tg3_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    grid* g = grid_create(n, n, n,
                          TG3_DOMAIN_MIN, TG3_DOMAIN_MAX,
                          TG3_DOMAIN_MIN, TG3_DOMAIN_MAX,
                          TG3_DOMAIN_MIN, TG3_DOMAIN_MAX);
    flow_field* field = flow_field_create(n, n, n);
    if (!g || !field) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create grid/field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }

    grid_initialize_uniform(g);

    /* Initialize with analytical solution at t=0 */
    for (size_t k = 0; k < n; k++) {
        double z = g->z[k];
        for (size_t j = 0; j < n; j++) {
            double y = g->y[j];
            for (size_t i = 0; i < n; i++) {
                double x = g->x[i];
                size_t idx = (k * n * n) + IDX_2D(i, j, n);

                field->u[idx] = tg3_analytical_u(x, y, z, 0.0, nu);
                field->v[idx] = tg3_analytical_v(x, y, z, 0.0, nu);
                field->w[idx] = 0.0;
                field->p[idx] = 0.0;
                field->rho[idx] = 1.0;
                field->T[idx] = 300.0;
            }
        }
    }

    /* Apply periodic BCs before recording initial metrics so that
     * initial and final metrics are computed on the same basis */
    {
        const char* init_bc_fields[] = {"u", "v", "w", "p"};
        double* init_bc_ptrs[] = {field->u, field->v, field->w, field->p};
        for (int f = 0; f < 4; f++) {
            cfd_status_t bc_status = bc_apply_scalar_3d(
                init_bc_ptrs[f], n, n, n, g->stride_z, BC_TYPE_PERIODIC);
            if (bc_status != CFD_SUCCESS) {
                snprintf(result.error_msg, sizeof(result.error_msg),
                         "Initial BC application failed for '%s': %d",
                         init_bc_fields[f], bc_status);
                flow_field_destroy(field);
                grid_destroy(g);
                return result;
            }
        }
    }

    result.initial_ke = tg3_compute_kinetic_energy(field, g);
    result.initial_max_velocity = tg3_compute_max_velocity(field);

    /* Set up solver */
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

    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create solver registry (out of memory)");
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver '%s' not available", solver_type);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    cfd_status_t init_status = solver_init(solver, g, &params);
    if (init_status == CFD_ERROR_UNSUPPORTED) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver '%s' returned UNSUPPORTED", solver_type);
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }
    if (init_status != CFD_SUCCESS) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Solver init failed: %d", init_status);
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();

    /* Run simulation */
    for (int step = 0; step < max_steps; step++) {
        /* Apply periodic BCs */
        const char* bc_fields[] = {"u", "v", "w", "p"};
        double* bc_ptrs[] = {field->u, field->v, field->w, field->p};
        for (int f = 0; f < 4; f++) {
            cfd_status_t bc_status = bc_apply_scalar_3d(bc_ptrs[f], n, n, n, g->stride_z, BC_TYPE_PERIODIC);
            if (bc_status != CFD_SUCCESS) {
                snprintf(result.error_msg, sizeof(result.error_msg),
                         "BC application failed for '%s' at step %d: %d",
                         bc_fields[f], step, bc_status);
                solver_destroy(solver);
                cfd_registry_destroy(registry);
                flow_field_destroy(field);
                grid_destroy(g);
                return result;
            }
        }

        cfd_status_t step_status = solver_step(solver, field, g, &params, &stats);
        if (step_status != CFD_SUCCESS) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Solver step failed at step %d: %d", step, step_status);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            flow_field_destroy(field);
            grid_destroy(g);
            return result;
        }

        double max_v = tg3_compute_max_velocity(field);
        if (!isfinite(max_v) || max_v > 1e6) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            flow_field_destroy(field);
            grid_destroy(g);
            return result;
        }

        result.steps_completed = step + 1;
    }

    result.final_time = max_steps * dt;
    result.final_ke = tg3_compute_kinetic_energy(field, g);
    result.final_max_velocity = tg3_compute_max_velocity(field);
    result.max_divergence = tg3_compute_max_divergence(field, g);

    /* Compute L2 errors on physical interior only (exclude periodic ghost layers) */
    double sum_sq_err_u = 0.0, sum_sq_exact_u = 0.0;
    double sum_sq_err_v = 0.0, sum_sq_exact_v = 0.0;
    size_t n_interior = 0;
    for (size_t k = g->k_start; k < g->k_end; k++) {
        double z = g->z[k];
        for (size_t j = 1; j < n - 1; j++) {
            double y = g->y[j];
            for (size_t i = 1; i < n - 1; i++) {
                double x = g->x[i];
                size_t idx = (k * n * n) + IDX_2D(i, j, n);

                double u_exact = tg3_analytical_u(x, y, z, result.final_time, nu);
                double v_exact = tg3_analytical_v(x, y, z, result.final_time, nu);
                double eu = field->u[idx] - u_exact;
                double ev = field->v[idx] - v_exact;
                sum_sq_err_u += eu * eu;
                sum_sq_exact_u += u_exact * u_exact;
                sum_sq_err_v += ev * ev;
                sum_sq_exact_v += v_exact * v_exact;
                n_interior++;
            }
        }
    }
    result.l2_error_u = (sum_sq_exact_u > 1e-15) ? sqrt(sum_sq_err_u / sum_sq_exact_u)
                                                  : sqrt(sum_sq_err_u / (double)n_interior);
    result.l2_error_v = (sum_sq_exact_v > 1e-15) ? sqrt(sum_sq_err_v / sum_sq_exact_v)
                                                  : sqrt(sum_sq_err_v / (double)n_interior);

    /* Decay rates */
    if (result.initial_max_velocity > 1e-10) {
        result.measured_velocity_decay = result.final_max_velocity / result.initial_max_velocity;
    }
    result.expected_velocity_decay = tg3_velocity_decay_factor(result.final_time, nu);

    if (result.initial_ke > 1e-10) {
        result.measured_ke_decay = result.final_ke / result.initial_ke;
    }
    result.expected_ke_decay = tg3_ke_decay_factor(result.final_time, nu);

    /* Compute max |w| — analytical w=0, so any nonzero value is numerical error */
    {
        size_t total = n * n * n;
        double mw = 0.0;
        for (size_t i = 0; i < total; i++) {
            double aw = fabs(field->w[i]);
            if (aw > mw) { mw = aw; }
        }
        result.max_w = mw;
    }

    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return result;
}

/* ============================================================================
 * PRINT UTILITY
 * ============================================================================ */

static inline void tg3_print_result(const tg3_result_t* result, const char* solver_name) {
    printf("\n    %s 3D Taylor-Green Vortex Validation:\n", solver_name);
    printf("      Steps: %d, Final time: %.4f\n", result->steps_completed, result->final_time);
    printf("\n      Velocity decay:\n");
    if (result->expected_velocity_decay > 1e-15) {
        printf("        Measured: %.6f, Expected: %.6f (error: %.2f%%)\n",
               result->measured_velocity_decay, result->expected_velocity_decay,
               100.0 * fabs(result->measured_velocity_decay - result->expected_velocity_decay) /
                   result->expected_velocity_decay);
    }
    printf("\n      Kinetic energy decay:\n");
    if (result->expected_ke_decay > 1e-15) {
        printf("        Measured: %.6f, Expected: %.6f (error: %.2f%%)\n",
               result->measured_ke_decay, result->expected_ke_decay,
               100.0 * fabs(result->measured_ke_decay - result->expected_ke_decay) /
                   result->expected_ke_decay);
    }
    printf("\n      L2 errors: u=%.6f, v=%.6f\n", result->l2_error_u, result->l2_error_v);
    printf("      Max divergence: %.2e\n", result->max_divergence);
}

#endif /* TAYLOR_GREEN_3D_REFERENCE_H */
