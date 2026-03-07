/**
 * @file poiseuille_3d_reference.h
 * @brief 3D Poiseuille flow analytical solution and validation utilities
 *
 * Extends the 2D Poiseuille validation to 3D by using periodic BCs in z.
 * The analytical solution remains the 2D parabola u(y) = 4*U_max*(y/H)*(1-y/H),
 * independent of z. This validates that 3D code paths produce correct results
 * without requiring the complex rectangular-duct Fourier series solution.
 *
 * Tests verify:
 *   - Velocity profile remains parabolic (within discretization limits)
 *   - Transverse velocities v and w remain near zero
 *   - Approximate mass conservation
 *   - Pressure gradient in correct direction and magnitude
 *   - Inlet BC accuracy
 *   - Z-uniformity: solution is identical across z-planes (periodic)
 */

#ifndef POISEUILLE_3D_REFERENCE_H
#define POISEUILLE_3D_REFERENCE_H

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
#include <string.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/* Domain */
#define POIS3D_DOMAIN_LENGTH  4.0
#define POIS3D_CHANNEL_HEIGHT 1.0
#define POIS3D_CHANNEL_DEPTH  1.0

/* Flow parameters */
#define POIS3D_U_MAX  1.0
#define POIS3D_RE     5.0
#define POIS3D_NU     (POIS3D_U_MAX * POIS3D_CHANNEL_HEIGHT / POIS3D_RE)  /* 0.2 */

/* Grid — match 2D resolution (41x21) with nz for 3D */
#define POIS3D_NX  41
#define POIS3D_NY  21
#define POIS3D_NZ  8

/* Time stepping */
#define POIS3D_DT     0.001
#define POIS3D_STEPS  200

/* Tolerances — slightly wider than 2D due to 3D discretization */
#define POIS3D_PROFILE_RMS_TOL     0.12   /* 12% RMS error */
#define POIS3D_MAX_VW_TOL          0.03   /* Max |v| and |w| */
#define POIS3D_MASS_FLUX_TOL       0.25   /* 25% mass flux variation */
#define POIS3D_PRESSURE_GRAD_TOL   0.40   /* 40% pressure gradient error */
#define POIS3D_INLET_BC_TOL        1e-10  /* Inlet BC should be exact */
#define POIS3D_Z_UNIFORMITY_TOL    1e-3   /* Small z-variation from Neumann BCs in Poisson solve */

/* ============================================================================
 * ANALYTICAL SOLUTION
 * ============================================================================ */

static inline double pois3d_analytical_u(double y) {
    return 4.0 * POIS3D_U_MAX * (y / POIS3D_CHANNEL_HEIGHT) *
           (1.0 - y / POIS3D_CHANNEL_HEIGHT);
}

static inline double pois3d_analytical_dpdx(void) {
    return -8.0 * POIS3D_NU * POIS3D_U_MAX /
           (POIS3D_CHANNEL_HEIGHT * POIS3D_CHANNEL_HEIGHT);
}

/* ============================================================================
 * RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    int success;
    char error_msg[256];

    double profile_rms_error;
    double max_v_magnitude;
    double max_w_magnitude;

    double mass_flux_in;
    double mass_flux_out;
    double mass_flux_mid;

    double measured_dpdx;
    double expected_dpdx;

    double inlet_max_error;
    double z_uniformity_error;

    int steps_completed;
} pois3d_result_t;

/* ============================================================================
 * SIMULATION RUNNER
 * ============================================================================ */

static inline pois3d_result_t pois3d_run_simulation(void) {
    pois3d_result_t result = {0};
    result.expected_dpdx = pois3d_analytical_dpdx();

    grid* g = grid_create(POIS3D_NX, POIS3D_NY, POIS3D_NZ,
                          0.0, POIS3D_DOMAIN_LENGTH,
                          0.0, POIS3D_CHANNEL_HEIGHT,
                          0.0, POIS3D_CHANNEL_DEPTH);
    flow_field* field = flow_field_create(POIS3D_NX, POIS3D_NY, POIS3D_NZ);
    if (!g || !field) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create grid or flow field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }
    grid_initialize_uniform(g);

    double dx = g->dx[0];
    double dy = g->dy[0];
    double dz = g->dz[0];
    size_t stride_z = g->stride_z;
    double dpdx_analytical = pois3d_analytical_dpdx();

    /* Initialize with analytical solution — same parabola on every z-plane */
    for (size_t k = 0; k < POIS3D_NZ; k++) {
        for (size_t j = 0; j < POIS3D_NY; j++) {
            double y = g->y[j];
            double u_exact = pois3d_analytical_u(y);
            for (size_t i = 0; i < POIS3D_NX; i++) {
                double x = g->x[i];
                size_t idx = k * stride_z + IDX_2D(i, j, POIS3D_NX);
                field->u[idx] = u_exact;
                field->v[idx] = 0.0;
                field->w[idx] = 0.0;
                field->p[idx] = dpdx_analytical * x;
                field->rho[idx] = 1.0;
                field->T[idx] = 300.0;
            }
        }
    }

    /* Configure boundary conditions */
    bc_inlet_config_t inlet = bc_inlet_config_parabolic(POIS3D_U_MAX);
    bc_outlet_config_t outlet = bc_outlet_config_zero_gradient();

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create solver registry");
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create projection solver");
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return result;
    }

    ns_solver_params_t params = {
        .dt = POIS3D_DT,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = POIS3D_NU,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    cfd_status_t init_status = solver_init(solver, g, &params);
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

    /* Time-stepping loop */
    for (int step = 0; step < POIS3D_STEPS; step++) {
        /* Apply 2D BCs on each z-plane — matches the working 2D test pattern.
         * Using 2D functions avoids bc_apply_noslip_3d setting all 6 faces to
         * zero (including z-faces intended for periodic). */
        for (size_t k = 0; k < POIS3D_NZ; k++) {
            double* u_k = field->u + k * stride_z;
            double* v_k = field->v + k * stride_z;
            bc_apply_noslip(u_k, v_k, POIS3D_NX, POIS3D_NY);
            bc_apply_inlet(u_k, v_k, POIS3D_NX, POIS3D_NY, &inlet);
            bc_apply_outlet_velocity(u_k, v_k, POIS3D_NX, POIS3D_NY, &outlet);
        }
        /* Enforce w BCs for each z-plane:
         *   - No-slip on top/bottom walls: w = 0 at j = 0 and j = ny-1
         *   - Inlet (x = 0): w = 0 for all j
         *   - Outlet (x = nx-1): zero-gradient, w(nx-1,j) = w(nx-2,j)
         */
        if (field->w) {
            for (size_t k = 0; k < POIS3D_NZ; k++) {
                double* w_k = field->w + k * stride_z;

                /* y-walls: j = 0 and j = ny-1 */
                for (size_t i = 0; i < POIS3D_NX; i++) {
                    w_k[IDX_2D(i, 0,              POIS3D_NX)] = 0.0;
                    w_k[IDX_2D(i, POIS3D_NY - 1,  POIS3D_NX)] = 0.0;
                }

                /* x-faces on interior j: inlet and outlet */
                for (size_t j = 1; j + 1 < POIS3D_NY; j++) {
                    /* Inlet: w = 0 at i = 0 */
                    w_k[IDX_2D(0,             j, POIS3D_NX)] = 0.0;
                    /* Outlet: zero-gradient, copy from i = nx-2 */
                    w_k[IDX_2D(POIS3D_NX - 1, j, POIS3D_NX)] =
                        w_k[IDX_2D(POIS3D_NX - 2, j, POIS3D_NX)];
                }
            }
        }

        /* Periodic BCs in z-direction ONLY (not x/y — those have wall/inlet/outlet).
         * bc_apply_scalar_3d(BC_TYPE_PERIODIC) applies periodic on ALL 6 faces,
         * which would destroy the inlet/outlet/wall BCs we just set. */
        {
            double* flds[] = {field->u, field->v, field->w, field->p};
            size_t plane_bytes = stride_z * sizeof(double);
            for (int f = 0; f < 4; f++) {
                if (!flds[f]) continue;
                double* d = flds[f];
                memcpy(d, d + (POIS3D_NZ - 2) * stride_z, plane_bytes);
                memcpy(d + (POIS3D_NZ - 1) * stride_z, d + stride_z, plane_bytes);
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

        /* Check for blow-up */
        size_t center_idx = (POIS3D_NZ / 2) * stride_z +
                            IDX_2D(POIS3D_NX / 2, POIS3D_NY / 2, POIS3D_NX);
        if (!isfinite(field->u[center_idx])) {
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

    /* Apply BCs one final time for measurements */
    for (size_t k = 0; k < POIS3D_NZ; k++) {
        double* u_k = field->u + k * stride_z;
        double* v_k = field->v + k * stride_z;
        bc_apply_noslip(u_k, v_k, POIS3D_NX, POIS3D_NY);
        bc_apply_inlet(u_k, v_k, POIS3D_NX, POIS3D_NY, &inlet);
        bc_apply_outlet_velocity(u_k, v_k, POIS3D_NX, POIS3D_NY, &outlet);
    }

    /* --- Velocity profile at measurement station (x = 75%L) --- */
    {
        size_t ix = (size_t)(0.75 * (POIS3D_NX - 1));
        size_t kc = POIS3D_NZ / 2;  /* Measure at center z-plane */
        double sum_sq_error = 0.0;
        double max_v = 0.0;
        double max_w = 0.0;

        for (size_t j = 0; j < POIS3D_NY; j++) {
            double y = g->y[j];
            double u_exact = pois3d_analytical_u(y);
            size_t idx = kc * stride_z + IDX_2D(ix, j, POIS3D_NX);
            double u_num = field->u[idx];
            double err = u_num - u_exact;
            sum_sq_error += err * err;

            double v_abs = fabs(field->v[idx]);
            double w_abs = fabs(field->w[idx]);
            if (v_abs > max_v) max_v = v_abs;
            if (w_abs > max_w) max_w = w_abs;
        }
        result.profile_rms_error = sqrt(sum_sq_error / POIS3D_NY);
        result.max_v_magnitude = max_v;
        result.max_w_magnitude = max_w;
    }

    /* --- Mass flux (integrate u over y-z plane) --- */
    {
        double flux_in = 0.0, flux_out = 0.0, flux_mid = 0.0;
        size_t ix_mid = POIS3D_NX / 2;

        for (size_t k = g->k_start; k < g->k_end; k++) {
            for (size_t j = 0; j < POIS3D_NY; j++) {
                size_t idx_in  = k * stride_z + IDX_2D(0, j, POIS3D_NX);
                size_t idx_out = k * stride_z + IDX_2D(POIS3D_NX - 1, j, POIS3D_NX);
                size_t idx_mid = k * stride_z + IDX_2D(ix_mid, j, POIS3D_NX);
                flux_in  += field->u[idx_in]  * dy * dz;
                flux_out += field->u[idx_out] * dy * dz;
                flux_mid += field->u[idx_mid] * dy * dz;
            }
        }
        result.mass_flux_in = flux_in;
        result.mass_flux_out = flux_out;
        result.mass_flux_mid = flux_mid;
    }

    /* --- Pressure gradient (central difference at centerline) --- */
    {
        size_t jc = POIS3D_NY / 2;
        size_t kc = POIS3D_NZ / 2;
        double sum_dpdx = 0.0;
        int count = 0;
        size_t ix_start = (size_t)(0.5 * (POIS3D_NX - 1));

        for (size_t i = ix_start + 1; i < POIS3D_NX - 1; i++) {
            size_t idx_p = kc * stride_z + IDX_2D(i + 1, jc, POIS3D_NX);
            size_t idx_m = kc * stride_z + IDX_2D(i - 1, jc, POIS3D_NX);
            double dpdx = (field->p[idx_p] - field->p[idx_m]) / (2.0 * dx);
            sum_dpdx += dpdx;
            count++;
        }
        result.measured_dpdx = (count > 0) ? sum_dpdx / count : 0.0;
    }

    /* --- Inlet BC accuracy --- */
    {
        double max_err = 0.0;
        size_t kc = POIS3D_NZ / 2;
        for (size_t j = 0; j < POIS3D_NY; j++) {
            double y = g->y[j];
            double u_exact = pois3d_analytical_u(y);
            size_t idx = kc * stride_z + IDX_2D(0, j, POIS3D_NX);
            double err = fabs(field->u[idx] - u_exact);
            if (err > max_err) max_err = err;
        }
        result.inlet_max_error = max_err;
    }

    /* --- Z-uniformity: solution should be identical across z-planes --- */
    {
        size_t ix = (size_t)(0.75 * (POIS3D_NX - 1));
        double max_diff = 0.0;

        for (size_t j = 1; j < POIS3D_NY - 1; j++) {
            /* Compare each z-plane to the center z-plane */
            size_t kc = POIS3D_NZ / 2;
            double u_ref = field->u[kc * stride_z + IDX_2D(ix, j, POIS3D_NX)];

            for (size_t k = g->k_start; k < g->k_end; k++) {
                if (k == kc) continue;
                double u_k = field->u[k * stride_z + IDX_2D(ix, j, POIS3D_NX)];
                double diff = fabs(u_k - u_ref);
                if (diff > max_diff) max_diff = diff;
            }
        }
        result.z_uniformity_error = max_diff;
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

static inline void pois3d_print_result(const pois3d_result_t* r) {
    printf("\n    3D Poiseuille Flow Validation:\n");
    printf("      Steps: %d\n", r->steps_completed);
    printf("      Profile RMS error: %.6f (tol: %.4f)\n",
           r->profile_rms_error, POIS3D_PROFILE_RMS_TOL);
    printf("      Max |v|: %.6f, Max |w|: %.6f (tol: %.4f)\n",
           r->max_v_magnitude, r->max_w_magnitude, POIS3D_MAX_VW_TOL);
    printf("      Z-uniformity: %.2e (tol: %.2e)\n",
           r->z_uniformity_error, POIS3D_Z_UNIFORMITY_TOL);
    printf("      Mass flux in/mid/out: %.6f / %.6f / %.6f\n",
           r->mass_flux_in, r->mass_flux_mid, r->mass_flux_out);
    printf("      dp/dx measured: %.6f, expected: %.6f\n",
           r->measured_dpdx, r->expected_dpdx);
}

#endif /* POISEUILLE_3D_REFERENCE_H */
