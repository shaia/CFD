/**
 * @file test_poiseuille_flow.c
 * @brief Poiseuille (channel) flow validation tests
 *
 * Tests the numerical solver against the analytical Poiseuille flow solution.
 * Poiseuille flow is a steady-state solution to the incompressible Navier-Stokes
 * equations for pressure-driven flow between parallel plates.
 *
 * Analytical solution for channel height H, max velocity U_max:
 *   u(y) = 4 * U_max * (y/H) * (1 - y/H)
 *   v(y) = 0
 *   dp/dx = -8 * mu * U_max / H^2
 *
 * Strategy: Initialize with analytical solution, run a few projection steps,
 * and verify the solution stays close to analytical. This avoids long spinup
 * from quiescent state while still validating that the solver preserves the
 * correct physics.
 *
 * Tests verify:
 *   - Velocity profile stability (analytical solution preserved by solver)
 *   - Mass conservation (flux in = flux out)
 *   - Pressure gradient matches analytical value
 *   - Inlet BC accuracy (parabolic profile applied exactly)
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
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

/* Domain */
#define POIS_DOMAIN_LENGTH  4.0
#define POIS_CHANNEL_HEIGHT 1.0

/* Flow parameters */
#define POIS_U_MAX  1.0
#define POIS_RE     5.0
#define POIS_NU     (POIS_U_MAX * POIS_CHANNEL_HEIGHT / POIS_RE)  /* 0.2 */

/* Grid */
#define POIS_NX  21
#define POIS_NY  11

/* Time stepping — just enough to verify stability, not full convergence */
#define POIS_DT     0.001
#define POIS_STEPS  200

/* Tolerances */
#define POIS_PROFILE_RMS_TOL     0.01   /* 1% RMS error on velocity profile */
#define POIS_MAX_V_TOL           0.01   /* Max |v| should be near zero */
#define POIS_MASS_FLUX_TOL       0.01   /* 1% mass flux conservation */
#define POIS_PRESSURE_GRAD_TOL   0.05   /* 5% pressure gradient error */
#define POIS_INLET_BC_TOL        1e-10  /* Inlet BC should be exact */

/* ============================================================================
 * ANALYTICAL SOLUTION
 * ============================================================================ */

static inline double pois_analytical_u(double y) {
    return 4.0 * POIS_U_MAX * (y / POIS_CHANNEL_HEIGHT) *
           (1.0 - y / POIS_CHANNEL_HEIGHT);
}

static inline double pois_analytical_dpdx(void) {
    return -8.0 * POIS_NU * POIS_U_MAX /
           (POIS_CHANNEL_HEIGHT * POIS_CHANNEL_HEIGHT);
}

/* ============================================================================
 * RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    int success;
    char error_msg[256];

    double profile_rms_error;
    double max_v_magnitude;

    double mass_flux_in;
    double mass_flux_out;
    double mass_flux_mid;

    double measured_dpdx;
    double expected_dpdx;

    double inlet_max_error;

    int steps_completed;
} pois_result_t;

/* ============================================================================
 * SIMULATION RUNNER
 * ============================================================================ */

static pois_result_t s_result;
static int s_result_valid = 0;

static pois_result_t run_poiseuille(void) {
    pois_result_t result = {0};
    result.expected_dpdx = pois_analytical_dpdx();

    grid* g = grid_create(POIS_NX, POIS_NY,
                          0.0, POIS_DOMAIN_LENGTH,
                          0.0, POIS_CHANNEL_HEIGHT);
    flow_field* field = flow_field_create(POIS_NX, POIS_NY);
    if (!g || !field) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create grid or flow field");
        if (g) grid_destroy(g);
        if (field) flow_field_destroy(field);
        return result;
    }
    grid_initialize_uniform(g);

    double dx = POIS_DOMAIN_LENGTH / (POIS_NX - 1);
    double dy = POIS_CHANNEL_HEIGHT / (POIS_NY - 1);
    double dpdx_analytical = pois_analytical_dpdx();

    /* Initialize with analytical solution */
    for (size_t j = 0; j < POIS_NY; j++) {
        double y = j * dy;
        double u_exact = pois_analytical_u(y);
        for (size_t i = 0; i < POIS_NX; i++) {
            double x = i * dx;
            size_t idx = j * POIS_NX + i;
            field->u[idx] = u_exact;
            field->v[idx] = 0.0;
            /* Linear pressure: p(x) = dpdx * x (plus arbitrary constant) */
            field->p[idx] = dpdx_analytical * x;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    /* Configure boundary conditions */
    bc_inlet_config_t inlet = bc_inlet_config_parabolic(POIS_U_MAX);
    bc_outlet_config_t outlet = bc_outlet_config_zero_gradient();

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Failed to create projection solver");
        cfd_registry_destroy(registry);
        grid_destroy(g);
        flow_field_destroy(field);
        return result;
    }

    ns_solver_params_t params = {
        .dt = POIS_DT,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = POIS_NU,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    solver_init(solver, g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    /* Time-stepping loop */
    for (int step = 0; step < POIS_STEPS; step++) {
        bc_apply_noslip(field->u, field->v, POIS_NX, POIS_NY);
        bc_apply_inlet(field->u, field->v, POIS_NX, POIS_NY, &inlet);
        bc_apply_outlet_velocity(field->u, field->v, POIS_NX, POIS_NY, &outlet);

        solver_step(solver, field, g, &params, &stats);

        if (!isfinite(field->u[POIS_NX / 2 + (POIS_NY / 2) * POIS_NX])) {
            snprintf(result.error_msg, sizeof(result.error_msg),
                     "Simulation blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            grid_destroy(g);
            flow_field_destroy(field);
            return result;
        }

        result.steps_completed = step + 1;
    }

    /* Apply BCs one final time for measurements */
    bc_apply_noslip(field->u, field->v, POIS_NX, POIS_NY);
    bc_apply_inlet(field->u, field->v, POIS_NX, POIS_NY, &inlet);
    bc_apply_outlet_velocity(field->u, field->v, POIS_NX, POIS_NY, &outlet);

    /* --- Velocity profile at measurement station --- */
    {
        size_t ix = (size_t)(0.75 * (POIS_NX - 1));
        double sum_sq_error = 0.0;
        double max_v = 0.0;

        for (size_t j = 0; j < POIS_NY; j++) {
            double y = j * dy;
            double u_exact = pois_analytical_u(y);
            double u_num = field->u[j * POIS_NX + ix];
            double err = u_num - u_exact;
            sum_sq_error += err * err;

            double v_abs = fabs(field->v[j * POIS_NX + ix]);
            if (v_abs > max_v) max_v = v_abs;
        }
        result.profile_rms_error = sqrt(sum_sq_error / POIS_NY);
        result.max_v_magnitude = max_v;
    }

    /* --- Mass flux --- */
    {
        double flux_in = 0.0, flux_out = 0.0, flux_mid = 0.0;
        size_t ix_mid = POIS_NX / 2;

        for (size_t j = 0; j < POIS_NY; j++) {
            flux_in  += field->u[j * POIS_NX] * dy;
            flux_out += field->u[j * POIS_NX + (POIS_NX - 1)] * dy;
            flux_mid += field->u[j * POIS_NX + ix_mid] * dy;
        }
        result.mass_flux_in = flux_in;
        result.mass_flux_out = flux_out;
        result.mass_flux_mid = flux_mid;
    }

    /* --- Pressure gradient --- */
    {
        size_t jc = POIS_NY / 2;
        double sum_dpdx = 0.0;
        int count = 0;
        size_t ix_start = (size_t)(0.5 * (POIS_NX - 1));

        for (size_t i = ix_start + 1; i < POIS_NX - 1; i++) {
            double dpdx = (field->p[jc * POIS_NX + i + 1] -
                           field->p[jc * POIS_NX + i - 1]) / (2.0 * dx);
            sum_dpdx += dpdx;
            count++;
        }
        result.measured_dpdx = (count > 0) ? sum_dpdx / count : 0.0;
    }

    /* --- Inlet BC accuracy --- */
    {
        double max_err = 0.0;
        for (size_t j = 0; j < POIS_NY; j++) {
            double y = j * dy;
            double u_exact = pois_analytical_u(y);
            double u_inlet = field->u[j * POIS_NX];
            double err = fabs(u_inlet - u_exact);
            if (err > max_err) max_err = err;
        }
        result.inlet_max_error = max_err;
    }

    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    grid_destroy(g);
    flow_field_destroy(field);

    return result;
}

static void ensure_simulation_run(void) {
    if (!s_result_valid) {
        printf("\n    Running Poiseuille flow simulation...\n");
        printf("    Re=%.0f, nx=%d, ny=%d, dt=%.4f, steps=%d\n",
               POIS_RE, POIS_NX, POIS_NY, POIS_DT, POIS_STEPS);
        printf("    Initialized with analytical solution\n");
        s_result = run_poiseuille();
        s_result_valid = 1;

        if (s_result.success) {
            printf("    Simulation completed: %d steps\n", s_result.steps_completed);
        }
    }
}

/* ============================================================================
 * TEST FUNCTIONS
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

void test_velocity_profile_accuracy(void) {
    printf("\n    Testing velocity profile accuracy at x=75%%L...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Profile RMS error: %.6f (tolerance: %.4f)\n",
           s_result.profile_rms_error, POIS_PROFILE_RMS_TOL);
    printf("      Max |v|: %.6f (tolerance: %.4f)\n",
           s_result.max_v_magnitude, POIS_MAX_V_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.profile_rms_error < POIS_PROFILE_RMS_TOL,
        "Velocity profile RMS error exceeds tolerance");
    TEST_ASSERT_TRUE_MESSAGE(s_result.max_v_magnitude < POIS_MAX_V_TOL,
        "Transverse velocity |v| exceeds tolerance");
}

void test_mass_conservation(void) {
    printf("\n    Testing mass conservation...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Mass flux in:  %.6f\n", s_result.mass_flux_in);
    printf("      Mass flux mid: %.6f\n", s_result.mass_flux_mid);
    printf("      Mass flux out: %.6f\n", s_result.mass_flux_out);

    double q_in = fabs(s_result.mass_flux_in);
    TEST_ASSERT_TRUE_MESSAGE(q_in > 1e-10, "Inlet mass flux is zero");

    double err_out = fabs(s_result.mass_flux_in - s_result.mass_flux_out) / q_in;
    double err_mid = fabs(s_result.mass_flux_in - s_result.mass_flux_mid) / q_in;

    printf("      |Q_in - Q_out|/Q_in: %.6f (tolerance: %.4f)\n",
           err_out, POIS_MASS_FLUX_TOL);
    printf("      |Q_in - Q_mid|/Q_in: %.6f (tolerance: %.4f)\n",
           err_mid, POIS_MASS_FLUX_TOL);

    TEST_ASSERT_TRUE_MESSAGE(err_out < POIS_MASS_FLUX_TOL,
        "Mass flux not conserved between inlet and outlet");
    TEST_ASSERT_TRUE_MESSAGE(err_mid < POIS_MASS_FLUX_TOL,
        "Mass flux not conserved between inlet and mid-channel");
}

void test_pressure_gradient(void) {
    printf("\n    Testing pressure gradient...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Measured dp/dx: %.6f\n", s_result.measured_dpdx);
    printf("      Expected dp/dx: %.6f\n", s_result.expected_dpdx);

    double expected_abs = fabs(s_result.expected_dpdx);
    TEST_ASSERT_TRUE_MESSAGE(expected_abs > 1e-10, "Expected dp/dx is zero");

    double rel_err = fabs(s_result.measured_dpdx - s_result.expected_dpdx) / expected_abs;
    printf("      Relative error: %.4f (tolerance: %.4f)\n",
           rel_err, POIS_PRESSURE_GRAD_TOL);

    TEST_ASSERT_TRUE_MESSAGE(rel_err < POIS_PRESSURE_GRAD_TOL,
        "Pressure gradient does not match analytical value");
}

void test_inlet_bc_accuracy(void) {
    printf("\n    Testing inlet BC accuracy...\n");

    ensure_simulation_run();
    TEST_ASSERT_TRUE_MESSAGE(s_result.success, s_result.error_msg);

    printf("      Inlet max error: %.2e (tolerance: %.2e)\n",
           s_result.inlet_max_error, POIS_INLET_BC_TOL);

    TEST_ASSERT_TRUE_MESSAGE(s_result.inlet_max_error < POIS_INLET_BC_TOL,
        "Inlet BC does not match analytical parabolic profile");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("POISEUILLE FLOW VALIDATION\n");
    printf("========================================\n");
    printf("\nAnalytical: u(y) = 4*U_max*(y/H)*(1-y/H)\n");
    printf("Domain: [0, %.0f] x [0, %.0f]\n", POIS_DOMAIN_LENGTH, POIS_CHANNEL_HEIGHT);
    printf("Re = %.0f, U_max = %.1f, nu = %.3f\n", POIS_RE, POIS_U_MAX, POIS_NU);

    RUN_TEST(test_velocity_profile_accuracy);
    RUN_TEST(test_mass_conservation);
    RUN_TEST(test_pressure_gradient);
    RUN_TEST(test_inlet_bc_accuracy);

    return UNITY_END();
}
