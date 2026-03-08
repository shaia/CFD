/**
 * @file test_energy_solver.c
 * @brief Unit tests for the energy equation solver
 *
 * Tests:
 * 1. Pure diffusion: 1D Gaussian temperature profile in quiescent flow
 *    decays according to the analytical heat equation solution.
 * 2. Pure advection: uniform flow carries temperature profile downstream.
 * 3. Energy equation disabled: alpha=0 should leave temperature unchanged.
 * 4. Buoyancy source: verify energy_compute_buoyancy produces correct forces.
 * 5. Backward compatibility: existing solvers work when alpha=0.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/energy_solver.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST SETUP / TEARDOWN
 * ============================================================================ */

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* ============================================================================
 * TEST 1: Pure Diffusion
 *
 * Initialize a Gaussian temperature pulse in quiescent flow (u=v=w=0).
 * After time t, the analytical solution is:
 *   T(x,y,t) = T0 / (1 + 4*alpha*t/sigma^2) * exp(-r^2 / (sigma^2 + 4*alpha*t))
 * where sigma is the initial width and r^2 = (x-x0)^2 + (y-y0)^2.
 *
 * We verify the peak temperature decays and the profile broadens.
 * ============================================================================ */

#define DIFF_NX 33
#define DIFF_NY 33
#define DIFF_ALPHA 0.01
#define DIFF_DT 0.0001
#define DIFF_STEPS 100
#define DIFF_SIGMA 0.1
#define DIFF_T0 10.0

static void test_pure_diffusion(void) {
    grid* g = grid_create(DIFF_NX, DIFF_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(DIFF_NX, DIFF_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Initialize: zero velocity, Gaussian temperature pulse at center */
    double cx = 0.5, cy = 0.5;
    for (size_t j = 0; j < DIFF_NY; j++) {
        for (size_t i = 0; i < DIFF_NX; i++) {
            size_t idx = j * DIFF_NX + i;
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->w[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;

            double x = g->x[i];
            double y = g->y[j];
            double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            field->T[idx] = DIFF_T0 * exp(-r2 / (DIFF_SIGMA * DIFF_SIGMA));
        }
    }

    /* Record initial peak temperature */
    double T_peak_initial = 0.0;
    for (size_t n = 0; n < (size_t)(DIFF_NX * DIFF_NY); n++) {
        if (field->T[n] > T_peak_initial) T_peak_initial = field->T[n];
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = DIFF_ALPHA;

    /* Run diffusion steps */
    for (int step = 0; step < DIFF_STEPS; step++) {
        cfd_status_t status = energy_step_explicit(field, g, &params, DIFF_DT, step * DIFF_DT);
        if (status != CFD_SUCCESS) {
            printf("  DIVERGED at step %d, status=%d\n", step, status);
            /* Print min/max T for debugging */
            double tmin = 1e30, tmax = -1e30;
            for (size_t n = 0; n < (size_t)(DIFF_NX * DIFF_NY); n++) {
                if (field->T[n] < tmin) tmin = field->T[n];
                if (field->T[n] > tmax) tmax = field->T[n];
                if (!isfinite(field->T[n])) {
                    printf("  NaN/Inf at index %zu\n", n);
                    break;
                }
            }
            printf("  T range: [%e, %e]\n", tmin, tmax);
        }
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    /* Check peak temperature has decreased (diffusion spreads the pulse) */
    double T_peak_final = 0.0;
    for (size_t n = 0; n < (size_t)(DIFF_NX * DIFF_NY); n++) {
        if (field->T[n] > T_peak_final) T_peak_final = field->T[n];
    }

    printf("  Diffusion test: T_peak initial=%.4f, final=%.4f\n",
           T_peak_initial, T_peak_final);

    TEST_ASSERT_TRUE_MESSAGE(T_peak_final < T_peak_initial,
                              "Peak temperature should decrease due to diffusion");
    TEST_ASSERT_TRUE_MESSAGE(T_peak_final > 0.0,
                              "Temperature should remain positive");

    /* Verify analytical decay rate (approximate).
     * Analytical peak: T0 / (1 + 4*alpha*t/sigma^2)
     * t = DIFF_STEPS * DIFF_DT = 0.01 */
    double t_final = DIFF_STEPS * DIFF_DT;
    double analytical_peak = DIFF_T0 / (1.0 + 4.0 * DIFF_ALPHA * t_final /
                                         (DIFF_SIGMA * DIFF_SIGMA));
    double rel_error = fabs(T_peak_final - analytical_peak) / analytical_peak;
    printf("  Analytical peak=%.4f, numerical=%.4f, rel_error=%.4f\n",
           analytical_peak, T_peak_final, rel_error);

    /* Allow 20% error due to boundary effects and discretization */
    TEST_ASSERT_TRUE_MESSAGE(rel_error < 0.20,
                              "Peak temperature should match analytical within 20%");

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 2: Pure Advection
 *
 * Uniform horizontal flow (u=1, v=0) with a temperature step.
 * After advection, the temperature profile should shift downstream.
 * ============================================================================ */

#define ADV_NX 41
#define ADV_NY 5
#define ADV_U  1.0
#define ADV_DT 0.001
#define ADV_STEPS 50

static void test_pure_advection(void) {
    grid* g = grid_create(ADV_NX, ADV_NY, 1, 0.0, 2.0, 0.0, 0.5, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(ADV_NX, ADV_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Initialize: uniform rightward flow, temperature hot on left half */
    for (size_t j = 0; j < ADV_NY; j++) {
        for (size_t i = 0; i < ADV_NX; i++) {
            size_t idx = j * ADV_NX + i;
            field->u[idx] = ADV_U;
            field->v[idx] = 0.0;
            field->w[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            /* Smooth tanh profile centered at x=0.5 */
            double x = g->x[i];
            field->T[idx] = 0.5 * (1.0 - tanh(20.0 * (x - 0.5)));
        }
    }

    /* Measure initial center of mass of temperature */
    double sum_Tx_initial = 0.0, sum_T_initial = 0.0;
    size_t j_mid = ADV_NY / 2;
    for (size_t i = 0; i < ADV_NX; i++) {
        size_t idx = j_mid * ADV_NX + i;
        sum_Tx_initial += field->T[idx] * g->x[i];
        sum_T_initial += field->T[idx];
    }
    double com_initial = sum_Tx_initial / sum_T_initial;

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.0001;  /* Small diffusivity to stabilize */

    for (int step = 0; step < ADV_STEPS; step++) {
        cfd_status_t status = energy_step_explicit(field, g, &params, ADV_DT,
                                                     step * ADV_DT);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    /* Measure final center of mass */
    double sum_Tx_final = 0.0, sum_T_final = 0.0;
    for (size_t i = 0; i < ADV_NX; i++) {
        size_t idx = j_mid * ADV_NX + i;
        sum_Tx_final += field->T[idx] * g->x[i];
        sum_T_final += field->T[idx];
    }
    double com_final = sum_Tx_final / sum_T_final;

    double expected_shift = ADV_U * ADV_STEPS * ADV_DT;
    double actual_shift = com_final - com_initial;

    printf("  Advection test: expected shift=%.4f, actual=%.4f\n",
           expected_shift, actual_shift);

    /* Advection should move the profile rightward */
    TEST_ASSERT_TRUE_MESSAGE(actual_shift > 0.0,
                              "Temperature profile should shift downstream");
    /* Allow 50% tolerance on shift magnitude (central differencing has
     * numerical diffusion and dispersion) */
    TEST_ASSERT_TRUE_MESSAGE(actual_shift > expected_shift * 0.3,
                              "Shift should be at least 30% of expected");

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 3: Energy Equation Disabled
 *
 * When alpha=0, energy_step_explicit should be a no-op.
 * ============================================================================ */

static void test_energy_disabled(void) {
    grid* g = grid_create(9, 9, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(9, 9, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Set non-uniform temperature */
    for (size_t n = 0; n < 81; n++) {
        field->u[n] = 1.0;
        field->v[n] = 0.5;
        field->rho[n] = 1.0;
        field->T[n] = 300.0 + (double)n;
    }

    /* Save original temperature */
    double T_orig[81];
    memcpy(T_orig, field->T, 81 * sizeof(double));

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.0;  /* Disabled */

    cfd_status_t status = energy_step_explicit(field, g, &params, 0.001, 0.0);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Temperature should be unchanged */
    for (size_t n = 0; n < 81; n++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[n], field->T[n]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4: Buoyancy Source Computation
 * ============================================================================ */

static void test_buoyancy_source(void) {
    ns_solver_params_t params = ns_solver_params_default();
    params.beta = 0.001;    /* Thermal expansion coefficient */
    params.T_ref = 300.0;   /* Reference temperature */
    params.gravity[0] = 0.0;
    params.gravity[1] = -9.81;
    params.gravity[2] = 0.0;

    double source_u = 0.0, source_v = 0.0, source_w = 0.0;

    /* Hot fluid (T > T_ref): should produce upward force (positive v for g_y < 0) */
    energy_compute_buoyancy(310.0, &params, &source_u, &source_v, &source_w);

    /* F = -beta * (T - T_ref) * g
     * source_v = -0.001 * (310 - 300) * (-9.81) = +0.0981 */
    printf("  Buoyancy: source_u=%.6f, source_v=%.6f, source_w=%.6f\n",
           source_u, source_v, source_w);

    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, source_u);
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, 0.0981, source_v);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, 0.0, source_w);

    /* Cold fluid (T < T_ref): should produce downward force */
    source_u = 0.0; source_v = 0.0; source_w = 0.0;
    energy_compute_buoyancy(290.0, &params, &source_u, &source_v, &source_w);

    /* source_v = -0.001 * (290 - 300) * (-9.81) = -0.0981 */
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, -0.0981, source_v);

    /* Zero beta: no buoyancy */
    params.beta = 0.0;
    source_u = 0.0; source_v = 0.0; source_w = 0.0;
    energy_compute_buoyancy(500.0, &params, &source_u, &source_v, &source_w);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, source_u);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, source_v);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, source_w);
}

/* ============================================================================
 * TEST 5: Backward Compatibility
 *
 * Existing solvers should work identically when alpha=0 (default).
 * Run a projection step with default params and verify no NaN/crash.
 * ============================================================================ */

static void test_backward_compatibility(void) {
    grid* g = grid_create(17, 17, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(17, 17, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Initialize with quiescent flow */
    for (size_t j = 0; j < 17; j++) {
        for (size_t i = 0; i < 17; i++) {
            size_t idx = j * 17 + i;
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->w[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.max_iter = 1;
    params.source_func = NULL;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;
    /* alpha=0 by default, so energy equation is disabled */

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);

    cfd_status_t init_status = solver_init(solver, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t step_status = solver_step(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_status);

    /* Temperature should remain at 300 (energy equation disabled) */
    for (size_t n = 0; n < 17 * 17; n++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-10, 300.0, field->T[n]);
    }

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 6: Heat Source Function
 *
 * Apply a uniform heat source Q=1.0 to a quiescent flow with zero initial
 * temperature. After N steps, T should increase by approximately Q*N*dt
 * in the interior (no advection, diffusion of uniform field is zero).
 * ============================================================================ */

#define HS_NX 17
#define HS_NY 17
#define HS_DT 0.001
#define HS_STEPS 50
#define HS_Q 1.0

static double uniform_heat_source(double x, double y, double z, double t,
                                   void* context) {
    (void)x; (void)y; (void)z; (void)t; (void)context;
    return HS_Q;
}

static void test_heat_source(void) {
    grid* g = grid_create(HS_NX, HS_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(HS_NX, HS_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Quiescent flow, uniform T=0, rho=1 */
    for (size_t n = 0; n < (size_t)(HS_NX * HS_NY); n++) {
        field->u[n] = 0.0;
        field->v[n] = 0.0;
        field->w[n] = 0.0;
        field->p[n] = 1.0;
        field->rho[n] = 1.0;
        field->T[n] = 0.0;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;  /* Need alpha > 0 to enable energy equation */
    params.heat_source_func = uniform_heat_source;
    params.heat_source_context = NULL;

    for (int step = 0; step < HS_STEPS; step++) {
        cfd_status_t status = energy_step_explicit(field, g, &params, HS_DT,
                                                    step * HS_DT);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
    }

    /* Interior points should have T ≈ Q * N * dt = 1.0 * 50 * 0.001 = 0.05
     * (boundary points stay at 0 since the stencil doesn't update them) */
    double expected = HS_Q * HS_STEPS * HS_DT;
    size_t ci = HS_NX / 2;
    size_t cj = HS_NY / 2;
    double T_center = field->T[cj * HS_NX + ci];

    printf("  Heat source test: T_center=%.6f, expected≈%.6f\n",
           T_center, expected);

    /* Uniform field has zero Laplacian, so diffusion doesn't contribute.
     * The only driver is Q. Allow 10% tolerance for boundary effects. */
    TEST_ASSERT_TRUE_MESSAGE(T_center > 0.0,
                              "Temperature should increase with heat source");
    TEST_ASSERT_DOUBLE_WITHIN(expected * 0.10, expected, T_center);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_pure_diffusion);
    RUN_TEST(test_pure_advection);
    RUN_TEST(test_energy_disabled);
    RUN_TEST(test_buoyancy_source);
    RUN_TEST(test_backward_compatibility);
    RUN_TEST(test_heat_source);
    return UNITY_END();
}
