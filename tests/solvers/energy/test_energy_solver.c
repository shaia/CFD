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

#define DIS_NX 9
#define DIS_NY 9
#define DIS_TOTAL ((size_t)(DIS_NX) * DIS_NY)

static void test_energy_disabled(void) {
    grid* g = grid_create(DIS_NX, DIS_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(DIS_NX, DIS_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Set non-uniform temperature */
    for (size_t n = 0; n < DIS_TOTAL; n++) {
        field->u[n] = 1.0;
        field->v[n] = 0.5;
        field->rho[n] = 1.0;
        field->T[n] = 300.0 + (double)n;
    }

    /* Save original temperature */
    double T_orig[DIS_TOTAL];
    memcpy(T_orig, field->T, DIS_TOTAL * sizeof(double));

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.0;  /* Disabled */

    cfd_status_t status = energy_step_explicit(field, g, &params, 0.001, 0.0);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Temperature should be unchanged */
    for (size_t n = 0; n < DIS_TOTAL; n++) {
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
 * TEST 7: Thermal BC — Mixed Dirichlet / Neumann
 *
 * Configure Dirichlet on left/right (fixed T) and Neumann on top/bottom
 * (zero-gradient). After calling energy_apply_thermal_bcs, verify:
 *   - Left/right boundaries have the Dirichlet values
 *   - Top/bottom boundaries equal the adjacent interior row
 * ============================================================================ */

#define TBC_NX 11
#define TBC_NY 11

static void test_thermal_bc_dirichlet_neumann(void) {
    grid* g = grid_create(TBC_NX, TBC_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(TBC_NX, TBC_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Fill T with a recognizable pattern: T = 100 + i + j*10 */
    for (size_t j = 0; j < TBC_NY; j++) {
        for (size_t i = 0; i < TBC_NX; i++) {
            size_t idx = j * TBC_NX + i;
            field->T[idx] = 100.0 + (double)i + (double)j * 10.0;
            field->rho[idx] = 1.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;  /* Must be > 0 to activate thermal BCs */
    params.thermal_bc.left   = BC_TYPE_DIRICHLET;
    params.thermal_bc.right  = BC_TYPE_DIRICHLET;
    params.thermal_bc.bottom = BC_TYPE_NEUMANN;
    params.thermal_bc.top    = BC_TYPE_NEUMANN;
    params.thermal_bc.dirichlet_values.left  = 500.0;
    params.thermal_bc.dirichlet_values.right = 200.0;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, energy_apply_thermal_bcs(field, &params));

    /* Verify Dirichlet on left (i=0) and right (i=nx-1) */
    for (size_t j = 0; j < TBC_NY; j++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, 500.0, field->T[j * TBC_NX + 0]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, 200.0, field->T[j * TBC_NX + (TBC_NX - 1)]);
    }

    /* Verify Neumann on bottom (j=0): T[0,i] == T[1,i] */
    for (size_t i = 0; i < TBC_NX; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, field->T[1 * TBC_NX + i],
                                   field->T[0 * TBC_NX + i]);
    }

    /* Verify Neumann on top (j=ny-1): T[ny-1,i] == T[ny-2,i] */
    for (size_t i = 0; i < TBC_NX; i++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, field->T[(TBC_NY - 2) * TBC_NX + i],
                                   field->T[(TBC_NY - 1) * TBC_NX + i]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 8: Thermal BC — All Periodic
 *
 * Default thermal_bc config (all PERIODIC) copies boundary cells from the
 * opposite interior cell.  Verify:
 *   - Interior cells are unchanged
 *   - Boundary cells equal the opposite interior cell
 * ============================================================================ */

#define PER_NX 9
#define PER_NY 9

static void test_thermal_bc_all_periodic(void) {
    grid* g = grid_create(PER_NX, PER_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(PER_NX, PER_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Fill T with a recognizable pattern: T = 100 + i + j*10 */
    for (size_t j = 0; j < PER_NY; j++) {
        for (size_t i = 0; i < PER_NX; i++) {
            size_t idx = j * PER_NX + i;
            field->T[idx] = 100.0 + (double)i + (double)j * 10.0;
            field->rho[idx] = 1.0;
        }
    }

    /* Save original interior T for comparison */
    double T_orig[PER_NX * PER_NY];
    memcpy(T_orig, field->T, sizeof(T_orig));

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;  /* Energy enabled, all-periodic by default */

    TEST_ASSERT_EQUAL(CFD_SUCCESS, energy_apply_thermal_bcs(field, &params));

    /* Interior cells (1..nx-2, 1..ny-2) should be unchanged */
    for (size_t j = 1; j < PER_NY - 1; j++) {
        for (size_t i = 1; i < PER_NX - 1; i++) {
            size_t idx = j * PER_NX + i;
            TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[idx], field->T[idx]);
        }
    }

    /* Left face (i=0): should equal interior column i=nx-2 (before bottom/top overwrite) */
    /* Bottom/top periodic run after left/right, so j=0 and j=ny-1 are overwritten.
     * Check interior rows only. */
    for (size_t j = 1; j < PER_NY - 1; j++) {
        size_t left_idx = j * PER_NX;
        size_t src_idx  = j * PER_NX + (PER_NX - 2);
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[src_idx], field->T[left_idx]);
    }

    /* Right face (i=nx-1): should equal interior column i=1 */
    for (size_t j = 1; j < PER_NY - 1; j++) {
        size_t right_idx = j * PER_NX + (PER_NX - 1);
        size_t src_idx   = j * PER_NX + 1;
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[src_idx], field->T[right_idx]);
    }

    /* Bottom face (j=0): copies from j=ny-2, but left/right periodic may have
     * modified i=0 and i=nx-1 in that row. Check interior columns. */
    for (size_t i = 1; i < PER_NX - 1; i++) {
        size_t bot_idx = i;
        size_t src_idx = (PER_NY - 2) * PER_NX + i;
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[src_idx], field->T[bot_idx]);
    }

    /* Top face (j=ny-1): copies from j=1 */
    for (size_t i = 1; i < PER_NX - 1; i++) {
        size_t top_idx = (PER_NY - 1) * PER_NX + i;
        size_t src_idx = PER_NX + i;
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, T_orig[src_idx], field->T[top_idx]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 9: 3D Thermal BCs — Front/Back Dirichlet + Neumann
 *
 * Create a small 3D field (nz=5) and verify:
 *   - Back face (k=0): Dirichlet sets T to specified value
 *   - Front face (k=nz-1): Neumann copies from adjacent interior plane
 *   - Left/right: Periodic copies from opposite interior column
 *   - Bottom/top: Dirichlet values overwrite corners correctly
 * ============================================================================ */

#define TBC3D_NX 7
#define TBC3D_NY 7
#define TBC3D_NZ 5
#define TBC3D_PLANE ((size_t)(TBC3D_NX) * TBC3D_NY)

static void test_thermal_bc_3d_front_back(void) {
    grid* g = grid_create(TBC3D_NX, TBC3D_NY, TBC3D_NZ,
                          0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(TBC3D_NX, TBC3D_NY, TBC3D_NZ);
    TEST_ASSERT_NOT_NULL(field);

    /* Fill T with a recognizable 3D pattern: T = k*1000 + j*100 + i */
    for (size_t k = 0; k < TBC3D_NZ; k++) {
        for (size_t j = 0; j < TBC3D_NY; j++) {
            for (size_t i = 0; i < TBC3D_NX; i++) {
                size_t idx = k * TBC3D_PLANE + j * TBC3D_NX + i;
                field->T[idx] = (double)k * 1000.0 + (double)j * 100.0 + (double)i;
                field->rho[idx] = 1.0;
            }
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;  /* Enable thermal BCs */

    /* Back = Dirichlet(50), Front = Neumann, Left/Right = Periodic,
     * Bottom = Dirichlet(99), Top = Dirichlet(77) */
    params.thermal_bc.back   = BC_TYPE_DIRICHLET;
    params.thermal_bc.front  = BC_TYPE_NEUMANN;
    params.thermal_bc.left   = BC_TYPE_PERIODIC;
    params.thermal_bc.right  = BC_TYPE_PERIODIC;
    params.thermal_bc.bottom = BC_TYPE_DIRICHLET;
    params.thermal_bc.top    = BC_TYPE_DIRICHLET;
    params.thermal_bc.dirichlet_values.back   = 50.0;
    params.thermal_bc.dirichlet_values.bottom = 99.0;
    params.thermal_bc.dirichlet_values.top    = 77.0;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, energy_apply_thermal_bcs(field, &params));

    /* Verify back face (k=0): back Dirichlet runs after bottom/top,
     * so the entire k=0 plane should be 50.0. */
    for (size_t j = 0; j < TBC3D_NY; j++) {
        for (size_t i = 0; i < TBC3D_NX; i++) {
            size_t idx = j * TBC3D_NX + i;
            TEST_ASSERT_DOUBLE_WITHIN(1e-15, 50.0, field->T[idx]);
        }
    }

    /* Verify front face (k=nz-1): Neumann copies from k=nz-2 */
    size_t front_base = (TBC3D_NZ - 1) * TBC3D_PLANE;
    size_t interior_base = (TBC3D_NZ - 2) * TBC3D_PLANE;
    for (size_t j = 1; j < TBC3D_NY - 1; j++) {
        for (size_t i = 1; i < TBC3D_NX - 1; i++) {
            size_t off = j * TBC3D_NX + i;
            TEST_ASSERT_DOUBLE_WITHIN(1e-15,
                                       field->T[interior_base + off],
                                       field->T[front_base + off]);
        }
    }

    /* Verify left face (i=0): Periodic copies from i=nx-2 */
    for (size_t k = 1; k < TBC3D_NZ - 1; k++) {
        size_t base = k * TBC3D_PLANE;
        for (size_t j = 1; j < TBC3D_NY - 1; j++) {
            size_t left_idx = base + j * TBC3D_NX;
            size_t src_idx = base + j * TBC3D_NX + (TBC3D_NX - 2);
            TEST_ASSERT_DOUBLE_WITHIN(1e-15,
                                       field->T[src_idx],
                                       field->T[left_idx]);
        }
    }

    /* Verify right face (i=nx-1): Periodic copies from i=1 */
    for (size_t k = 1; k < TBC3D_NZ - 1; k++) {
        size_t base = k * TBC3D_PLANE;
        for (size_t j = 1; j < TBC3D_NY - 1; j++) {
            size_t right_idx = base + j * TBC3D_NX + (TBC3D_NX - 1);
            size_t src_idx = base + j * TBC3D_NX + 1;
            TEST_ASSERT_DOUBLE_WITHIN(1e-15,
                                       field->T[src_idx],
                                       field->T[right_idx]);
        }
    }

    /* Verify bottom face (j=0): Dirichlet 99.0 at interior k planes.
     * k=0 is overwritten by back Dirichlet, k=nz-1 by front Neumann. */
    for (size_t k = 1; k < TBC3D_NZ - 1; k++) {
        size_t base = k * TBC3D_PLANE;
        for (size_t i = 0; i < TBC3D_NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(1e-15, 99.0, field->T[base + i]);
        }
    }

    /* Verify top face (j=ny-1): Dirichlet 77.0 at interior k planes */
    for (size_t k = 1; k < TBC3D_NZ - 1; k++) {
        size_t base = k * TBC3D_PLANE;
        for (size_t i = 0; i < TBC3D_NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(1e-15, 77.0,
                                       field->T[base + (TBC3D_NY - 1) * TBC3D_NX + i]);
        }
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 11: Thermal BC invalid configuration is rejected
 *
 * energy_apply_thermal_bcs must surface misconfiguration as CFD_ERROR_INVALID
 * rather than silently leaving a face unchanged: NULL inputs and unsupported
 * per-face BC types (e.g. BC_TYPE_NOSLIP) are errors. alpha<=0 stays a no-op.
 * ============================================================================ */

static void test_thermal_bc_invalid_config(void) {
    const size_t nx = 5, ny = 5;
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;

    /* NULL field/params -> CFD_ERROR_INVALID */
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, energy_apply_thermal_bcs(NULL, &params));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, energy_apply_thermal_bcs(field, NULL));

    /* Unsupported per-face BC type -> CFD_ERROR_INVALID */
    params.thermal_bc.left = BC_TYPE_NOSLIP;
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, energy_apply_thermal_bcs(field, &params));

    /* A valid configuration still succeeds */
    params.thermal_bc.left = BC_TYPE_DIRICHLET;
    params.thermal_bc.dirichlet_values.left = 300.0;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, energy_apply_thermal_bcs(field, &params));

    /* alpha<=0 is a no-op success regardless of (otherwise invalid) config */
    params.alpha = 0.0;
    params.thermal_bc.left = BC_TYPE_NOSLIP;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, energy_apply_thermal_bcs(field, &params));

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 12: Energy equation cross-backend consistency (OMP vs scalar)
 *
 * The energy equation + thermal BCs are now implemented on the OMP backends.
 * Each OMP solver must (a) accept alpha>0 instead of returning
 * CFD_ERROR_UNSUPPORTED, and (b) produce a temperature field that matches its
 * scalar counterpart, since both share identical numerics. A heat-conduction
 * setup (left-hot / right-cold Dirichlet faces, quiescent flow) is advanced on
 * both backends and the final T fields are compared.
 * ============================================================================ */

#define CONSIST_NX 9
#define CONSIST_NY 9
#define CONSIST_N  (CONSIST_NX * CONSIST_NY)

/* Advance `solver_name` for `steps` steps of left-hot/right-cold heat
 * conduction, writing the final temperature field into T_out (size CONSIST_N).
 *
 * Return value distinguishes "backend genuinely unavailable" (caller should
 * skip) from "backend present but rejected the energy run" (caller should
 * fail):
 *   CFD_ERROR_NOT_FOUND   - solver name not registered, or init returned
 *                           CFD_ERROR_UNSUPPORTED (backend/sub-solver absent)
 *   other init error      - propagated as-is (real failure)
 *   solver_solve's status - when init succeeded; a solve-time
 *                           CFD_ERROR_UNSUPPORTED therefore surfaces as a
 *                           failure rather than a silent skip. */
static cfd_status_t run_energy_case(ns_solver_registry_t* registry,
                                    const char* solver_name, int steps,
                                    double* T_out) {
    grid* g = grid_create(CONSIST_NX, CONSIST_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    if (!g) return CFD_ERROR_NOMEM;
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(CONSIST_NX, CONSIST_NY, 1);
    if (!field) { grid_destroy(g); return CFD_ERROR_NOMEM; }
    for (size_t n = 0; n < CONSIST_N; n++) {
        field->rho[n] = 1.0;
        field->T[n] = 300.0;
    }

    ns_solver_t* solver = cfd_solver_create(registry, solver_name);
    if (!solver) {
        /* Backend not registered — signal skip */
        flow_field_destroy(field);
        grid_destroy(g);
        return CFD_ERROR_NOT_FOUND;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.05;
    params.dt = 0.001;
    params.max_iter = steps;
    params.thermal_bc.left = BC_TYPE_DIRICHLET;
    params.thermal_bc.right = BC_TYPE_DIRICHLET;
    params.thermal_bc.bottom = BC_TYPE_NEUMANN;
    params.thermal_bc.top = BC_TYPE_NEUMANN;
    params.thermal_bc.dirichlet_values.left = 320.0;
    params.thermal_bc.dirichlet_values.right = 280.0;

    cfd_status_t status = solver_init(solver, g, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        /* Backend present but cannot initialize this config (e.g. a required
         * SIMD sub-solver is absent) — treat as a skip, not a failure. */
        status = CFD_ERROR_NOT_FOUND;
    } else if (status == CFD_SUCCESS) {
        ns_solver_stats_t stats = ns_solver_stats_default();
        /* A solve-time CFD_ERROR_UNSUPPORTED is returned as-is so the caller
         * fails: init accepted the solver, so rejecting the energy params now
         * is a real regression, not an absent backend. */
        status = solver_solve(solver, field, g, &params, &stats);
        if (status == CFD_SUCCESS) {
            memcpy(T_out, field->T, CONSIST_N * sizeof(double));
        }
    }

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    return status;
}

static void test_energy_optimized_matches_scalar(void) {
    /* Each optimized backend (OMP and AVX2) must accept the energy equation and
     * reproduce the scalar reference temperature field. */
    const struct {
        const char* scalar;
        const char* optimized;
    } pairs[] = {
        { NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP },
        { NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP },
        { NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OMP },
        { NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED },
        { NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OPTIMIZED },
        { NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED },
    };
    const size_t n_pairs = sizeof(pairs) / sizeof(pairs[0]);
    const int steps = 20;

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    for (size_t p = 0; p < n_pairs; p++) {
        double T_scalar[CONSIST_N];
        double T_opt[CONSIST_N];

        /* Scalar reference must always succeed. */
        cfd_status_t ss = run_energy_case(registry, pairs[p].scalar, steps, T_scalar);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, ss,
                                  "scalar energy solve must succeed");

        /* Optimized backend: skip cleanly only if it (or its sub-solver) is
         * genuinely absent. A present backend that rejects the energy params
         * at solve time returns CFD_ERROR_UNSUPPORTED here and must fail. */
        cfd_status_t os = run_energy_case(registry, pairs[p].optimized, steps, T_opt);
        if (os == CFD_ERROR_NOT_FOUND) {
            continue;
        }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, os,
                                  "optimized solve must accept energy equation params");

        /* Tolerance is moderate (not machine-epsilon): the AVX2 momentum kernel
         * uses fused multiply-add, so the advecting velocity field differs from
         * the scalar path at FP-epsilon and accumulates over the run. A real
         * stencil/BC bug would diverge by O(1), far above this bound. */
        for (size_t n = 0; n < CONSIST_N; n++) {
            TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(
                1e-3, T_scalar[n], T_opt[n],
                "optimized temperature field must match scalar reference");
        }
    }

    cfd_registry_destroy(registry);
}

/* ============================================================================
 * TEST 13: GPU energy equation — diffusion decay sanity
 *
 * The GPU projection backend uses a different Poisson algorithm (Jacobi +
 * relaxation) than the scalar/OMP/AVX2 CG path, so its advected temperature
 * does not match the scalar reference to 1e-3 — a cross-backend equality test
 * (test_energy_optimized_matches_scalar) is inappropriate for it. Instead we
 * check physical behavior: a Gaussian temperature pulse in quiescent flow with
 * adiabatic walls must diffuse (peak decreases, field stays finite/positive).
 * Skips cleanly when no CUDA device is present.
 * ============================================================================ */

#define GPU_NX 33
#define GPU_NY 33

static void test_energy_gpu_diffusion_decay(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (!solver) {
        /* No CUDA device / GPU backend not registered — skip. */
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("GPU projection solver unavailable");
        return;
    }

    grid* g = grid_create(GPU_NX, GPU_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(GPU_NX, GPU_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    const double cx = 0.5, cy = 0.5, sigma = 0.15;
    double T_peak_initial = 0.0;
    for (size_t j = 0; j < GPU_NY; j++) {
        for (size_t i = 0; i < GPU_NX; i++) {
            size_t idx = j * GPU_NX + i;
            field->u[idx] = 0.0; field->v[idx] = 0.0; field->w[idx] = 0.0;
            field->p[idx] = 0.0; field->rho[idx] = 1.0;
            double x = g->x[i], y = g->y[j];
            double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            field->T[idx] = 10.0 * exp(-r2 / (sigma * sigma));
            if (field->T[idx] > T_peak_initial) T_peak_initial = field->T[idx];
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;
    params.dt = 0.0002;
    params.max_iter = 50;
    /* Adiabatic (zero-gradient) walls so the pulse simply diffuses. */
    params.thermal_bc.left = BC_TYPE_NEUMANN;
    params.thermal_bc.right = BC_TYPE_NEUMANN;
    params.thermal_bc.bottom = BC_TYPE_NEUMANN;
    params.thermal_bc.top = BC_TYPE_NEUMANN;

    cfd_status_t init_status = solver_init(solver, g, &params);
    if (init_status == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(solver);
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("GPU projection init unsupported");
        return;
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t solve_status = solver_solve(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, solve_status,
                              "GPU energy solve must succeed");

    double T_peak_final = 0.0;
    int all_finite = 1;
    for (size_t n = 0; n < (size_t)(GPU_NX * GPU_NY); n++) {
        if (!isfinite(field->T[n])) all_finite = 0;
        if (field->T[n] > T_peak_final) T_peak_final = field->T[n];
    }
    printf("  GPU diffusion: T_peak initial=%.4f, final=%.4f\n",
           T_peak_initial, T_peak_final);

    TEST_ASSERT_TRUE_MESSAGE(all_finite, "GPU temperature field must stay finite");
    TEST_ASSERT_TRUE_MESSAGE(T_peak_final < T_peak_initial,
                             "GPU diffusion must lower the peak temperature");
    TEST_ASSERT_TRUE_MESSAGE(T_peak_final > 0.0,
                             "GPU temperature must remain positive");

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

/* ============================================================================
 * TEST 14: GPU energy equation rejects host heat-source callbacks
 *
 * A host C function pointer cannot run on the device, so the GPU projection
 * solver must surface CFD_ERROR_UNSUPPORTED when heat_source_func is set with
 * the energy equation enabled (rather than silently ignoring the source).
 * Skips cleanly when no CUDA device is present.
 * ============================================================================ */

static void test_energy_gpu_rejects_heat_source(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU);
    if (!solver) {
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("GPU projection solver unavailable");
        return;
    }

    grid* g = grid_create(GPU_NX, GPU_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(GPU_NX, GPU_NY, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t n = 0; n < (size_t)(GPU_NX * GPU_NY); n++) {
        field->rho[n] = 1.0;
        field->T[n] = 300.0;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.alpha = 0.01;
    params.dt = 0.0002;
    params.max_iter = 1;
    params.heat_source_func = uniform_heat_source;  /* not supported on GPU */

    cfd_status_t init_status = solver_init(solver, g, &params);
    if (init_status == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(solver);
        flow_field_destroy(field);
        grid_destroy(g);
        cfd_registry_destroy(registry);
        TEST_IGNORE_MESSAGE("GPU projection init unsupported");
        return;
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t solve_status = solver_solve(solver, field, g, &params, &stats);
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, solve_status,
                              "GPU energy must reject host heat_source_func");

    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
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
    RUN_TEST(test_thermal_bc_dirichlet_neumann);
    RUN_TEST(test_thermal_bc_all_periodic);
    RUN_TEST(test_thermal_bc_3d_front_back);
    RUN_TEST(test_thermal_bc_invalid_config);
    RUN_TEST(test_energy_optimized_matches_scalar);
    RUN_TEST(test_energy_gpu_diffusion_decay);
    RUN_TEST(test_energy_gpu_rejects_heat_source);
    return UNITY_END();
}
