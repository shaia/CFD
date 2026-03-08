/**
 * @file test_natural_convection.c
 * @brief Natural convection validation test (differentially heated cavity)
 *
 * Tests the coupled energy equation + Boussinesq buoyancy using the
 * de Vahl Davis (1983) benchmark: a square cavity with:
 *   - Hot left wall (T = T_hot)
 *   - Cold right wall (T = T_cold)
 *   - Insulated (adiabatic) top and bottom walls
 *   - No-slip velocity BCs on all walls
 *
 * At steady state, natural convection creates a clockwise circulation
 * (for gravity pointing downward). The Rayleigh number is:
 *   Ra = g * beta * dT * L^3 / (nu * alpha)
 *
 * For Ra = 1000, the benchmark Nusselt number is Nu ~ 1.118 (de Vahl Davis).
 *
 * This test runs a short simulation and verifies:
 * 1. Flow develops in the correct direction (buoyancy-driven)
 * 2. Temperature gradient is established between hot and cold walls
 * 3. Maximum velocity is physically reasonable
 * 4. No divergence (NaN/Inf)
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/* Domain: unit square cavity */
#define NC_NX 21
#define NC_NY 21
#define NC_LENGTH 1.0

/* Physical parameters for Ra = 1000 */
#define NC_T_HOT    310.0
#define NC_T_COLD   290.0
#define NC_T_REF    300.0   /* (T_hot + T_cold) / 2 */
#define NC_DT_TEMP  (NC_T_HOT - NC_T_COLD)  /* 20 K */
#define NC_BETA     0.003333  /* 1/T_ref [1/K] */
#define NC_G        9.81

/* Derived: Ra = g * beta * dT * L^3 / (nu * alpha)
 * For Ra = 1000 with the above beta, dT, L=1:
 *   nu * alpha = g * beta * dT * L^3 / Ra = 9.81 * 0.003333 * 20 / 1000 = 6.54e-4
 * Choose Pr = nu/alpha = 0.71 (air)
 *   alpha = sqrt(6.54e-4 / 0.71) = 0.0303
 *   nu = 0.71 * alpha = 0.0215 */
#define NC_ALPHA    0.0303
#define NC_NU       0.0215

/* Time stepping */
#define NC_DT       0.0005
#define NC_STEPS    500

/* ============================================================================
 * TEST SETUP / TEARDOWN
 * ============================================================================ */

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* ============================================================================
 * Apply thermal boundary conditions for differentially heated cavity
 * ============================================================================ */

static void apply_cavity_bcs(flow_field* field, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        /* Left wall: hot */
        field->T[j * nx + 0] = NC_T_HOT;
        /* Right wall: cold */
        field->T[j * nx + (nx - 1)] = NC_T_COLD;

        /* No-slip on left and right walls */
        field->u[j * nx + 0] = 0.0;
        field->v[j * nx + 0] = 0.0;
        field->u[j * nx + (nx - 1)] = 0.0;
        field->v[j * nx + (nx - 1)] = 0.0;
    }

    for (size_t i = 0; i < nx; i++) {
        /* Bottom wall: adiabatic (Neumann dT/dy=0 via copy from interior) */
        field->T[0 * nx + i] = field->T[1 * nx + i];
        /* Top wall: adiabatic */
        field->T[(ny - 1) * nx + i] = field->T[(ny - 2) * nx + i];

        /* No-slip on top and bottom walls */
        field->u[0 * nx + i] = 0.0;
        field->v[0 * nx + i] = 0.0;
        field->u[(ny - 1) * nx + i] = 0.0;
        field->v[(ny - 1) * nx + i] = 0.0;
    }
}

/* ============================================================================
 * TEST: Buoyancy-Driven Flow Development
 * ============================================================================ */

static void test_natural_convection_development(void) {
    grid* g = grid_create(NC_NX, NC_NY, 1, 0.0, NC_LENGTH, 0.0, NC_LENGTH, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(NC_NX, NC_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Initialize: quiescent flow, linear temperature profile */
    for (size_t j = 0; j < NC_NY; j++) {
        for (size_t i = 0; i < NC_NX; i++) {
            size_t idx = j * NC_NX + i;
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->w[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            /* Linear temperature from hot (left) to cold (right) */
            double x = g->x[i];
            field->T[idx] = NC_T_HOT - NC_DT_TEMP * (x / NC_LENGTH);
        }
    }

    /* Set up solver params with energy equation and Boussinesq */
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = NC_DT;
    params.mu = NC_NU;
    params.alpha = NC_ALPHA;
    params.beta = NC_BETA;
    params.T_ref = NC_T_REF;
    params.gravity[0] = 0.0;
    params.gravity[1] = -NC_G;
    params.gravity[2] = 0.0;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    /* Use projection solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);

    cfd_status_t init_status = solver_init(solver, g, &params);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    /* Run simulation */
    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int step = 0; step < NC_STEPS; step++) {
        /* Apply thermal BCs before each step */
        apply_cavity_bcs(field, NC_NX, NC_NY);

        cfd_status_t status = solver_step(solver, field, g, &params, &stats);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status,
                                   "Solver step should succeed");

        /* Re-apply thermal BCs after step */
        apply_cavity_bcs(field, NC_NX, NC_NY);
    }

    /* ---- Verify results ---- */

    /* 1. Check that flow has developed (max velocity > 0) */
    double max_vel = 0.0;
    for (size_t n = 0; n < (size_t)(NC_NX * NC_NY); n++) {
        double vel = sqrt(field->u[n] * field->u[n] + field->v[n] * field->v[n]);
        if (vel > max_vel) max_vel = vel;
    }
    printf("  Max velocity: %.6f\n", max_vel);
    TEST_ASSERT_TRUE_MESSAGE(max_vel > 1e-6,
                              "Buoyancy should drive flow (max_vel > 0)");

    /* 2. Check vertical velocity direction near hot wall (should be upward = positive v)
     * and near cold wall (should be downward = negative v) */
    size_t j_mid = NC_NY / 2;
    double v_near_hot = field->v[j_mid * NC_NX + 2];   /* Near left (hot) wall */
    double v_near_cold = field->v[j_mid * NC_NX + (NC_NX - 3)]; /* Near right (cold) wall */

    printf("  v near hot wall: %.6f, v near cold wall: %.6f\n",
           v_near_hot, v_near_cold);

    TEST_ASSERT_TRUE_MESSAGE(v_near_hot > 0.0,
                              "Hot fluid should rise (positive v near hot wall)");
    TEST_ASSERT_TRUE_MESSAGE(v_near_cold < 0.0,
                              "Cold fluid should sink (negative v near cold wall)");

    /* 3. Check temperature gradient is maintained */
    double T_left = field->T[j_mid * NC_NX + 0];
    double T_right = field->T[j_mid * NC_NX + (NC_NX - 1)];
    printf("  T_left=%.2f, T_right=%.2f\n", T_left, T_right);

    TEST_ASSERT_DOUBLE_WITHIN(0.1, NC_T_HOT, T_left);
    TEST_ASSERT_DOUBLE_WITHIN(0.1, NC_T_COLD, T_right);

    /* 4. Check no NaN/Inf in any field */
    for (size_t n = 0; n < (size_t)(NC_NX * NC_NY); n++) {
        TEST_ASSERT_TRUE(isfinite(field->u[n]));
        TEST_ASSERT_TRUE(isfinite(field->v[n]));
        TEST_ASSERT_TRUE(isfinite(field->p[n]));
        TEST_ASSERT_TRUE(isfinite(field->T[n]));
    }

    /* 5. Max velocity should be physically reasonable (not exploded) */
    TEST_ASSERT_TRUE_MESSAGE(max_vel < 10.0,
                              "Max velocity should be physically reasonable");

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_natural_convection_development);
    return UNITY_END();
}
