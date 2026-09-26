/**
 * @file test_turbulence_wall_functions.c
 * @brief Unit tests for the Spalding's-law wall functions
 *
 * Tests:
 * 1. turbulence_wall_u_tau recovers a known friction velocity from a
 *    Spalding-consistent (u_p, y_p) pair in the sublayer, buffer and log layer.
 * 2. Limits: the linear law deep in the sublayer, the log law far out in the
 *    log layer; degenerate inputs return 0.
 * 3. Continuity across the buffer layer: u_tau rises smoothly through
 *    y+ = 11.63, where the former linear/log switch made it jump 3.3%.
 * 4. Full BC application on a small grid: equilibrium k/eps at the first
 *    interior node and a wall-face viscosity that reproduces the wall-law
 *    shear exactly: (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p = u_tau^2.
 * ============================================================================ */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "unity.h"

#include <math.h>

/* Constants (must match turbulence_solver_internal.h) */
#define WF_KAPPA 0.41
#define WF_B     5.2
#define WF_C_MU  0.09

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* Spalding's law y+(u+), written out independently of the library */
static double spalding_yplus(double uplus) {
    double ku = WF_KAPPA * uplus;
    return uplus + exp(-WF_KAPPA * WF_B) *
                       (exp(ku) - 1.0 - ku - ku * ku / 2.0 - ku * ku * ku / 6.0);
}

/* ============================================================================
 * TEST 1: u_tau recovery across the sublayer, buffer and log layer
 * ============================================================================ */

static void check_u_tau_recovery(double ut_exact, double uplus) {
    const double nu = 1.5e-5;
    double y_p = spalding_yplus(uplus) * nu / ut_exact;
    double u_p = ut_exact * uplus;

    double ut = turbulence_wall_u_tau(u_p, y_p, nu);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10 * ut_exact, ut_exact, ut);
}

static void test_u_tau_recovery(void) {
    const double uplus[] = {3.0, 8.0, 11.63, 14.0, 17.0, 25.0}; /* y+ ~ 3 .. 1900 */
    for (size_t n = 0; n < sizeof(uplus) / sizeof(uplus[0]); n++) {
        check_u_tau_recovery(0.05, uplus[n]);
        check_u_tau_recovery(1.0, uplus[n]);
    }
}

/* ============================================================================
 * TEST 2: linear-law and log-law limits, degenerate inputs
 * ============================================================================ */

static void test_u_tau_limits(void) {
    const double nu = 1.5e-5;
    const double ut_exact = 0.2;

    /* Deep sublayer (y+ = 0.5): the linear law u+ = y+, to O((kappa u+)^4) */
    double yplus = 0.5;
    double y_p = yplus * nu / ut_exact;
    double u_p = ut_exact * yplus;
    TEST_ASSERT_DOUBLE_WITHIN(1e-4 * ut_exact, ut_exact,
                              turbulence_wall_u_tau(u_p, y_p, nu));

    /* Far out in the log layer (y+ = 1000): the log law. Spalding approaches
     * it from above; the gap is ~1e-4 here, ~3% at y+ = 40. */
    yplus = 1000.0;
    y_p = yplus * nu / ut_exact;
    u_p = ut_exact * (log(yplus) / WF_KAPPA + WF_B);
    TEST_ASSERT_DOUBLE_WITHIN(1e-3 * ut_exact, ut_exact,
                              turbulence_wall_u_tau(u_p, y_p, nu));

    /* Degenerate inputs return 0 */
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(0.0, y_p, nu));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(u_p, 0.0, nu));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(u_p, y_p, 0.0));
}

/* ============================================================================
 * TEST 3: continuity through the buffer layer
 *
 * Sweep Re_p = u_p*y_p/nu across 11.63^2 (the former linear/log switch) in
 * relative steps d. u_tau = u_p/u+ grows like Re_p^s with 1/2 <= s <= 1, so each
 * step must raise it by at most ~d. The piecewise law jumped 3.3% at the switch.
 * ============================================================================ */

static void test_u_tau_continuous_through_buffer_layer(void) {
    const double nu = 1e-5;
    const double y_p = 1e-3;
    const double d = 1e-3;

    double prev = 0.0;
    double max_step = 0.0;
    for (double re = 100.0; re < 200.0; re *= 1.0 + d) {
        double ut = turbulence_wall_u_tau(re * nu / y_p, y_p, nu);
        if (prev > 0.0) {
            TEST_ASSERT_TRUE_MESSAGE(ut > prev, "u_tau must increase with u_p");
            max_step = fmax(max_step, ut / prev - 1.0);
        }
        prev = ut;
    }
    TEST_ASSERT_TRUE_MESSAGE(max_step < 2.0 * d, "u_tau jumps in the buffer layer");
}

/* ============================================================================
 * TEST 4: full wall-function BC application (bottom wall, k-epsilon)
 * ============================================================================ */

static void test_wall_function_bc_application(void) {
    const size_t nx = 8, ny = 8;
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);

    const double u_p = 2.0;
    const double mu = 1e-4; /* with rho=1: nu=1e-4, first node lands in log layer */
    for (size_t n = 0; n < nx * ny; n++) {
        field->rho[n] = 1.0;
        field->u[n] = u_p;
        field->v[n] = 0.0;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = mu;
    params.turb_model = TURB_MODEL_K_EPSILON;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.left = BC_TYPE_NEUMANN;
    params.turb_bc.right = BC_TYPE_NEUMANN;
    params.turb_bc.top = BC_TYPE_NEUMANN;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    const double y_p = g->y[1] - g->y[0];
    const double nu = mu; /* rho = 1 */
    const double ut = turbulence_wall_u_tau(u_p, y_p, nu);
    TEST_ASSERT_TRUE(ut > 0.0);
    /* Sanity: this configuration puts the first node in the log layer */
    TEST_ASSERT_TRUE(ut * y_p / nu > 30.0);

    size_t i = nx / 2;
    size_t idx_w = i;        /* wall node (j=0) */
    size_t idx_p = i + nx;   /* first interior node (j=1) */

    /* Equilibrium values at the first interior node */
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut / sqrt(WF_C_MU), field->turb_k[idx_p]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut * ut / (WF_KAPPA * y_p),
                              field->turb_eps[idx_p]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, field->turb_k[idx_w]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, field->turb_eps[idx_p], field->turb_eps[idx_w]);

    /* Discrete wall shear must reproduce the wall law exactly:
     * (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p == u_tau^2 */
    double nu_face = nu + 0.5 * (field->nu_t[idx_w] + field->nu_t[idx_p]);
    double shear = nu_face * u_p / y_p;
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut, shear);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 5: sublayer wall function degenerates to laminar shear
 * ============================================================================ */

static void test_wall_function_sublayer_laminar(void) {
    const size_t nx = 8, ny = 8;
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Large viscosity puts the first node deep in the viscous sublayer */
    const double u_p = 0.01;
    const double mu = 0.1;
    for (size_t n = 0; n < nx * ny; n++) {
        field->rho[n] = 1.0;
        field->u[n] = u_p;
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = mu;
    params.turb_model = TURB_MODEL_K_EPSILON;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.left = BC_TYPE_NEUMANN;
    params.turb_bc.right = BC_TYPE_NEUMANN;
    params.turb_bc.top = BC_TYPE_NEUMANN;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    /* In the sublayer u_tau^2 -> nu*u_p/y_p, so the wall-face eddy viscosity
     * falls to laminar: 2*nu*(y+/u+ - 1) ~ (kappa u+)^4, here u+ ~ 0.12 and
     * nu_t_p ~ 4.8e-7 nu. */
    size_t i = nx / 2;
    TEST_ASSERT_EQUAL_DOUBLE(0.0, field->nu_t[i]);                /* wall node */
    TEST_ASSERT_TRUE(field->nu_t[i + nx] >= 0.0);                 /* first interior */
    TEST_ASSERT_TRUE(field->nu_t[i + nx] < 1e-6 * mu);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_u_tau_recovery);
    RUN_TEST(test_u_tau_limits);
    RUN_TEST(test_u_tau_continuous_through_buffer_layer);
    RUN_TEST(test_wall_function_bc_application);
    RUN_TEST(test_wall_function_sublayer_laminar);
    return UNITY_END();
}
