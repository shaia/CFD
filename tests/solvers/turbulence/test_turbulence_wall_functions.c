/**
 * @file test_turbulence_wall_functions.c
 * @brief Unit tests for the log-law wall functions
 *
 * Tests:
 * 1. turbulence_wall_u_tau recovers a known friction velocity from a
 *    log-law-consistent (u_p, y_p) pair at y+ = 30 and y+ = 100.
 * 2. Viscous sublayer branch (y+ = 5): u_tau = sqrt(nu*u_p/y_p).
 * 3. Full BC application on a small grid: equilibrium k/eps at the first
 *    interior node and a wall-face viscosity that reproduces the log-law
 *    wall shear exactly: (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p = u_tau^2.
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

/* ============================================================================
 * TEST 1: u_tau recovery in the log layer
 * ============================================================================ */

static void check_u_tau_recovery(double ut_exact, double yplus) {
    const double nu = 1.5e-5;
    double y_p = yplus * nu / ut_exact;
    double u_p = ut_exact * (log(yplus) / WF_KAPPA + WF_B);

    double ut = turbulence_wall_u_tau(u_p, y_p, nu);
    TEST_ASSERT_DOUBLE_WITHIN(1e-8 * ut_exact, ut_exact, ut);
}

static void test_u_tau_recovery_log_layer(void) {
    check_u_tau_recovery(0.05, 30.0);
    check_u_tau_recovery(0.05, 100.0);
    check_u_tau_recovery(1.0, 30.0);
    check_u_tau_recovery(1.0, 100.0);
}

/* ============================================================================
 * TEST 2: viscous sublayer branch
 * ============================================================================ */

static void test_u_tau_sublayer(void) {
    const double nu = 1.5e-5;
    const double ut_exact = 0.2;
    const double yplus = 5.0;
    double y_p = yplus * nu / ut_exact;
    double u_p = ut_exact * yplus; /* linear law u+ = y+ */

    double ut = turbulence_wall_u_tau(u_p, y_p, nu);
    /* sqrt(nu*u_p/y_p) = sqrt(ut^2) = ut exactly */
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, ut_exact, ut);

    /* Degenerate inputs return 0 */
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(0.0, y_p, nu));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(u_p, 0.0, nu));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, turbulence_wall_u_tau(u_p, y_p, 0.0));
}

/* ============================================================================
 * TEST 3: full wall-function BC application (bottom wall, k-epsilon)
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
    /* Sanity: this configuration must exercise the log-law branch */
    TEST_ASSERT_TRUE(ut * y_p / nu > 11.63);

    size_t i = nx / 2;
    size_t idx_w = i;        /* wall node (j=0) */
    size_t idx_p = i + nx;   /* first interior node (j=1) */

    /* Equilibrium values at the first interior node */
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut / sqrt(WF_C_MU), field->turb_k[idx_p]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut * ut / (WF_KAPPA * y_p),
                              field->turb_eps[idx_p]);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, field->turb_k[idx_w]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, field->turb_eps[idx_p], field->turb_eps[idx_w]);

    /* Discrete wall shear must reproduce the log law exactly:
     * (nu + 0.5*(nu_t_w + nu_t_p)) * u_p/y_p == u_tau^2 */
    double nu_face = nu + 0.5 * (field->nu_t[idx_w] + field->nu_t[idx_p]);
    double shear = nu_face * u_p / y_p;
    TEST_ASSERT_DOUBLE_WITHIN(1e-10, ut * ut, shear);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4: sublayer wall function degenerates to laminar shear
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

    /* In the sublayer u_tau^2 = nu*u_p/y_p, so the wall-face eddy viscosity
     * must vanish (pure laminar shear) */
    size_t i = nx / 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.0, field->nu_t[i]);        /* wall node */
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.0, field->nu_t[i + nx]);   /* first interior */

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_u_tau_recovery_log_layer);
    RUN_TEST(test_u_tau_sublayer);
    RUN_TEST(test_wall_function_bc_application);
    RUN_TEST(test_wall_function_sublayer_laminar);
    return UNITY_END();
}
