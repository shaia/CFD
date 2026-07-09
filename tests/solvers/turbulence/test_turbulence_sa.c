/**
 * @file test_turbulence_sa.c
 * @brief Unit tests for the Spalart-Allmaras turbulence model (no-ft2 variant)
 *
 * Tests:
 * 1. turbulence_init_uniform sets nu_t = nu_tilde * fv1(chi) exactly.
 * 2. No-wall invariance: with zero velocity, uniform nu_tilde, and no
 *    wall-function faces, nu_tilde is constant to machine precision.
 * 3. Production under prescribed linear shear (no wall): one step gives
 *    exactly nt_new = nt * (1 + dt*cb1*Omega).
 * 4. Positivity under an abusive time step with wall destruction active.
 * 5. Wall-distance helper: nearest NOSLIP face distance, has_wall flag.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "unity.h"

#include <math.h>

/* SA constants (must match turbulence_solver_internal.h) */
#define SA_T_CB1 0.1355
#define SA_T_CV1 7.1

/* Internal helper under test: exported from the static library but not part
 * of the public API (declared in turbulence_solver_internal.h). */
extern double turb_wall_distance(const grid* grid, const ns_turbulence_bc_config_t* tbc,
                                 size_t i, size_t j, int* has_wall);

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

static grid* make_grid(size_t nx, size_t ny, double lx, double ly) {
    grid* g = grid_create(nx, ny, 1, 0.0, lx, 0.0, ly, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    return g;
}

static flow_field* make_field(size_t nx, size_t ny) {
    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t n = 0; n < nx * ny; n++) {
        field->rho[n] = 1.0;
    }
    return field;
}

/* ============================================================================
 * TEST 1: nu_t = nu_tilde * fv1(chi)
 * ============================================================================ */

static void test_init_uniform_fv1(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny, 1.0, 1.0);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-3;
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;

    const double nt0 = 3e-3; /* chi = 3 */
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, nt0));

    double chi = nt0 / params.mu; /* rho = 1 */
    double chi3 = chi * chi * chi;
    double fv1 = chi3 / (chi3 + SA_T_CV1 * SA_T_CV1 * SA_T_CV1);
    double nu_t_expected = nt0 * fv1;

    TEST_ASSERT_DOUBLE_WITHIN(1e-15, nu_t_expected, field->nu_t[0]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, nu_t_expected, field->nu_t[nx * ny - 1]);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 2: no-wall invariance
 * ============================================================================ */

static void test_no_wall_invariance(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny, 1.0, 1.0);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-3;
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;
    /* all faces PERIODIC (zero-init): no wall anywhere */

    const double nt0 = 5e-3;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, nt0));

    for (int s = 0; s < 100; s++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS,
                          turbulence_step_explicit(field, g, &params, 1e-2, s * 1e-2));
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));
    }

    for (size_t n = 0; n < nx * ny; n++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, nt0, field->turb_nu_tilde[n]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 3: production under linear shear u = S*y (no wall)
 *
 * Omega = |dv/dx - du/dy| = S, uniform nu_tilde: advection, diffusion, and
 * the cb2 gradient term vanish; no wall means no destruction and S_tilde =
 * Omega. One step is exactly nt_new = nt + dt*cb1*S*nt.
 * ============================================================================ */

static void test_production_linear_shear(void) {
    const size_t nx = 17, ny = 17;
    const double S = 2.0;
    const double dt = 1e-3;
    grid* g = make_grid(nx, ny, 1.0, 1.0);
    flow_field* field = make_field(nx, ny);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = S * g->y[j];
            field->v[idx] = 0.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-3;
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;

    const double nt0 = 3e-3;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, nt0));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_step_explicit(field, g, &params, dt, 0.0));

    double nt_expected = nt0 * (1.0 + dt * SA_T_CB1 * S);
    size_t center = (ny / 2) * nx + nx / 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, nt_expected, field->turb_nu_tilde[center]);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4a: Patankar destruction cannot cross zero, even with a huge dt
 *
 * Zero velocity, uniform nu_tilde, wall faces configured (destruction active
 * via turb_bc without applying BCs, so no gradients ever form and only the
 * semi-implicit sink acts). A single step with dt = 1000 must leave nu_tilde
 * positive and finite everywhere.
 * ============================================================================ */

static void test_positivity_huge_dt_sink(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny, 1.0, 1.0);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-4;
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.top = BC_TYPE_NOSLIP;

    const double nt0 = 1e-2;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, nt0));

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      turbulence_step_explicit(field, g, &params, 1000.0, 0.0));

    for (size_t n = 0; n < nx * ny; n++) {
        TEST_ASSERT_TRUE(isfinite(field->turb_nu_tilde[n]));
        TEST_ASSERT_TRUE(field->turb_nu_tilde[n] >= 0.0);
        TEST_ASSERT_TRUE(field->turb_nu_tilde[n] <= nt0);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4b: multi-step robustness with shear, wall functions, and a time step
 * at the explicit diffusion stability margin
 * ============================================================================ */

static void test_positivity_robustness(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny, 1.0, 1.0);
    flow_field* field = make_field(nx, ny);

    /* Shear flow to activate production alongside wall destruction */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            field->u[j * nx + i] = 5.0 * g->y[j];
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-4;
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.top = BC_TYPE_NOSLIP;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, 1e-2));

    const double dt = 0.04;
    for (int s = 0; s < 500; s++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS,
                          turbulence_step_explicit(field, g, &params, dt, s * dt));
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));
    }

    for (size_t n = 0; n < nx * ny; n++) {
        TEST_ASSERT_TRUE(isfinite(field->turb_nu_tilde[n]));
        TEST_ASSERT_TRUE(field->turb_nu_tilde[n] >= 0.0);
        TEST_ASSERT_TRUE(isfinite(field->nu_t[n]));
        TEST_ASSERT_TRUE(field->nu_t[n] >= 0.0);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 5: wall-distance helper
 * ============================================================================ */

static void test_wall_distance(void) {
    const size_t nx = 11, ny = 11;
    grid* g = make_grid(nx, ny, 4.0, 2.0);

    ns_turbulence_bc_config_t tbc = {0};
    int has_wall = 0;

    /* No wall anywhere */
    double d = turb_wall_distance(g, &tbc, 5, 5, &has_wall);
    TEST_ASSERT_EQUAL(0, has_wall);
    TEST_ASSERT_EQUAL_DOUBLE(0.0, d);

    /* Channel: bottom and top walls */
    tbc.bottom = BC_TYPE_NOSLIP;
    tbc.top = BC_TYPE_NOSLIP;
    d = turb_wall_distance(g, &tbc, 5, 2, &has_wall);
    TEST_ASSERT_EQUAL(1, has_wall);
    TEST_ASSERT_DOUBLE_WITHIN(1e-14, g->y[2] - g->y[0], d);

    d = turb_wall_distance(g, &tbc, 5, 8, &has_wall);
    TEST_ASSERT_EQUAL(1, has_wall);
    TEST_ASSERT_DOUBLE_WITHIN(1e-14, g->y[ny - 1] - g->y[8], d);

    /* Add a left wall closer than the y-walls */
    tbc.left = BC_TYPE_NOSLIP;
    d = turb_wall_distance(g, &tbc, 1, 5, &has_wall);
    TEST_ASSERT_EQUAL(1, has_wall);
    TEST_ASSERT_DOUBLE_WITHIN(1e-14, g->x[1] - g->x[0], d);

    grid_destroy(g);
}

/* ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_init_uniform_fv1);
    RUN_TEST(test_no_wall_invariance);
    RUN_TEST(test_production_linear_shear);
    RUN_TEST(test_positivity_huge_dt_sink);
    RUN_TEST(test_positivity_robustness);
    RUN_TEST(test_wall_distance);
    return UNITY_END();
}
