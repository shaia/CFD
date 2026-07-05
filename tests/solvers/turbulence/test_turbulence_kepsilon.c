/**
 * @file test_turbulence_kepsilon.c
 * @brief Unit tests for the standard k-epsilon turbulence model
 *
 * Tests:
 * 1. Decaying homogeneous turbulence: with zero velocity and uniform k/eps,
 *    the model reduces to dk/dt = -eps, deps/dt = -C2*eps^2/k with the
 *    analytical solution k(t) = k0 * (1 + (C2-1)*eps0*t/k0)^(-1/(C2-1)).
 * 2. First-order temporal convergence of the decay solution.
 * 3. Positivity under an abusive time step (Patankar sink treatment).
 * 4. TURB_MODEL_NONE is a strict no-op.
 * 5. Production term under prescribed linear shear matches P_k = nu_t*S^2.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "unity.h"

#include <math.h>
#include <string.h>

/* Model constants (must match turbulence_solver_internal.h) */
#define KE_C_MU 0.09
#define KE_C1   1.44
#define KE_C2   1.92

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* ============================================================================
 * Helpers
 * ============================================================================ */

static grid* make_grid(size_t nx, size_t ny) {
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
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

/* Run the homogeneous decay case and return k at the domain center. */
static double run_decay(double dt, int steps) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-5;
    params.turb_model = TURB_MODEL_K_EPSILON;
    /* turb_bc zero-init = all PERIODIC */

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 1.0, 1.0, 0.0));

    for (int s = 0; s < steps; s++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS,
                          turbulence_step_explicit(field, g, &params, dt, s * dt));
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));
    }

    double k_center = field->turb_k[(ny / 2) * nx + nx / 2];

    flow_field_destroy(field);
    grid_destroy(g);
    return k_center;
}

static double decay_exact(double k0, double eps0, double t) {
    return k0 * pow(1.0 + (KE_C2 - 1.0) * eps0 * t / k0, -1.0 / (KE_C2 - 1.0));
}

/* ============================================================================
 * TEST 1: analytic decay of homogeneous turbulence
 * ============================================================================ */

static void test_homogeneous_decay(void) {
    const double dt = 1e-3;
    const int steps = 500;
    double k_num = run_decay(dt, steps);
    double k_ref = decay_exact(1.0, 1.0, steps * dt);

    TEST_ASSERT_TRUE(k_num > 0.0);
    double rel_err = fabs(k_num - k_ref) / k_ref;
    TEST_ASSERT_TRUE_MESSAGE(rel_err < 0.02, "decay error exceeds 2%");
}

/* ============================================================================
 * TEST 2: first-order temporal convergence
 * ============================================================================ */

static void test_decay_convergence(void) {
    const double t_final = 0.4;
    double e_coarse = fabs(run_decay(2e-3, 200) - decay_exact(1.0, 1.0, t_final));
    double e_fine = fabs(run_decay(1e-3, 400) - decay_exact(1.0, 1.0, t_final));

    TEST_ASSERT_TRUE(e_fine > 0.0);
    double ratio = e_coarse / e_fine;
    /* First-order scheme: halving dt should roughly halve the error */
    TEST_ASSERT_TRUE_MESSAGE(ratio > 1.5 && ratio < 3.0,
                             "decay error does not converge at first order");
}

/* ============================================================================
 * TEST 3: positivity under an abusive time step
 * ============================================================================ */

static void test_positivity_abusive_dt(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-5;
    params.turb_model = TURB_MODEL_K_EPSILON;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 1.0, 1.0, 0.0));

    /* dt orders of magnitude above the source-term stability estimate */
    const double dt = 10.0;
    for (int s = 0; s < 100; s++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS,
                          turbulence_step_explicit(field, g, &params, dt, s * dt));
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));
    }

    for (size_t n = 0; n < nx * ny; n++) {
        TEST_ASSERT_TRUE(isfinite(field->turb_k[n]));
        TEST_ASSERT_TRUE(isfinite(field->turb_eps[n]));
        TEST_ASSERT_TRUE(field->turb_k[n] > 0.0);
        TEST_ASSERT_TRUE(field->turb_eps[n] > 0.0);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4: TURB_MODEL_NONE is a strict no-op
 * ============================================================================ */

static void test_disabled_noop(void) {
    const size_t nx = 9, ny = 9;
    grid* g = make_grid(nx, ny);
    flow_field* field = make_field(nx, ny);

    ns_solver_params_t params = ns_solver_params_default();
    TEST_ASSERT_EQUAL(TURB_MODEL_NONE, params.turb_model);

    for (size_t n = 0; n < nx * ny; n++) {
        field->turb_k[n] = 5.0;
        field->turb_eps[n] = 7.0;
        field->nu_t[n] = 3.0;
    }

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_step_explicit(field, g, &params, 0.1, 0.0));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    for (size_t n = 0; n < nx * ny; n++) {
        TEST_ASSERT_EQUAL_DOUBLE(5.0, field->turb_k[n]);
        TEST_ASSERT_EQUAL_DOUBLE(7.0, field->turb_eps[n]);
        TEST_ASSERT_EQUAL_DOUBLE(3.0, field->nu_t[n]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 5: production under prescribed linear shear u = S*y
 *
 * With uniform k/eps (no advection or diffusion of k) and central velocity
 * gradients, S^2 = (du/dy)^2 and one Patankar step gives exactly
 *   k_new = (k + dt*P_k) / (1 + dt*eps/k),  P_k = min(nu_t*S^2, 10*eps)
 * ============================================================================ */

static void test_production_linear_shear(void) {
    const size_t nx = 17, ny = 17;
    const double S = 2.0;
    const double dt = 1e-3;
    grid* g = make_grid(nx, ny);
    flow_field* field = make_field(nx, ny);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = S * g->y[j];
            field->v[idx] = 0.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-5;
    params.turb_model = TURB_MODEL_K_EPSILON;

    const double k0 = 1.0, eps0 = 1.0;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, k0, eps0, 0.0));
    double nu_t0 = field->nu_t[0];
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, KE_C_MU * k0 * k0 / eps0, nu_t0);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_step_explicit(field, g, &params, dt, 0.0));

    double P_k = nu_t0 * S * S;
    if (P_k > 10.0 * eps0) P_k = 10.0 * eps0;
    double k_expected = (k0 + dt * P_k) / (1.0 + dt * eps0 / k0);
    double e_expected = (eps0 + dt * KE_C1 * (eps0 / k0) * P_k) /
                        (1.0 + dt * KE_C2 * eps0 / k0);

    size_t center = (ny / 2) * nx + nx / 2;
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, k_expected, field->turb_k[center]);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, e_expected, field->turb_eps[center]);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_homogeneous_decay);
    RUN_TEST(test_decay_convergence);
    RUN_TEST(test_positivity_abusive_dt);
    RUN_TEST(test_disabled_noop);
    RUN_TEST(test_production_linear_shear);
    return UNITY_END();
}
