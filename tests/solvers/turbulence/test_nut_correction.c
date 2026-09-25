/**
 * @file test_nut_correction.c
 * @brief Contract tests for the optional eddy-viscosity correction.
 *
 * The algebraic strain-rate power law (params.turb_nut_correction). Accuracy
 * is measured elsewhere -- test_turbulent_channel against DNS -- so what is
 * tested here is the contract it must honour:
 *
 * 1. Switched off is BIT-IDENTICAL to a build without the field. Anything else
 *    means the hook perturbs the base model.
 * 2. nu_t is scaled by exactly the documented power law, and the realizability
 *    clamp lands after the correction, not before and after.
 * 3. It vanishes where turbulence does.
 * 4. Every refusal is loud. An unknown value, or a correction set with anything
 *    but k-epsilon, fails with a specific status instead of silently leaving
 *    nu_t uncorrected -- at solver init, and again at the step for a caller who
 *    drives the turbulence step directly.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"

#include "unity.h"

#include <math.h>
#include <string.h>

#define NX 9
#define NY 9
#define CELLS (NX * NY)

/* Field/model constants. nu = mu/rho = 1e-3, so the realizability clamp sits at
 * TURB_NU_T_MAX_FACTOR * nu = 100 -- far above the nu_t ~ 0.09 these cases
 * produce, which keeps the clamp out of the scaling assertions. */
#define TEST_MU 1e-3
#define TEST_K0 1.0
#define TEST_EPS0 1.0
#define TEST_DT 1e-4

/* Must match TURB_ALG_BETA_A / _B in turbulence_solver_internal.h. Duplicated
 * rather than included so a coefficient change has to be made deliberately in
 * both places, and cannot pass unnoticed because the test recomputes it. */
#define ALG_A 1.6945
#define ALG_B (-0.2778)

/* Mirror of the internal header's constants: private to the library, but the
 * composition they define is the documented contract. */
#define TEST_C_MU       0.09
#define TEST_NU_T_CAP_F 1e5
#define TEST_EPS_FLOOR  1e-10

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* ------------------------------------------------------------------ helpers */

static grid* make_grid(void) {
    grid* g = grid_create(NX, NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    return g;
}

/* Linear shear, so the strain rate is nonzero: u = j/(NY-1), v = 0. */
static flow_field* make_field(void) {
    flow_field* field = flow_field_create(NX, NY, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            size_t n = j * NX + i;
            field->rho[n] = 1.0;
            field->u[n] = (double)j / (double)(NY - 1);
            field->v[n] = 0.0;
        }
    }
    return field;
}

static ns_solver_params_t make_params(turbulence_model_t model) {
    ns_solver_params_t params = ns_solver_params_default();
    params.mu = TEST_MU;
    params.turb_model = model;
    /* turb_bc zero-init = all PERIODIC */
    return params;
}

/**
 * One k-epsilon step with the given correction, leaving the state in the out
 * arrays.
 *
 * A single step is deliberate: the transport kernel consumes the PREVIOUS
 * step's nu_t, so after exactly one step the corrected and un-corrected runs
 * differ by the correction alone. Over several steps the correction feeds back
 * into k and eps and the comparison would no longer be exact.
 */
static cfd_status_t step_once_with(ns_nut_correction_t correction, double k0,
                                   double* out_nu_t, double* out_k, double* out_eps) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_nut_correction = correction;

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      turbulence_init_uniform(field, &params, k0, TEST_EPS0, 0.0));
    cfd_status_t status = turbulence_step_explicit(field, g, &params, TEST_DT, 0.0);
    if (status == CFD_SUCCESS) {
        if (out_nu_t) memcpy(out_nu_t, field->nu_t, CELLS * sizeof(double));
        if (out_k) memcpy(out_k, field->turb_k, CELLS * sizeof(double));
        if (out_eps) memcpy(out_eps, field->turb_eps, CELLS * sizeof(double));
    }

    flow_field_destroy(field);
    grid_destroy(g);
    return status;
}

/* The documented multiplier at one cell, clamped to [0.1, 10]. The field's only
 * nonzero strain is du/dy, and the central difference gives |S| = du/dy = 1
 * exactly, so S* = k/eps. */
static double expected_beta(double k, double eps) {
    double beta = ALG_A * pow(k / eps, ALG_B);
    if (beta < 0.1) beta = 0.1;
    if (beta > 10.0) beta = 10.0;
    return beta;
}

/* Take one step without a correction, then one with it, from the same start. */
static void run_pair(double k0, double* base, double* k, double* eps, double* corrected) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once_with(NS_NUT_CORRECTION_NONE, k0, base, k, eps));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_with(NS_NUT_CORRECTION_S_STAR, k0, corrected, NULL, NULL));
}

/* ============================================================================
 * 1. Switched off is bit-identical
 * ============================================================================ */

void test_default_is_off(void) {
    ns_solver_params_t params = ns_solver_params_default();
    TEST_ASSERT_EQUAL_INT(NS_NUT_CORRECTION_NONE, params.turb_nut_correction);
}

void test_off_is_repeatable_bit_for_bit(void) {
    double base[CELLS];
    double again[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_with(NS_NUT_CORRECTION_NONE, TEST_K0, base, NULL, NULL));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_with(NS_NUT_CORRECTION_NONE, TEST_K0, again, NULL, NULL));
    TEST_ASSERT_EQUAL_MEMORY(base, again, sizeof(base));
}

/* ============================================================================
 * 2. The power law is applied, and clamped after
 * ============================================================================ */

void test_applies_the_documented_power_law(void) {
    double base[CELLS], k[CELLS], eps[CELLS], corrected[CELLS];
    run_pair(TEST_K0, base, k, eps, corrected);

    /* k and eps are the same in both runs: the correction lands after the
     * transport step and does not feed back within one step. */
    for (size_t n = 0; n < CELLS; n++) {
        const double beta = expected_beta(k[n], eps[n]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-12 * base[n] * beta, base[n] * beta, corrected[n]);
    }
}

/* The realizability clamp lands AFTER the correction, never before AND after.
 *
 * turb_update_nu_t already clamps, so scaling its output would give cap*beta
 * wherever the raw k-epsilon value had been clamped -- and beta > 1 could not
 * lift a clamped cell at all. At k0 = 40 the raw value (~144) sits above the
 * cap (100) while raw*beta (~88, beta ~0.61) still sits below it, so the
 * correct answer (raw*beta) and the wrong one (cap*beta ~61) differ. */
void test_correction_scales_the_unclamped_viscosity(void) {
    double base[CELLS], k[CELLS], eps[CELLS], nu_t[CELLS];
    run_pair(40.0, base, k, eps, nu_t);

    const double cap = TEST_NU_T_CAP_F * TEST_MU; /* rho = 1, so nu = mu */
    int clamp_bound = 0;
    for (size_t n = 0; n < CELLS; n++) {
        const double raw = TEST_C_MU * k[n] * k[n] / fmax(eps[n], TEST_EPS_FLOOR);
        const double expect = fmin(raw * expected_beta(k[n], eps[n]), cap);
        TEST_ASSERT_DOUBLE_WITHIN(1e-12 * expect, expect, nu_t[n]);
        if (raw > cap) {
            clamp_bound = 1;
        }
    }
    /* Without this the case is vacuous: it would pass on the old code too. */
    TEST_ASSERT_TRUE_MESSAGE(clamp_bound,
                             "raw nu_t must exceed the cap for this to test anything");
}

/* ============================================================================
 * 3. It vanishes where turbulence does
 * ============================================================================ */

void test_vanishes_where_turbulence_does(void) {
    /* The ship gate requires the correction to vanish in laminar regions. It
     * does so structurally rather than by fitting: the correction is
     * multiplicative and bounded by TURB_CLOSURE_BETA_MAX, and nu_t = C_mu
     * k^2/eps is already negligible where k is, so the most it can do is scale
     * a vanishing viscosity by ten. Started from k = 0; the transport leaves a
     * rounding-level residue rather than an exact zero, which is why this
     * bounds the ratio instead of asserting equality. */
    double base[CELLS], corrected[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_with(NS_NUT_CORRECTION_NONE, 0.0, base, NULL, NULL));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_with(NS_NUT_CORRECTION_S_STAR, 0.0, corrected, NULL, NULL));
    for (size_t n = 0; n < CELLS; n++) {
        TEST_ASSERT_TRUE_MESSAGE(corrected[n] <= 10.0 * base[n] + 1e-300,
                                 "correction exceeded the clamp in a laminar region");
        /* And negligible against the molecular viscosity it sits beside. */
        TEST_ASSERT_TRUE_MESSAGE(corrected[n] < 1e-8 * TEST_MU,
                                 "nu_t is not negligible in a laminar region");
    }
}

/* ============================================================================
 * 4. Failures are loud: at the step
 * ============================================================================ */

static cfd_status_t step_with_model(turbulence_model_t model, ns_nut_correction_t correction) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(model);
    params.turb_nut_correction = correction;

    if (model == TURB_MODEL_SPALART_ALLMARAS) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, 1e-3));
    } else if (model == TURB_MODEL_K_EPSILON) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS,
                          turbulence_init_uniform(field, &params, TEST_K0, TEST_EPS0, 0.0));
    }
    cfd_status_t status = turbulence_step_explicit(field, g, &params, TEST_DT, 0.0);

    flow_field_destroy(field);
    grid_destroy(g);
    return status;
}

void test_step_refuses_spalart_allmaras(void) {
    /* SA carries no k/epsilon, so S* does not exist. Refused, not ignored -- a
     * caller who set a correction asked for it to run. */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      step_with_model(TURB_MODEL_SPALART_ALLMARAS, NS_NUT_CORRECTION_S_STAR));
}

void test_step_refuses_correction_without_a_model(void) {
    /* The step's TURB_MODEL_NONE exit used to run first, which turned this into
     * a silent no-op that reported success. */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      step_with_model(TURB_MODEL_NONE, NS_NUT_CORRECTION_S_STAR));
}

void test_step_refuses_unknown_value(void) {
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
                      step_with_model(TURB_MODEL_K_EPSILON, (ns_nut_correction_t)99));
}

/* ============================================================================
 * 5. The refusal happens at init, while the caller can still choose
 * ============================================================================ */

static cfd_status_t init_solver_with(turbulence_model_t model, ns_nut_correction_t correction) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);
    grid* g = make_grid();
    ns_solver_params_t params = make_params(model);
    params.turb_nut_correction = correction;

    cfd_status_t status = solver_init(solver, g, &params);

    grid_destroy(g);
    solver_destroy(solver);
    cfd_registry_destroy(registry);
    return status;
}

void test_solver_init_refuses_correction_without_kepsilon(void) {
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      init_solver_with(TURB_MODEL_SPALART_ALLMARAS, NS_NUT_CORRECTION_S_STAR));
    /* The configuration that used to be the quietest of all: a correction with
     * no turbulence model at all, which simply never ran. */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      init_solver_with(TURB_MODEL_NONE, NS_NUT_CORRECTION_S_STAR));
}

void test_solver_init_refuses_unknown_value(void) {
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
                      init_solver_with(TURB_MODEL_K_EPSILON, (ns_nut_correction_t)99));
}

void test_solver_init_accepts_correction_with_kepsilon(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      init_solver_with(TURB_MODEL_K_EPSILON, NS_NUT_CORRECTION_S_STAR));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_default_is_off);
    RUN_TEST(test_off_is_repeatable_bit_for_bit);
    RUN_TEST(test_applies_the_documented_power_law);
    RUN_TEST(test_correction_scales_the_unclamped_viscosity);
    RUN_TEST(test_vanishes_where_turbulence_does);
    RUN_TEST(test_step_refuses_spalart_allmaras);
    RUN_TEST(test_step_refuses_correction_without_a_model);
    RUN_TEST(test_step_refuses_unknown_value);
    RUN_TEST(test_solver_init_refuses_correction_without_kepsilon);
    RUN_TEST(test_solver_init_refuses_unknown_value);
    RUN_TEST(test_solver_init_accepts_correction_with_kepsilon);
    return UNITY_END();
}
