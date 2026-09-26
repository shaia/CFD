/**
 * @file test_nut_correction.c
 * @brief Contract tests for the optional eddy-viscosity corrections.
 *
 * Two alternatives share one seam: the algebraic strain-rate power law
 * (params.turb_nut_correction) and the learned closure (params.turb_closure).
 * Accuracy is measured elsewhere -- test_turbulent_channel against DNS -- so
 * what is tested here is the contract both must honour:
 *
 * 1. Switched off is BIT-IDENTICAL to a build without the field, and a learned
 *    beta = 1 is bit-identical to no closure. Anything else means the hook
 *    perturbs the base model.
 * 2. nu_t is scaled by exactly the documented power law or the predicted
 *    factor, clamped in BOTH directions -- the correction the channel DNS asks
 *    for is below 1, so this is not a dissipation-only path -- and the
 *    realizability clamp lands after the correction, not before and after.
 * 3. It vanishes where turbulence does.
 * 4. Every refusal is loud. An unknown value, a non-finite prediction, a model
 *    of the wrong shape, both corrections at once, or a correction set with
 *    anything but k-epsilon, fails with a specific status instead of silently
 *    leaving nu_t uncorrected -- at solver init, and again at the step for a
 *    caller who drives the turbulence step directly.
 * 5. Tiling is invisible: a context with capacity 1 gives bit-identical results
 *    to one sized for the whole grid.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/nn/cfdnn.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"

#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TMP_MODEL "test_learned_closure_tmp.cfdnn"

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
void tearDown(void) {
    cfd_finalize();
    remove(TMP_MODEL);
}

/* ------------------------------------------------------------------ helpers */

/* A 3 -> 1 dense layer with zero weights and bias = value, identity
 * activation: every cell predicts exactly `value`, whatever the features are.
 * That is what makes the scaling assertions exact rather than approximate. */
static cfd_status_t write_constant_model(float value) {
    static float w[3];
    static float b[1];
    memset(w, 0, sizeof(w));
    b[0] = value;

    cfd_nn_layer_desc_t l = {CFD_NN_LAYER_DENSE, CFD_NN_ACT_IDENTITY, 0.0f, 3, 1, w, b};
    cfd_nn_model_desc_t desc = {"constant-beta", &l, 1};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

/* Two layers whose product overflows binary32: 3e38 * 10 = inf. */
static cfd_status_t write_overflowing_model(void) {
    static float w1[3];
    static float b1[1];
    static float w2[1];
    memset(w1, 0, sizeof(w1));
    b1[0] = 3.0e38f;
    w2[0] = 10.0f;

    cfd_nn_layer_desc_t l[2];
    l[0] = (cfd_nn_layer_desc_t){CFD_NN_LAYER_DENSE, CFD_NN_ACT_IDENTITY, 0.0f, 3, 1, w1, b1};
    l[1] = (cfd_nn_layer_desc_t){CFD_NN_LAYER_DENSE, CFD_NN_ACT_IDENTITY, 0.0f, 1, 1, w2, NULL};
    cfd_nn_model_desc_t desc = {"overflow", l, 2};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

/* A 2 -> 1 model: the right idea, the wrong feature count. */
static cfd_status_t write_wrong_shape_model(void) {
    static float w[2];
    static float b[1];
    memset(w, 0, sizeof(w));
    b[0] = 1.0f;

    cfd_nn_layer_desc_t l = {CFD_NN_LAYER_DENSE, CFD_NN_ACT_IDENTITY, 0.0f, 2, 1, w, b};
    cfd_nn_model_desc_t desc = {"wrong-shape", &l, 1};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

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
 * One k-epsilon step with the given corrections, leaving the state in the out
 * arrays.
 *
 * A single step is deliberate: the transport kernel consumes the PREVIOUS
 * step's nu_t, so after exactly one step the corrected and un-corrected runs
 * differ by the correction alone. Over several steps the correction feeds back
 * into k and eps and the comparison would no longer be exact.
 */
static cfd_status_t step_once_full(cfd_nn_context_t* closure, ns_nut_correction_t correction,
                                   double k0, double* out_nu_t, double* out_k,
                                   double* out_eps) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_closure = closure;
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

static cfd_status_t step_once_with(ns_nut_correction_t correction, double k0,
                                   double* out_nu_t, double* out_k, double* out_eps) {
    return step_once_full(NULL, correction, k0, out_nu_t, out_k, out_eps);
}

static cfd_status_t step_once(cfd_nn_context_t* closure, double* out_nu_t) {
    return step_once_full(closure, NS_NUT_CORRECTION_NONE, TEST_K0, out_nu_t, NULL, NULL);
}

/* Load TMP_MODEL and create a context of the given capacity. */
static void open_closure(size_t capacity, cfd_nn_model_t** out_model,
                         cfd_nn_context_t** out_ctx) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, out_model));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_nn_context_create(*out_model, capacity,
                                            CFD_NN_BACKEND_SCALAR, out_ctx));
}

static void close_closure(cfd_nn_model_t* model, cfd_nn_context_t* ctx) {
    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(model);
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

static cfd_status_t init_solver_with_params(ns_solver_params_t* params) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(solver);
    grid* g = make_grid();

    cfd_status_t status = solver_init(solver, g, params);

    grid_destroy(g);
    solver_destroy(solver);
    cfd_registry_destroy(registry);
    return status;
}

static cfd_status_t init_solver_with(turbulence_model_t model, ns_nut_correction_t correction) {
    ns_solver_params_t params = make_params(model);
    params.turb_nut_correction = correction;
    return init_solver_with_params(&params);
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

/* ============================================================================
 * 6. The learned closure on the same seam
 * ============================================================================ */

void test_beta_one_is_bit_identical_to_no_closure(void) {
    double base[CELLS];
    double corrected[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(NULL, base));

    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(1.0f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(ctx, corrected));
    /* Memory equality, not a tolerance: multiplying by exactly 1.0 must not
     * perturb a single bit, or "correction off" is not a safe default. */
    TEST_ASSERT_EQUAL_MEMORY(base, corrected, sizeof(base));

    close_closure(model, ctx);
}

static void assert_scaled_by(float predicted, double expected_factor) {
    double base[CELLS];
    double corrected[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(NULL, base));

    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(predicted));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(ctx, corrected));
    for (size_t n = 0; n < CELLS; n++) {
        TEST_ASSERT_TRUE_MESSAGE(base[n] > 0.0, "base nu_t must be positive");
        TEST_ASSERT_DOUBLE_WITHIN(1e-12 * base[n] * expected_factor,
                                  base[n] * expected_factor, corrected[n]);
    }
    close_closure(model, ctx);
}

void test_predicted_factor_scales_nu_t(void) {
    assert_scaled_by(2.0f, 2.0);
}

void test_factor_above_the_ceiling_is_clamped(void) {
    /* 100 -> TURB_CLOSURE_BETA_MAX. The clamp is what bounds a wrong model. */
    assert_scaled_by(100.0f, 10.0);
}

void test_factor_below_the_floor_is_clamped(void) {
    /* 1e-4 -> TURB_CLOSURE_BETA_MIN. A reduction is legitimate (the DNS asks
     * for one in the outer layer), so the floor -- not zero -- is the bound. */
    assert_scaled_by(1.0e-4f, 0.1);
}

/* The learned path has its own apply loop, so the clamp ordering is checked on
 * it too: see test_correction_scales_the_unclamped_viscosity for the case. */
void test_closure_scales_the_unclamped_viscosity(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(0.5f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    double nu_t[CELLS], k[CELLS], eps[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      step_once_full(ctx, NS_NUT_CORRECTION_NONE, 40.0, nu_t, k, eps));

    const double cap = TEST_NU_T_CAP_F * TEST_MU; /* rho = 1, so nu = mu */
    int clamp_bound = 0;
    for (size_t n = 0; n < CELLS; n++) {
        const double raw = TEST_C_MU * k[n] * k[n] / fmax(eps[n], TEST_EPS_FLOOR);
        const double expect = fmin(raw * 0.5, cap);
        TEST_ASSERT_DOUBLE_WITHIN(1e-12 * expect, expect, nu_t[n]);
        if (raw > cap) {
            clamp_bound = 1;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(clamp_bound,
                             "raw nu_t must exceed the cap for this to test anything");
    close_closure(model, ctx);
}

void test_small_context_matches_grid_sized_context(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(2.0f));

    cfd_nn_model_t* model_big = NULL;
    cfd_nn_context_t* ctx_big = NULL;
    open_closure(CELLS, &model_big, &ctx_big);
    double big[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(ctx_big, big));
    close_closure(model_big, ctx_big);

    /* Capacity 1 forces one inference call per cell. Same arithmetic per cell,
     * so the result must be identical bit for bit. */
    cfd_nn_model_t* model_small = NULL;
    cfd_nn_context_t* ctx_small = NULL;
    open_closure(1, &model_small, &ctx_small);
    double small[CELLS];
    TEST_ASSERT_EQUAL(CFD_SUCCESS, step_once(ctx_small, small));
    close_closure(model_small, ctx_small);

    TEST_ASSERT_EQUAL_MEMORY(big, small, sizeof(big));
}

void test_non_finite_prediction_fails_the_step(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_overflowing_model());
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    /* Not a silent skip: the step fails, so a corrupt model cannot quietly
     * turn into an uncorrected run that still reports success. */
    TEST_ASSERT_EQUAL(CFD_ERROR_DIVERGED, step_once(ctx, NULL));

    close_closure(model, ctx);
}

void test_wrong_model_shape_is_rejected(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_wrong_shape_model());
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, step_once(ctx, NULL));

    close_closure(model, ctx);
}

void test_closure_with_spalart_allmaras_is_refused(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(2.0f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_SPALART_ALLMARAS);
    params.turb_closure = ctx;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, 0.0, 0.0, 1e-3));
    /* SA carries no k/epsilon, so the features do not exist. Refused, not
     * ignored -- a caller who set a closure asked for it to run. */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      turbulence_step_explicit(field, g, &params, TEST_DT, 0.0));

    flow_field_destroy(field);
    grid_destroy(g);
    close_closure(model, ctx);
}

void test_both_corrections_at_once_are_refused(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(2.0f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    /* Two multipliers on nu_t would compound into something that is neither
     * the fitted correction nor the trained one. */
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      step_once_full(ctx, NS_NUT_CORRECTION_S_STAR, TEST_K0, NULL, NULL, NULL));

    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_closure = ctx;
    params.turb_nut_correction = NS_NUT_CORRECTION_S_STAR;
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, init_solver_with_params(&params));

    close_closure(model, ctx);
}

void test_solver_init_refuses_closure_without_kepsilon(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(2.0f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    ns_solver_params_t sa = make_params(TURB_MODEL_SPALART_ALLMARAS);
    sa.turb_closure = ctx;
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, init_solver_with_params(&sa));

    ns_solver_params_t none = make_params(TURB_MODEL_NONE);
    none.turb_closure = ctx;
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED, init_solver_with_params(&none));

    close_closure(model, ctx);
}

void test_solver_init_refuses_wrong_model_shape(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_wrong_shape_model());
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_closure = ctx;
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, init_solver_with_params(&params));

    close_closure(model, ctx);
}

void test_solver_init_accepts_a_valid_closure(void) {
    TEST_ASSERT_EQUAL(CFD_SUCCESS, write_constant_model(2.0f));
    cfd_nn_model_t* model = NULL;
    cfd_nn_context_t* ctx = NULL;
    open_closure(CELLS, &model, &ctx);

    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_closure = ctx;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_solver_with_params(&params));

    close_closure(model, ctx);
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
    RUN_TEST(test_beta_one_is_bit_identical_to_no_closure);
    RUN_TEST(test_predicted_factor_scales_nu_t);
    RUN_TEST(test_factor_above_the_ceiling_is_clamped);
    RUN_TEST(test_factor_below_the_floor_is_clamped);
    RUN_TEST(test_closure_scales_the_unclamped_viscosity);
    RUN_TEST(test_small_context_matches_grid_sized_context);
    RUN_TEST(test_non_finite_prediction_fails_the_step);
    RUN_TEST(test_wrong_model_shape_is_rejected);
    RUN_TEST(test_closure_with_spalart_allmaras_is_refused);
    RUN_TEST(test_both_corrections_at_once_are_refused);
    RUN_TEST(test_solver_init_refuses_closure_without_kepsilon);
    RUN_TEST(test_solver_init_refuses_wrong_model_shape);
    RUN_TEST(test_solver_init_accepts_a_valid_closure);
    return UNITY_END();
}
