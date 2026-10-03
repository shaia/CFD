/**
 * @file test_cfdnn_backends.c
 * @brief Cross-backend agreement for `.cfdnn` inference.
 *
 * The contract in cfdnn_internal.h says SIMD lanes and OpenMP threads carry
 * distinct SAMPLES, never distinct input features, so no backend ever reduces
 * across lanes or threads. Two consequences are asserted here:
 *
 *   OMP vs scalar   BIT-IDENTICAL. The per-sample accumulation is untouched,
 *                   so exact equality is the right assertion. If this ever
 *                   fails, someone parallelised a reduction axis -- which is
 *                   precisely what this test exists to catch.
 *
 *   SIMD vs scalar  within 1e-5 relative, and the observed maximum is PRINTED
 *                   so drift toward the tolerance is visible before it becomes
 *                   a failure. Bit-exactness is not promised: SIMD contracts
 *                   its dense sums with FMA and evaluates tanh, sigmoid and
 *                   softplus with vector approximations. Checked at every
 *                   batch size around the vector widths (4 and 8), where the
 *                   full blocks meet the scalar tail, on a layer far wider
 *                   than a vector, which exercises the transpose scratch, and
 *                   per activation, isolated, to 1e-6.
 *
 *   NaN             survives every vectorised activation, so a corrupt model
 *                   still fails with CFD_ERROR_DIVERGED on SIMD.
 *
 * Backends that are not built are skipped, not failed, per the project's
 * optional-backend policy.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/nn/cfdnn.h"

#include "unity.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_MODEL "test_cfdnn_backends_tmp.cfdnn"

#define N_IN    6
#define N_HID   16
#define N_OUT   1
#define N_BATCH 257 /* deliberately not a multiple of any vector width, so the
                     * scalar remainder tail is always exercised */

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
    remove(TMP_MODEL);
}

static float g_w1[N_HID * N_IN];
static float g_b1[N_HID];
static float g_w2[N_OUT * N_HID];
static float g_b2[N_OUT];

/* Deterministic pseudo-random weights: a fixed LCG, so the model is
 * reproducible without shipping a data file. */
static float next_weight(uint32_t* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float)((*state >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
}

static cfd_status_t write_two_layer_model(void) {
    uint32_t st = 12345u;
    for (int i = 0; i < N_HID * N_IN; i++) {
        g_w1[i] = next_weight(&st);
    }
    for (int i = 0; i < N_HID; i++) {
        g_b1[i] = next_weight(&st);
    }
    for (int i = 0; i < N_OUT * N_HID; i++) {
        g_w2[i] = next_weight(&st);
    }
    for (int i = 0; i < N_OUT; i++) {
        g_b2[i] = next_weight(&st);
    }

    cfd_nn_layer_desc_t layers[2];
    layers[0].kind         = CFD_NN_LAYER_DENSE;
    layers[0].activation   = CFD_NN_ACT_TANH;
    layers[0].act_param    = 0.0f;
    layers[0].in_features  = N_IN;
    layers[0].out_features = N_HID;
    layers[0].weights      = g_w1;
    layers[0].bias         = g_b1;

    layers[1].kind         = CFD_NN_LAYER_DENSE;
    layers[1].activation   = CFD_NN_ACT_SOFTPLUS;
    layers[1].act_param    = 0.0f;
    layers[1].in_features  = N_HID;
    layers[1].out_features = N_OUT;
    layers[1].weights      = g_w2;
    layers[1].bias         = g_b2;

    cfd_nn_model_desc_t desc = {"backend-consistency", layers, 2};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

/* Run the model on a fixed input set with the given backend.
 * Returns 0 when the backend is unavailable. */
static int run_backend(cfd_nn_backend_t backend, const double* in, double* out,
                       const char** resolved) {
    cfd_nn_model_t* m = NULL;
    if (cfd_nn_model_load(TMP_MODEL, &m) != CFD_SUCCESS) {
        TEST_FAIL_MESSAGE("model load failed");
    }
    cfd_nn_context_t* ctx = NULL;
    cfd_status_t      st  = cfd_nn_context_create(m, N_BATCH, backend, &ctx);
    if (st == CFD_ERROR_UNSUPPORTED) {
        cfd_nn_model_destroy(m);
        return 0;
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);
    *resolved = cfd_nn_context_backend(ctx);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_predict_batch(ctx, N_BATCH, in, N_BATCH * N_IN,
                                               out, N_BATCH * N_OUT));
    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(m);
    return 1;
}

static void fill_inputs(double* in) {
    /* Spread over several decades so the comparison is not dominated by one
     * magnitude, and include negatives so both activation branches run. */
    for (int s = 0; s < N_BATCH; s++) {
        for (int f = 0; f < N_IN; f++) {
            double t        = (double)(s * N_IN + f);
            in[s * N_IN + f] = sin(t * 0.37) * pow(10.0, (double)(f % 4) - 2.0);
        }
    }
}

void test_omp_is_bit_identical_to_scalar(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_two_layer_model());

    double* in  = (double*)malloc(N_BATCH * N_IN * sizeof(double));
    double* ref = (double*)malloc(N_BATCH * N_OUT * sizeof(double));
    double* got = (double*)malloc(N_BATCH * N_OUT * sizeof(double));
    TEST_ASSERT_NOT_NULL(in);
    fill_inputs(in);

    const char* rs = "";
    const char* ro = "";
    TEST_ASSERT_TRUE(run_backend(CFD_NN_BACKEND_SCALAR, in, ref, &rs));

    if (!run_backend(CFD_NN_BACKEND_OMP, in, got, &ro)) {
        free(in);
        free(ref);
        free(got);
        TEST_IGNORE_MESSAGE("OpenMP backend not built");
        return;
    }
    printf("    scalar=%s omp=%s batch=%d\n", rs, ro, N_BATCH);

    /* Exact equality, deliberately: see the file header. */
    TEST_ASSERT_EQUAL_MEMORY_MESSAGE(ref, got, N_BATCH * N_OUT * sizeof(double),
                                     "OMP diverged from scalar -- a reduction "
                                     "axis was parallelised");
    free(in);
    free(ref);
    free(got);
}

void test_simd_matches_scalar_within_tolerance(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_two_layer_model());

    double* in  = (double*)malloc(N_BATCH * N_IN * sizeof(double));
    double* ref = (double*)malloc(N_BATCH * N_OUT * sizeof(double));
    double* got = (double*)malloc(N_BATCH * N_OUT * sizeof(double));
    TEST_ASSERT_NOT_NULL(in);
    fill_inputs(in);

    const char* rs = "";
    const char* rv = "";
    TEST_ASSERT_TRUE(run_backend(CFD_NN_BACKEND_SCALAR, in, ref, &rs));

    if (!run_backend(CFD_NN_BACKEND_SIMD, in, got, &rv)) {
        free(in);
        free(ref);
        free(got);
        TEST_IGNORE_MESSAGE("SIMD backend not built or not supported by this CPU");
        return;
    }

    /* Assert the backend really is vectorised. Without this a run that
     * silently resolved to scalar would pass vacuously -- the same trap the
     * CI AVX2 job greps its own output to avoid. */
    TEST_ASSERT_TRUE_MESSAGE(strcmp(rv, "scalar") != 0,
                             "SIMD request resolved to scalar; test would pass vacuously");

    double worst = 0.0;
    for (int i = 0; i < N_BATCH * N_OUT; i++) {
        double denom = fabs(ref[i]) > 1e-30 ? fabs(ref[i]) : 1.0;
        double rel   = fabs(got[i] - ref[i]) / denom;
        if (rel > worst) {
            worst = rel;
        }
    }
    printf("    simd=%s max relative difference vs scalar: %.3e\n", rv, worst);
    TEST_ASSERT_TRUE_MESSAGE(worst < 1e-5, "SIMD drifted from scalar beyond 1e-5");

    free(in);
    free(ref);
    free(got);
}

/* A wide two-layer model (WIDE_IN -> WIDE_HID tanh -> 3 identity) for the
 * block/tail boundary sweep. Identity output so no activation hides a
 * difference, and three outputs so the per-output scatter is exercised. */
#define WIDE_IN   37
#define WIDE_HID  300
#define WIDE_OUT  3
#define MAX_SWEEP 33

static float g_ww1[WIDE_HID * WIDE_IN];
static float g_wb1[WIDE_HID];
static float g_ww2[WIDE_OUT * WIDE_HID];
static float g_wb2[WIDE_OUT];

static cfd_status_t write_wide_model(void) {
    uint32_t st = 777u;
    for (int i = 0; i < WIDE_HID * WIDE_IN; i++) {
        g_ww1[i] = next_weight(&st) * 0.2f;
    }
    for (int i = 0; i < WIDE_HID; i++) {
        g_wb1[i] = next_weight(&st);
    }
    for (int i = 0; i < WIDE_OUT * WIDE_HID; i++) {
        g_ww2[i] = next_weight(&st) * 0.1f;
    }
    for (int i = 0; i < WIDE_OUT; i++) {
        g_wb2[i] = next_weight(&st);
    }
    cfd_nn_layer_desc_t layers[2] = {
        {CFD_NN_LAYER_DENSE, CFD_NN_ACT_TANH, 0.0f, WIDE_IN, WIDE_HID, g_ww1, g_wb1},
        {CFD_NN_LAYER_DENSE, CFD_NN_ACT_IDENTITY, 0.0f, WIDE_HID, WIDE_OUT, g_ww2, g_wb2},
    };
    cfd_nn_model_desc_t desc = {"backend-wide", layers, 2};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

static void predict_with(cfd_nn_context_t* ctx, size_t batch, const double* in, double* out) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_predict_batch(ctx, batch, in, batch * WIDE_IN, out,
                                               batch * WIDE_OUT));
}

void test_simd_matches_scalar_at_every_batch_size(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_wide_model());
    cfd_nn_model_t* m = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));

    cfd_nn_context_t* ref_ctx = NULL;
    cfd_nn_context_t* simd_ctx = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(m, MAX_SWEEP, CFD_NN_BACKEND_SCALAR, &ref_ctx));
    /* Only an unavailable backend skips; any other failure is a regression. */
    cfd_status_t st = cfd_nn_context_create(m, MAX_SWEEP, CFD_NN_BACKEND_SIMD, &simd_ctx);
    if (st == CFD_ERROR_UNSUPPORTED) {
        cfd_nn_context_destroy(ref_ctx);
        cfd_nn_model_destroy(m);
        TEST_IGNORE_MESSAGE("SIMD backend not built or not supported by this CPU");
        return;
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);

    static double in[MAX_SWEEP * WIDE_IN];
    static double ref[MAX_SWEEP * WIDE_OUT];
    static double got[MAX_SWEEP * WIDE_OUT];
    for (int k = 0; k < MAX_SWEEP * WIDE_IN; k++) {
        in[k] = cos((double)k * 0.61) * (1.0 + (double)(k % 5));
    }

    double worst = 0.0;
    for (size_t batch = 1; batch <= MAX_SWEEP; batch++) {
        predict_with(ref_ctx, batch, in, ref);
        predict_with(simd_ctx, batch, in, got);
        for (size_t k = 0; k < batch * WIDE_OUT; k++) {
            /* Absolute floor: identity outputs can pass near zero. */
            double rel = fabs(got[k] - ref[k]) / fmax(fabs(ref[k]), 1e-2);
            worst = fmax(worst, rel);
        }
    }
    printf("    simd=%s batches 1..%d, %d-wide hidden layer: max difference %.3e\n",
           cfd_nn_context_backend(simd_ctx), MAX_SWEEP, WIDE_HID, worst);
    TEST_ASSERT_TRUE_MESSAGE(worst < 1e-5, "SIMD drifted from scalar beyond 1e-5");

    cfd_nn_context_destroy(simd_ctx);
    cfd_nn_context_destroy(ref_ctx);
    cfd_nn_model_destroy(m);
}

/* ============================================================================
 * Vectorised activations
 *
 * The SIMD backend evaluates tanh, sigmoid and softplus with polynomial
 * approximations rather than the scalar tanhf/expf/log1pf. Each is isolated
 * here behind a 1 -> 1 identity layer (weight 1, bias 0), so the activation is
 * the only arithmetic between input and output and its error is not diluted
 * by a dense sum.
 * ============================================================================ */

#define ACT_SAMPLES 4096

static float g_one = 1.0f;
static float g_zero = 0.0f;

static cfd_status_t write_activation_model(cfd_nn_activation_t act) {
    cfd_nn_layer_desc_t l = {CFD_NN_LAYER_DENSE, act, 0.1f, 1, 1, &g_one, &g_zero};
    cfd_nn_model_desc_t desc = {"activation-only", &l, 1};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

/* Opens scalar and SIMD contexts on TMP_MODEL; returns 0 (and frees) when SIMD
 * is unavailable, failing on any other error. */
static int open_pair(size_t cap, cfd_nn_model_t** m, cfd_nn_context_t** ref,
                     cfd_nn_context_t** simd) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, m));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_context_create(*m, cap, CFD_NN_BACKEND_SCALAR, ref));
    cfd_status_t st = cfd_nn_context_create(*m, cap, CFD_NN_BACKEND_SIMD, simd);
    if (st == CFD_ERROR_UNSUPPORTED) {
        cfd_nn_context_destroy(*ref);
        cfd_nn_model_destroy(*m);
        return 0;
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);
    return 1;
}

static void check_activation(cfd_nn_activation_t act, const char* name, double lo, double hi) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_activation_model(act));
    cfd_nn_model_t* m = NULL;
    cfd_nn_context_t* ref_ctx = NULL;
    cfd_nn_context_t* simd_ctx = NULL;
    if (!open_pair(ACT_SAMPLES, &m, &ref_ctx, &simd_ctx)) {
        TEST_IGNORE_MESSAGE("SIMD backend not built or not supported by this CPU");
        return;
    }

    /* A uniform sweep of [lo, hi], with the first 64 samples replaced by
     * +-1e-k so small arguments (where tanh must stay odd and accurate) are
     * covered too. */
    static double in[ACT_SAMPLES], ref[ACT_SAMPLES], got[ACT_SAMPLES];
    for (int i = 0; i < ACT_SAMPLES; i++) {
        in[i] = lo + (hi - lo) * (double)i / (double)(ACT_SAMPLES - 1);
    }
    for (int k = 0; k < 64; k++) {
        in[k] = (k % 2 ? -1.0 : 1.0) * pow(10.0, -(double)(k / 2) * 0.4);
    }
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_predict_batch(ref_ctx, ACT_SAMPLES, in, ACT_SAMPLES, ref, ACT_SAMPLES));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_predict_batch(simd_ctx, ACT_SAMPLES, in, ACT_SAMPLES, got, ACT_SAMPLES));

    double worst = 0.0;
    for (int i = 0; i < ACT_SAMPLES; i++) {
        double rel = fabs(got[i] - ref[i]) / fmax(fabs(ref[i]), 1e-30);
        worst = fmax(worst, rel);
    }
    printf("    simd=%s %-8s on [%g, %g]: max relative difference %.3e\n",
           cfd_nn_context_backend(simd_ctx), name, lo, hi, worst);
    /* About 2.7e-7 in the float32 prototype; scalar libm adds its own ~1 ulp. */
    TEST_ASSERT_TRUE_MESSAGE(worst < 1e-6, "vectorised activation drifted from scalar");

    cfd_nn_context_destroy(simd_ctx);
    cfd_nn_context_destroy(ref_ctx);
    cfd_nn_model_destroy(m);
}

void test_simd_tanh_matches_scalar(void) { check_activation(CFD_NN_ACT_TANH, "tanh", -20.0, 20.0); }
void test_simd_sigmoid_matches_scalar(void) {
    check_activation(CFD_NN_ACT_SIGMOID, "sigmoid", -60.0, 60.0);
}
/* Down to -80, where softplus is exp(x) ~ 1e-35: log1p must not round it to 0. */
void test_simd_softplus_matches_scalar(void) {
    check_activation(CFD_NN_ACT_SOFTPLUS, "softplus", -80.0, 60.0);
}

/* A NaN must survive every vectorised activation, so predict still reports
 * CFD_ERROR_DIVERGED. minps/maxps return their second operand when either is
 * NaN, so a clamp written the wrong way round would turn NaN into a finite
 * value and hand a corrupt model's output back as a prediction. The NaN sits
 * in the vector body (sample 3 of 16), not the scalar tail. */
void test_simd_activations_propagate_nan(void) {
    const cfd_nn_activation_t acts[] = {CFD_NN_ACT_TANH, CFD_NN_ACT_SIGMOID, CFD_NN_ACT_SOFTPLUS};
    for (size_t a = 0; a < sizeof(acts) / sizeof(acts[0]); a++) {
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_activation_model(acts[a]));
        cfd_nn_model_t* m = NULL;
        cfd_nn_context_t* ref_ctx = NULL;
        cfd_nn_context_t* simd_ctx = NULL;
        if (!open_pair(16, &m, &ref_ctx, &simd_ctx)) {
            TEST_IGNORE_MESSAGE("SIMD backend not built or not supported by this CPU");
            return;
        }
        double in[16], out[16];
        for (int i = 0; i < 16; i++) {
            in[i] = 0.25 * (double)i - 2.0;
        }
        in[3] = NAN;
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_DIVERGED,
                                      cfd_nn_predict_batch(simd_ctx, 16, in, 16, out, 16),
                                      "a NaN input came out of the activation finite");
        cfd_nn_context_destroy(simd_ctx);
        cfd_nn_context_destroy(ref_ctx);
        cfd_nn_model_destroy(m);
    }
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_omp_is_bit_identical_to_scalar);
    RUN_TEST(test_simd_matches_scalar_within_tolerance);
    RUN_TEST(test_simd_matches_scalar_at_every_batch_size);
    RUN_TEST(test_simd_tanh_matches_scalar);
    RUN_TEST(test_simd_sigmoid_matches_scalar);
    RUN_TEST(test_simd_softplus_matches_scalar);
    RUN_TEST(test_simd_activations_propagate_nan);
    return UNITY_END();
}
