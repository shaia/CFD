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
 *                   a failure. Bit-exactness is not promised across vector
 *                   widths because of FMA contraction.
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

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_omp_is_bit_identical_to_scalar);
    RUN_TEST(test_simd_matches_scalar_within_tolerance);
    return UNITY_END();
}
