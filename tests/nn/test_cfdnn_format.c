/**
 * @file test_cfdnn_format.c
 * @brief Round-trip, rejection and arithmetic tests for the `.cfdnn` format.
 *
 * The rejection matrix is the point of this file. A weight file is untrusted
 * input: it can be truncated, bit-rotted, produced by a mismatched exporter or
 * hand-edited. Every one of those must fail with a SPECIFIC status rather than
 * be guessed at, which is the rule lib/src/io/checkpoint.c already follows for
 * `.cfdchk`.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/nn/cfdnn.h"

#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_MODEL "test_cfdnn_tmp.cfdnn"

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
    remove(TMP_MODEL);
}

/* ------------------------------------------------------------------ helpers */

/* A 2 -> 3 dense layer with hand-checkable values.
 * W is row-major [out][in]: rows (1,2), (3,4), (5,6); bias (10,20,30). */
static const float W1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
static const float B1[3] = {10.0f, 20.0f, 30.0f};

static cfd_nn_layer_desc_t make_layer(cfd_nn_activation_t act) {
    cfd_nn_layer_desc_t l;
    l.kind         = CFD_NN_LAYER_DENSE;
    l.activation   = act;
    l.act_param    = 0.0f;
    l.in_features  = 2;
    l.out_features = 3;
    l.weights      = W1;
    l.bias         = B1;
    return l;
}

static cfd_status_t write_simple(cfd_nn_activation_t act) {
    cfd_nn_layer_desc_t l    = make_layer(act);
    cfd_nn_model_desc_t desc = {"unit-test", &l, 1};
    return cfd_nn_model_write(TMP_MODEL, &desc);
}

static unsigned char* slurp(const char* path, size_t* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc((size_t)n);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) {
        free(b);
        b = NULL;
    }
    fclose(f);
    if (b) {
        *out_n = (size_t)n;
    }
    return b;
}

static void spit(const char* path, const unsigned char* b, size_t n) {
    FILE* f = fopen(path, "wb");
    TEST_ASSERT_NOT_NULL(f);
    fwrite(b, 1, n, f);
    fclose(f);
}

/* Corrupt one byte of a good file and assert the loader's verdict. */
static void expect_status_after_poke(size_t offset, unsigned char value,
                                     cfd_status_t expect, const char* what) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    size_t         n = 0;
    unsigned char* b = slurp(TMP_MODEL, &n);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_TRUE_MESSAGE(offset < n, what);
    b[offset] = value;
    spit(TMP_MODEL, b, n);
    free(b);

    cfd_nn_model_t* m  = NULL;
    cfd_status_t    st = cfd_nn_model_load(TMP_MODEL, &m);
    TEST_ASSERT_EQUAL_INT_MESSAGE((int)expect, (int)st, what);
    TEST_ASSERT_NULL_MESSAGE(m, what);
}

/* ------------------------------------------------------------------- tests */

void test_write_then_load_roundtrip(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));

    cfd_nn_model_t* m = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));
    TEST_ASSERT_NOT_NULL(m);
    TEST_ASSERT_EQUAL_UINT(2, (unsigned)cfd_nn_model_inputs(m));
    TEST_ASSERT_EQUAL_UINT(3, (unsigned)cfd_nn_model_outputs(m));
    TEST_ASSERT_EQUAL_STRING("unit-test", cfd_nn_model_name(m));
    cfd_nn_model_destroy(m);
}

/* load() and load_memory() must agree: they are the same parser, and the
 * memory path is what lets a golden model be embedded in a test binary. */
void test_file_and_memory_paths_agree(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    size_t         n = 0;
    unsigned char* b = slurp(TMP_MODEL, &n);
    TEST_ASSERT_NOT_NULL(b);

    cfd_nn_model_t* a = NULL;
    cfd_nn_model_t* c = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &a));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load_memory(b, n, &c));

    double in[2]  = {1.0, 1.0};
    double oa[3]  = {0};
    double oc[3]  = {0};
    cfd_nn_context_t* ca = NULL;
    cfd_nn_context_t* cc = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(a, 1, CFD_NN_BACKEND_SCALAR, &ca));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(c, 1, CFD_NN_BACKEND_SCALAR, &cc));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_predict_batch(ca, 1, in, 2, oa, 3));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_predict_batch(cc, 1, in, 2, oc, 3));
    TEST_ASSERT_EQUAL_MEMORY(oa, oc, sizeof(oa));

    cfd_nn_context_destroy(ca);
    cfd_nn_context_destroy(cc);
    cfd_nn_model_destroy(a);
    cfd_nn_model_destroy(c);
    free(b);
}

/* Hand-computed: y = W.x + b with x = (1,1) gives (13, 27, 41). */
void test_dense_matches_hand_computation(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    cfd_nn_model_t*   m   = NULL;
    cfd_nn_context_t* ctx = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(m, 2, CFD_NN_BACKEND_SCALAR, &ctx));

    double in[2] = {1.0, 1.0};
    double out[3];
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_predict_batch(ctx, 1, in, 2, out, 3));
    TEST_ASSERT_EQUAL_FLOAT(13.0f, (float)out[0]);
    TEST_ASSERT_EQUAL_FLOAT(27.0f, (float)out[1]);
    TEST_ASSERT_EQUAL_FLOAT(41.0f, (float)out[2]);

    /* Two samples at once: the second must not disturb the first. */
    double in2[4]  = {1.0, 1.0, 0.0, 0.0};
    double out2[6] = {0};
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_predict_batch(ctx, 2, in2, 4, out2, 6));
    TEST_ASSERT_EQUAL_FLOAT(13.0f, (float)out2[0]);
    TEST_ASSERT_EQUAL_FLOAT(27.0f, (float)out2[1]);
    TEST_ASSERT_EQUAL_FLOAT(41.0f, (float)out2[2]);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, (float)out2[3]); /* bias only */
    TEST_ASSERT_EQUAL_FLOAT(20.0f, (float)out2[4]);
    TEST_ASSERT_EQUAL_FLOAT(30.0f, (float)out2[5]);

    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(m);
}

/* Softplus is the closure's non-negativity guarantee, so it must stay finite
 * and positive at magnitudes where a naive expf() would overflow. */
void test_softplus_is_positive_and_finite(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_SOFTPLUS));
    cfd_nn_model_t*   m   = NULL;
    cfd_nn_context_t* ctx = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(m, 1, CFD_NN_BACKEND_SCALAR, &ctx));

    /* Large negative input drives the pre-activation very negative. */
    double in[2] = {-1.0e5, -1.0e5};
    double out[3];
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_predict_batch(ctx, 1, in, 2, out, 3));
    for (int i = 0; i < 3; i++) {
        TEST_ASSERT_TRUE_MESSAGE(isfinite(out[i]), "softplus produced non-finite");
        TEST_ASSERT_TRUE_MESSAGE(out[i] >= 0.0, "softplus produced a negative");
    }
    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(m);
}

void test_rejects_corrupt_files(void) {
    /* Header layout is fixed, so these offsets are part of the contract. */
    expect_status_after_poke(0, 'X', CFD_ERROR_INVALID, "bad magic");
    expect_status_after_poke(8, 99, CFD_ERROR_UNSUPPORTED, "unknown version");
    expect_status_after_poke(12, 0xFF, CFD_ERROR_UNSUPPORTED, "foreign endianness");
    expect_status_after_poke(24, 2, CFD_ERROR_UNSUPPORTED, "f64 dtype reserved");
    expect_status_after_poke(24, 99, CFD_ERROR_UNSUPPORTED, "unknown dtype");
    expect_status_after_poke(28, 0xFF, CFD_ERROR_INVALID, "layer_count past cap");
    /* Reject-unknown covers the fields a later format version would use,
     * not just the ones this one reads: a file that sets them means
     * something this build cannot honour. */
    expect_status_after_poke(23, 0x80, CFD_ERROR_UNSUPPORTED, "unassigned flag bit");
    expect_status_after_poke(25, 1, CFD_ERROR_UNSUPPORTED, "unknown tensor layout");
    expect_status_after_poke(26, 1, CFD_ERROR_UNSUPPORTED, "nonzero reserved u16");
    expect_status_after_poke(32, 1, CFD_ERROR_UNSUPPORTED, "nonzero reserved u32 #1");
    expect_status_after_poke(36, 1, CFD_ERROR_UNSUPPORTED, "nonzero reserved u32 #2");
}

/* A flipped weight bit leaves every declared size valid, so only the CRC can
 * catch it. This is the test that proves the checksum is actually wired up. */
void test_rejects_bitrot_via_crc(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    size_t         n = 0;
    unsigned char* b = slurp(TMP_MODEL, &n);
    TEST_ASSERT_NOT_NULL(b);
    b[n - 8] ^= 0x01; /* inside the weight block, before the trailing CRC */
    spit(TMP_MODEL, b, n);
    free(b);

    cfd_nn_model_t* m = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_IO, cfd_nn_model_load(TMP_MODEL, &m));
    TEST_ASSERT_NULL(m);
}

/* Truncating at every length exercises each of the loader's early-return
 * paths in one loop -- and under ASan, each one's cleanup as well. */
void test_truncation_at_every_length_is_rejected(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    size_t         n = 0;
    unsigned char* b = slurp(TMP_MODEL, &n);
    TEST_ASSERT_NOT_NULL(b);

    for (size_t k = 1; k < n; k++) {
        cfd_nn_model_t* m  = NULL;
        cfd_status_t    st = cfd_nn_model_load_memory(b, k, &m);
        TEST_ASSERT_TRUE_MESSAGE(st != CFD_SUCCESS, "truncated image loaded");
        TEST_ASSERT_NULL_MESSAGE(m, "model returned for a truncated image");
    }
    free(b);
}

void test_api_guards(void) {
    cfd_nn_model_t* m = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_IO, cfd_nn_model_load("does_not_exist.cfdnn", &m));
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_model_load_memory(NULL, 0, &m));

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));

    cfd_nn_context_t* ctx = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID,
                          cfd_nn_context_create(m, 0, CFD_NN_BACKEND_SCALAR, &ctx));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
                          cfd_nn_context_create(m, 1, CFD_NN_BACKEND_SCALAR, &ctx));
    TEST_ASSERT_EQUAL_STRING("scalar", cfd_nn_context_backend(ctx));

    double in[2] = {1.0, 1.0};
    double out[3];
    /* Count mismatches and oversized batches are caller errors, not crashes. */
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_predict_batch(ctx, 1, in, 3, out, 3));
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_predict_batch(ctx, 1, in, 2, out, 2));
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_predict_batch(ctx, 9, in, 18, out, 27));

    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(m);
    cfd_nn_model_destroy(NULL); /* NULL is safe */
    cfd_nn_context_destroy(NULL);
}

/* An explicitly requested backend that is not built must be reported, never
 * silently swapped for another one. */
void test_unavailable_backend_is_reported(void) {
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, write_simple(CFD_NN_ACT_IDENTITY));
    cfd_nn_model_t* m = NULL;
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, cfd_nn_model_load(TMP_MODEL, &m));

    cfd_nn_context_t* ctx = NULL;
    cfd_status_t      st  = cfd_nn_context_create(m, 1, CFD_NN_BACKEND_SIMD, &ctx);
    if (cfd_nn_backend_available(CFD_NN_BACKEND_SIMD)) {
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, st);
        cfd_nn_context_destroy(ctx);
    } else {
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED, st);
        TEST_ASSERT_NULL(ctx);
    }
    cfd_nn_model_destroy(m);
}

/* The writer runs the reader's shape rules, so a malformed model cannot be
 * produced in the first place. */
void test_writer_rejects_bad_shapes(void) {
    cfd_nn_layer_desc_t l = make_layer(CFD_NN_ACT_IDENTITY);
    l.in_features         = 0;
    cfd_nn_model_desc_t d = {"bad", &l, 1};
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_model_write(TMP_MODEL, &d));

    /* Layer 2's input width must match layer 1's output width. */
    cfd_nn_layer_desc_t two[2];
    two[0]                = make_layer(CFD_NN_ACT_IDENTITY); /* 2 -> 3 */
    two[1]                = make_layer(CFD_NN_ACT_IDENTITY); /* 2 -> 3, mismatched */
    cfd_nn_model_desc_t d2 = {"bad-chain", two, 2};
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, cfd_nn_model_write(TMP_MODEL, &d2));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_write_then_load_roundtrip);
    RUN_TEST(test_file_and_memory_paths_agree);
    RUN_TEST(test_dense_matches_hand_computation);
    RUN_TEST(test_softplus_is_positive_and_finite);
    RUN_TEST(test_rejects_corrupt_files);
    RUN_TEST(test_rejects_bitrot_via_crc);
    RUN_TEST(test_truncation_at_every_length_is_rejected);
    RUN_TEST(test_api_guards);
    RUN_TEST(test_unavailable_backend_is_reported);
    RUN_TEST(test_writer_rejects_bad_shapes);
    return UNITY_END();
}
