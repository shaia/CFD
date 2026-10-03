/**
 * @file test_cfdnn_python_export.c
 * @brief The Python exporter and the C library agree on the `.cfdnn` format.
 *
 * tools/cfdnn/ is outside the build and outside CI, so this is the only place
 * CI sees its output. The model in cfdnn_python_golden.h was written by
 * tools/cfdnn/cfdnn.py: an MLP distilled from the algebraic correction
 * beta = A (S*)^B (tools/cfdnn/distill_algebraic.py). Distilling a known
 * answer is what makes every link checkable:
 *
 * 1. The C reader accepts the Python writer's bytes.
 * 2. Every available kernel backend reproduces the Python float64 reference
 *    to float32 accuracy.
 * 3. The C writer produces the same bytes from the same weights, so there is
 *    one format, not two that happen to agree on one file.
 * 4. Run as params.turb_closure, the network reproduces
 *    NS_NUT_CORRECTION_S_STAR to within its fit error. That checks what the
 *    format cannot: feature order, input normalization folded into the first
 *    layer, and that the output is the nu_t multiplier -- the design note
 *    records one sign inversion on exactly that lever already.
 *
 * Regenerate the header with:
 *   python tools/cfdnn/distill_algebraic.py --out <tmp>.cfdnn \
 *          --c-header tests/nn/cfdnn_python_golden.h
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/nn/cfdnn.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"

#include "../../lib/src/nn/cfdnn_internal.h"

#include "cfdnn_python_golden.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_MODEL "test_cfdnn_python_export_tmp.cfdnn"

/* The C kernels accumulate in float32; the expectation is float64 arithmetic on
 * the same float32 weights. Same budget as the SIMD-vs-scalar tolerance. */
#define KERNEL_REL_TOL 1e-5

/* Must match TURB_ALG_BETA_A / _B in turbulence_solver_internal.h and ALG_A /
 * ALG_B in tools/cfdnn/distill_algebraic.py. */
#define ALG_A 1.6945
#define ALG_B (-0.2778)

/* Header bytes [16, 22) are the writer's library version, which the reader
 * ignores and which changes on every release; the trailing CRC covers them. */
#define LIB_VERSION_OFFSET 16
#define LIB_VERSION_BYTES  6
#define CRC_BYTES          4

void setUp(void) {
    cfd_init();
}
void tearDown(void) {
    cfd_finalize();
    remove(TMP_MODEL);
}

static cfd_nn_model_t* load_golden(void) {
    cfd_nn_model_t* model = NULL;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_nn_model_load_memory(k_python_golden, sizeof(k_python_golden), &model));
    TEST_ASSERT_NOT_NULL(model);
    return model;
}

/* ============================================================================
 * 1-2. The C reader and kernels accept and reproduce the Python model
 * ============================================================================ */

void test_c_reader_loads_the_python_model(void) {
    cfd_nn_model_t* model = load_golden();
    TEST_ASSERT_EQUAL_STRING("beta-s-star-distilled", cfd_nn_model_name(model));
    TEST_ASSERT_EQUAL_size_t(3, cfd_nn_model_inputs(model));
    TEST_ASSERT_EQUAL_size_t(1, cfd_nn_model_outputs(model));
    cfd_nn_model_destroy(model);
}

static void check_backend(cfd_nn_backend_t backend) {
    if (!cfd_nn_backend_available(backend)) {
        TEST_IGNORE_MESSAGE("backend not available on this build/CPU");
    }
    cfd_nn_model_t* model = load_golden();
    cfd_nn_context_t* ctx = NULL;
    cfd_status_t st = cfd_nn_context_create(model, PYTHON_GOLDEN_PROBES, backend, &ctx);
    if (st == CFD_ERROR_UNSUPPORTED) {
        cfd_nn_model_destroy(model);
        TEST_IGNORE_MESSAGE("backend does not implement the Dense op yet");
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, st);

    double out[PYTHON_GOLDEN_PROBES];
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_nn_predict_batch(ctx, PYTHON_GOLDEN_PROBES, &k_python_golden_probes[0][0],
                                           PYTHON_GOLDEN_PROBES * 3, out, PYTHON_GOLDEN_PROBES));
    double worst = 0.0;
    for (size_t p = 0; p < PYTHON_GOLDEN_PROBES; p++) {
        const double rel = fabs(out[p] / k_python_golden_expect[p] - 1.0);
        worst = fmax(worst, rel);
        TEST_ASSERT_DOUBLE_WITHIN(KERNEL_REL_TOL * k_python_golden_expect[p],
                                  k_python_golden_expect[p], out[p]);
    }
    printf("  %s vs Python float64 reference: max relative difference %.3e\n",
           cfd_nn_context_backend(ctx), worst);

    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(model);
}

void test_scalar_kernel_reproduces_python(void) {
    check_backend(CFD_NN_BACKEND_SCALAR);
}
void test_omp_kernel_reproduces_python(void) {
    check_backend(CFD_NN_BACKEND_OMP);
}
void test_simd_kernel_reproduces_python(void) {
    check_backend(CFD_NN_BACKEND_SIMD);
}

/* The network was distilled from beta = A (S*)^B, which depends on ln S* alone.
 * At every probe it must track that law to within the fit error the exporter
 * measured -- including the two probes that vary only ln Re_t and ln nu_t/nu,
 * which it has to have learned to ignore. */
void test_python_model_is_the_algebraic_law(void) {
    for (size_t p = 0; p < PYTHON_GOLDEN_PROBES; p++) {
        const double law = ALG_A * exp(ALG_B * k_python_golden_probes[p][0]);
        TEST_ASSERT_DOUBLE_WITHIN(PYTHON_GOLDEN_MAX_REL_ERR * law, law, k_python_golden_expect[p]);
    }
}

/* ============================================================================
 * 3. One format: the C writer reproduces the Python writer's bytes
 * ============================================================================ */

static unsigned char* slurp(const char* path, size_t* size) {
    FILE* fp = fopen(path, "rb");
    TEST_ASSERT_NOT_NULL(fp);
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc((size_t)n);
    TEST_ASSERT_NOT_NULL(buf);
    TEST_ASSERT_EQUAL_size_t((size_t)n, fread(buf, 1, (size_t)n, fp));
    fclose(fp);
    *size = (size_t)n;
    return buf;
}

void test_c_writer_emits_the_python_bytes(void) {
    cfd_nn_model_t* model = load_golden();

    cfd_nn_layer_desc_t layers[CFD_NN_MAX_LAYERS];
    for (size_t i = 0; i < model->layer_count; i++) {
        const cfd_nn_layer_t* l = &model->layers[i];
        layers[i] =
            (cfd_nn_layer_desc_t){CFD_NN_LAYER_DENSE, l->activation, l->act_param, l->in_features,
                                  l->out_features,    l->weights,    l->bias};
    }
    cfd_nn_model_desc_t desc = {cfd_nn_model_name(model), layers, model->layer_count};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_nn_model_write(TMP_MODEL, &desc));
    cfd_nn_model_destroy(model);

    size_t size = 0;
    unsigned char* written = slurp(TMP_MODEL, &size);
    TEST_ASSERT_EQUAL_size_t(sizeof(k_python_golden), size);

    /* Everything but the library-version stamp and the CRC that covers it. */
    TEST_ASSERT_EQUAL_MEMORY(k_python_golden, written, LIB_VERSION_OFFSET);
    const size_t tail = LIB_VERSION_OFFSET + LIB_VERSION_BYTES;
    TEST_ASSERT_EQUAL_MEMORY(k_python_golden + tail, written + tail, size - tail - CRC_BYTES);
    free(written);
}

/* ============================================================================
 * 4. End to end: the learned closure reproduces the algebraic correction
 * ============================================================================ */

#define NX        9
#define NY        9
#define CELLS     (NX * NY)
#define TEST_MU   1e-3
#define TEST_EPS0 1.0
#define TEST_DT   1e-4

/* One k-epsilon step on a linear shear u = y (|S| = 1, so S* = k/eps), with
 * either the learned or the algebraic correction. One step, because the
 * transport consumes the previous step's nu_t: after exactly one the two runs
 * differ by the correction alone. See test_nut_correction.c. */
static void step_once(cfd_nn_context_t* closure, ns_nut_correction_t correction, double k0,
                      double* out_nu_t) {
    grid* g = grid_create(NX, NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(NX, NY, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            field->rho[j * NX + i] = 1.0;
            field->u[j * NX + i] = (double)j / (double)(NY - 1);
            field->v[j * NX + i] = 0.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = TEST_MU;
    params.turb_model = TURB_MODEL_K_EPSILON;
    params.turb_closure = closure;
    params.turb_nut_correction = correction;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_init_uniform(field, &params, k0, TEST_EPS0, 0.0));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_step_explicit(field, g, &params, TEST_DT, 0.0));
    memcpy(out_nu_t, field->nu_t, CELLS * sizeof(double));

    flow_field_destroy(field);
    grid_destroy(g);
}

/* k0 spans S* ~ 0.2..5, Re_t ~ 40..2.5e4 and nu_t/nu ~ 4..2e3: all inside the
 * box the network was trained on, so the fit error bounds the difference. */
void test_learned_closure_reproduces_the_algebraic_correction(void) {
    cfd_nn_model_t* model = load_golden();
    cfd_nn_context_t* ctx = NULL;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_nn_context_create(model, CELLS, CFD_NN_BACKEND_SCALAR, &ctx));

    const double k0s[] = {0.2, 1.0, 5.0};
    double worst = 0.0;
    for (size_t c = 0; c < sizeof(k0s) / sizeof(k0s[0]); c++) {
        double learned[CELLS], algebraic[CELLS];
        step_once(ctx, NS_NUT_CORRECTION_NONE, k0s[c], learned);
        step_once(NULL, NS_NUT_CORRECTION_S_STAR, k0s[c], algebraic);
        for (size_t n = 0; n < CELLS; n++) {
            TEST_ASSERT_TRUE(algebraic[n] > 0.0);
            const double rel = fabs(learned[n] / algebraic[n] - 1.0);
            worst = fmax(worst, rel);
            /* Fit error plus float32 inference; well short of the sign or
             * feature-order errors this exists to catch, which are O(1). */
            TEST_ASSERT_DOUBLE_WITHIN((PYTHON_GOLDEN_MAX_REL_ERR + KERNEL_REL_TOL) * algebraic[n],
                                      algebraic[n], learned[n]);
        }
    }
    printf("  learned vs algebraic nu_t: max relative difference %.3e (fit error %.3e)\n", worst,
           PYTHON_GOLDEN_MAX_REL_ERR);

    cfd_nn_context_destroy(ctx);
    cfd_nn_model_destroy(model);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_c_reader_loads_the_python_model);
    RUN_TEST(test_scalar_kernel_reproduces_python);
    RUN_TEST(test_omp_kernel_reproduces_python);
    RUN_TEST(test_simd_kernel_reproduces_python);
    RUN_TEST(test_python_model_is_the_algebraic_law);
    RUN_TEST(test_c_writer_emits_the_python_bytes);
    RUN_TEST(test_learned_closure_reproduces_the_algebraic_correction);
    return UNITY_END();
}
