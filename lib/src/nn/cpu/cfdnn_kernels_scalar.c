/**
 * @file cfdnn_kernels_scalar.c
 * @brief Portable reference kernels for `.cfdnn` inference.
 *
 * This is the correctness reference: every other backend must agree with it.
 * Per the project's scalar-backend policy it is used for validation and small
 * runs, never for long-running performance tests.
 *
 * Accumulation order is part of the contract, not an implementation detail.
 * The sum over input features runs in ascending index order, and nothing is
 * ever reduced across samples. SIMD and OpenMP backends parallelise the SAMPLE
 * axis, which is not reduced, so OpenMP stays bit-identical to this file and
 * SIMD differs only by FMA contraction.
 */

#include "../cfdnn_internal.h"

#include <math.h>

void cfd_nn_apply_activation(cfd_nn_activation_t act, float param, float* v, size_t n) {
    switch (act) {
        case CFD_NN_ACT_IDENTITY:
            break;
        case CFD_NN_ACT_RELU:
            for (size_t i = 0; i < n; i++) {
                if (v[i] < 0.0f) {
                    v[i] = 0.0f;
                }
            }
            break;
        case CFD_NN_ACT_LEAKY_RELU:
            for (size_t i = 0; i < n; i++) {
                if (v[i] < 0.0f) {
                    v[i] *= param;
                }
            }
            break;
        case CFD_NN_ACT_TANH:
            for (size_t i = 0; i < n; i++) {
                v[i] = tanhf(v[i]);
            }
            break;
        case CFD_NN_ACT_SIGMOID:
            for (size_t i = 0; i < n; i++) {
                v[i] = 1.0f / (1.0f + expf(-v[i]));
            }
            break;
        case CFD_NN_ACT_SOFTPLUS:
            for (size_t i = 0; i < n; i++) {
                /* log1p(exp(x)) via the max form, which stays finite for large
                 * x where expf() would overflow. The closure's non-negativity
                 * guarantee rests on this layer, so it must not produce inf. */
                float x  = v[i];
                float mx = x > 0.0f ? x : 0.0f;
                v[i]     = mx + log1pf(expf(-fabsf(x)));
            }
            break;
        default:
            break; /* unreachable: the loader rejects unknown activations */
    }
}

static void dense_scalar(const cfd_nn_layer_t* l, size_t batch, const float* in,
                         float* out) {
    const size_t ni = l->in_features;
    const size_t no = l->out_features;

    for (size_t s = 0; s < batch; s++) {
        const float* x = in + s * ni;
        float*       y = out + s * no;
        for (size_t o = 0; o < no; o++) {
            const float* w   = l->weights + o * ni; /* row-major [out][in] */
            float        acc = l->bias ? l->bias[o] : 0.0f;
            for (size_t i = 0; i < ni; i++) {
                acc += x[i] * w[i];
            }
            y[o] = acc;
        }
        cfd_nn_apply_activation(l->activation, l->act_param, y, no);
    }
}

const cfd_nn_backend_impl_t cfd_nn_impl_scalar = {
    "scalar",
    dense_scalar,
};
