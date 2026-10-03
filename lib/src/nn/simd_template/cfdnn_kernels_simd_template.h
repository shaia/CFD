/**
 * @file cfdnn_kernels_simd_template.h
 * @brief Dense-layer kernel shared by the AVX2 and NEON `.cfdnn` backends.
 *
 * NOT a standalone header: it is #include-d by avx2/cfdnn_kernels_avx2.c and
 * neon/cfdnn_kernels_neon.c after they define
 *
 *   SIMD_SUFFIX          token pasted onto function names (avx2, neon)
 *   SIMD_VEC             vector type holding SIMD_WIDTH floats
 *   SIMD_WIDTH           floats per vector
 *   SIMD_LOAD(p)         unaligned load of SIMD_WIDTH floats
 *   SIMD_STORE(p, v)     unaligned store
 *   SIMD_SET1(x)         broadcast
 *   SIMD_FMA(a, b, c)    a * b + c
 *
 * Vectorised across SAMPLES, never across input features -- the contract in
 * cfdnn_internal.h. Each lane carries one sample, and its sum over input
 * features runs in the same ascending order, from the same bias, as the
 * scalar reference. Nothing is reduced across lanes, so the only difference
 * from scalar is the rounding FMA saves per term.
 *
 * The input is sample-major, so a block of SIMD_WIDTH samples is first
 * transposed into feature-major scratch (one vector per feature). That costs
 * one pass over the block's inputs and turns every inner-loop load into a
 * contiguous vector load; the alternative, a strided gather per weight, would
 * repeat the gather for every output. Samples past the last full block run
 * the scalar arithmetic unchanged, per the vector-loop + scalar-tail rule.
 *
 * Activations reuse the shared scalar cfd_nn_apply_activation() over the
 * finished outputs, so every backend evaluates exactly the same activation
 * function and the only cross-backend difference stays FMA rounding in the
 * dense sums. That exactness has a measured cost: at closure size (3 -> 16
 * tanh -> 16 tanh -> 1 softplus) the scalar transcendentals take about 85% of
 * AVX2 inference time. Vectorising them would mean an approximation the
 * scalar backend does not make; that trade is an open decision, recorded in
 * docs/technical-notes/ml-integration-design.md section 2.3 and ROADMAP.md.
 */

#define CFDNN_SIMD_CAT_(a, b) a##_##b
#define CFDNN_SIMD_CAT(a, b)  CFDNN_SIMD_CAT_(a, b)
#define CFDNN_SIMD_FUNC(name) CFDNN_SIMD_CAT(name, SIMD_SUFFIX)

/* One sample, exactly as cpu/cfdnn_kernels_scalar.c computes it. */
static void CFDNN_SIMD_FUNC(dense_tail)(const cfd_nn_layer_t* l, const float* x, float* y) {
    const size_t ni = l->in_features;
    const size_t no = l->out_features;
    for (size_t o = 0; o < no; o++) {
        const float* w = l->weights + o * ni;
        float acc = l->bias ? l->bias[o] : 0.0f;
        for (size_t i = 0; i < ni; i++) {
            acc += x[i] * w[i];
        }
        y[o] = acc;
    }
}

static void CFDNN_SIMD_FUNC(dense)(const cfd_nn_layer_t* l, size_t batch, const float* in,
                                   float* out, float* scratch) {
    const size_t ni = l->in_features;
    const size_t no = l->out_features;
    size_t s = 0;

    for (; s + SIMD_WIDTH <= batch; s += SIMD_WIDTH) {
        const float* xb = in + s * ni;
        float* yb = out + s * no;

        /* scratch[i][lane] = x[lane][i]: ni <= widest, the context's sizing. */
        for (size_t lane = 0; lane < SIMD_WIDTH; lane++) {
            for (size_t i = 0; i < ni; i++) {
                scratch[i * SIMD_WIDTH + lane] = xb[lane * ni + i];
            }
        }

        for (size_t o = 0; o < no; o++) {
            const float* w = l->weights + o * ni; /* row-major [out][in] */
            SIMD_VEC acc = SIMD_SET1(l->bias ? l->bias[o] : 0.0f);
            for (size_t i = 0; i < ni; i++) {
                acc = SIMD_FMA(SIMD_LOAD(scratch + i * SIMD_WIDTH), SIMD_SET1(w[i]), acc);
            }
            float lanes[SIMD_WIDTH];
            SIMD_STORE(lanes, acc);
            for (size_t lane = 0; lane < SIMD_WIDTH; lane++) {
                yb[lane * no + o] = lanes[lane];
            }
        }
    }

    for (; s < batch; s++) {
        CFDNN_SIMD_FUNC(dense_tail)(l, in + s * ni, out + s * no);
    }

    cfd_nn_apply_activation(l->activation, l->act_param, out, batch * no);
}
