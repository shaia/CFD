/**
 * @file cfdnn_kernels_simd_template.h
 * @brief Dense-layer kernel and activations shared by the AVX2 and NEON `.cfdnn` backends.
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
 *   SIMD_ADD/SUB/MUL/DIV(a, b)
 *   SIMD_MIN(a, b), SIMD_MAX(a, b)   NaN in b MUST propagate (see the clamps)
 *   SIMD_ABS(a), SIMD_FLOOR(a)
 *   SIMD_SELECT_LT(a, b, t, f)       per lane: a < b ? t : f
 *   SIMD_SELECT_EQ(a, b, t, f)       per lane: a == b ? t : f
 *   SIMD_POW2N(n)        2^n for a vector of integral floats n in [-127, 128], built
 *                        in the exponent bits: -127 gives +0 and 128 gives +inf,
 *                        the flush and overflow vexp's clamps rely on
 *   SIMD_FREXP_E(x)      exponent e of x > 0 (normal), as float: x = m * 2^e
 *   SIMD_FREXP_M(x)      mantissa m of x > 0 (normal), in [0.5, 1)
 *
 * Vectorised across SAMPLES, never across input features -- the contract in
 * cfdnn_internal.h. Each lane carries one sample, and its sum over input
 * features runs in the same ascending order, from the same bias, as the
 * scalar reference. Nothing is reduced across lanes, so the only difference
 * from scalar in the dense sums is the rounding FMA saves per term.
 *
 * The input is sample-major, so a block of SIMD_WIDTH samples is first
 * transposed into feature-major scratch (one vector per feature). That costs
 * one pass over the block's inputs and turns every inner-loop load into a
 * contiguous vector load; the alternative, a strided gather per weight, would
 * repeat the gather for every output. Samples past the last full block run
 * the scalar arithmetic unchanged, per the vector-loop + scalar-tail rule.
 *
 * Activations. tanh, sigmoid and softplus are vectorised with polynomial
 * approximations; identity, ReLU and leaky ReLU, which cost nothing, use the
 * shared scalar cfd_nn_apply_activation(). This is a deliberate departure from
 * scalar: the scalar backend calls tanhf/expf/log1pf, these do not, so SIMD
 * and scalar evaluate slightly different activation functions. Measured in
 * float32 against float64 truth (design note section 2.3): tanh 4.3 ulp,
 * softplus 2.7 ulp, sigmoid 2.3 ulp, all under 2.7e-7 relative -- about 40x
 * inside the 1e-5 cross-backend tolerance -- where the scalar transcendentals
 * held AVX2 to about 1.6x over scalar on a closure-sized network.
 *
 * NaN must survive every activation, because cfd_nn_predict_batch() turns a
 * non-finite output into CFD_ERROR_DIVERGED: a clamp that swallowed NaN would
 * hand a corrupt model's garbage back as a finite prediction. Every clamp
 * therefore passes x as the SECOND operand of SIMD_MIN/SIMD_MAX, which is the
 * one an x86 minps/maxps returns when either operand is NaN.
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

/* ==========================================================================
 * Vector transcendentals
 * ========================================================================== */

/**
 * exp(x), Cephes expf: x = n ln2 + r with |r| <= ln2/2, a degree-5 polynomial
 * for e^r, scaled by 2^n built in the exponent bits. Clamped to +-88.376 so
 * 2^n stays representable; below about -87.3 the result flushes to zero,
 * which for every caller here (exp(-|x|) in softplus, exp(-x) in sigmoid) is
 * the value the exact function rounds to anyway or is irrelevant beside 1.
 */
static inline SIMD_VEC CFDNN_SIMD_FUNC(vexp)(SIMD_VEC x) {
    x = SIMD_MAX(SIMD_SET1(-88.3762626647949f), SIMD_MIN(SIMD_SET1(88.3762626647949f), x));
    SIMD_VEC n = SIMD_FLOOR(SIMD_FMA(x, SIMD_SET1(1.44269504088896341f), SIMD_SET1(0.5f)));
    /* r = x - n ln2, with ln2 split so n * C1 is exact. */
    SIMD_VEC r = SIMD_FMA(n, SIMD_SET1(-0.693359375f), x);
    r = SIMD_FMA(n, SIMD_SET1(2.12194440e-4f), r);
    SIMD_VEC z = SIMD_MUL(r, r);
    SIMD_VEC y = SIMD_SET1(1.9875691500e-4f);
    y = SIMD_FMA(y, r, SIMD_SET1(1.3981999507e-3f));
    y = SIMD_FMA(y, r, SIMD_SET1(8.3334519073e-3f));
    y = SIMD_FMA(y, r, SIMD_SET1(4.1665795894e-2f));
    y = SIMD_FMA(y, r, SIMD_SET1(1.6666665459e-1f));
    y = SIMD_FMA(y, r, SIMD_SET1(5.0000001201e-1f));
    y = SIMD_ADD(SIMD_FMA(y, z, r), SIMD_SET1(1.0f));
    return SIMD_MUL(y, SIMD_POW2N(n));
}

/**
 * log(x) for normal x > 0, Cephes logf: x = m 2^e with m folded into
 * [sqrt(1/2), sqrt(2)), then a degree-9 polynomial in m - 1. The only caller
 * passes 1 + t with t in [0, 1], so x is always in [1, 2].
 */
static inline SIMD_VEC CFDNN_SIMD_FUNC(vlog)(SIMD_VEC x) {
    SIMD_VEC e = SIMD_FREXP_E(x);
    SIMD_VEC m = SIMD_FREXP_M(x); /* [0.5, 1) */
    /* m < sqrt(1/2): use 2m - 1 and e - 1, keeping the argument near zero. */
    SIMD_VEC sqrth = SIMD_SET1(0.707106781186547524f);
    e = SIMD_SELECT_LT(m, sqrth, SIMD_SUB(e, SIMD_SET1(1.0f)), e);
    m = SIMD_SELECT_LT(m, sqrth, SIMD_SUB(SIMD_ADD(m, m), SIMD_SET1(1.0f)),
                       SIMD_SUB(m, SIMD_SET1(1.0f)));
    SIMD_VEC z = SIMD_MUL(m, m);
    SIMD_VEC y = SIMD_SET1(7.0376836292e-2f);
    y = SIMD_FMA(y, m, SIMD_SET1(-1.1514610310e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(1.1676998740e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(-1.2420140846e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(1.4249322787e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(-1.6668057665e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(2.0000714765e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(-2.4999993993e-1f));
    y = SIMD_FMA(y, m, SIMD_SET1(3.3333331174e-1f));
    y = SIMD_MUL(SIMD_MUL(y, m), z);
    y = SIMD_FMA(e, SIMD_SET1(-2.12194440e-4f), y);
    y = SIMD_FMA(z, SIMD_SET1(-0.5f), y);
    y = SIMD_ADD(m, y);
    return SIMD_FMA(e, SIMD_SET1(0.693359375f), y);
}

/**
 * tanh(x): odd rational minimax p(x)/q(x) of degrees 13/6, clamped at +-7.905
 * where tanh rounds to +-1 in float32. Odd in x, so small arguments keep full
 * relative accuracy instead of cancelling as 1 - 2/(e^2x + 1) would.
 */
static inline SIMD_VEC CFDNN_SIMD_FUNC(vtanh)(SIMD_VEC x) {
    x = SIMD_MAX(SIMD_SET1(-7.90531110763549805f), SIMD_MIN(SIMD_SET1(7.90531110763549805f), x));
    SIMD_VEC x2 = SIMD_MUL(x, x);
    SIMD_VEC p = SIMD_SET1(-2.76076847742355e-16f);
    p = SIMD_FMA(p, x2, SIMD_SET1(2.00018790482477e-13f));
    p = SIMD_FMA(p, x2, SIMD_SET1(-8.60467152213735e-11f));
    p = SIMD_FMA(p, x2, SIMD_SET1(5.12229709037114e-08f));
    p = SIMD_FMA(p, x2, SIMD_SET1(1.48572235717979e-05f));
    p = SIMD_FMA(p, x2, SIMD_SET1(6.37261928875436e-04f));
    p = SIMD_FMA(p, x2, SIMD_SET1(4.89352455891786e-03f));
    p = SIMD_MUL(p, x);
    SIMD_VEC q = SIMD_SET1(1.19825839466702e-06f);
    q = SIMD_FMA(q, x2, SIMD_SET1(1.18534705686654e-04f));
    q = SIMD_FMA(q, x2, SIMD_SET1(2.26843463243900e-03f));
    q = SIMD_FMA(q, x2, SIMD_SET1(4.89352518554385e-03f));
    return SIMD_DIV(p, q);
}

/** sigmoid(x) = 1 / (1 + exp(-x)). */
static inline SIMD_VEC CFDNN_SIMD_FUNC(vsigmoid)(SIMD_VEC x) {
    SIMD_VEC one = SIMD_SET1(1.0f);
    return SIMD_DIV(one, SIMD_ADD(one, CFDNN_SIMD_FUNC(vexp)(SIMD_SUB(SIMD_SET1(0.0f), x))));
}

/**
 * softplus(x) = max(x, 0) + log1p(exp(-|x|)), the scalar kernel's overflow-safe
 * form. log1p(t) uses Goldberg's identity log(u) * t / (u - 1), u = 1 + t,
 * which stays accurate where u - 1 loses digits, and returns t itself where u
 * rounds to 1 -- so softplus of a very negative x is exp(x), not 0.
 */
static inline SIMD_VEC CFDNN_SIMD_FUNC(vsoftplus)(SIMD_VEC x) {
    SIMD_VEC one = SIMD_SET1(1.0f);
    SIMD_VEC zero = SIMD_SET1(0.0f);
    SIMD_VEC t = CFDNN_SIMD_FUNC(vexp)(SIMD_SUB(zero, SIMD_ABS(x)));
    SIMD_VEC u = SIMD_ADD(one, t);
    SIMD_VEC d = SIMD_SUB(u, one);
    SIMD_VEC safe = SIMD_SELECT_EQ(d, zero, one, d);
    SIMD_VEC l1p =
        SIMD_SELECT_EQ(d, zero, t, SIMD_DIV(SIMD_MUL(CFDNN_SIMD_FUNC(vlog)(u), t), safe));
    return SIMD_ADD(SIMD_MAX(zero, x), l1p);
}

/* In place over n contiguous values: vector body, shared scalar tail. */
static void CFDNN_SIMD_FUNC(activate)(cfd_nn_activation_t act, float param, float* v, size_t n) {
    size_t i = 0;
    switch (act) {
        case CFD_NN_ACT_TANH:
            for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
                SIMD_STORE(v + i, CFDNN_SIMD_FUNC(vtanh)(SIMD_LOAD(v + i)));
            }
            break;
        case CFD_NN_ACT_SIGMOID:
            for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
                SIMD_STORE(v + i, CFDNN_SIMD_FUNC(vsigmoid)(SIMD_LOAD(v + i)));
            }
            break;
        case CFD_NN_ACT_SOFTPLUS:
            for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
                SIMD_STORE(v + i, CFDNN_SIMD_FUNC(vsoftplus)(SIMD_LOAD(v + i)));
            }
            break;
        default:
            break; /* identity / ReLU / leaky ReLU: the scalar loop below is already cheap */
    }
    cfd_nn_apply_activation(act, param, v + i, n - i);
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

    CFDNN_SIMD_FUNC(activate)(l->activation, l->act_param, out, batch * no);
}
