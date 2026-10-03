/**
 * @file cfdnn_kernels_neon.c
 * @brief NEON backend for `.cfdnn` inference (4 samples per vector).
 *
 * Architecture macros plus the shared template in
 * ../simd_template/cfdnn_kernels_simd_template.h; see there for the kernel.
 *
 * Guarded on the architecture alone, not on CFD_ENABLE_OPENMP: the kernel is
 * single-threaded (see the matching note in avx2/cfdnn_kernels_avx2.c).
 *
 * AArch64 only. vfmaq_f32 is an AArch64 intrinsic, so a 32-bit ARM build with
 * NEON (__ARM_NEON alone) takes the NULL-kernel branch instead of failing to
 * compile; the dispatcher then reports the backend unavailable, as it does on
 * every non-ARM platform. The build targets arm64 only, so this costs nothing.
 */

#include "../cfdnn_internal.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#define SIMD_SUFFIX       neon
#define SIMD_VEC          float32x4_t
#define SIMD_WIDTH        4
#define SIMD_LOAD(p)      vld1q_f32(p)
#define SIMD_STORE(p, v)  vst1q_f32(p, v)
#define SIMD_SET1(x)      vdupq_n_f32(x)
#define SIMD_FMA(a, b, c) vfmaq_f32(c, a, b) /* c + a * b */
#define SIMD_ADD(a, b)    vaddq_f32(a, b)
#define SIMD_SUB(a, b)    vsubq_f32(a, b)
#define SIMD_MUL(a, b)    vmulq_f32(a, b)
#define SIMD_DIV(a, b)    vdivq_f32(a, b)
/* fmin/fmax on AArch64 propagate NaN from either operand, which satisfies the
 * template's contract (NaN in the second operand propagates). */
#define SIMD_MIN(a, b)             vminq_f32(a, b)
#define SIMD_MAX(a, b)             vmaxq_f32(a, b)
#define SIMD_ABS(a)                vabsq_f32(a)
#define SIMD_FLOOR(a)              vrndmq_f32(a)
#define SIMD_SELECT_LT(a, b, t, f) vbslq_f32(vcltq_f32(a, b), t, f)
#define SIMD_SELECT_EQ(a, b, t, f) vbslq_f32(vceqq_f32(a, b), t, f)
#define SIMD_POW2N(n)              cfdnn_pow2n_neon(n)
#define SIMD_FREXP_E(x)            cfdnn_frexp_e_neon(x)
#define SIMD_FREXP_M(x)            cfdnn_frexp_m_neon(x)

/* 2^n for integral n: (n + 127) placed in the exponent field. */
static inline float32x4_t cfdnn_pow2n_neon(float32x4_t n) {
    int32x4_t e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    return vreinterpretq_f32_s32(vshlq_n_s32(e, 23));
}

/* x = m * 2^e with m in [0.5, 1), for normal x > 0. */
static inline float32x4_t cfdnn_frexp_e_neon(float32x4_t x) {
    int32x4_t bits = vreinterpretq_s32_f32(x);
    int32x4_t e = vandq_s32(vshrq_n_s32(bits, 23), vdupq_n_s32(0xFF));
    return vcvtq_f32_s32(vsubq_s32(e, vdupq_n_s32(126)));
}

static inline float32x4_t cfdnn_frexp_m_neon(float32x4_t x) {
    int32x4_t bits = vandq_s32(vreinterpretq_s32_f32(x), vdupq_n_s32(0x007FFFFF));
    return vreinterpretq_f32_s32(vorrq_s32(bits, vdupq_n_s32(0x3F000000)));
}

#include "../simd_template/cfdnn_kernels_simd_template.h"

const cfd_nn_backend_impl_t cfd_nn_impl_neon = {
    "neon",
    dense_neon,
    SIMD_WIDTH,
};

#else /* not ARM */

const cfd_nn_backend_impl_t cfd_nn_impl_neon = {
    "neon",
    NULL, /* not compiled in: reported, never silently substituted */
    0,
};

#endif
