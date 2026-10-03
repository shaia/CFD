/**
 * @file cfdnn_kernels_avx2.c
 * @brief AVX2 backend for `.cfdnn` inference (8 samples per vector).
 *
 * Architecture macros plus the shared template in
 * ../simd_template/cfdnn_kernels_simd_template.h; see there for the kernel.
 *
 * Guarded on CFD_HAS_AVX2 alone. Some linear-solver AVX2 files also require
 * CFD_ENABLE_OPENMP, so a build without OpenMP silently loses their
 * vectorisation; this kernel is single-threaded and has no reason to inherit
 * that. Without AVX2 the table exports a NULL kernel and the dispatcher
 * reports the backend unavailable.
 */

#include "../cfdnn_internal.h"

#if defined(CFD_HAS_AVX2)

#include <immintrin.h>

#define SIMD_SUFFIX       avx2
#define SIMD_VEC          __m256
#define SIMD_WIDTH        8
#define SIMD_LOAD(p)      _mm256_loadu_ps(p)
#define SIMD_STORE(p, v)  _mm256_storeu_ps(p, v)
#define SIMD_SET1(x)      _mm256_set1_ps(x)
#define SIMD_FMA(a, b, c) _mm256_fmadd_ps(a, b, c)
#define SIMD_ADD(a, b)    _mm256_add_ps(a, b)
#define SIMD_SUB(a, b)    _mm256_sub_ps(a, b)
#define SIMD_MUL(a, b)    _mm256_mul_ps(a, b)
#define SIMD_DIV(a, b)    _mm256_div_ps(a, b)
/* minps/maxps return the SECOND operand when either is NaN: the template's
 * NaN-propagation contract. */
#define SIMD_MIN(a, b)             _mm256_min_ps(a, b)
#define SIMD_MAX(a, b)             _mm256_max_ps(a, b)
#define SIMD_ABS(a)                _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#define SIMD_FLOOR(a)              _mm256_floor_ps(a)
#define SIMD_SELECT_LT(a, b, t, f) _mm256_blendv_ps(f, t, _mm256_cmp_ps(a, b, _CMP_LT_OQ))
#define SIMD_SELECT_EQ(a, b, t, f) _mm256_blendv_ps(f, t, _mm256_cmp_ps(a, b, _CMP_EQ_OQ))
#define SIMD_POW2N(n)              cfdnn_pow2n_avx2(n)
#define SIMD_FREXP_E(x)            cfdnn_frexp_e_avx2(x)
#define SIMD_FREXP_M(x)            cfdnn_frexp_m_avx2(x)

/* 2^n for integral n: (n + 127) placed in the exponent field. */
static inline __m256 cfdnn_pow2n_avx2(__m256 n) {
    __m256i e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
    return _mm256_castsi256_ps(_mm256_slli_epi32(e, 23));
}

/* x = m * 2^e with m in [0.5, 1), for normal x > 0. */
static inline __m256 cfdnn_frexp_e_avx2(__m256 x) {
    __m256i bits = _mm256_castps_si256(x);
    __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
    return _mm256_cvtepi32_ps(_mm256_sub_epi32(e, _mm256_set1_epi32(126)));
}

static inline __m256 cfdnn_frexp_m_avx2(__m256 x) {
    __m256i bits = _mm256_castps_si256(x);
    bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF));
    return _mm256_castsi256_ps(_mm256_or_si256(bits, _mm256_set1_epi32(0x3F000000)));
}

#include "../simd_template/cfdnn_kernels_simd_template.h"

const cfd_nn_backend_impl_t cfd_nn_impl_avx2 = {
    "avx2",
    dense_avx2,
    SIMD_WIDTH,
};

#else /* !CFD_HAS_AVX2 */

const cfd_nn_backend_impl_t cfd_nn_impl_avx2 = {
    "avx2",
    NULL, /* not compiled in: reported, never silently substituted */
    0,
};

#endif /* CFD_HAS_AVX2 */
