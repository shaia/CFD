/**
 * @file upwind_avx2.h
 * @brief AVX2 first-order upwind derivative, the vector form of
 *        stencil_upwind_diff() in cfd/math/stencils.h.
 *
 * Shared by the AVX2 Navier-Stokes and energy kernels.
 */
#ifndef CFD_NS_UPWIND_AVX2_H
#define CFD_NS_UPWIND_AVX2_H

#if defined(CFD_HAS_AVX2)
#include <immintrin.h>

/**
 * Per lane: (f_c - f_m) * inv_h where vel >= 0, otherwise (f_p - f_c) * inv_h.
 * A NaN velocity selects the forward difference, matching the scalar stencil.
 *
 * @param inv_h  1/h per lane (0.0 on an inactive axis)
 */
static inline __m256d upwind_deriv_avx2(__m256d vel, __m256d f_m, __m256d f_c, __m256d f_p,
                                        __m256d inv_h) {
    __m256d backward = _mm256_mul_pd(_mm256_sub_pd(f_c, f_m), inv_h);
    __m256d forward = _mm256_mul_pd(_mm256_sub_pd(f_p, f_c), inv_h);
    __m256d use_backward = _mm256_cmp_pd(vel, _mm256_setzero_pd(), _CMP_GE_OQ);
    return _mm256_blendv_pd(forward, backward, use_backward);
}
#endif /* CFD_HAS_AVX2 */

#endif /* CFD_NS_UPWIND_AVX2_H */
