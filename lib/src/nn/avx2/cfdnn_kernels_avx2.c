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
