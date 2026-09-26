/**
 * @file cfdnn_kernels_neon.c
 * @brief NEON backend for `.cfdnn` inference (4 samples per vector).
 *
 * Architecture macros plus the shared template in
 * ../simd_template/cfdnn_kernels_simd_template.h; see there for the kernel.
 *
 * Guarded on the architecture alone, not on CFD_ENABLE_OPENMP: the kernel is
 * single-threaded (see the matching note in avx2/cfdnn_kernels_avx2.c).
 * Elsewhere the table exports a NULL kernel and the dispatcher reports the
 * backend unavailable.
 */

#include "../cfdnn_internal.h"

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

#define SIMD_SUFFIX       neon
#define SIMD_VEC          float32x4_t
#define SIMD_WIDTH        4
#define SIMD_LOAD(p)      vld1q_f32(p)
#define SIMD_STORE(p, v)  vst1q_f32(p, v)
#define SIMD_SET1(x)      vdupq_n_f32(x)
#define SIMD_FMA(a, b, c) vfmaq_f32(c, a, b) /* c + a * b */

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
