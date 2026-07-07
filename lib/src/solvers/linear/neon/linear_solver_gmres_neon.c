/**
 * @file linear_solver_gmres_neon.c
 * @brief Restarted GMRES(m) linear solver - ARM NEON + OpenMP backend
 *
 * Provides NEON-specific macro definitions and includes the parameterized SIMD
 * template. All algorithm logic lives in the template header. Only the O(n)
 * vector primitives are vectorized; dense Givens/Hessenberg work stays scalar.
 */

#include "../linear_solver_internal.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <string.h>
#include <limits.h>

#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define GMRES_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>
#endif

#if defined(GMRES_HAS_NEON)

//=============================================================================
// NEON-SPECIFIC MACRO DEFINITIONS
//=============================================================================

#define SIMD_SUFFIX neon
#define SIMD_VEC float64x2_t
#define SIMD_WIDTH 2

#define SIMD_LOAD(ptr) vld1q_f64(ptr)
#define SIMD_STORE(ptr, vec) vst1q_f64(ptr, vec)
#define SIMD_SET1(val) vdupq_n_f64(val)
#define SIMD_SETZERO() vdupq_n_f64(0.0)

#define SIMD_ADD(a, b) vaddq_f64(a, b)
#define SIMD_SUB(a, b) vsubq_f64(a, b)
#define SIMD_MUL(a, b) vmulq_f64(a, b)
#define SIMD_FMA(a, b, c) vfmaq_f64(c, a, b)

/* NEON horizontal sum (2 doubles -> 1 double) */
#define SIMD_HSUM(vec) (vgetq_lane_f64(vec, 0) + vgetq_lane_f64(vec, 1))

#include "../simd_template/linear_solver_gmres_simd_template.h"

#undef SIMD_SUFFIX
#undef SIMD_VEC
#undef SIMD_WIDTH
#undef SIMD_LOAD
#undef SIMD_STORE
#undef SIMD_SET1
#undef SIMD_SETZERO
#undef SIMD_ADD
#undef SIMD_SUB
#undef SIMD_MUL
#undef SIMD_FMA
#undef SIMD_HSUM

#else  /* !GMRES_HAS_NEON */

/* Stub for platforms without NEON */
poisson_solver_t* create_gmres_neon_solver(void) {
    return NULL;
}

#endif  /* GMRES_HAS_NEON */
