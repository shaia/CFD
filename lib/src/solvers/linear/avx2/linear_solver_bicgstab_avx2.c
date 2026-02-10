/**
 * @file linear_solver_bicgstab_avx2.c
 * @brief BiCGSTAB linear solver - AVX2 + OpenMP backend
 *
 * This file provides AVX2-specific macro definitions and includes the
 * parameterized SIMD template. All algorithm logic is in the template header.
 *
 * SIMD optimizations:
 * - AVX2 intrinsics (4 doubles per vector, 256-bit)
 * - Fused multiply-add (FMA3)
 * - OpenMP parallelization
 */

#include "../linear_solver_internal.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <string.h>
#include <limits.h>

/* Platform detection for AVX2 + OpenMP
 * CFD_HAS_AVX2 is set by CMake when -DCFD_ENABLE_AVX2=ON.
 * This works consistently across all compilers (GCC, Clang, MSVC).
 */
#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define BICGSTAB_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>
#endif

#if defined(BICGSTAB_HAS_AVX2)

//=============================================================================
// AVX2-SPECIFIC MACRO DEFINITIONS
//=============================================================================

#define SIMD_SUFFIX avx2
#define SIMD_VEC __m256d
#define SIMD_WIDTH 4

#define SIMD_LOAD(ptr) _mm256_loadu_pd(ptr)
#define SIMD_STORE(ptr, vec) _mm256_storeu_pd(ptr, vec)
#define SIMD_SET1(val) _mm256_set1_pd(val)
#define SIMD_SETZERO() _mm256_setzero_pd()

#define SIMD_ADD(a, b) _mm256_add_pd(a, b)
#define SIMD_SUB(a, b) _mm256_sub_pd(a, b)
#define SIMD_MUL(a, b) _mm256_mul_pd(a, b)
#define SIMD_FMA(a, b, c) _mm256_fmadd_pd(a, b, c)

/* AVX2 horizontal sum (4 doubles -> 1 double) - MSVC-compatible */
static inline double simd_hsum_avx2(__m256d vec) {
    __m128d low = _mm256_castpd256_pd128(vec);
    __m128d high = _mm256_extractf128_pd(vec, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_hadd_pd(sum128, sum128);
    return _mm_cvtsd_f64(sum128);
}

#define SIMD_HSUM(vec) simd_hsum_avx2(vec)

//=============================================================================
// INCLUDE SIMD TEMPLATE
//=============================================================================

#include "../simd_template/linear_solver_bicgstab_simd_template.h"

//=============================================================================
// CLEANUP MACROS
//=============================================================================

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

#else  /* !BICGSTAB_HAS_AVX2 */

/* Stub for platforms without AVX2 */
poisson_solver_t* create_bicgstab_avx2_solver(void) {
    return NULL;
}

#endif  /* BICGSTAB_HAS_AVX2 */
