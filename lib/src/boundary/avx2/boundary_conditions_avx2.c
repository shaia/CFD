/**
 * Boundary Conditions - AVX2 + OpenMP Implementation
 *
 * Delegates to parameterized SIMD template with AVX2 intrinsics.
 * Neumann, Periodic, and Dirichlet are generated from the shared SIMD template.
 * Inlet delegates to scalar (no SIMD benefit for 1D boundary loops).
 * Outlet is in boundary_conditions_outlet_avx2.c.
 */

#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#include <immintrin.h>
#include <omp.h>

#define BC_SIMD_STORE(dst, val)  _mm256_storeu_pd((dst), (val))
#define BC_SIMD_LOAD(src)        _mm256_loadu_pd((src))
#define BC_SIMD_BROADCAST(val)   _mm256_set1_pd((val))
#define BC_SIMD_VEC_TYPE         __m256d
#define BC_SIMD_WIDTH            4
#define BC_SIMD_MASK             3
#define BC_SIMD_THRESHOLD        256
#define BC_SIMD_FUNC_PREFIX      avx2

#include "../boundary_conditions_simd_impl.h"

/* AVX2 backend implementation table
 * Note: Inlet delegates to scalar (no SIMD benefit for 1D boundary loops)
 * Note: bc_apply_outlet_avx2_impl is defined in boundary_conditions_outlet_avx2.c */
const bc_backend_impl_t bc_impl_avx2 = {
    .apply_neumann = bc_apply_neumann_avx2_impl,
    .apply_periodic = bc_apply_periodic_avx2_impl,
    .apply_dirichlet = bc_apply_dirichlet_avx2_impl,
    .apply_inlet = bc_apply_inlet_scalar_impl,
    .apply_outlet = bc_apply_outlet_avx2_impl,
    .apply_symmetry = NULL  /* Falls back to scalar */
};

#else /* !CFD_HAS_AVX2 || !CFD_ENABLE_OPENMP */

#include "../boundary_conditions_internal.h"

/* AVX2 not available - provide empty table */
const bc_backend_impl_t bc_impl_avx2 = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL,
    .apply_outlet = NULL,
    .apply_symmetry = NULL
};

#endif /* CFD_HAS_AVX2 && CFD_ENABLE_OPENMP */
