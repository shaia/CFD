/**
 * Outlet Boundary Conditions - AVX2 + OpenMP Implementation
 *
 * Delegates to parameterized SIMD template with AVX2 intrinsics.
 */

#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>

#define BC_SIMD_STORE(dst, val) _mm256_storeu_pd((dst), (val))
#define BC_SIMD_LOAD(src)       _mm256_loadu_pd((src))
#define BC_SIMD_WIDTH           4
#define BC_SIMD_MASK            3
#define BC_SIMD_THRESHOLD       256
#define BC_OUTLET_FUNC_NAME     bc_apply_outlet_avx2_impl

#include "../boundary_conditions_outlet_simd.h"

#else /* !BC_HAS_AVX2 */

#include "../boundary_conditions_outlet_common.h"

cfd_status_t bc_apply_outlet_avx2_impl(double* field, size_t nx, size_t ny,
                                            const bc_outlet_config_t* config) {
    (void)field; (void)nx; (void)ny; (void)config;
    return CFD_ERROR_UNSUPPORTED;
}

#endif /* BC_HAS_AVX2 */
