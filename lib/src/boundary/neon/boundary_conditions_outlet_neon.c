/**
 * Outlet Boundary Conditions - ARM NEON Implementation
 *
 * Delegates to parameterized SIMD template with NEON intrinsics.
 */

#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_NEON 1
#include <arm_neon.h>
#include <omp.h>

#define BC_SIMD_STORE(dst, val) vst1q_f64((dst), (val))
#define BC_SIMD_LOAD(src)       vld1q_f64((src))
#define BC_SIMD_WIDTH           2
#define BC_SIMD_MASK            1
#define BC_SIMD_THRESHOLD       256
#define BC_OUTLET_FUNC_NAME     bc_apply_outlet_neon_impl

#include "../boundary_conditions_outlet_simd.h"

#else /* !BC_HAS_NEON */

#include "../boundary_conditions_outlet_common.h"

cfd_status_t bc_apply_outlet_neon_impl(double* field, size_t nx, size_t ny,
                                        const bc_outlet_config_t* config) {
    (void)field; (void)nx; (void)ny; (void)config;
    return CFD_ERROR_UNSUPPORTED;
}

#endif /* BC_HAS_NEON */
