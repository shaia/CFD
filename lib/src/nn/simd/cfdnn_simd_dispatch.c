/**
 * @file cfdnn_simd_dispatch.c
 * @brief Runtime SIMD backend selection for `.cfdnn` inference.
 *
 * Mirrors lib/src/solvers/linear/simd/linear_solver_simd_dispatch.c: this file
 * holds only the dispatch, while the AVX2 and NEON kernels live in their own
 * sibling directories and share ../simd_template/cfdnn_kernels_simd_template.h.
 *
 * Selection is by the CPU actually running, not only by what was compiled:
 * a binary built with AVX2 enabled but run on a CPU without it gets NULL here,
 * so an explicit CFD_NN_BACKEND_SIMD request returns CFD_ERROR_UNSUPPORTED and
 * AUTO resolves to OMP or scalar -- the project's no-silent-fallback rule.
 * cfd_detect_simd_arch() caches its answer and is thread-safe, so this is
 * cheap enough to call at every context creation.
 */

#include "../cfdnn_internal.h"

#include "cfd/core/cpu_features.h"

const cfd_nn_backend_impl_t* cfd_nn_simd_impl(void) {
    switch (cfd_detect_simd_arch()) {
        case CFD_SIMD_AVX2:
            return cfd_nn_impl_avx2.dense ? &cfd_nn_impl_avx2 : NULL;
        case CFD_SIMD_NEON:
            return cfd_nn_impl_neon.dense ? &cfd_nn_impl_neon : NULL;
        default:
            return NULL;
    }
}
