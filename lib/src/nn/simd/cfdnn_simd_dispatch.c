/**
 * @file cfdnn_simd_dispatch.c
 * @brief Runtime SIMD backend selection for `.cfdnn` inference.
 *
 * Mirrors lib/src/solvers/linear/simd/linear_solver_simd_dispatch.c: this file
 * holds only the dispatch, while the actual AVX2 and NEON kernels live in
 * their own sibling directories.
 *
 * No SIMD kernels are implemented yet, so the table exports a NULL kernel and
 * the backend reports itself unavailable. That is deliberate rather than
 * provisional: an explicit CFD_NN_BACKEND_SIMD request returns
 * CFD_ERROR_UNSUPPORTED and AUTO resolves to OMP or scalar, per the project's
 * no-silent-fallback rule. Adding the kernels is a later change that only has
 * to populate this table.
 *
 * When they land, the vectorisation axis is fixed by the contract in
 * cfdnn_internal.h: each SIMD lane carries a distinct SAMPLE, never a distinct
 * input feature. Reducing across lanes would make the accumulation order
 * depend on vector width and break cross-backend reproducibility.
 */

#include "../cfdnn_internal.h"

bool cfd_nn_simd_available(void) {
    return cfd_nn_impl_simd.dense != NULL;
}

const char* cfd_nn_simd_arch_name(void) {
    return cfd_nn_impl_simd.dense ? "simd" : "none";
}

const cfd_nn_backend_impl_t cfd_nn_impl_simd = {
    "simd",
    NULL, /* not implemented yet; reported as unavailable */
};
