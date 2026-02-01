/**
 * Boundary Conditions - ARM NEON + OpenMP Implementation
 *
 * Delegates to parameterized SIMD template with NEON intrinsics.
 * Neumann, Periodic, and Dirichlet are generated from the shared SIMD template.
 * Inlet delegates to scalar (no SIMD benefit for 1D boundary loops).
 * Outlet is in boundary_conditions_outlet_neon.c.
 */

#if (defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(CFD_ENABLE_OPENMP)
#include <arm_neon.h>
#include <omp.h>

#define BC_SIMD_STORE(dst, val)  vst1q_f64((dst), (val))
#define BC_SIMD_LOAD(src)        vld1q_f64((src))
#define BC_SIMD_BROADCAST(val)   vdupq_n_f64((val))
#define BC_SIMD_VEC_TYPE         float64x2_t
#define BC_SIMD_WIDTH            2
#define BC_SIMD_MASK             1
#define BC_SIMD_THRESHOLD        256
#define BC_SIMD_FUNC_PREFIX      neon

#include "../boundary_conditions_simd_impl.h"

/* NEON backend implementation table
 * Note: Inlet delegates to scalar (no SIMD benefit for 1D boundary loops)
 * Note: bc_apply_outlet_neon_impl is defined in boundary_conditions_outlet_neon.c */
const bc_backend_impl_t bc_impl_neon = {
    .apply_neumann = bc_apply_neumann_neon_impl,
    .apply_periodic = bc_apply_periodic_neon_impl,
    .apply_dirichlet = bc_apply_dirichlet_neon_impl,
    .apply_inlet = bc_apply_inlet_scalar_impl,
    .apply_outlet = bc_apply_outlet_neon_impl,
    .apply_symmetry = NULL  /* Falls back to scalar */
};

#else /* !NEON || !CFD_ENABLE_OPENMP */

#include "../boundary_conditions_internal.h"

/* NEON not available - provide empty table */
const bc_backend_impl_t bc_impl_neon = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL,
    .apply_outlet = NULL,
    .apply_symmetry = NULL
};

#endif /* NEON && CFD_ENABLE_OPENMP */
