/**
 * Boundary Conditions - OpenMP Implementation
 *
 * OpenMP parallelized boundary condition implementations.
 * Neumann, Periodic, and Dirichlet are generated from the shared template.
 */

#ifdef CFD_ENABLE_OPENMP

#include <omp.h>

#define BC_CORE_FUNC_PREFIX omp
#define BC_CORE_USE_OMP 1
#include "../boundary_conditions_core_impl.h"

/* OpenMP backend implementation table
 * Note: Inlet delegates to scalar (no SIMD benefit for 1D boundary loops)
 * Note: bc_apply_outlet_omp_impl is defined in boundary_conditions_outlet_omp.c */
const bc_backend_impl_t bc_impl_omp = {
    .apply_neumann = bc_apply_neumann_omp_impl,
    .apply_periodic = bc_apply_periodic_omp_impl,
    .apply_dirichlet = bc_apply_dirichlet_omp_impl,
    .apply_inlet = bc_apply_inlet_scalar_impl,
    .apply_outlet = bc_apply_outlet_omp_impl,
    .apply_symmetry = NULL  /* Falls back to scalar */
};

#else /* !CFD_ENABLE_OPENMP */

#include "../boundary_conditions_internal.h"

/* OpenMP not available - provide empty table */
const bc_backend_impl_t bc_impl_omp = {
    .apply_neumann = NULL,
    .apply_periodic = NULL,
    .apply_dirichlet = NULL,
    .apply_inlet = NULL,
    .apply_outlet = NULL,
    .apply_symmetry = NULL
};

#endif /* CFD_ENABLE_OPENMP */
