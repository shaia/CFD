/**
 * Inlet Boundary Conditions - AVX2 + OpenMP Implementation
 *
 * OpenMP parallelized inlet velocity boundary condition implementation for x86-64.
 * Uses OpenMP for thread-level parallelism over boundary points.
 *
 * Supports:
 * - Uniform velocity profile
 * - Parabolic profile (fully-developed flow)
 * - Custom user-defined profiles via callback
 * - Velocity specification by components, magnitude+direction, or mass flow rate
 */

#include "../boundary_conditions_inlet_common.h"

#if (defined(__AVX2__) || defined(CFD_HAS_AVX2)) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_AVX2_OMP 1
#include <omp.h>
#include <limits.h>
#endif

#if defined(BC_HAS_AVX2_OMP)

static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

cfd_status_t bc_apply_inlet_avx2_impl(double* u, double* v, size_t nx, size_t ny,
                                       const bc_inlet_config_t* config) {
    if (!u || !v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_inlet_is_valid_edge(config->edge)) {
        return CFD_ERROR_INVALID;
    }

    int edge_idx = bc_inlet_edge_to_index(config->edge);
    const bc_inlet_edge_loop_t* loop = &bc_inlet_edge_loops[edge_idx];
    size_t count = loop->use_ny_for_count ? ny : nx;
    double pos_denom = (count > 1) ? (double)(count - 1) : 1.0;
    bc_inlet_idx_func_t idx_fn = loop->idx_fn;

    int idx;
#pragma omp parallel for schedule(static)
    for (idx = 0; idx < size_to_int(count); idx++) {
        double position = (count > 1) ? (double)idx / pos_denom : 0.5;
        double u_val, v_val;
        bc_inlet_compute_velocity(config, position, &u_val, &v_val);

        size_t arr_idx = idx_fn((size_t)idx, nx, ny);
        u[arr_idx] = u_val;
        v[arr_idx] = v_val;
    }

    return CFD_SUCCESS;
}

#endif /* BC_HAS_AVX2_OMP */
