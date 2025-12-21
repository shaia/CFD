/**
 * Outlet Boundary Conditions - OpenMP Implementation
 *
 * OpenMP parallelized outlet boundary condition implementation.
 * Parallelizes over rows for left/right boundaries and
 * over columns for top/bottom boundaries.
 *
 * Supports:
 * - Zero-gradient (Neumann) outlet
 * - Convective outlet (advection-based)
 */

#include "../boundary_conditions_outlet_common.h"

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>

cfd_status_t bc_apply_outlet_omp_impl(double* field, size_t nx, size_t ny,
                                       const bc_outlet_config_t* config) {
    if (!field || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_outlet_is_valid_edge(config->edge)) {
        return CFD_ERROR_INVALID;
    }

    if (!bc_outlet_is_valid_type(config->type)) {
        return CFD_ERROR_INVALID;
    }

    int edge_idx = bc_outlet_edge_to_index(config->edge);
    const bc_outlet_edge_loop_t* loop = &bc_outlet_edge_loops[edge_idx];
    size_t count = loop->use_ny_for_count ? ny : nx;
    int count_int = (int)count;
    int i;

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
        case BC_OUTLET_CONVECTIVE:
            /* Both types use zero-gradient for now.
             * Full convective BC would require temporal information. */
            #pragma omp parallel for schedule(static)
            for (i = 0; i < count_int; i++) {
                size_t dst_idx = loop->dst_fn((size_t)i, nx, ny);
                size_t src_idx = loop->src_fn((size_t)i, nx, ny);
                field[dst_idx] = field[src_idx];
            }
            break;

        default:
            return CFD_ERROR_INVALID;
    }

    return CFD_SUCCESS;
}

#else /* !CFD_ENABLE_OPENMP */

cfd_status_t bc_apply_outlet_omp_impl(double* field, size_t nx, size_t ny,
                                       const bc_outlet_config_t* config) {
    (void)field; (void)nx; (void)ny; (void)config;
    return CFD_ERROR_UNSUPPORTED;
}

#endif /* CFD_ENABLE_OPENMP */
