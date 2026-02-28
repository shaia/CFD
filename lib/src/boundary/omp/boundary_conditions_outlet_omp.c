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
 * - 3D z-face outlets (FRONT/BACK edges)
 */

#include "../boundary_conditions_outlet_common.h"

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>

cfd_status_t bc_apply_outlet_omp_impl(double* field, size_t nx, size_t ny,
                                       size_t nz, size_t stride_z,
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

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
        case BC_OUTLET_CONVECTIVE:
            /* Both types use zero-gradient for now.
             * Full convective BC would require temporal information. */
            if (loop->is_z_face) {
                /* Z-face outlet (FRONT or BACK): copy entire xy-plane */
                if (nz <= 1) {
                    return CFD_ERROR_INVALID;
                }
                size_t dst_plane, src_plane;
                if (config->edge == BC_EDGE_FRONT) {
                    dst_plane = (nz - 1) * stride_z;
                    src_plane = (nz - 2) * stride_z;
                } else { /* BC_EDGE_BACK */
                    dst_plane = 0;
                    src_plane = stride_z;
                }
                int plane_size_int = (int)(nx * ny);
                int idx;
                #pragma omp parallel for schedule(static)
                for (idx = 0; idx < plane_size_int; idx++) {
                    field[dst_plane + (size_t)idx] = field[src_plane + (size_t)idx];
                }
            } else {
                /* X/Y-face outlet: loop over each z-plane */
                size_t count = loop->use_ny_for_count ? ny : nx;
                int count_int = (int)count;
                for (size_t k = 0; k < nz; k++) {
                    size_t base = k * stride_z;
                    int i;
                    #pragma omp parallel for schedule(static)
                    for (i = 0; i < count_int; i++) {
                        size_t dst_idx = base + loop->dst_fn((size_t)i, nx, ny);
                        size_t src_idx = base + loop->src_fn((size_t)i, nx, ny);
                        field[dst_idx] = field[src_idx];
                    }
                }
            }
            break;

        default:
            return CFD_ERROR_INVALID;
    }

    return CFD_SUCCESS;
}

#else /* !CFD_ENABLE_OPENMP */

cfd_status_t bc_apply_outlet_omp_impl(double* field, size_t nx, size_t ny,
                                       size_t nz, size_t stride_z,
                                       const bc_outlet_config_t* config) {
    (void)field; (void)nx; (void)ny; (void)nz; (void)stride_z; (void)config;
    return CFD_ERROR_UNSUPPORTED;
}

#endif /* CFD_ENABLE_OPENMP */
