/**
 * Outlet Boundary Conditions - Parameterized SIMD Implementation
 *
 * This header is included by both AVX2 and NEON outlet implementations.
 * Before including, define:
 *   BC_SIMD_STORE(dst, val)  - store SIMD vector to memory
 *   BC_SIMD_LOAD(src)        - load SIMD vector from memory
 *   BC_SIMD_WIDTH            - number of doubles per SIMD vector (4 or 2)
 *   BC_SIMD_MASK             - alignment mask (~3 for AVX2, ~1 for NEON)
 *   BC_SIMD_THRESHOLD        - min width for OMP on SIMD loops
 *   BC_OUTLET_FUNC_NAME      - function name to define
 */

#include "boundary_conditions_outlet_common.h"
#include <limits.h>

static inline int bc_outlet_simd_size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

cfd_status_t BC_OUTLET_FUNC_NAME(double* field, size_t nx, size_t ny,
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

    int j, i;

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
        case BC_OUTLET_CONVECTIVE:
            switch (config->edge) {
                case BC_EDGE_LEFT:
                    #pragma omp parallel for schedule(static)
                    for (j = 0; j < bc_outlet_simd_size_to_int(ny); j++) {
                        size_t row = (size_t)j * nx;
                        field[row] = field[row + 1];
                    }
                    break;

                case BC_EDGE_RIGHT:
                    #pragma omp parallel for schedule(static)
                    for (j = 0; j < bc_outlet_simd_size_to_int(ny); j++) {
                        size_t row = (size_t)j * nx;
                        field[row + nx - 1] = field[row + nx - 2];
                    }
                    break;

                case BC_EDGE_BOTTOM: {
                    double* dst = field;
                    double* src = field + nx;
                    size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

                    if (nx >= BC_SIMD_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < bc_outlet_simd_size_to_int(simd_end); i += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + i, BC_SIMD_LOAD(src + i));
                        }
                    } else {
                        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + ii, BC_SIMD_LOAD(src + ii));
                        }
                    }
                    for (size_t ii = simd_end; ii < nx; ii++) {
                        dst[ii] = src[ii];
                    }
                    break;
                }

                case BC_EDGE_TOP: {
                    double* dst = field + ((ny - 1) * nx);
                    double* src = field + ((ny - 2) * nx);
                    size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

                    if (nx >= BC_SIMD_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < bc_outlet_simd_size_to_int(simd_end); i += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + i, BC_SIMD_LOAD(src + i));
                        }
                    } else {
                        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + ii, BC_SIMD_LOAD(src + ii));
                        }
                    }
                    for (size_t ii = simd_end; ii < nx; ii++) {
                        dst[ii] = src[ii];
                    }
                    break;
                }

                default:
                    return CFD_ERROR_INVALID;
            }
            break;

        default:
            return CFD_ERROR_INVALID;
    }

    return CFD_SUCCESS;
}
