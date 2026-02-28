/**
 * Outlet Boundary Conditions - Parameterized SIMD Implementation
 *
 * This header is included by both AVX2 and NEON outlet implementations.
 * Before including, define:
 *   BC_SIMD_STORE(dst, val)  - store SIMD vector to memory
 *   BC_SIMD_LOAD(src)        - load SIMD vector from memory
 *   BC_SIMD_WIDTH            - number of doubles per SIMD vector (4 or 2)
 *   BC_SIMD_MASK             - low-bit mask for rounding down (3 for AVX2, 1 for NEON)
 *   BC_SIMD_THRESHOLD        - min width for OMP on SIMD loops
 *   BC_OUTLET_FUNC_NAME      - function name to define
 *
 * 3D support: all functions accept (nz, stride_z). When nz <= 1,
 * z-face loops are skipped and x/y-face loops run for a single plane.
 */

#include "boundary_conditions_outlet_common.h"
#include "cfd/core/indexing.h"
#include <limits.h>

static inline int bc_outlet_simd_size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

cfd_status_t BC_OUTLET_FUNC_NAME(double* field, size_t nx, size_t ny,
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

    int j, i;

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
        case BC_OUTLET_CONVECTIVE:
            switch (config->edge) {
                case BC_EDGE_LEFT:
                    for (size_t k = 0; k < nz; k++) {
                        size_t base = k * stride_z;
                        #pragma omp parallel for schedule(static)
                        for (j = 0; j < bc_outlet_simd_size_to_int(ny); j++) {
                            size_t row = base + (size_t)j * nx;
                            field[row] = field[row + 1];
                        }
                    }
                    break;

                case BC_EDGE_RIGHT:
                    for (size_t k = 0; k < nz; k++) {
                        size_t base = k * stride_z;
                        #pragma omp parallel for schedule(static)
                        for (j = 0; j < bc_outlet_simd_size_to_int(ny); j++) {
                            size_t row = base + (size_t)j * nx;
                            field[row + nx - 1] = field[row + nx - 2];
                        }
                    }
                    break;

                case BC_EDGE_BOTTOM:
                    for (size_t k = 0; k < nz; k++) {
                        size_t base = k * stride_z;
                        double* dst = field + base;
                        double* src = field + base + nx;
                        size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

                        if (nx >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
                            #pragma omp parallel for schedule(static)
                            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
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
                    }
                    break;

                case BC_EDGE_TOP:
                    for (size_t k = 0; k < nz; k++) {
                        size_t base = k * stride_z;
                        double* dst = field + base + ((ny - 1) * nx);
                        double* src = field + base + ((ny - 2) * nx);
                        size_t simd_end = nx & ~(size_t)BC_SIMD_MASK;

                        if (nx >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
                            #pragma omp parallel for schedule(static)
                            for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
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
                    }
                    break;

                case BC_EDGE_FRONT:
                case BC_EDGE_BACK: {
                    /* Z-face outlet: copy entire xy-plane from adjacent interior */
                    if (nz <= 1) {
                        return CFD_ERROR_INVALID;
                    }
                    double* dst;
                    double* src;
                    if (config->edge == BC_EDGE_FRONT) {
                        dst = field + ((nz - 1) * stride_z);
                        src = field + ((nz - 2) * stride_z);
                    } else { /* BC_EDGE_BACK */
                        dst = field;
                        src = field + stride_z;
                    }
                    size_t plane_size = nx * ny;
                    size_t simd_end = plane_size & ~(size_t)BC_SIMD_MASK;

                    if (plane_size >= BC_SIMD_THRESHOLD && simd_end <= (size_t)INT_MAX) {
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < (int)simd_end; i += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + i, BC_SIMD_LOAD(src + i));
                        }
                    } else {
                        for (size_t ii = 0; ii < simd_end; ii += BC_SIMD_WIDTH) {
                            BC_SIMD_STORE(dst + ii, BC_SIMD_LOAD(src + ii));
                        }
                    }
                    for (size_t ii = simd_end; ii < plane_size; ii++) {
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

#undef BC_SIMD_STORE
#undef BC_SIMD_LOAD
#undef BC_SIMD_WIDTH
#undef BC_SIMD_MASK
#undef BC_SIMD_THRESHOLD
#undef BC_OUTLET_FUNC_NAME
