/**
 * Outlet Boundary Conditions - AVX2 + OpenMP Implementation
 *
 * Combined AVX2 SIMD and OpenMP parallelized outlet boundary condition implementation.
 * Uses OpenMP for thread-level parallelism across rows.
 * Uses AVX2 SIMD for instruction-level parallelism on contiguous memory.
 *
 * Supports:
 * - Zero-gradient (Neumann) outlet
 * - Convective outlet (advection-based)
 */

#include "../boundary_conditions_outlet_common.h"

#if defined(CFD_HAS_AVX2) && defined(CFD_ENABLE_OPENMP)
#define BC_HAS_AVX2 1
#include <immintrin.h>
#include <omp.h>
#include <limits.h>
#endif

#if defined(BC_HAS_AVX2)

#define BC_AVX2_THRESHOLD 256

static inline int size_to_int(size_t sz) {
    return (sz > (size_t)INT_MAX) ? INT_MAX : (int)sz;
}

cfd_status_t bc_apply_outlet_avx2_impl(double* field, size_t nx, size_t ny,
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

    /* For left/right edges, use OMP over rows.
     * For top/bottom edges, use AVX2 on contiguous memory. */
    int j, i;

    switch (config->type) {
        case BC_OUTLET_ZERO_GRADIENT:
        case BC_OUTLET_CONVECTIVE:
            /* Both types use zero-gradient for now. */
            switch (config->edge) {
                case BC_EDGE_LEFT:
                    #pragma omp parallel for schedule(static)
                    for (j = 0; j < size_to_int(ny); j++) {
                        size_t row = (size_t)j * nx;
                        field[row] = field[row + 1];
                    }
                    break;

                case BC_EDGE_RIGHT:
                    #pragma omp parallel for schedule(static)
                    for (j = 0; j < size_to_int(ny); j++) {
                        size_t row = (size_t)j * nx;
                        field[row + nx - 1] = field[row + nx - 2];
                    }
                    break;

                case BC_EDGE_BOTTOM: {
                    /* Bottom: row 0 = row 1 */
                    double* dst = field;
                    double* src = field + nx;
                    size_t simd_end = nx & ~(size_t)3;

                    if (nx >= BC_AVX2_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < size_to_int(simd_end); i += 4) {
                            _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
                        }
                    } else {
                        for (size_t ii = 0; ii < simd_end; ii += 4) {
                            _mm256_storeu_pd(dst + ii, _mm256_loadu_pd(src + ii));
                        }
                    }
                    /* Remainder */
                    for (size_t ii = simd_end; ii < nx; ii++) {
                        dst[ii] = src[ii];
                    }
                    break;
                }

                case BC_EDGE_TOP: {
                    /* Top: row ny-1 = row ny-2 */
                    double* dst = field + ((ny - 1) * nx);
                    double* src = field + ((ny - 2) * nx);
                    size_t simd_end = nx & ~(size_t)3;

                    if (nx >= BC_AVX2_THRESHOLD) {
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < size_to_int(simd_end); i += 4) {
                            _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
                        }
                    } else {
                        for (size_t ii = 0; ii < simd_end; ii += 4) {
                            _mm256_storeu_pd(dst + ii, _mm256_loadu_pd(src + ii));
                        }
                    }
                    /* Remainder */
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

#else /* !BC_HAS_AVX2 */

cfd_status_t bc_apply_outlet_avx2_impl(double* field, size_t nx, size_t ny,
                                            const bc_outlet_config_t* config) {
    (void)field; (void)nx; (void)ny; (void)config;
    return CFD_ERROR_UNSUPPORTED;
}

#endif /* BC_HAS_AVX2 */
