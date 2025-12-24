/**
 * @file boundary_copy_utils.h
 * @brief Shared boundary copying utilities for projection method solvers
 *
 * This header provides helper functions used across all projection solver
 * backends (CPU, AVX2, OpenMP) to copy boundary velocity values.
 */

#ifndef BOUNDARY_COPY_UTILS_H
#define BOUNDARY_COPY_UTILS_H

#include <stddef.h>

/**
 * Copy boundary values between velocity arrays
 *
 * Used to preserve caller-set boundary conditions (e.g., lid velocity for
 * cavity flow) during the projection method's predictor and corrector steps.
 *
 * @param dst_u  Destination u-velocity array
 * @param dst_v  Destination v-velocity array
 * @param src_u  Source u-velocity array
 * @param src_v  Source v-velocity array
 * @param nx     Grid width
 * @param ny     Grid height
 */
static inline void copy_boundary_velocities(double* dst_u, double* dst_v,
                                            const double* src_u, const double* src_v,
                                            size_t nx, size_t ny) {
    // Bottom and top boundaries (j = 0 and j = ny-1)
    for (size_t i = 0; i < nx; i++) {
        dst_u[i] = src_u[i];
        dst_v[i] = src_v[i];
        dst_u[(ny - 1) * nx + i] = src_u[(ny - 1) * nx + i];
        dst_v[(ny - 1) * nx + i] = src_v[(ny - 1) * nx + i];
    }
    // Left and right boundaries (i = 0 and i = nx-1)
    for (size_t j = 1; j < ny - 1; j++) {
        dst_u[j * nx] = src_u[j * nx];
        dst_v[j * nx] = src_v[j * nx];
        dst_u[j * nx + nx - 1] = src_u[j * nx + nx - 1];
        dst_v[j * nx + nx - 1] = src_v[j * nx + nx - 1];
    }
}

#endif /* BOUNDARY_COPY_UTILS_H */
