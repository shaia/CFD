/**
 * @file boundary_copy_utils.h
 * @brief Shared boundary copying utilities for projection method solvers
 *
 * This header provides helper functions used across all projection solver
 * backends (CPU, AVX2, OpenMP) to copy boundary velocity values.
 *
 * Include from backend subdirectories as: #include "../boundary_copy_utils.h"
 */

#ifndef BOUNDARY_COPY_UTILS_H
#define BOUNDARY_COPY_UTILS_H

#include "cfd/core/indexing.h"
#include <stddef.h>

/**
 * Copy boundary values between velocity arrays
 *
 * Used to preserve caller-set boundary conditions (e.g., lid velocity for
 * cavity flow) during the projection method's predictor and corrector steps.
 *
 * Copies ALL four boundaries (top, bottom, left, right). For flows with
 * outlet boundaries (Neumann BCs), use copy_dirichlet_boundaries instead.
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
        dst_u[IDX_2D(i, ny - 1, nx)] = src_u[IDX_2D(i, ny - 1, nx)];
        dst_v[IDX_2D(i, ny - 1, nx)] = src_v[IDX_2D(i, ny - 1, nx)];
    }
    // Left and right boundaries (i = 0 and i = nx-1)
    for (size_t j = 1; j < ny - 1; j++) {
        dst_u[IDX_2D(0, j, nx)] = src_u[IDX_2D(0, j, nx)];
        dst_v[IDX_2D(0, j, nx)] = src_v[IDX_2D(0, j, nx)];
        dst_u[IDX_2D(nx - 1, j, nx)] = src_u[IDX_2D(nx - 1, j, nx)];
        dst_v[IDX_2D(nx - 1, j, nx)] = src_v[IDX_2D(nx - 1, j, nx)];
    }
}

/**
 * Copy only Dirichlet boundary values (walls + inlet), skip outlet
 *
 * Used for flows with outlet boundaries (e.g., Poiseuille channel flow).
 * Copies top, bottom, and left boundaries, but NOT the right boundary
 * to allow the pressure correction to update the outlet velocity.
 *
 * @param dst_u  Destination u-velocity array
 * @param dst_v  Destination v-velocity array
 * @param src_u  Source u-velocity array
 * @param src_v  Source v-velocity array
 * @param nx     Grid width
 * @param ny     Grid height
 */
static inline void copy_dirichlet_boundaries(double* dst_u, double* dst_v,
                                             const double* src_u, const double* src_v,
                                             size_t nx, size_t ny) {
    // Bottom and top boundaries (j = 0 and j = ny-1)
    for (size_t i = 0; i < nx; i++) {
        dst_u[i] = src_u[i];
        dst_v[i] = src_v[i];
        dst_u[IDX_2D(i, ny - 1, nx)] = src_u[IDX_2D(i, ny - 1, nx)];
        dst_v[IDX_2D(i, ny - 1, nx)] = src_v[IDX_2D(i, ny - 1, nx)];
    }
    // Left boundary only (i = 0) - skip right boundary to preserve outlet
    for (size_t j = 1; j < ny - 1; j++) {
        dst_u[IDX_2D(0, j, nx)] = src_u[IDX_2D(0, j, nx)];
        dst_v[IDX_2D(0, j, nx)] = src_v[IDX_2D(0, j, nx)];
    }
}

#endif /* BOUNDARY_COPY_UTILS_H */
