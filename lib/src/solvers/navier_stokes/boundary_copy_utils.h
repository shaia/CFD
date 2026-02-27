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

/**
 * Copy boundary values for 3D velocity arrays (u, v, w on all 6 faces)
 *
 * When nz <= 1, copies only the 4 xy-boundary edges (same as 2D) and
 * ignores w. When nz > 1, also copies the front/back z-faces.
 *
 * @param dst_u, dst_v, dst_w  Destination velocity arrays
 * @param src_u, src_v, src_w  Source velocity arrays
 * @param nx, ny, nz           Grid dimensions (nz=1 for 2D)
 */
static inline void copy_boundary_velocities_3d(
    double* dst_u, double* dst_v, double* dst_w,
    const double* src_u, const double* src_v, const double* src_w,
    size_t nx, size_t ny, size_t nz)
{
    size_t plane = nx * ny;

    for (size_t k = 0; k < nz; k++) {
        size_t base = k * plane;

        /* Bottom and top y-boundaries (j = 0 and j = ny-1) */
        for (size_t i = 0; i < nx; i++) {
            size_t bot = base + i;
            size_t top = base + IDX_2D(i, ny - 1, nx);
            dst_u[bot] = src_u[bot];
            dst_v[bot] = src_v[bot];
            dst_u[top] = src_u[top];
            dst_v[top] = src_v[top];
            if (nz > 1) {
                dst_w[bot] = src_w[bot];
                dst_w[top] = src_w[top];
            }
        }
        /* Left and right x-boundaries (i = 0 and i = nx-1) */
        for (size_t j = 1; j < ny - 1; j++) {
            size_t left = base + IDX_2D(0, j, nx);
            size_t right = base + IDX_2D(nx - 1, j, nx);
            dst_u[left] = src_u[left];
            dst_v[left] = src_v[left];
            dst_u[right] = src_u[right];
            dst_v[right] = src_v[right];
            if (nz > 1) {
                dst_w[left] = src_w[left];
                dst_w[right] = src_w[right];
            }
        }
    }

    /* Front and back z-faces (k = 0 and k = nz-1) */
    if (nz > 1) {
        size_t back_base = (nz - 1) * plane;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t off = IDX_2D(i, j, nx);
                /* k = 0 face */
                dst_u[off] = src_u[off];
                dst_v[off] = src_v[off];
                dst_w[off] = src_w[off];
                /* k = nz-1 face */
                dst_u[back_base + off] = src_u[back_base + off];
                dst_v[back_base + off] = src_v[back_base + off];
                dst_w[back_base + off] = src_w[back_base + off];
            }
        }
    }
}

#endif /* BOUNDARY_COPY_UTILS_H */
