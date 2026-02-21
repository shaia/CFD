#ifndef CFD_INDEXING_H
#define CFD_INDEXING_H

/**
 * @file indexing.h
 * @brief Grid indexing macros for 2D and 3D structured grids
 *
 * Row-major layout: x-fastest, z-slowest.
 *   2D: idx = j * nx + i
 *   3D: idx = k * (nx * ny) + j * nx + i
 *
 * Solvers use  k * stride_z + IDX_2D(i, j, nx)  where stride_z is
 * precomputed in the grid struct (0 when nz==1, nx*ny when nz>1).
 */

/** 2D flat index from (i, j) in a row of width nx */
#define IDX_2D(i, j, nx) ((j) * (nx) + (i))

/** 3D flat index from (i, j, k) in a grid of dimensions nx * ny */
#define IDX_3D(i, j, k, nx, ny) ((k) * (nx) * (ny) + (j) * (nx) + (i))

#endif /* CFD_INDEXING_H */
