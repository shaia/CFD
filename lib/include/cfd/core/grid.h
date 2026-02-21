#ifndef CFD_GRID_H
#define CFD_GRID_H

#include "cfd/cfd_export.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Structured Curvilinear Grid
 *
 * Supports non-uniform spacing through coordinate arrays.
 * For uniform grids, all dx[i] are equal and all dy[j] are equal.
 */
typedef struct {
    double* x;    // x-coordinates of grid points [nx]
    double* y;    // y-coordinates of grid points [ny]
    double* dx;   // x-direction cell sizes [nx-1]
    double* dy;   // y-direction cell sizes [ny-1]
    size_t nx;    // number of points in x-direction
    size_t ny;    // number of points in y-direction
    double xmin;  // minimum x-coordinate
    double xmax;  // maximum x-coordinate
    double ymin;  // minimum y-coordinate
    double ymax;  // maximum y-coordinate

    // 3D extension (nz=1 reproduces 2D behavior)
    double* z;        // z-coordinates of grid points [nz] (NULL when nz==1)
    double* dz;       // z-direction cell sizes [nz-1] (NULL when nz==1)
    size_t nz;        // number of points in z-direction (1 for 2D)
    double zmin;      // minimum z-coordinate (0.0 for 2D)
    double zmax;      // maximum z-coordinate (0.0 for 2D)
    size_t stride_z;  // nx*ny when nz>1, 0 when nz==1
    double inv_dz2;   // 1/(dz*dz) when nz>1, 0.0 when nz==1
    size_t k_start;   // 1 when nz>1, 0 when nz==1
    size_t k_end;     // nz-1 when nz>1, 1 when nz==1
} grid;

/**
 * Create a grid
 *
 * @param nx Number of points in x-direction
 * @param ny Number of points in y-direction
 * @param xmin Minimum x-coordinate
 * @param xmax Maximum x-coordinate
 * @param ymin Minimum y-coordinate
 * @param ymax Maximum y-coordinate
 * @return Pointer to newly created grid
 */
CFD_LIBRARY_EXPORT grid* grid_create(size_t nx, size_t ny, double xmin, double xmax, double ymin,
                                     double ymax);

/**
 * Create a 3D grid
 *
 * @param nx   Number of points in x-direction
 * @param ny   Number of points in y-direction
 * @param nz   Number of points in z-direction (1 for 2D)
 * @param xmin Minimum x-coordinate
 * @param xmax Maximum x-coordinate
 * @param ymin Minimum y-coordinate
 * @param ymax Maximum y-coordinate
 * @param zmin Minimum z-coordinate (0.0 for 2D)
 * @param zmax Maximum z-coordinate (0.0 for 2D)
 * @return Pointer to newly created grid
 */
CFD_LIBRARY_EXPORT grid* grid_create_3d(size_t nx, size_t ny, size_t nz,
                                         double xmin, double xmax,
                                         double ymin, double ymax,
                                         double zmin, double zmax);

/**
 * Destroy grid and free memory
 */
CFD_LIBRARY_EXPORT void grid_destroy(grid* grid);

/**
 * Initialize grid with uniform spacing
 */
CFD_LIBRARY_EXPORT void grid_initialize_uniform(grid* grid);

/**
 * Initialize grid with stretched spacing using tanh stretching.
 *
 * Uses the formula: x[i] = xmin + L * (1 + tanh(beta * (2*xi - 1)) / tanh(beta)) / 2
 * where xi = i / (nx-1) ranges from 0 to 1, and L = xmax - xmin.
 *
 * This clusters grid points near both boundaries, which is useful for
 * resolving boundary layers in viscous flow simulations.
 *
 * @param grid Pointer to grid structure
 * @param beta Stretching parameter controlling point clustering:
 *             - beta = 0: Uniform spacing (falls back to grid_initialize_uniform)
 *             - beta = 1.0: Mild clustering near boundaries
 *             - beta = 2.0: Moderate clustering (recommended for most cases)
 *             - beta = 3.0+: Strong clustering near boundaries
 *             Higher beta values produce smaller cells near boundaries and
 *             larger cells in the domain center.
 */
CFD_LIBRARY_EXPORT void grid_initialize_stretched(grid* grid, double beta);

#ifdef __cplusplus
}
#endif

#endif  // CFD_GRID_H
