#ifndef CFD_GRID_H
#define CFD_GRID_H

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
    double* x;          // x-coordinates of grid points [nx]
    double* y;          // y-coordinates of grid points [ny]
    double* dx;         // x-direction cell sizes [nx-1]
    double* dy;         // y-direction cell sizes [ny-1]
    size_t nx;          // number of points in x-direction
    size_t ny;          // number of points in y-direction
    double xmin;        // minimum x-coordinate
    double xmax;        // maximum x-coordinate
    double ymin;        // minimum y-coordinate
    double ymax;        // maximum y-coordinate
} Grid;

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
Grid* grid_create(size_t nx, size_t ny,
                  double xmin, double xmax,
                  double ymin, double ymax);

/**
 * Destroy grid and free memory
 */
void grid_destroy(Grid* grid);

/**
 * Initialize grid with uniform spacing
 */
void grid_initialize_uniform(Grid* grid);

/**
 * Initialize grid with stretched spacing
 *
 * @param beta Stretching parameter (larger = more stretching)
 */
void grid_initialize_stretched(Grid* grid, double beta);

#ifdef __cplusplus
}
#endif

#endif // CFD_GRID_H
