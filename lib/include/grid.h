#ifndef CFD_GRID_H
#define CFD_GRID_H

#include <stddef.h>

// Grid structure to store mesh information
typedef struct {
    double* x;          // x-coordinates of grid points
    double* y;          // y-coordinates of grid points
    double* dx;         // x-direction cell sizes
    double* dy;         // y-direction cell sizes
    size_t nx;          // number of points in x-direction
    size_t ny;          // number of points in y-direction
    double xmin;        // minimum x-coordinate
    double xmax;        // maximum x-coordinate
    double ymin;        // minimum y-coordinate
    double ymax;        // maximum y-coordinate
} Grid;

// Function declarations
Grid* grid_create(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax);
void grid_destroy(Grid* grid);
void grid_initialize_uniform(Grid* grid);
void grid_initialize_stretched(Grid* grid, double beta);

#endif // CFD_GRID_H 