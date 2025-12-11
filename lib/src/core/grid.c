#include "cfd/core/grid.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"


#include <math.h>

Grid* grid_create(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax) {
    Grid* grid = (Grid*)cfd_malloc(sizeof(Grid));
    if (grid == NULL) {
        return NULL;
    }

    grid->nx = nx;
    grid->ny = ny;
    grid->xmin = xmin;
    grid->xmax = xmax;
    grid->ymin = ymin;
    grid->ymax = ymax;

    // Allocate memory for grid arrays
    grid->x = (double*)cfd_calloc(nx, sizeof(double));
    grid->y = (double*)cfd_calloc(ny, sizeof(double));
    grid->dx = (double*)cfd_calloc(nx - 1, sizeof(double));
    grid->dy = (double*)cfd_calloc(ny - 1, sizeof(double));

    if (!grid->x || !grid->y || !grid->dx || !grid->dy) {
        grid_destroy(grid);
        return NULL;
    }

    return grid;
}

void grid_destroy(Grid* grid) {
    if (grid != NULL) {
        cfd_free(grid->x);
        cfd_free(grid->y);
        cfd_free(grid->dx);
        cfd_free(grid->dy);
        cfd_free(grid);
    }
}

void grid_initialize_uniform(Grid* grid) {
    double dx = (grid->xmax - grid->xmin) / (grid->nx - 1);
    double dy = (grid->ymax - grid->ymin) / (grid->ny - 1);

    // Set x-coordinates
    for (size_t i = 0; i < grid->nx; i++) {
        grid->x[i] = grid->xmin + i * dx;
    }

    // Set y-coordinates
    for (size_t j = 0; j < grid->ny; j++) {
        grid->y[j] = grid->ymin + j * dy;
    }

    // Set cell sizes
    for (size_t i = 0; i < grid->nx - 1; i++) {
        grid->dx[i] = dx;
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        grid->dy[j] = dy;
    }
}

void grid_initialize_stretched(Grid* grid, double beta) {
    // Initialize x-direction with stretching
    for (size_t i = 0; i < grid->nx; i++) {
        double xi = (double)i / (grid->nx - 1);
        grid->x[i] = grid->xmin +
                     (grid->xmax - grid->xmin) * (1.0 - cosh(beta * (1.0 - 2.0 * xi)) / cosh(beta));
    }

    // Initialize y-direction with stretching
    for (size_t j = 0; j < grid->ny; j++) {
        double eta = (double)j / (grid->ny - 1);
        grid->y[j] = grid->ymin + (grid->ymax - grid->ymin) *
                                      (1.0 - cosh(beta * (1.0 - 2.0 * eta)) / cosh(beta));
    }

    // Calculate cell sizes
    for (size_t i = 0; i < grid->nx - 1; i++) {
        grid->dx[i] = grid->x[i + 1] - grid->x[i];
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        grid->dy[j] = grid->y[j + 1] - grid->y[j];
    }
}