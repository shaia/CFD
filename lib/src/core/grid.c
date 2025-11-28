#include "grid.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Grid* grid_create(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax) {
    // Validate input parameters
    // Need at least 2 points in each direction for a valid grid (to compute dx/dy)
    if (nx < 2) {
        fprintf(stderr, "ERROR: grid_create: nx must be at least 2 (got %zu)\n", nx);
        return NULL;
    }
    if (ny < 2) {
        fprintf(stderr, "ERROR: grid_create: ny must be at least 2 (got %zu)\n", ny);
        return NULL;
    }
    if (xmax <= xmin) {
        fprintf(stderr, "ERROR: grid_create: xmax must be greater than xmin\n");
        return NULL;
    }
    if (ymax <= ymin) {
        fprintf(stderr, "ERROR: grid_create: ymax must be greater than ymin\n");
        return NULL;
    }

    Grid* grid = (Grid*)malloc(sizeof(Grid));
    if (grid == NULL) {
        fprintf(stderr, "ERROR: grid_create: failed to allocate Grid structure\n");
        return NULL;
    }

    grid->nx = nx;
    grid->ny = ny;
    grid->xmin = xmin;
    grid->xmax = xmax;
    grid->ymin = ymin;
    grid->ymax = ymax;

    // Allocate memory for grid arrays
    grid->x = (double*)calloc(nx, sizeof(double));
    grid->y = (double*)calloc(ny, sizeof(double));
    grid->dx = (double*)calloc(nx - 1, sizeof(double));
    grid->dy = (double*)calloc(ny - 1, sizeof(double));

    // Check all allocations succeeded
    if (grid->x == NULL || grid->y == NULL || grid->dx == NULL || grid->dy == NULL) {
        fprintf(stderr, "ERROR: grid_create: failed to allocate grid arrays\n");
        // Clean up partial allocations
        free(grid->x);
        free(grid->y);
        free(grid->dx);
        free(grid->dy);
        free(grid);
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
    double dx = (grid->xmax - grid->xmin) / (grid->nx - 1);
    for (size_t i = 0; i < grid->nx; i++) {
        double xi = (double)i / (grid->nx - 1);
        grid->x[i] = grid->xmin + (grid->xmax - grid->xmin) * 
                     (1.0 - cosh(beta * (1.0 - 2.0 * xi)) / cosh(beta));
    }
    
    // Initialize y-direction with stretching
    double dy = (grid->ymax - grid->ymin) / (grid->ny - 1);
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