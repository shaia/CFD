#include "cfd/core/grid.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"


#include <math.h>
#include <stddef.h>

grid* grid_create_3d(size_t nx, size_t ny, size_t nz,
                     double xmin, double xmax,
                     double ymin, double ymax,
                     double zmin, double zmax) {
    if (nx == 0 || ny == 0 || nz == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "grid dimensions must be positive");
        return NULL;
    }
    if (xmax <= xmin || ymax <= ymin) {
        cfd_set_error(CFD_ERROR_INVALID, "grid bounds invalid (max must be > min)");
        return NULL;
    }
    if (nz > 1 && zmax <= zmin) {
        cfd_set_error(CFD_ERROR_INVALID, "grid z-bounds invalid (zmax must be > zmin when nz > 1)");
        return NULL;
    }

    grid* new_grid = (grid*)cfd_malloc(sizeof(grid));
    if (new_grid == NULL) {
        return NULL;
    }

    new_grid->nx = nx;
    new_grid->ny = ny;
    new_grid->nz = nz;
    new_grid->xmin = xmin;
    new_grid->xmax = xmax;
    new_grid->ymin = ymin;
    new_grid->ymax = ymax;

    // Allocate memory for x/y grid arrays
    new_grid->x = (double*)cfd_calloc(nx, sizeof(double));
    new_grid->y = (double*)cfd_calloc(ny, sizeof(double));
    new_grid->dx = (double*)cfd_calloc(nx - 1, sizeof(double));
    new_grid->dy = (double*)cfd_calloc(ny - 1, sizeof(double));

    if (!new_grid->x || !new_grid->y || !new_grid->dx || !new_grid->dy) {
        grid_destroy(new_grid);
        return NULL;
    }

    // Set z-dimension fields and precomputed constants
    if (nz > 1) {
        new_grid->zmin = zmin;
        new_grid->zmax = zmax;
        new_grid->z = (double*)cfd_calloc(nz, sizeof(double));
        new_grid->dz = (double*)cfd_calloc(nz - 1, sizeof(double));
        if (!new_grid->z || !new_grid->dz) {
            grid_destroy(new_grid);
            return NULL;
        }
        new_grid->stride_z = nx * ny;
        // inv_dz2 is set during grid_initialize_uniform/stretched
        new_grid->inv_dz2 = 0.0;
        new_grid->k_start = 1;
        new_grid->k_end = nz - 1;
    } else {
        // 2D mode: no z allocation, zero precomputed constants
        new_grid->zmin = 0.0;
        new_grid->zmax = 0.0;
        new_grid->z = NULL;
        new_grid->dz = NULL;
        new_grid->stride_z = 0;
        new_grid->inv_dz2 = 0.0;
        new_grid->k_start = 0;
        new_grid->k_end = 1;
    }

    return new_grid;
}

grid* grid_create(size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax) {
    return grid_create_3d(nx, ny, 1, xmin, xmax, ymin, ymax, 0.0, 0.0);
}

void grid_destroy(grid* grid) {
    if (grid != NULL) {
        cfd_free(grid->x);
        cfd_free(grid->y);
        cfd_free(grid->dx);
        cfd_free(grid->dy);
        cfd_free(grid->z);
        cfd_free(grid->dz);
        cfd_free(grid);
    }
}

void grid_initialize_uniform(grid* grid) {
    double dx = (grid->xmax - grid->xmin) / (grid->nx - 1);
    double dy = (grid->ymax - grid->ymin) / (grid->ny - 1);

    // Set x-coordinates
    for (size_t i = 0; i < grid->nx; i++) {
        grid->x[i] = grid->xmin + (i * dx);
    }

    // Set y-coordinates
    for (size_t j = 0; j < grid->ny; j++) {
        grid->y[j] = grid->ymin + (j * dy);
    }

    // Set cell sizes
    for (size_t i = 0; i < grid->nx - 1; i++) {
        grid->dx[i] = dx;
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        grid->dy[j] = dy;
    }

    // Initialize z-direction if 3D
    if (grid->nz > 1 && grid->z && grid->dz) {
        double dz_val = (grid->zmax - grid->zmin) / (grid->nz - 1);

        for (size_t k = 0; k < grid->nz; k++) {
            grid->z[k] = grid->zmin + (k * dz_val);
        }
        for (size_t k = 0; k < grid->nz - 1; k++) {
            grid->dz[k] = dz_val;
        }

        grid->inv_dz2 = 1.0 / (dz_val * dz_val);
    }
}

void grid_initialize_stretched(grid* grid, double beta) {
    // Tanh stretching: clusters points near both boundaries
    // Higher beta = more clustering near boundaries (useful for boundary layers)
    //
    // Formula: x[i] = xmin + (xmax - xmin) * (1 + tanh(beta * (2*xi - 1)) / tanh(beta)) / 2
    // where xi = i / (n-1) goes from 0 to 1
    //
    // At xi=0: tanh(-beta)/tanh(beta) = -1, so x = xmin
    // At xi=1: tanh(+beta)/tanh(beta) = +1, so x = xmax
    // At xi=0.5: tanh(0)/tanh(beta) = 0, so x = (xmin + xmax) / 2

    // Handle beta = 0 or very small beta: fall back to uniform grid
    // (avoids division by zero since tanh(0) = 0)
    if (fabs(beta) < 1e-10) {
        grid_initialize_uniform(grid);
        return;
    }

    double tanh_beta = tanh(beta);

    // Initialize x-direction with stretching
    for (size_t i = 0; i < grid->nx; i++) {
        double xi = (double)i / (grid->nx - 1);
        grid->x[i] = grid->xmin + (grid->xmax - grid->xmin) *
                     (1.0 + tanh(beta * (2.0 * xi - 1.0)) / tanh_beta) / 2.0;
    }

    // Initialize y-direction with stretching
    for (size_t j = 0; j < grid->ny; j++) {
        double eta = (double)j / (grid->ny - 1);
        grid->y[j] = grid->ymin + (grid->ymax - grid->ymin) *
                     (1.0 + tanh(beta * (2.0 * eta - 1.0)) / tanh_beta) / 2.0;
    }

    // Calculate cell sizes
    for (size_t i = 0; i < grid->nx - 1; i++) {
        grid->dx[i] = grid->x[i + 1] - grid->x[i];
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        grid->dy[j] = grid->y[j + 1] - grid->y[j];
    }

    // Initialize z-direction with stretching if 3D
    if (grid->nz > 1 && grid->z && grid->dz) {
        for (size_t k = 0; k < grid->nz; k++) {
            double zeta = (double)k / (grid->nz - 1);
            grid->z[k] = grid->zmin + (grid->zmax - grid->zmin) *
                         (1.0 + (tanh(beta * (2.0 * zeta - 1.0)) / tanh_beta)) / 2.0;
        }
        for (size_t k = 0; k < grid->nz - 1; k++) {
            grid->dz[k] = grid->z[k + 1] - grid->z[k];
        }

        // Use minimum dz for inv_dz2 (conservative for CFL)
        double dz_min = grid->dz[0];
        for (size_t k = 1; k < grid->nz - 1; k++) {
            if (grid->dz[k] < dz_min) {
                dz_min = grid->dz[k];
            }
        }
        grid->inv_dz2 = 1.0 / (dz_min * dz_min);
    }
}
