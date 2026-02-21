/**
 * Lid-Driven Cavity Example
 *
 * Classic CFD benchmark problem using Dirichlet boundary conditions.
 * A square cavity with:
 *   - Top wall (lid) moving at constant velocity u=1
 *   - All other walls stationary (no-slip)
 *
 * This example demonstrates:
 *   - Using bc_dirichlet_values_t for fixed-value boundary conditions
 *   - Setting up the lid-driven cavity problem
 *   - Running a time-stepping simulation
 *   - VTK output for visualization
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/io/vtk_output.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Apply lid-driven cavity boundary conditions using Dirichlet BCs.
 *
 * Boundary setup:
 *   - Top (lid): u = lid_velocity, v = 0
 *   - Bottom:    u = 0, v = 0
 *   - Left:      u = 0, v = 0
 *   - Right:     u = 0, v = 0
 */
void apply_lid_driven_cavity_bc(flow_field* field, double lid_velocity) {
    /* Dirichlet BC for u-velocity: lid moves, walls stationary */
    bc_dirichlet_values_t u_bc = {
        .left = 0.0,          /* Left wall: no-slip */
        .right = 0.0,         /* Right wall: no-slip */
        .top = lid_velocity,  /* Moving lid */
        .bottom = 0.0         /* Bottom wall: no-slip */
    };

    /* Dirichlet BC for v-velocity: all walls have v=0 */
    bc_dirichlet_values_t v_bc = {
        .left = 0.0,
        .right = 0.0,
        .top = 0.0,   /* No vertical velocity at lid */
        .bottom = 0.0
    };

    /* Apply Dirichlet BCs to velocity components */
    bc_apply_dirichlet_velocity(field->u, field->v, field->nx, field->ny, &u_bc, &v_bc);

    /* Apply Neumann BC for pressure (zero gradient at walls) */
    bc_apply_neumann(field->p, field->nx, field->ny);
}

/**
 * Initialize flow field for lid-driven cavity.
 * Starts with quiescent flow (zero velocity, constant pressure).
 */
void initialize_cavity_flow(flow_field* field) {
    size_t total = field->nx * field->ny;

    for (size_t i = 0; i < total; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->p[i] = 0.0;
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }
}

/**
 * Compute velocity magnitude for visualization.
 */
void compute_velocity_magnitude(const flow_field* field, double* vmag) {
    size_t total = field->nx * field->ny;

    for (size_t i = 0; i < total; i++) {
        vmag[i] = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }
}

/**
 * Print simulation info and Reynolds number.
 */
void print_simulation_info(size_t nx, size_t ny, double L, double U, double nu) {
    double Re = U * L / nu;

    printf("Lid-Driven Cavity Simulation\n");
    printf("============================\n");
    printf("Grid:           %zu x %zu\n", nx, ny);
    printf("Domain:         [0, %.1f] x [0, %.1f]\n", L, L);
    printf("Lid velocity:   %.2f m/s\n", U);
    printf("Viscosity:      %.6f m^2/s\n", nu);
    printf("Reynolds:       %.1f\n", Re);
    printf("\n");
}

int main(int argc, char* argv[]) {
    /* Simulation parameters */
    size_t nx = 64, ny = 64;         /* Grid resolution */
    double L = 1.0;                   /* Cavity side length */
    double lid_velocity = 1.0;        /* Lid velocity (m/s) */
    double nu = 0.01;                 /* Kinematic viscosity */
    double dt = 0.001;                /* Time step */
    int max_iter = 5000;              /* Maximum iterations */
    int output_interval = 500;        /* Output every N steps */

    /* Parse command line for Reynolds number */
    if (argc > 1) {
        double Re = atof(argv[1]);
        if (Re > 0) {
            nu = lid_velocity * L / Re;
            printf("Using Re = %.1f (nu = %.6f)\n\n", Re, nu);
        }
    }

    /* Print simulation info */
    print_simulation_info(nx, ny, L, lid_velocity, nu);

    /* Create grid and flow field */
    grid* g = grid_create(nx, ny, 0.0, L, 0.0, L);
    flow_field* field = flow_field_create(nx, ny);

    if (!g || !field) {
        fprintf(stderr, "Failed to allocate grid or flow field\n");
        return 1;
    }

    /* Initialize flow field */
    initialize_cavity_flow(field);

    /* Apply initial boundary conditions */
    apply_lid_driven_cavity_bc(field, lid_velocity);

    /* Allocate velocity magnitude array for output */
    double* vmag = (double*)malloc(nx * ny * sizeof(double));
    if (!vmag) {
        fprintf(stderr, "Failed to allocate velocity magnitude array\n");
        flow_field_destroy(field);
        grid_destroy(g);
        return 1;
    }

    /* Configure output directory */
    cfd_set_output_base_dir("output");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "lid_cavity", nx, ny);
    printf("Output directory: %s\n\n", run_dir);

    /* Set BC backend (optional - auto-selects best available) */
    printf("BC backend: %s\n\n", bc_get_backend_name());

    /* Simple explicit time-stepping loop */
    printf("Running simulation...\n");
    for (int iter = 0; iter <= max_iter; iter++) {
        /* Apply boundary conditions */
        apply_lid_driven_cavity_bc(field, lid_velocity);

        /*
         * Simplified diffusion update (for BC demonstration only).
         *
         * NOTE: This uses in-place updates which creates Gauss-Seidel-like
         * behavior rather than proper explicit Euler. A correct implementation
         * would use temporary arrays to store new values, ensuring all updates
         * use synchronized values from time t. For proper Navier-Stokes solving,
         * use the projection method solvers (e.g., solve_projection_method()).
         */
        double dx = L / (nx - 1);
        double dy = L / (ny - 1);

        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = IDX_2D(i, j, nx);

                /* Diffusion term (Laplacian) */
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * field->u[idx] + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * field->u[idx] + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * field->v[idx] + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * field->v[idx] + field->v[idx - nx]) / (dy * dy);

                /* Update velocities (diffusion only, in-place) */
                field->u[idx] += dt * nu * (d2u_dx2 + d2u_dy2);
                field->v[idx] += dt * nu * (d2v_dx2 + d2v_dy2);
            }
        }

        /* Output at intervals */
        if (iter % output_interval == 0) {
            /* Compute velocity magnitude */
            compute_velocity_magnitude(field, vmag);

            /* Build output filename */
            char filename[512];
#ifdef _WIN32
            snprintf(filename, sizeof(filename), "%s\\cavity_%05d.vtk", run_dir, iter);
#else
            snprintf(filename, sizeof(filename), "%s/cavity_%05d.vtk", run_dir, iter);
#endif

            /* Write VTK file */
            write_vtk_output(filename, "velocity_magnitude", vmag, nx, ny,
                             g->xmin, g->xmax, g->ymin, g->ymax);

            /* Find max velocity for progress output */
            double max_u = 0.0;
            for (size_t k = 0; k < nx * ny; k++) {
                if (fabs(field->u[k]) > max_u) max_u = fabs(field->u[k]);
            }

            printf("  Iteration %5d: max|u| = %.6f\n", iter, max_u);
        }
    }

    printf("\nSimulation completed!\n");
    printf("Output files: %s/cavity_*.vtk\n", run_dir);
    printf("\nVisualization tips:\n");
    printf("  - Open in ParaView and apply 'Glyph' filter to see velocity vectors\n");
    printf("  - The lid drives a clockwise vortex in the cavity\n");
    printf("  - Higher Re produces stronger secondary vortices in corners\n");

    /* Cleanup */
    free(vmag);
    flow_field_destroy(field);
    grid_destroy(g);

    return 0;
}
