/**
 * Custom Boundary Conditions Example
 *
 * Demonstrates how to set up different boundary conditions
 * and solve a flow around a cylinder problem.
 */

#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"
#include "cfd/core/logging.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/math_utils.h"

#include "cfd/io/vtk_output.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void setup_cylinder_flow(FlowField* field, Grid* grid) {
    printf("Setting up flow around cylinder...\n");

    // Initialize flow field with uniform flow
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;

            // Initial conditions: uniform flow in x-direction
            field->u[idx] = 1.0;    // u-velocity
            field->v[idx] = 0.0;    // v-velocity
            field->p[idx] = 0.0;    // pressure
            field->rho[idx] = 1.0;  // density
            field->T[idx] = 300.0;  // temperature
        }
    }

    // Add cylinder obstacle in the center
    double cx = (grid->xmax + grid->xmin) / 2.0;
    double cy = (grid->ymax + grid->ymin) / 2.0;
    double radius = 0.05;  // 5% of domain width

    printf("Cylinder center: (%.3f, %.3f), radius: %.3f\n", cx, cy, radius);

    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            double x = grid->xmin + i * (grid->xmax - grid->xmin) / (field->nx - 1);
            double y = grid->ymin + j * (grid->ymax - grid->ymin) / (field->ny - 1);

            double dist = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));

            if (dist < radius) {
                size_t idx = j * field->nx + i;
                // Inside cylinder: no-slip boundary conditions
                field->u[idx] = 0.0;
                field->v[idx] = 0.0;
            }
        }
    }
}

void apply_inlet_outlet_bc(FlowField* field, Grid* grid) {
    // Inlet boundary (left side): fixed velocity
    for (size_t j = 0; j < field->ny; j++) {
        size_t idx = j * field->nx + 0;  // i = 0 (left boundary)
        field->u[idx] = 1.0;             // Inlet velocity
        field->v[idx] = 0.0;
        field->p[idx] = 0.0;
    }

    // Outlet boundary (right side): zero gradient
    for (size_t j = 0; j < field->ny; j++) {
        size_t idx_out = j * field->nx + (field->nx - 1);  // i = nx-1 (right boundary)
        size_t idx_in = j * field->nx + (field->nx - 2);   // i = nx-2 (interior)

        field->u[idx_out] = field->u[idx_in];
        field->v[idx_out] = field->v[idx_in];
        field->p[idx_out] = field->p[idx_in];
    }

    // Top and bottom walls: no-slip
    for (size_t i = 0; i < field->nx; i++) {
        // Bottom wall (j = 0)
        size_t idx_bot = 0 * field->nx + i;
        field->u[idx_bot] = 0.0;
        field->v[idx_bot] = 0.0;

        // Top wall (j = ny-1)
        size_t idx_top = (field->ny - 1) * field->nx + i;
        field->u[idx_top] = 0.0;
        field->v[idx_top] = 0.0;
    }
}

int main() {
    printf("Custom Boundary Conditions Example\n");
    printf("=================================\n");

    // Create a larger grid for better resolution around cylinder
    size_t nx = 200, ny = 100;
    double xmin = 0.0, xmax = 2.0;  // Longer domain
    double ymin = 0.0, ymax = 1.0;

    printf("Grid: %zux%zu, Domain: [%.1f,%.1f] x [%.1f,%.1f]\n", nx, ny, xmin, xmax, ymin, ymax);

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    FlowField* field = flow_field_create(nx, ny);

    // Setup initial conditions and cylinder
    setup_cylinder_flow(field, grid);

    // Solver parameters for external flow
    SolverParams params = {
        .max_iter = 2000,  // More iterations for convergence
        .dt = 0.0005,      // Smaller time step for stability
        .cfl = 0.3,        // Conservative CFL number
        .gamma = 1.4,      // Specific heat ratio
        .mu = 0.005,       // Viscosity (corresponds to Re~200)
        .k = 0.1,          // Thermal conductivity
        .tolerance = 1e-6  // Convergence tolerance
    };

    printf("\nSolver parameters:\n");
    printf("  Max iterations: %d\n", params.max_iter);
    printf("  Time step: %.6f\n", params.dt);
    printf("  Viscosity: %.6f\n", params.mu);

    // Configure output directory
    cfd_set_output_base_dir("../../artifacts");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "cylinder_flow", nx, ny);
    printf("\nOutput directory: %s\n", run_dir);

    printf("\nRunning simulation...\n");
    for (int iter = 0; iter < params.max_iter; iter++) {
        // Apply custom boundary conditions each iteration
        apply_inlet_outlet_bc(field, grid);

        // Single solver step (you'd implement this in the actual solver)
        // For this example, we'll just output at intervals
        if (iter % 200 == 0) {
            char filename[512];
            char basename[128];
            snprintf(basename, sizeof(basename), "cylinder_flow_%d.vtk", iter);

            // Build full path: run_dir/basename
#ifdef _WIN32
            snprintf(filename, sizeof(filename), "%s\\%s", run_dir, basename);
#else
            snprintf(filename, sizeof(filename), "%s/%s", run_dir, basename);
#endif

            write_vtk_output(filename, "velocity_magnitude", field->u, nx, ny, grid->xmin,
                             grid->xmax, grid->ymin, grid->ymax);

            printf("  Iteration %d, output: %s\n", iter, basename);
        }
    }

    printf("\nSimulation completed!\n");
    printf("Output files saved to %s\\cylinder_flow_*.vtk\n", run_dir);
    printf("Use visualization tools to view the flow pattern around the cylinder.\n");

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);

    return 0;
}