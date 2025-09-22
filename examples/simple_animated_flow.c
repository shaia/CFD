#include "simulation_api.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple analytical flow patterns for demonstration
void set_analytical_flow(FlowField* field, const Grid* grid, double time) {
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = grid->x[i];
            double y = grid->y[j];

            // Rotating vortex that evolves over time
            double cx = 2.0 + 0.5 * sin(time);  // Vortex center moves
            double cy = 1.0 + 0.3 * cos(time);
            double r = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            double theta = atan2(y - cy, x - cx);

            // Vortex strength that varies with time
            double strength = 3.0 * exp(-r * r / 0.5) * (1.0 + 0.5 * sin(2.0 * time));

            // Velocity components for rotating flow
            field->u[idx] = -strength * sin(theta) * exp(-r);
            field->v[idx] = strength * cos(theta) * exp(-r);

            // Add background flow
            field->u[idx] += 0.5 * (1.0 + 0.3 * sin(time + x));
            field->v[idx] += 0.2 * sin(2.0 * time + y);

            // Pressure field
            field->p[idx] = 1.0 + 0.5 * sin(x + time) + 0.3 * cos(y + 2.0 * time);

            // Density
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

int main() {
    printf("CFD Framework - Simple Animated Flow Demo\n");
    printf("========================================\n");

    // Define grid and domain parameters
    size_t nx = 60, ny = 30;
    double xmin = 0.0, xmax = 4.0, ymin = 0.0, ymax = 2.0;

    printf("Grid size: %zu x %zu\n", nx, ny);
    printf("Domain: [%.1f, %.1f] x [%.1f, %.1f]\n", xmin, xmax, ymin, ymax);

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    if (!sim_data) {
        printf("Failed to initialize simulation\n");
        return 1;
    }

    // Create output directory
    ensure_directory_exists("../../output");
    ensure_directory_exists("../../output/vtk_files");

    // Animation parameters
    int max_steps = 100;
    int output_interval = 2;
    double dt = 0.05;

    printf("\nRunning simple analytical flow animation...\n");
    printf("Total steps: %d\n", max_steps);
    printf("Output interval: every %d steps\n", output_interval);
    printf("Expected frames: %d\n", max_steps / output_interval + 1);

    printf("\nGenerating analytical flow patterns...\n");
    for (int step = 0; step <= max_steps; step++) {
        double time = step * dt;

        // Set analytical flow field based on current time
        set_analytical_flow(sim_data->field, sim_data->grid, time);

        // Output visualization data at intervals
        if (step % output_interval == 0) {
            char filename[512];

            // Complete flow field for animation
            snprintf(filename, sizeof(filename), "../../output/vtk_files/simple_flow_%04d.vtk", step);
            write_flow_field_to_vtk(sim_data, filename);

            printf("Step %4d: Animation frame saved (t = %.3f)\n", step, time);
        }

        // Progress indicator
        if (step % (max_steps / 10) == 0 && step > 0) {
            printf("Progress: %d%% complete\n", (step * 100) / max_steps);
        }
    }

    // Cleanup
    free_simulation(sim_data);

    printf("\nSimple animation completed successfully!\n");
    printf("\nGenerated animation files:\n");
    printf("  - simple_flow_*.vtk : %d frames\n", (max_steps / output_interval) + 1);
    printf("\nTo create animation:\n");
    printf("  python visualization/animate_flow.py ../../output/vtk_files\n");

    return 0;
}