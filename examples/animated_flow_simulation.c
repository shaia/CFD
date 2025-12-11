#include "cfd/api/simulation_api.h"
#include "cfd/core/utils.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    printf("CFD Framework - Animated Flow Simulation\n");
    printf("========================================\n");

    // Define grid and domain parameters for better resolution
    size_t nx = 80, ny = 40;
    double xmin = 0.0, xmax = 4.0, ymin = 0.0, ymax = 2.0;

    printf("Grid size: %zu x %zu\n", nx, ny);
    printf("Domain: [%.1f, %.1f] x [%.1f, %.1f]\n", xmin, xmax, ymin, ymax);

    // Configure output directory (optional)
    simulation_set_output_dir("../../artifacts");

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    if (!sim_data) {
        printf("Failed to initialize simulation\n");
        return 1;
    }

    // Set run prefix
    simulation_set_run_prefix(sim_data, "animated_flow");

    // Enhanced simulation parameters for better dynamics
    int max_steps = 200;      // More time steps for animation
    int output_interval = 5;  // Output every 5 steps for smooth animation

    // Register outputs
    simulation_register_output(sim_data, OUTPUT_FULL_FIELD, output_interval, "flow_field");
    simulation_register_output(sim_data, OUTPUT_VELOCITY, output_interval, "velocity_vectors");

    printf("\nRunning enhanced simulation for animation...\n");
    printf("Total steps: %d\n", max_steps);
    printf("Output interval: every %d steps\n", output_interval);
    printf("Expected frames: %d\n", max_steps / output_interval + 1);

    // Add more dramatic initial conditions by modifying the flow field
    FlowField* field = sim_data->field;
    Grid* grid = sim_data->grid;

    printf("\nSetting up enhanced initial conditions...\n");
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = grid->x[i];
            double y = grid->y[j];

            // Create multiple vortices and pressure waves

            // Primary vortex at (1.0, 1.0)
            double r1 = sqrt((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0));
            if (r1 < 0.6) {
                double theta1 = atan2(y - 1.0, x - 1.0);
                double vortex_strength = 2.0 * exp(-r1 * r1 / 0.2);
                field->u[idx] += -vortex_strength * sin(theta1);
                field->v[idx] += vortex_strength * cos(theta1);
                field->p[idx] += 0.5 * exp(-r1 * r1 / 0.1);
            }

            // Secondary vortex at (3.0, 1.0) - counter-rotating
            double r2 = sqrt((x - 3.0) * (x - 3.0) + (y - 1.0) * (y - 1.0));
            if (r2 < 0.5) {
                double theta2 = atan2(y - 1.0, x - 3.0);
                double vortex_strength = -1.5 * exp(-r2 * r2 / 0.15);
                field->u[idx] += -vortex_strength * sin(theta2);
                field->v[idx] += vortex_strength * cos(theta2);
                field->p[idx] += 0.3 * exp(-r2 * r2 / 0.08);
            }

            // Pressure wave from left boundary
            if (x < 0.5) {
                field->p[idx] += 0.2 * sin(2.0 * M_PI * y) * exp(-x * x / 0.1);
                field->u[idx] += 1.0 * exp(-x * x / 0.05);
            }

            // Background shear flow
            field->u[idx] += 0.5 * (1.0 + 0.3 * sin(M_PI * y / 2.0));
            field->v[idx] += 0.1 * sin(M_PI * x / 2.0);
        }
    }

    printf("Enhanced initial conditions applied!\n");
    printf("- Primary vortex at (1.0, 1.0)\n");
    printf("- Counter-rotating vortex at (3.0, 1.0)\n");
    printf("- Pressure wave from left boundary\n");
    printf("- Background shear flow\n");

    printf("\nStarting time integration...\n");
    double time = 0.0;  // Track time manually
    double dt = 0.005;  // Fixed time step

    for (int step = 0; step <= max_steps; step++) {
        // Run simulation step
        if (step > 0) {
            run_simulation_step(sim_data);
            time += dt;  // Increment time manually
        }

        // Automatically write registered outputs
        simulation_write_outputs(sim_data, step);

        if (step % output_interval == 0) {
            printf("Step %4d: Animation frame saved (t = %.4f)\n", step, time);
        }

        // Progress indicator
        if (step % (max_steps / 10) == 0 && step > 0) {
            printf("Progress: %d%% complete\n", (step * 100) / max_steps);
        }
    }

    // Cleanup
    free_simulation(sim_data);

    printf("\nAnimated simulation completed successfully!\n");
    printf("\nGenerated animation files automatically:\n");
    printf("  - flow_field_*.vtk       : Complete flow field frames\n");
    printf("  - velocity_vectors_*.vtk : Velocity vector frames\n");
    printf("  - Total frames: %d\n", (max_steps / output_interval) + 1);
    printf("\nAnimation data ready for visualization!\n");
    printf("Use: python visualization/animate_flow.py artifacts\\output\n");

    return 0;
}