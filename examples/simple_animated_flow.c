#include "simulation_api.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Configurable vortex characteristics
#define VORTEX_CENTER_AMPLITUDE_X 0.5    // Amplitude of vortex center x-movement
#define VORTEX_CENTER_AMPLITUDE_Y 0.3    // Amplitude of vortex center y-movement
#define VORTEX_STRENGTH_BASE 3.0         // Base strength of the vortex
#define VORTEX_STRENGTH_VARIATION 0.5    // Amplitude of strength variation over time
#define VORTEX_STRENGTH_FREQUENCY 2.0    // Frequency of strength variation
#define VORTEX_SIZE_FACTOR 0.5           // Controls vortex size (1/σ² in Gaussian)
#define VORTEX_DECAY_RATE 1.0            // Exponential decay rate with distance

// Configurable background flow characteristics
#define BACKGROUND_FLOW_U_BASE 0.5       // Base u-velocity of background flow
#define BACKGROUND_FLOW_U_VARIATION 0.3  // Amplitude of u-velocity variation
#define BACKGROUND_FLOW_V_AMPLITUDE 0.2  // Amplitude of v-velocity oscillation
#define BACKGROUND_FLOW_V_FREQUENCY 2.0  // Frequency of v-velocity oscillation

// Configurable pressure field characteristics
#define PRESSURE_BASE 1.0                // Base pressure level
#define PRESSURE_X_AMPLITUDE 0.5         // Amplitude of pressure variation in x
#define PRESSURE_Y_AMPLITUDE 0.3         // Amplitude of pressure variation in y
#define PRESSURE_Y_FREQUENCY 2.0         // Frequency of pressure variation in y

// Simple analytical flow patterns for demonstration
void set_analytical_flow(FlowField* field, const Grid* grid, double time) {
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = grid->x[i];
            double y = grid->y[j];

            // Rotating vortex that evolves over time
            double cx = 2.0 + VORTEX_CENTER_AMPLITUDE_X * sin(time);  // Vortex center moves
            double cy = 1.0 + VORTEX_CENTER_AMPLITUDE_Y * cos(time);
            double r = sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            double theta = atan2(y - cy, x - cx);

            // Vortex strength that varies with time
            double strength = VORTEX_STRENGTH_BASE * exp(-r * r / VORTEX_SIZE_FACTOR) *
                            (1.0 + VORTEX_STRENGTH_VARIATION * sin(VORTEX_STRENGTH_FREQUENCY * time));

            // Velocity components for rotating flow
            field->u[idx] = -strength * sin(theta) * exp(-VORTEX_DECAY_RATE * r);
            field->v[idx] = strength * cos(theta) * exp(-VORTEX_DECAY_RATE * r);

            // Add background flow
            field->u[idx] += BACKGROUND_FLOW_U_BASE * (1.0 + BACKGROUND_FLOW_U_VARIATION * sin(time + x));
            field->v[idx] += BACKGROUND_FLOW_V_AMPLITUDE * sin(BACKGROUND_FLOW_V_FREQUENCY * time + y);

            // Pressure field
            field->p[idx] = PRESSURE_BASE + PRESSURE_X_AMPLITUDE * sin(x + time) +
                          PRESSURE_Y_AMPLITUDE * cos(y + PRESSURE_Y_FREQUENCY * time);

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

    // Configure output directory
    simulation_set_output_dir("../../artifacts");

    // Initialize simulation
    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    if (!sim_data) {
        printf("Failed to initialize simulation\n");
        return 1;
    }

    // Set run prefix
    simulation_set_run_prefix(sim_data, "simple_flow");

    // Animation parameters
    int max_steps = 100;
    int output_interval = 2;
    double dt = 0.05;

    // Register output
    simulation_register_output(sim_data, OUTPUT_FULL_FIELD, output_interval, "flow");

    printf("\nRunning simple analytical flow animation...\n");
    printf("Total steps: %d\n", max_steps);
    printf("Output interval: every %d steps\n", output_interval);
    printf("Expected frames: %d\n", max_steps / output_interval + 1);

    printf("\nGenerating analytical flow patterns...\n");
    for (int step = 0; step <= max_steps; step++) {
        double time = step * dt;

        // Set analytical flow field based on current time
        set_analytical_flow(sim_data->field, sim_data->grid, time);

        // Automatically write registered outputs
        simulation_write_outputs(sim_data, step);

        if (step % output_interval == 0) {
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
    printf("  - flow_*.vtk : %d frames\n", (max_steps / output_interval) + 1);
    printf("\nTo create animation:\n");
    printf("  python visualization/animate_flow.py artifacts\\output\n");

    return 0;
}