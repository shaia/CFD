#include "simulation_api.h"
#include "utils.h"
#include <stdio.h>

int main() {
    printf("CFD Framework - CSV Data Export Example\n");
    printf("========================================\n");

    // Define grid and domain parameters
    size_t nx = 100, ny = 50;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

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
    simulation_set_run_prefix(sim_data, "csv_export");

    printf("\nRegistering outputs:\n");

    // Register VTK outputs for visualization
    printf("  - VTK flow field (every 20 steps)\n");
    simulation_register_output(sim_data, OUTPUT_FULL_FIELD, 20, "flow_field");

    // Register CSV outputs for data analysis
    printf("  - CSV timeseries (every step) - tracks global stats over time\n");
    simulation_register_output(sim_data, OUTPUT_CSV_TIMESERIES, 1, "timeseries");

    printf("  - CSV statistics (every 5 steps) - min/max/avg for all fields\n");
    simulation_register_output(sim_data, OUTPUT_CSV_STATISTICS, 5, "statistics");

    printf("  - CSV centerline (every 10 steps) - velocity/pressure profiles\n");
    simulation_register_output(sim_data, OUTPUT_CSV_CENTERLINE, 10, "centerline");

    // Run simulation
    int max_steps = 100;

    printf("\nRunning simulation...\n");
    printf("Total steps: %d\n\n", max_steps);

    for (int step = 0; step < max_steps; step++) {
        // Run simulation step
        run_simulation_step(sim_data);

        // Automatically write all registered outputs
        simulation_write_outputs(sim_data, step);

        if (step % 20 == 0) {
            printf("Step %3d: t = %.4f\n", step, sim_data->current_time);
        }
    }

    // Cleanup
    free_simulation(sim_data);

    printf("\nSimulation completed successfully!\n");
    printf("\nGenerated files:\n");
    printf("  VTK Files (3D visualization):\n");
    printf("    - flow_field_*.vtk : Complete flow field snapshots\n");
    printf("\n  CSV Files (data analysis):\n");
    printf("    - timeseries.csv   : Time history (100 rows)\n");
    printf("      Columns: step, time, dt, max_u, max_v, max_p, avg_u, avg_v, avg_p, ...\n");
    printf("    - statistics.csv   : Detailed statistics (20 rows)\n");
    printf("      Columns: step, time, min_u, max_u, avg_u, min_v, max_v, avg_v, ...\n");
    printf("    - centerline.csv   : Velocity/pressure profiles (10 files)\n");
    printf("      Columns: x, u, v, p, rho, T\n");
    printf("\nUse these CSV files with:\n");
    printf("  - Excel/LibreOffice for quick plots\n");
    printf("  - Python (pandas, matplotlib) for custom analysis\n");
    printf("  - MATLAB/Octave for numerical analysis\n");
    printf("  - R for statistical analysis\n");

    return 0;
}
