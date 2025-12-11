/**
 * Performance Comparison Example
 *
 * Demonstrates the performance difference between basic and optimized solvers
 * using the modern pluggable solver interface.
 */

#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"
#include "cfd/core/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void benchmark_solver(const char* solver_name, const char* solver_type, size_t nx, size_t ny,
                      int iterations) {
    printf("\n=== %s Benchmark ===\n", solver_name);
    printf("Grid size: %zux%zu, Iterations: %d\n", nx, ny, iterations);

    // Create solver using modern interface
    Solver* solver = solver_create(solver_type);
    if (!solver) {
        fprintf(stderr, "Failed to create solver: %s\n", solver_type);
        return;
    }

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, 0.0, 1.0, 0.0, 0.5);
    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Initialize solver parameters
    SolverParams params = solver_params_default();
    params.dt = 0.001;
    params.cfl = 0.5;
    params.tolerance = 1e-6;

    // Initialize solver
    solver_init(solver, grid, &params);

    // Measure execution time
    clock_t start = clock();
    SolverStats stats = solver_stats_default();

    for (int i = 0; i < iterations; i++) {
        solver_step(solver, field, grid, &params, &stats);
    }

    clock_t end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double cells_per_second = (double)(nx * ny * iterations) / cpu_time;

    printf("Execution time: %.3f seconds\n", cpu_time);
    printf("Performance: %.0f cell-updates/second\n", cells_per_second);
    printf("Memory usage: %.2f MB\n", (double)(nx * ny * 5 * sizeof(double)) / (1024 * 1024));

    // Cleanup
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(grid);
}

int main() {
    printf("CFD Library Performance Comparison\n");
    printf("==================================\n");

    // Test different grid sizes
    size_t grid_sizes[][2] = {
        {50, 25},    // Small
        {100, 50},   // Medium
        {200, 100},  // Large
        {400, 200}   // Very Large
    };

    int iterations = 100;

    for (int i = 0; i < 4; i++) {
        size_t nx = grid_sizes[i][0];
        size_t ny = grid_sizes[i][1];

        printf("\n");
        for (int j = 0; j < 50; j++)
            printf("=");
        printf("\n");
        printf("Grid Size: %zux%zu (%zu total cells)\n", nx, ny, nx * ny);
        for (int j = 0; j < 50; j++)
            printf("=");
        printf("\n");

        // Benchmark basic solver
        benchmark_solver("Basic Solver", SOLVER_TYPE_EXPLICIT_EULER, nx, ny, iterations);

        // Benchmark optimized solver
        benchmark_solver("Optimized Solver", SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, nx, ny,
                         iterations);
        
        // Benchmark OpenMP solver
        benchmark_solver("OpenMP Solver", SOLVER_TYPE_EXPLICIT_EULER_OMP, nx, ny, iterations);

        // Benchmark Projection solvers
        benchmark_solver("Projection Solver", SOLVER_TYPE_PROJECTION, nx, ny, iterations);
        benchmark_solver("Projection Optimized", SOLVER_TYPE_PROJECTION_OPTIMIZED, nx, ny, iterations);
        benchmark_solver("Projection OpenMP", SOLVER_TYPE_PROJECTION_OMP, nx, ny, iterations);



        // Calculate speedup
        // Note: This is a simplified example - for accurate benchmarking,
        // you'd want to run multiple trials and take averages
    }

    printf("\n");
    for (int j = 0; j < 50; j++)
        printf("=");
    printf("\n");
    printf("Benchmark completed!\n");
    printf("Note: Performance varies by hardware and system load.\n");
    printf("For production use, consider the optimized solver for large grids.\n");
    printf("\nModern Solver Interface Benefits:\n");
    printf("- Easy to switch between solver types\n");
    printf("- Consistent API across all solvers\n");
    printf("- Access to detailed solver statistics\n");

    return 0;
}
