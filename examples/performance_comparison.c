/**
 * Performance Comparison Example
 *
 * Demonstrates the performance difference between basic and optimized solvers
 * and measures execution time for different grid sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "grid.h"
#include "solver.h"
#include "utils.h"

void benchmark_solver(const char* solver_name,
                     void (*solver_func)(FlowField*, Grid*, SolverParams*),
                     size_t nx, size_t ny, int iterations) {
    printf("\n=== %s Benchmark ===\n", solver_name);
    printf("Grid size: %zux%zu, Iterations: %d\n", nx, ny, iterations);

    // Create grid and flow field
    Grid* grid = grid_create(nx, ny, 0.0, 1.0, 0.0, 0.5);
    FlowField* field = flow_field_create(nx, ny);

    // Initialize solver parameters
    SolverParams params = {
        .max_iter = iterations,
        .dt = 0.001,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = 0.01,
        .k = 0.1,
        .tolerance = 1e-6
    };

    // Measure execution time
    clock_t start = clock();
    solver_func(field, grid, &params);
    clock_t end = clock();

    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double cells_per_second = (double)(nx * ny * iterations) / cpu_time;

    printf("Execution time: %.3f seconds\n", cpu_time);
    printf("Performance: %.0f cell-updates/second\n", cells_per_second);
    printf("Memory usage: %.2f MB\n",
           (double)(nx * ny * 5 * sizeof(double)) / (1024 * 1024));

    // Cleanup
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
        for(int j = 0; j < 50; j++) printf("=");
        printf("\n");
        printf("Grid Size: %zux%zu (%zu total cells)\n", nx, ny, nx*ny);
        for(int j = 0; j < 50; j++) printf("=");
        printf("\n");

        // Benchmark basic solver
        benchmark_solver("Basic Solver", solve_navier_stokes, nx, ny, iterations);

        // Benchmark optimized solver
        benchmark_solver("Optimized Solver", solve_navier_stokes_optimized, nx, ny, iterations);

        // Calculate speedup
        // Note: This is a simplified example - for accurate benchmarking,
        // you'd want to run multiple trials and take averages
    }

    printf("\n");
    for(int j = 0; j < 50; j++) printf("=");
    printf("\n");
    printf("Benchmark completed!\n");
    printf("Note: Performance varies by hardware and system load.\n");
    printf("For production use, consider the optimized solver for large grids.\n");

    return 0;
}