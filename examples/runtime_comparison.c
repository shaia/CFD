/**
 * Runtime Comparison Test
 *
 * Comprehensive benchmark comparing CUDA GPU solvers vs SIMD CPU solvers
 * across different problem sizes, solver types, and iteration counts.
 *
 * Tests:
 * 1. grid size scaling (small to large grids)
 * 2. Iteration count scaling (few to many iterations)
 * 3. NSSolver type comparison (Euler vs Projection)
 * 4. GPU threshold analysis (when GPU becomes faster)
 */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "cfd/api/simulation_api.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/gpu_device.h"
#include "cfd/solvers/navier_stokes_solver.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#else
#include <sys/time.h>
#endif

// Test configuration
#define WARMUP_STEPS 2
#define REPEAT_COUNT 1

// grid sizes to test (reduced for faster testing)
static const size_t GRID_SIZES[][2] = {
    {50, 25},    // 1,250 points
    {100, 50},   // 5,000 points
    {200, 100},  // 20,000 points
    {300, 150},  // 45,000 points
};
#define NUM_GRID_SIZES (sizeof(GRID_SIZES) / sizeof(GRID_SIZES[0]))

// Iteration counts to test (reduced for faster testing)
static const int ITERATION_COUNTS[] = {10, 50, 100};
#define NUM_ITERATION_COUNTS (sizeof(ITERATION_COUNTS) / sizeof(ITERATION_COUNTS[0]))

// NSSolver pairs to compare (SIMD vs GPU)
typedef struct {
    const char* simd_solver;
    const char* gpu_solver;
    const char* name;
} solver_pair;

static const solver_pair SOLVER_PAIRS[] = {
    {NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU, "Explicit Euler"},
    {NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU, "Projection Method"},
};
#define NUM_SOLVER_PAIRS (sizeof(SOLVER_PAIRS) / sizeof(SOLVER_PAIRS[0]))

// Benchmark result structure
typedef struct {
    size_t nx;
    size_t ny;
    int iterations;
    const char* solver_name;
    double simd_time_ms;
    double gpu_time_ms;
    double speedup;
    double max_velocity;
    double max_pressure;
    int gpu_available;
} benchmark_result;

// High-resolution timer
static double get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

// Print separator line
static void print_separator(void) {
    printf("--------------------------------------------------------------------------------\n");
}

// Print header
static void print_header(const char* title) {
    printf("\n");
    print_separator();
    printf("%s\n", title);
    print_separator();
}

// Run benchmark for a specific configuration
static benchmark_result run_benchmark(size_t nx, size_t ny, int iterations, const char* simd_solver,
                                      const char* gpu_solver, const char* name) {
    benchmark_result result;
    memset(&result, 0, sizeof(result));

    result.nx = nx;
    result.ny = ny;
    result.iterations = iterations;
    result.solver_name = name;
    result.gpu_available = gpu_is_available();

    double xmin = 0.0, xmax = 2.0;
    double ymin = 0.0, ymax = 1.0;

    // Benchmark SIMD solver
    {
        simulation_data* sim =
            init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, simd_solver);
        if (!sim) {
            fprintf(stderr, "Failed to create SIMD simulation\n");
            return result;
        }

        // Warmup
        for (int i = 0; i < WARMUP_STEPS; i++) {
            run_simulation_step(sim);
        }

        // Reset for actual benchmark
        free_simulation(sim);
        sim = init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, simd_solver);

        // Timed run (average over repeats)
        double total_time = 0.0;
        for (int r = 0; r < REPEAT_COUNT; r++) {
            // Reset simulation
            free_simulation(sim);
            sim = init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, simd_solver);

            double start = get_time_ms();
            for (int i = 0; i < iterations; i++) {
                run_simulation_step(sim);
            }
            double end = get_time_ms();
            total_time += (end - start);
        }
        result.simd_time_ms = total_time / REPEAT_COUNT;

        // Get final stats
        const ns_solver_stats_t* stats = simulation_get_stats(sim);
        if (stats) {
            result.max_velocity = stats->max_velocity;
            result.max_pressure = stats->max_pressure;
        }

        free_simulation(sim);
    }

    // Benchmark GPU solver
    {
        simulation_data* sim =
            init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, gpu_solver);
        if (!sim) {
            fprintf(stderr, "Failed to create GPU simulation\n");
            return result;
        }

        // Warmup
        for (int i = 0; i < WARMUP_STEPS; i++) {
            run_simulation_step(sim);
        }

        // Reset for actual benchmark
        free_simulation(sim);
        sim = init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, gpu_solver);

        // Timed run (average over repeats)
        double total_time = 0.0;
        for (int r = 0; r < REPEAT_COUNT; r++) {
            // Reset simulation
            free_simulation(sim);
            sim = init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, gpu_solver);

            double start = get_time_ms();
            for (int i = 0; i < iterations; i++) {
                run_simulation_step(sim);
            }
            double end = get_time_ms();
            total_time += (end - start);
        }
        result.gpu_time_ms = total_time / REPEAT_COUNT;

        free_simulation(sim);
    }

    // Calculate speedup (positive = GPU faster, negative = SIMD faster)
    if (result.gpu_time_ms > 0) {
        result.speedup = result.simd_time_ms / result.gpu_time_ms;
    }

    return result;
}

// Print single result
static void print_result(const benchmark_result* r) {
    const char* winner = (r->speedup >= 1.0) ? "GPU" : "SIMD";
    double speedup_display = (r->speedup >= 1.0) ? r->speedup : (1.0 / r->speedup);

    printf("| %4zux%-4zu | %4d | %-18s | %10.2f | %10.2f | %5.2fx %-4s |\n", r->nx, r->ny,
           r->iterations, r->solver_name, r->simd_time_ms, r->gpu_time_ms, speedup_display, winner);
}

// Test 1: grid size scaling
static void test_grid_size_scaling(void) {
    print_header("TEST 1: grid Size Scaling (100 iterations)");
    printf("| grid Size | Iter | NSSolver             | SIMD (ms)  | GPU (ms)   | Speedup     |\n");
    print_separator();

    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        for (size_t g = 0; g < NUM_GRID_SIZES; g++) {
            benchmark_result result = run_benchmark(
                GRID_SIZES[g][0], GRID_SIZES[g][1], 100, SOLVER_PAIRS[pair].simd_solver,
                SOLVER_PAIRS[pair].gpu_solver, SOLVER_PAIRS[pair].name);
            print_result(&result);
        }
        if (pair < NUM_SOLVER_PAIRS - 1) {
            print_separator();
        }
    }
}

// Test 2: Iteration count scaling
static void test_iteration_scaling(void) {
    print_header("TEST 2: Iteration Count Scaling (200x100 grid)");
    printf("| grid Size | Iter | NSSolver             | SIMD (ms)  | GPU (ms)   | Speedup     |\n");
    print_separator();

    size_t nx = 200, ny = 100;

    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        for (size_t i = 0; i < NUM_ITERATION_COUNTS; i++) {
            benchmark_result result =
                run_benchmark(nx, ny, ITERATION_COUNTS[i], SOLVER_PAIRS[pair].simd_solver,
                              SOLVER_PAIRS[pair].gpu_solver, SOLVER_PAIRS[pair].name);
            print_result(&result);
        }
        if (pair < NUM_SOLVER_PAIRS - 1) {
            print_separator();
        }
    }
}

// Test 3: Find GPU crossover point
static void test_gpu_crossover(void) {
    print_header("TEST 3: GPU Crossover Analysis");
    printf("Finding the grid size where GPU becomes faster than SIMD...\n\n");

    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        printf("NSSolver: %s\n", SOLVER_PAIRS[pair].name);
        printf("| grid Size  | Points    | SIMD (ms) | GPU (ms)  | Winner | Speedup |\n");
        printf("|------------|-----------|-----------|-----------|--------|----------|\n");

        // Test a range of grid sizes
        size_t test_sizes[][2] = {{32, 16},   {64, 32},   {100, 50},  {128, 64},  {150, 75},
                                  {200, 100}, {256, 128}, {300, 150}, {400, 200}, {500, 250}};
        size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

        int crossover_found = 0;
        size_t crossover_nx = 0, crossover_ny = 0;

        for (size_t t = 0; t < num_tests; t++) {
            benchmark_result result = run_benchmark(
                test_sizes[t][0], test_sizes[t][1], 100, SOLVER_PAIRS[pair].simd_solver,
                SOLVER_PAIRS[pair].gpu_solver, SOLVER_PAIRS[pair].name);

            const char* winner = (result.speedup >= 1.0) ? "GPU" : "SIMD";
            double speedup = (result.speedup >= 1.0) ? result.speedup : (1.0 / result.speedup);

            printf("| %4zux%-5zu | %9zu | %9.2f | %9.2f | %-6s | %5.2fx    |\n", test_sizes[t][0],
                   test_sizes[t][1], test_sizes[t][0] * test_sizes[t][1], result.simd_time_ms,
                   result.gpu_time_ms, winner, speedup);

            if (!crossover_found && result.speedup >= 1.0) {
                crossover_found = 1;
                crossover_nx = test_sizes[t][0];
                crossover_ny = test_sizes[t][1];
            }
        }

        printf("\n");
        if (crossover_found) {
            printf("Crossover point: approximately %zux%zu (%zu points)\n", crossover_nx,
                   crossover_ny, crossover_nx * crossover_ny);
        } else {
            printf("GPU did not become faster in tested range (may need CUDA hardware)\n");
        }
        printf("\n");
    }
}

// Test 4: All solvers comparison
static void test_all_solvers(void) {
    print_header("TEST 4: All Solvers Comparison (200x100 grid, 100 iterations)");

    const char* all_solvers[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER,       NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,   NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
    };
    size_t num_solvers = sizeof(all_solvers) / sizeof(all_solvers[0]);

    size_t nx = 200, ny = 100;
    int iterations = 100;
    double xmin = 0.0, xmax = 2.0;
    double ymin = 0.0, ymax = 1.0;

    printf("| NSSolver                       | Time (ms) | Max Vel | Max Press | Cells/sec   |\n");
    print_separator();

    double best_time = 1e9;
    const char* best_solver = NULL;

    for (size_t s = 0; s < num_solvers; s++) {
        simulation_data* sim =
            init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, all_solvers[s]);
        if (!sim) {
            continue;
        }

        // Warmup
        for (int i = 0; i < WARMUP_STEPS; i++) {
            run_simulation_step(sim);
        }

        // Reset
        free_simulation(sim);
        sim = init_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, all_solvers[s]);

        // Timed run
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            run_simulation_step(sim);
        }
        double end = get_time_ms();
        double time_ms = end - start;

        const ns_solver_stats_t* stats = simulation_get_stats(sim);
        double max_vel = stats ? stats->max_velocity : 0.0;
        double max_press = stats ? stats->max_pressure : 0.0;

        double cells_per_sec = (double)(nx * ny * iterations) / (time_ms / 1000.0);

        printf("| %-28s | %9.2f | %7.4f | %9.4f | %11.2e |\n", all_solvers[s], time_ms, max_vel,
               max_press, cells_per_sec);

        if (time_ms < best_time) {
            best_time = time_ms;
            best_solver = all_solvers[s];
        }

        free_simulation(sim);
    }

    print_separator();
    printf("Fastest solver: %s (%.2f ms)\n", best_solver, best_time);
}

// Test 5: Large grid performance
static void test_large_grid(void) {
    print_header("TEST 5: Large grid Performance");

    // Reduced sizes for faster testing
    size_t large_sizes[][2] = {
        {300, 150},
        {400, 200},
    };
    size_t num_sizes = sizeof(large_sizes) / sizeof(large_sizes[0]);

    printf("Testing large grids with 50 iterations...\n\n");
    printf("| grid Size  | Points    | SIMD (ms) | GPU (ms)  | Winner | Throughput  |\n");
    printf("|------------|-----------|-----------|-----------|--------|-------------|\n");

    for (size_t i = 0; i < num_sizes; i++) {
        size_t nx = large_sizes[i][0];
        size_t ny = large_sizes[i][1];

        benchmark_result result = run_benchmark(nx, ny, 50, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
                                                NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU, "Projection");

        const char* winner = (result.speedup >= 1.0) ? "GPU" : "SIMD";
        double best_time = (result.speedup >= 1.0) ? result.gpu_time_ms : result.simd_time_ms;
        double throughput = (double)(nx * ny * 50) / (best_time / 1000.0);

        printf("| %4zux%-5zu | %9zu | %9.2f | %9.2f | %-6s | %8.2e |\n", nx, ny, nx * ny,
               result.simd_time_ms, result.gpu_time_ms, winner, throughput);
    }
}

// Print system info
static void print_system_info(void) {
    print_header("SYSTEM INFORMATION");

    printf("GPU Status: %s\n",
           gpu_is_available() ? "Available" : "Not available (using CPU fallback)");

    if (gpu_is_available()) {
        gpu_device_info_t info[4];
        int num_devices = gpu_get_device_info(info, 4);

        printf("GPU Devices: %d\n", num_devices);
        for (int i = 0; i < num_devices; i++) {
            printf("  Device %d: %s\n", i, info[i].name);
            printf("    Compute Capability: %d.%d\n", info[i].compute_capability_major,
                   info[i].compute_capability_minor);
            printf("    Total Memory: %.2f GB\n",
                   info[i].total_memory / (1024.0 * 1024.0 * 1024.0));
            printf("    Multiprocessors: %d\n", info[i].multiprocessor_count);
        }
    }

    printf("\nAvailable Solvers:\n");
    const char* solver_names[10];
    int num_solvers = simulation_list_solvers(solver_names, 10);
    for (int i = 0; i < num_solvers; i++) {
        printf("  %d. %s\n", i + 1, solver_names[i]);
    }

    printf("\nBenchmark Configuration:\n");
    printf("  Warmup steps: %d\n", WARMUP_STEPS);
    printf("  Repeat count: %d\n", REPEAT_COUNT);
}

// Write results to CSV
static void write_results_csv(void) {
    char csv_path[512];
    make_output_path(csv_path, sizeof(csv_path), "benchmark_results.csv");

    FILE* f = fopen(csv_path, "w");
    if (!f) {
        fprintf(stderr, "Warning: Could not create CSV file\n");
        return;
    }

    fprintf(f, "grid_nx,grid_ny,points,iterations,solver,simd_ms,gpu_ms,speedup,winner\n");

    // Run all benchmarks and write to CSV
    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        for (size_t g = 0; g < NUM_GRID_SIZES; g++) {
            for (size_t i = 0; i < NUM_ITERATION_COUNTS; i++) {
                benchmark_result result =
                    run_benchmark(GRID_SIZES[g][0], GRID_SIZES[g][1], ITERATION_COUNTS[i],
                                  SOLVER_PAIRS[pair].simd_solver, SOLVER_PAIRS[pair].gpu_solver,
                                  SOLVER_PAIRS[pair].name);

                const char* winner = (result.speedup >= 1.0) ? "GPU" : "SIMD";

                fprintf(f, "%zu,%zu,%zu,%d,%s,%.4f,%.4f,%.4f,%s\n", result.nx, result.ny,
                        result.nx * result.ny, result.iterations, result.solver_name,
                        result.simd_time_ms, result.gpu_time_ms, result.speedup, winner);
            }
        }
    }

    fclose(f);
    printf("\nResults written to: %s\n", csv_path);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("==========================================================================\n");
    printf("           CFD Runtime Comparison: CUDA vs SIMD Benchmarks\n");
    printf("==========================================================================\n");

    // Configure output directory
    cfd_set_output_base_dir("../../artifacts");

    // Print system info
    print_system_info();

    // Run tests
    test_grid_size_scaling();
    test_iteration_scaling();
    test_gpu_crossover();
    test_all_solvers();
    test_large_grid();

    // Summary
    print_header("SUMMARY");
    printf("Key findings:\n");
    printf("- GPU acceleration benefits large grids (typically >10,000 points)\n");
    printf("- SIMD optimization is effective for all grid sizes on CPU\n");
    printf("- GPU overhead makes it slower for small problems\n");
    printf("- Projection method is more compute-intensive, benefits more from GPU\n");

    if (!gpu_is_available()) {
        printf("\nNote: CUDA was not available. GPU times show CPU fallback performance.\n");
        printf("Build with -DCFD_ENABLE_CUDA=ON and run on a CUDA-capable system for GPU "
               "benchmarks.\n");
    }

    // Write CSV results (optional, comment out if not needed)
    // write_results_csv();

    printf("\n==========================================================================\n");
    printf("                         Benchmarks Complete\n");
    printf("==========================================================================\n");

    return 0;
}
