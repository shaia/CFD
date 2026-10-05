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
    {NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_GPU, "Projection Method"},
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
    int simd_ran;  /* 0 when the SIMD solver's backend is missing here */
    int gpu_ran;   /* 0 when the GPU solver's backend is missing here */
    int failed;    /* a solver that could run here failed */
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

// Create a simulation for `solver`. Returns NULL with *unavailable = 1 when this build
// or machine lacks the solver's backend: cfd_solver_create_checked() refuses it with
// CFD_ERROR_UNSUPPORTED before looking the name up (a GPU solver without CUDA), or its
// init does (an AVX2 solver without AVX2). NULL with *unavailable = 0 is a failure,
// already reported on stderr -- an unregistered name included: that is a misspelled or
// removed solver, not a missing backend.
static simulation_data* create_sim(const char* solver, size_t nx, size_t ny, int* unavailable) {
    *unavailable = 0;
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        fprintf(stderr, "  %s: out of memory\n", solver);
        return NULL;
    }
    cfd_registry_register_defaults(registry);
    cfd_clear_error();
    ns_solver_t* probe = cfd_solver_create_checked(registry, solver);
    cfd_status_t probe_status = cfd_get_last_status();
    if (probe) {
        solver_destroy(probe);
    }
    cfd_registry_destroy(registry);
    if (!probe) {
        *unavailable = (probe_status == CFD_ERROR_UNSUPPORTED);
        if (!*unavailable) {
            fprintf(stderr, "  %s: not created: %s\n", solver, cfd_get_error_string(probe_status));
        }
        return NULL;
    }

    cfd_clear_error();
    simulation_data* sim = init_simulation_with_solver(nx, ny, 1, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, solver);
    if (!sim) {
        cfd_status_t status = cfd_get_last_status();
        *unavailable = (status == CFD_ERROR_UNSUPPORTED);
        if (!*unavailable) {
            fprintf(stderr, "  %s: init failed: %s\n", solver, cfd_get_error_string(status));
        }
        return NULL;
    }
    sim->params.dt = 0.005; /* the step this example has always run at */
    return sim;
}

// Time `iterations` steps of `solver`, after WARMUP_STEPS on a separate simulation,
// averaged over REPEAT_COUNT fresh simulations. Returns CFD_SUCCESS with *time_ms set,
// CFD_ERROR_UNSUPPORTED when the backend is missing here (*unavailable = 1), or the
// status of the failure. max_vel and max_press may be NULL.
static cfd_status_t time_solver(const char* solver, size_t nx, size_t ny, int iterations,
                                double* time_ms, double* max_vel, double* max_press,
                                int* unavailable) {
    double total_time = 0.0;
    for (int run = -1; run < REPEAT_COUNT; run++) {  // run -1 is the warmup
        simulation_data* sim = create_sim(solver, nx, ny, unavailable);
        if (!sim) {
            return *unavailable ? CFD_ERROR_UNSUPPORTED : CFD_ERROR;
        }
        int steps = (run < 0) ? WARMUP_STEPS : iterations;
        double start = get_time_ms();
        for (int i = 0; i < steps; i++) {
            cfd_status_t status = run_simulation_step(sim);
            if (status != CFD_SUCCESS) {
                fprintf(stderr, "  %s: step %d failed: %s\n", solver, i,
                        cfd_get_error_string(status));
                free_simulation(sim);
                return status;
            }
        }
        double end = get_time_ms();
        if (run >= 0) {
            total_time += (end - start);
            const ns_solver_stats_t* stats = simulation_get_stats(sim);
            if (stats && max_vel) *max_vel = stats->max_velocity;
            if (stats && max_press) *max_press = stats->max_pressure;
        }
        free_simulation(sim);
    }
    *time_ms = total_time / REPEAT_COUNT;
    return CFD_SUCCESS;
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

    int unavailable = 0;
    cfd_status_t status = time_solver(simd_solver, nx, ny, iterations, &result.simd_time_ms,
                                      &result.max_velocity, &result.max_pressure, &unavailable);
    result.simd_ran = (status == CFD_SUCCESS);
    if (status != CFD_SUCCESS && !unavailable) {
        result.failed = 1;
    }

    status = time_solver(gpu_solver, nx, ny, iterations, &result.gpu_time_ms, NULL, NULL,
                         &unavailable);
    result.gpu_ran = (status == CFD_SUCCESS);
    if (status != CFD_SUCCESS && !unavailable) {
        result.failed = 1;
    }

    // Speedup only means something when both ran (above 1 = GPU faster)
    if (result.simd_ran && result.gpu_ran && result.gpu_time_ms > 0) {
        result.speedup = result.simd_time_ms / result.gpu_time_ms;
    }

    return result;
}

// Print one timing cell: the time, or n/a when that solver's backend is missing here
static void print_time_cell(int ran, double time_ms) {
    if (ran) {
        printf(" %10.2f |", time_ms);
    } else {
        printf(" %10s |", "n/a");
    }
}

// Print single result
static void print_result(const benchmark_result* r) {
    printf("| %4zux%-4zu | %4d | %-18s |", r->nx, r->ny, r->iterations, r->solver_name);
    print_time_cell(r->simd_ran, r->simd_time_ms);
    print_time_cell(r->gpu_ran, r->gpu_time_ms);
    if (r->simd_ran && r->gpu_ran) {
        const char* winner = (r->speedup >= 1.0) ? "GPU" : "SIMD";
        double speedup_display = (r->speedup >= 1.0) ? r->speedup : (1.0 / r->speedup);
        printf(" %5.2fx %-4s |\n", speedup_display, winner);
    } else {
        printf(" %-11s |\n", "n/a");
    }
}

// Test 1: grid size scaling. Returns the number of failed benchmarks.
static int test_grid_size_scaling(void) {
    print_header("TEST 1: grid Size Scaling (100 iterations)");
    printf("| grid Size | Iter | NSSolver             | SIMD (ms)  | GPU (ms)   | Speedup     |\n");
    print_separator();

    int failures = 0;
    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        for (size_t g = 0; g < NUM_GRID_SIZES; g++) {
            benchmark_result result = run_benchmark(
                GRID_SIZES[g][0], GRID_SIZES[g][1], 100, SOLVER_PAIRS[pair].simd_solver,
                SOLVER_PAIRS[pair].gpu_solver, SOLVER_PAIRS[pair].name);
            print_result(&result);
            failures += result.failed;
        }
        if (pair < NUM_SOLVER_PAIRS - 1) {
            print_separator();
        }
    }
    return failures;
}

// Test 2: Iteration count scaling. Returns the number of failed benchmarks.
static int test_iteration_scaling(void) {
    print_header("TEST 2: Iteration Count Scaling (200x100 grid)");
    printf("| grid Size | Iter | NSSolver             | SIMD (ms)  | GPU (ms)   | Speedup     |\n");
    print_separator();

    size_t nx = 200, ny = 100;

    int failures = 0;
    for (size_t pair = 0; pair < NUM_SOLVER_PAIRS; pair++) {
        for (size_t i = 0; i < NUM_ITERATION_COUNTS; i++) {
            benchmark_result result =
                run_benchmark(nx, ny, ITERATION_COUNTS[i], SOLVER_PAIRS[pair].simd_solver,
                              SOLVER_PAIRS[pair].gpu_solver, SOLVER_PAIRS[pair].name);
            print_result(&result);
            failures += result.failed;
        }
        if (pair < NUM_SOLVER_PAIRS - 1) {
            print_separator();
        }
    }
    return failures;
}

// Test 3: Find GPU crossover point. Returns the number of failed benchmarks.
static int test_gpu_crossover(void) {
    print_header("TEST 3: GPU Crossover Analysis");
    printf("Finding the grid size where GPU becomes faster than SIMD...\n\n");

    int failures = 0;
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
            failures += result.failed;

            if (!(result.simd_ran && result.gpu_ran)) {
                printf("| %4zux%-5zu | %9zu | %9s | %9s | %-6s | %-9s |\n", test_sizes[t][0],
                       test_sizes[t][1], test_sizes[t][0] * test_sizes[t][1],
                       result.simd_ran ? "ran" : "n/a", result.gpu_ran ? "ran" : "n/a", "n/a",
                       "n/a");
                continue;
            }

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
    return failures;
}

// Test 4: All solvers comparison. Returns the number of solvers that failed.
static int test_all_solvers(void) {
    print_header("TEST 4: All Solvers Comparison (200x100 grid, 100 iterations)");

    const char* all_solvers[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER,       NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,   NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_GPU,
    };
    size_t num_solvers = sizeof(all_solvers) / sizeof(all_solvers[0]);

    size_t nx = 200, ny = 100;
    int iterations = 100;

    printf("| NSSolver                       | Time (ms) | Max Vel | Max Press | Cells/sec   |\n");
    print_separator();

    double best_time = 1e9;
    const char* best_solver = NULL;
    int failures = 0;

    for (size_t s = 0; s < num_solvers; s++) {
        double time_ms = 0.0, max_vel = 0.0, max_press = 0.0;
        int unavailable = 0;
        cfd_status_t status = time_solver(all_solvers[s], nx, ny, iterations, &time_ms, &max_vel,
                                          &max_press, &unavailable);
        if (status != CFD_SUCCESS) {
            printf("| %-28s | %-9s |\n", all_solvers[s], unavailable ? "n/a" : "FAILED");
            failures += !unavailable;
            continue;
        }

        double cells_per_sec = (double)(nx * ny * iterations) / (time_ms / 1000.0);

        printf("| %-28s | %9.2f | %7.4f | %9.4f | %11.2e |\n", all_solvers[s], time_ms, max_vel,
               max_press, cells_per_sec);

        if (time_ms < best_time) {
            best_time = time_ms;
            best_solver = all_solvers[s];
        }
    }

    print_separator();
    if (best_solver) {
        printf("Fastest solver: %s (%.2f ms)\n", best_solver, best_time);
    }
    return failures;
}

// Test 5: Large grid performance. Returns the number of failed benchmarks.
static int test_large_grid(void) {
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

    int failures = 0;
    for (size_t i = 0; i < num_sizes; i++) {
        size_t nx = large_sizes[i][0];
        size_t ny = large_sizes[i][1];

        benchmark_result result = run_benchmark(nx, ny, 50, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
                                                NS_SOLVER_TYPE_PROJECTION_GPU, "Projection");
        failures += result.failed;

        if (!(result.simd_ran && result.gpu_ran)) {
            printf("| %4zux%-5zu | %9zu | %9s | %9s | %-6s | %-11s |\n", nx, ny, nx * ny,
                   result.simd_ran ? "ran" : "n/a", result.gpu_ran ? "ran" : "n/a", "n/a", "n/a");
            continue;
        }

        const char* winner = (result.speedup >= 1.0) ? "GPU" : "SIMD";
        double best_time = (result.speedup >= 1.0) ? result.gpu_time_ms : result.simd_time_ms;
        double throughput = (double)(nx * ny * 50) / (best_time / 1000.0);

        printf("| %4zux%-5zu | %9zu | %9.2f | %9.2f | %-6s | %8.2e |\n", nx, ny, nx * ny,
               result.simd_time_ms, result.gpu_time_ms, winner, throughput);
    }
    return failures;
}

// Print system info
static void print_system_info(void) {
    print_header("SYSTEM INFORMATION");

    printf("GPU Status: %s\n",
           gpu_is_available() ? "Available" : "Not available");

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
    int failures = test_grid_size_scaling();
    failures += test_iteration_scaling();
    failures += test_gpu_crossover();
    failures += test_all_solvers();
    failures += test_large_grid();

    // Summary
    print_header("SUMMARY");
    printf("Key findings:\n");
    printf("- GPU acceleration benefits large grids (typically >10,000 points)\n");
    printf("- SIMD optimization is effective for all grid sizes on CPU\n");
    printf("- GPU overhead makes it slower for small problems\n");
    printf("- Projection method is more compute-intensive, benefits more from GPU\n");

    if (!gpu_is_available()) {
        printf("\nNote: CUDA was not available. GPU solver benchmarks were skipped.\n");
        printf("Build with -DCFD_ENABLE_CUDA=ON and run on a CUDA-capable system for GPU "
               "benchmarks.\n");
    }

    // Write CSV results (optional, comment out if not needed)
    // write_results_csv();

    printf("\n==========================================================================\n");
    if (failures > 0) {
        printf("        Benchmarks FAILED: %d solver run(s) failed; see stderr\n", failures);
        printf("==========================================================================\n");
        return 1;
    }
    printf("                         Benchmarks Complete\n");
    printf("==========================================================================\n");

    return 0;
}
