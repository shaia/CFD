/**
 * Performance Comparison Example
 *
 * Demonstrates the performance difference between basic and optimized solvers
 * using the modern pluggable solver interface.
 */

#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Elapsed wall time. clock() would charge every OpenMP thread's CPU time on Linux and
// macOS, which makes threaded rows look slower than they run.
static double wall_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (1e-9 * (double)ts.tv_nsec);
}

// The library's message for a failure, falling back to the status name. The stored
// message is used only when it belongs to this status: a failure that sets none would
// otherwise be reported with an earlier, unrelated one.
static const char* failure_reason(cfd_status_t status) {
    const char* reason = cfd_get_last_error();
    return (reason && cfd_get_last_status() == status) ? reason : cfd_get_error_string(status);
}

// Benchmark one solver type. Returns CFD_SUCCESS when it ran, or was skipped because
// this build or CPU lacks its backend; any other status is a failure of the run.
cfd_status_t benchmark_solver(const char* solver_name, const char* solver_type, size_t nx,
                              size_t ny, int iterations) {
    printf("\n=== %s Benchmark ===\n", solver_name);
    printf("Grid size: %zux%zu, Iterations: %d\n", nx, ny, iterations);

    // A previous benchmark's message must not be reported as this one's reason
    cfd_clear_error();
    cfd_status_t status = CFD_SUCCESS;
    struct NSSolverRegistry* registry = cfd_registry_create();
    struct NSSolver* solver = NULL;
    // grid_create only allocates; the coordinates and spacings are zero until
    // grid_initialize_uniform fills them.
    grid* grid = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
    flow_field* field = flow_field_create(nx, ny, 1);
    if (!registry || !grid || !field) {
        status = CFD_ERROR_NOMEM;
        printf("Failed: %s\n", failure_reason(status));
        goto cleanup;
    }
    cfd_registry_register_defaults(registry);
    grid_initialize_uniform(grid);
    initialize_flow_field(field, grid);

    // Create the solver. The checked variant refuses a backend this build or CPU lacks
    // with CFD_ERROR_UNSUPPORTED; plain cfd_solver_create() would report an OpenMP
    // solver on a build without OpenMP as an unregistered name (CFD_ERROR_NOT_FOUND).
    solver = cfd_solver_create_checked(registry, solver_type);
    if (!solver) {
        status = cfd_get_last_status() != CFD_SUCCESS ? cfd_get_last_status() : CFD_ERROR;
        if (status == CFD_ERROR_UNSUPPORTED) {
            printf("Skipped: %s\n", failure_reason(status));
            status = CFD_SUCCESS;
        } else {
            printf("Failed to create solver %s: %s\n", solver_type, failure_reason(status));
        }
        goto cleanup;
    }

    // Initialize solver parameters
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 0.001;
    params.cfl = 0.5;
    params.tolerance = 1e-6;

    // A backend this build or CPU lacks refuses with CFD_ERROR_UNSUPPORTED: report it
    // rather than time it. Any other refusal is a real failure.
    status = solver_init(solver, grid, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        printf("Skipped: %s\n", failure_reason(status));
        status = CFD_SUCCESS;
        goto cleanup;
    }
    if (status != CFD_SUCCESS) {
        printf("Init failed: %s\n", failure_reason(status));
        goto cleanup;
    }

    // Measure execution time
    double start = wall_seconds();
    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int i = 0; i < iterations; i++) {
        status = solver_step(solver, field, grid, &params, &stats);
        if (status != CFD_SUCCESS) {
            printf("Failed at step %d: %s\n", i, failure_reason(status));
            goto cleanup;
        }
    }

    double elapsed = wall_seconds() - start;
    double cells_per_second = (double)(nx * ny * iterations) / elapsed;

    printf("Execution time: %.3f seconds\n", elapsed);
    printf("Performance: %.0f cell-updates/second\n", cells_per_second);
    printf("Memory usage: %.2f MB\n", (double)(nx * ny * 5 * sizeof(double)) / (1024 * 1024));

cleanup:
    solver_destroy(solver);
    flow_field_destroy(field);
    grid_destroy(grid);
    cfd_registry_destroy(registry);
    return status;
}

int main() {
    printf("CFD Library Performance Comparison\n");
    printf("==================================\n");

    // Benchmarks that could run but failed; any one makes the process exit nonzero
    int failures = 0;

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
        for (int j = 0; j < 50; j++) {
            printf("=");
        }
        printf("\n");
        printf("Grid Size: %zux%zu (%zu total cells)\n", nx, ny, nx * ny);
        for (int j = 0; j < 50; j++) {
            printf("=");
        }
        printf("\n");

        static const struct {
            const char* name;
            const char* type;
        } solvers[] = {
            {"Basic NSSolver", NS_SOLVER_TYPE_EXPLICIT_EULER},
            {"Optimized NSSolver", NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED},
            {"OpenMP NSSolver", NS_SOLVER_TYPE_EXPLICIT_EULER_OMP},
            {"Projection NSSolver", NS_SOLVER_TYPE_PROJECTION},
            {"Projection Optimized", NS_SOLVER_TYPE_PROJECTION_OPTIMIZED},
            {"Projection OpenMP", NS_SOLVER_TYPE_PROJECTION_OMP},
        };
        for (size_t s = 0; s < sizeof(solvers) / sizeof(solvers[0]); s++) {
            if (benchmark_solver(solvers[s].name, solvers[s].type, nx, ny, iterations) !=
                CFD_SUCCESS) {
                failures++;
            }
        }


        // Calculate speedup
        // Note: This is a simplified example - for accurate benchmarking,
        // you'd want to run multiple trials and take averages
    }

    printf("\n");
    for (int j = 0; j < 50; j++) {
        printf("=");
    }
    printf("\n");
    if (failures > 0) {
        printf("Benchmark FAILED: %d solver run(s) failed; see the messages above.\n", failures);
        return EXIT_FAILURE;
    }
    printf("Benchmark completed!\n");
    printf("Note: Performance varies by hardware and system load.\n");
    printf("For production use, consider the optimized solver for large grids.\n");
    printf("\nModern NSSolver Interface Benefits:\n");
    printf("- Easy to switch between solver types\n");
    printf("- Consistent API across all solvers\n");
    printf("- Access to detailed solver statistics\n");

    return 0;
}
