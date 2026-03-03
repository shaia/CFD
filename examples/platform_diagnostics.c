/**
 * Platform Diagnostics Example
 *
 * Queries runtime platform capabilities, demonstrates derived field
 * computation and statistics, and shows error handling patterns.
 *
 * This example demonstrates:
 *   - SIMD architecture detection (AVX2, NEON)
 *   - Backend availability checking (BC, Poisson solvers)
 *   - NS solver registration listing
 *   - Derived field computation (velocity magnitude, statistics)
 *   - Error handling API (status codes, error messages)
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void print_simd_info(void) {
    printf("1. SIMD Capabilities\n");
    printf("   Architecture: %s\n", cfd_get_simd_name());
    printf("   AVX2:  %s\n", cfd_has_avx2() ? "yes" : "no");
    printf("   NEON:  %s\n", cfd_has_neon() ? "yes" : "no");
    printf("   Any:   %s\n", cfd_has_simd() ? "yes" : "no");
}

static void print_backend_availability(void) {
    printf("\n2. Backend Availability\n");

    printf("   Boundary Conditions:\n");
    printf("     Scalar:  %s\n", bc_backend_available(BC_BACKEND_SCALAR) ? "available" : "not available");
    printf("     SIMD:    %s\n", bc_backend_available(BC_BACKEND_SIMD) ? "available" : "not available");
    printf("     OpenMP:  %s\n", bc_backend_available(BC_BACKEND_OMP) ? "available" : "not available");

    printf("   Poisson Solvers:\n");
    printf("     Scalar:  %s\n", poisson_solver_backend_available(POISSON_BACKEND_SCALAR) ? "available" : "not available");
    printf("     SIMD:    %s (%s)\n",
           poisson_solver_backend_available(POISSON_BACKEND_SIMD) ? "available" : "not available",
           poisson_solver_get_simd_arch_name());
    printf("     OpenMP:  %s\n", poisson_solver_backend_available(POISSON_BACKEND_OMP) ? "available" : "not available");
}

static void print_available_solvers(void) {
    printf("\n3. Available NS Solvers\n");

    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        printf("   Failed to create registry\n");
        return;
    }
    cfd_registry_register_defaults(registry);

    const char* names[32];
    int count = cfd_registry_list(registry, names, 32);
    int to_print = count < 32 ? count : 32;
    printf("   Found %d solver(s):\n", count);
    for (int i = 0; i < to_print; i++) {
        printf("     - %s\n", names[i]);
    }
    if (count > 32) {
        printf("     ... and %d more not shown\n", count - 32);
    }

    cfd_registry_destroy(registry);
}

static void demonstrate_derived_fields(void) {
    printf("\n4. Derived Fields & Statistics\n");

    size_t nx = 32, ny = 32;
    grid* g = grid_create(nx, ny, 1, 0.0, 2.0 * M_PI, 0.0, 2.0 * M_PI, 0.0, 0.0);
    if (!g) { printf("   Failed to create grid\n"); return; }
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, 1);
    if (!field) { printf("   Failed to create flow field\n"); grid_destroy(g); return; }
    initialize_flow_field(field, g);

    /* Initialize with Taylor-Green vortex pattern */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = g->x[i];
            double y = g->y[j];
            field->u[j * nx + i] = 0.1 * cos(x) * sin(y);
            field->v[j * nx + i] = -0.1 * sin(x) * cos(y);
            field->p[j * nx + i] = 0.0;
        }
    }

    /* Compute derived fields and statistics */
    derived_fields* df = derived_fields_create(nx, ny, 1);
    if (!df) {
        printf("   Failed to create derived fields\n");
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }

    derived_fields_compute_velocity_magnitude(df, field);
    derived_fields_compute_statistics(df, field);

    printf("   Taylor-Green vortex (%zux%zu):\n", nx, ny);
    printf("   u-velocity:  min=%.4f, max=%.4f, avg=%.4f\n",
           df->u_stats.min_val, df->u_stats.max_val, df->u_stats.avg_val);
    printf("   v-velocity:  min=%.4f, max=%.4f, avg=%.4f\n",
           df->v_stats.min_val, df->v_stats.max_val, df->v_stats.avg_val);
    printf("   pressure:    min=%.4f, max=%.4f, avg=%.4f\n",
           df->p_stats.min_val, df->p_stats.max_val, df->p_stats.avg_val);
    printf("   |V| mag:     min=%.4f, max=%.4f, avg=%.4f\n",
           df->vel_mag_stats.min_val, df->vel_mag_stats.max_val, df->vel_mag_stats.avg_val);

    /* Also demonstrate standalone calculate_field_statistics */
    field_stats u_stats = calculate_field_statistics(field->u, nx * ny);
    printf("   (standalone) u sum=%.4f\n", u_stats.sum_val);

    derived_fields_destroy(df);
    flow_field_destroy(field);
    grid_destroy(g);
}

static void demonstrate_error_handling(void) {
    printf("\n5. Error Handling Patterns\n");

    /* Example 1: Request a nonexistent solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        printf("   Failed to create registry\n");
        return;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, "nonexistent_solver");
    if (!solver) {
        printf("   Requesting 'nonexistent_solver'... NULL (expected)\n");
        printf("     Last error:  \"%s\"\n", cfd_get_last_error());
        printf("     Status code: %s (%d)\n",
               cfd_get_error_string(cfd_get_last_status()),
               cfd_get_last_status());
        cfd_clear_error();
    }

    /* Example 2: Graceful fallback for unavailable backend */
    solver = cfd_solver_create(registry, "projection_jacobi_gpu");
    if (!solver) {
        printf("   Requesting GPU solver... not available\n");
        printf("     Falling back to scalar...\n");
        solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
        if (solver) {
            printf("     Fallback to '%s': success\n", solver->name);
            solver_destroy(solver);
            cfd_clear_error();
        }
    } else {
        printf("   GPU solver available: %s\n", solver->name);
        solver_destroy(solver);
    }

    cfd_registry_destroy(registry);
}

int main(void) {
    printf("CFD Platform Diagnostics\n");
    printf("========================\n\n");

    print_simd_info();
    print_backend_availability();
    print_available_solvers();
    demonstrate_derived_fields();
    demonstrate_error_handling();

    printf("\nDiagnostics complete.\n");
    return 0;
}
