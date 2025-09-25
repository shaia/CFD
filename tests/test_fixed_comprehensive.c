#include "unity.h"
#include "simulation_api.h"
#include "solver.h"
#include "grid.h"
#include "utils.h"
#include "vtk_output.h"
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
    #include <direct.h>
    #include <io.h>
    #define access _access
    #define F_OK 0
#else
    #include <unistd.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

int file_exists(const char* filename) {
    return access(filename, F_OK) == 0;
}

// Test basic simulation functionality
void test_simulation_basic(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    SimulationData* sim_data = init_simulation(nx, ny, xmin, xmax, ymin, ymax);
    TEST_ASSERT_NOT_NULL(sim_data);
    TEST_ASSERT_NOT_NULL(sim_data->field);
    TEST_ASSERT_NOT_NULL(sim_data->grid);

    // Check initial values are finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(sim_data->field->u[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->v[i]));
        TEST_ASSERT_TRUE(isfinite(sim_data->field->p[i]));
    }

    // Run one step
    run_simulation_step(sim_data);

    // Check values are still finite
    int finite_count = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (isfinite(sim_data->field->u[i]) && isfinite(sim_data->field->v[i]) &&
            isfinite(sim_data->field->p[i])) {
            finite_count++;
        }
    }
    TEST_ASSERT_GREATER_THAN((int)(0.8 * nx * ny), finite_count);

    free_simulation(sim_data);
}

// Test that both solvers produce reasonable results
void test_solver_consistency(void) {
    size_t nx = 12, ny = 8;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid1 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    Grid* grid2 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid1);
    grid_initialize_uniform(grid2);

    FlowField* field1 = flow_field_create(nx, ny);
    FlowField* field2 = flow_field_create(nx, ny);
    initialize_flow_field(field1, grid1);
    initialize_flow_field(field2, grid2);

    // Copy initial conditions to ensure identical start
    for (size_t i = 0; i < nx * ny; i++) {
        field2->u[i] = field1->u[i];
        field2->v[i] = field1->v[i];
        field2->p[i] = field1->p[i];
        field2->rho[i] = field1->rho[i];
        field2->T[i] = field1->T[i];
    }

    SolverParams params = {
        .dt = 0.001, .cfl = 0.2, .gamma = 1.4, .mu = 0.01, .k = 0.0242,
        .max_iter = 2, .tolerance = 1e-6,
        .source_amplitude_u = 0.1, .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1, .pressure_coupling = 0.1
    };

    // Run both solvers
    solve_navier_stokes(field1, grid1, &params);
    solve_navier_stokes_optimized(field2, grid2, &params);

    // Both should produce finite results
    int finite1 = 0, finite2 = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (isfinite(field1->u[i]) && isfinite(field1->v[i]) && isfinite(field1->p[i])) {
            finite1++;
        }
        if (isfinite(field2->u[i]) && isfinite(field2->v[i]) && isfinite(field2->p[i])) {
            finite2++;
        }
    }

    TEST_ASSERT_GREATER_THAN((int)(0.9 * nx * ny), finite1);
    TEST_ASSERT_GREATER_THAN((int)(0.9 * nx * ny), finite2);

    printf("Solver results: Basic=%d/%d finite, Optimized=%d/%d finite\n",
           finite1, (int)(nx*ny), finite2, (int)(nx*ny));

    flow_field_destroy(field1);
    flow_field_destroy(field2);
    grid_destroy(grid1);
    grid_destroy(grid2);
}

// Test that physics improvements work (pressure gradient induces flow)
void test_physics_improvements(void) {
    size_t nx = 10, ny = 8;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);
    FlowField* field = flow_field_create(nx, ny);

    // Set up pressure gradient with zero initial velocity
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double x = grid->x[i];
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->p[idx] = 1.0 + x;  // Positive pressure gradient in x
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    double initial_velocity = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_velocity += fabs(field->u[i]) + fabs(field->v[i]);
    }

    SolverParams params = {
        .dt = 0.001, .cfl = 0.2, .gamma = 1.4, .mu = 0.001, .k = 0.0242,
        .max_iter = 3, .tolerance = 1e-6,
        .source_amplitude_u = 0.1, .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1, .pressure_coupling = 0.1
    };

    solve_navier_stokes(field, grid, &params);

    // Check that pressure gradient induced some velocity
    double final_velocity = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        final_velocity += fabs(field->u[i]) + fabs(field->v[i]);
    }

    printf("Physics test: Initial velocity sum=%.6f, Final=%.6f\n",
           initial_velocity, final_velocity);

    // Pressure gradient should induce velocity (final > initial)
    TEST_ASSERT_GREATER_THAN(initial_velocity, final_velocity);

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that source terms prevent decay
void test_decay_prevention(void) {
    size_t nx = 10, ny = 8;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);
    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Calculate initial kinetic energy
    double initial_energy = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_energy += field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }

    SolverParams params = {
        .dt = 0.001, .cfl = 0.2, .gamma = 1.4, .mu = 0.01, .k = 0.0242,
        .max_iter = 10, .tolerance = 1e-6,
        .source_amplitude_u = 0.1, .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1, .pressure_coupling = 0.1
    };

    solve_navier_stokes(field, grid, &params);

    // Calculate final energy
    double final_energy = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        final_energy += field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }

    double energy_ratio = final_energy / initial_energy;

    printf("Decay prevention: Initial=%.3f, Final=%.3f, Ratio=%.3f\n",
           initial_energy, final_energy, energy_ratio);

    // With source terms, energy should be maintained or grow (not decay rapidly)
    TEST_ASSERT_TRUE(energy_ratio > 0.5);  // Should not decay to <50%
    TEST_ASSERT_TRUE(energy_ratio < 2.0);  // Should not grow too much either

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test output paths are working
void test_output_paths(void) {
    // Test directory creation
    ensure_directory_exists("../../output");
    ensure_directory_exists("..\\..\\artifacts\\output");

    TEST_ASSERT_TRUE(file_exists("../../output"));
    TEST_ASSERT_TRUE(file_exists("..\\..\\artifacts\\output"));

    // Test VTK file creation
    size_t nx = 5, ny = 5;
    double data[25];
    for (int i = 0; i < 25; i++) data[i] = i * 0.1;

    const char* test_file = "..\\..\\artifacts\\output\\test_fixed.vtk";
    remove(test_file);

    write_vtk_output(test_file, "test_data", data, nx, ny, 0.0, 1.0, 0.0, 1.0);

    TEST_ASSERT_TRUE(file_exists(test_file));

    // Check file has reasonable size
    FILE* file = fopen(test_file, "r");
    TEST_ASSERT_NOT_NULL(file);
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    TEST_ASSERT_GREATER_THAN(100, size);

    remove(test_file);  // Clean up
}

int main(void) {
    UNITY_BEGIN();

    printf("\n=== Fixed CFD Solver Tests ===\n");

    printf("\n1. Basic simulation test...\n");
    RUN_TEST(test_simulation_basic);

    printf("\n2. Solver consistency test...\n");
    RUN_TEST(test_solver_consistency);

    printf("\n3. Physics improvements test...\n");
    RUN_TEST(test_physics_improvements);

    printf("\n4. Decay prevention test...\n");
    RUN_TEST(test_decay_prevention);

    printf("\n5. Output paths test...\n");
    RUN_TEST(test_output_paths);

    printf("\n=== Tests Complete ===\n");

    return UNITY_END();
}