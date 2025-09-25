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

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Helper function to check if file exists
int file_exists(const char* filename) {
    return access(filename, F_OK) == 0;
}

// Basic simulation test
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

// Test solver consistency between regular and optimized versions
void test_solver_consistency(void) {
    size_t nx = 15, ny = 10;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    // Create identical setups
    Grid* grid1 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    Grid* grid2 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid1);
    grid_initialize_uniform(grid2);

    FlowField* field1 = flow_field_create(nx, ny);
    FlowField* field2 = flow_field_create(nx, ny);
    initialize_flow_field(field1, grid1);
    initialize_flow_field(field2, grid2);

    // Ensure identical initial conditions
    for (size_t i = 0; i < nx * ny; i++) {
        field2->u[i] = field1->u[i];
        field2->v[i] = field1->v[i];
        field2->p[i] = field1->p[i];
        field2->rho[i] = field1->rho[i];
        field2->T[i] = field1->T[i];
    }

    SolverParams params = {
        .dt = 0.001, .cfl = 0.2, .gamma = 1.4, .mu = 0.01, .k = 0.0242,
        .max_iter = 3, .tolerance = 1e-6,
        .source_amplitude_u = 0.1, .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1, .pressure_coupling = 0.1
    };

    // Run both solvers
    solve_navier_stokes(field1, grid1, &params);
    solve_navier_stokes_optimized(field2, grid2, &params);

    // Compare results
    int close_values = 0;
    for (size_t i = 0; i < nx * ny; i++) {
        double u_diff = fabs(field1->u[i] - field2->u[i]);
        double v_diff = fabs(field1->v[i] - field2->v[i]);
        double p_diff = fabs(field1->p[i] - field2->p[i]);

        if (u_diff < 1e-3 && v_diff < 1e-3 && p_diff < 1e-2) {
            close_values++;
        }
    }

    // At least 80% should be close
    TEST_ASSERT_GREATER_THAN((int)(0.8 * nx * ny), close_values);

    flow_field_destroy(field1);
    flow_field_destroy(field2);
    grid_destroy(grid1);
    grid_destroy(grid2);
}

// Test that physics improvements are working (viscosity, pressure gradients)
void test_physics_improvements(void) {
    size_t nx = 12, ny = 8;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);
    FlowField* field = flow_field_create(nx, ny);

    // Set up pressure gradient with zero velocity
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double x = grid->x[i];
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->p[idx] = 1.0 + x;  // Pressure gradient
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    SolverParams params = {
        .dt = 0.001, .cfl = 0.2, .gamma = 1.4, .mu = 0.001, .k = 0.0242,
        .max_iter = 5, .tolerance = 1e-6,
        .source_amplitude_u = 0.1, .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1, .pressure_coupling = 0.1
    };

    solve_navier_stokes(field, grid, &params);

    // Pressure gradient should induce velocity
    double avg_u = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        avg_u += field->u[i];
    }
    avg_u /= (nx * ny);

    // Pressure gradient should induce velocity change
    printf("Physics test - Average u-velocity: %.6f\n", avg_u);
    TEST_ASSERT_TRUE(fabs(avg_u) > 1e-6);  // Some velocity should be induced

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test decay prevention (source terms working)
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
        .max_iter = 15, .tolerance = 1e-6,
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

    // Source terms should prevent rapid decay
    TEST_ASSERT_TRUE(energy_ratio > 0.95);  // Should not decay below 95%
    TEST_ASSERT_TRUE(energy_ratio < 10.0);  // Should not blow up

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test output paths are correct
void test_output_paths(void) {
    // Ensure directories exist
    // Create cross-platform paths and directories
    char artifacts_path[256];
    char output_path[256];
    char test_file[256];

    make_artifacts_path(artifacts_path, sizeof(artifacts_path), "");
    make_artifacts_path(output_path, sizeof(output_path), "output");

    ensure_directory_exists(artifacts_path);
    ensure_directory_exists(output_path);

    TEST_ASSERT_TRUE(file_exists(artifacts_path));
    TEST_ASSERT_TRUE(file_exists(output_path));

    // Test VTK file creation
    size_t nx = 5, ny = 5;
    double data[25];
    for (int i = 0; i < 25; i++) data[i] = i * 0.1;

    make_output_path(test_file, sizeof(test_file), "test_comprehensive.vtk");
    remove(test_file);  // Clean up first

    write_vtk_output(test_file, "test_data", data, nx, ny, 0.0, 1.0, 0.0, 1.0);

    TEST_ASSERT_TRUE(file_exists(test_file));

    // Check file size is reasonable
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

    printf("\n=== CFD Solver Comprehensive Tests ===\n");

    printf("\n1. Running basic simulation test...\n");
    RUN_TEST(test_simulation_basic);

    printf("\n2. Running solver consistency test...\n");
    RUN_TEST(test_solver_consistency);

    printf("\n3. Running physics improvements test...\n");
    RUN_TEST(test_physics_improvements);

    printf("\n4. Running decay prevention test...\n");
    RUN_TEST(test_decay_prevention);

    printf("\n5. Running output paths test...\n");
    RUN_TEST(test_output_paths);

    printf("\n=== All Tests Complete ===\n");

    return UNITY_END();
}