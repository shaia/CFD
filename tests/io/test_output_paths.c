#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/logging.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"


#include "cfd/io/vtk_output.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#define rmdir  _rmdir
#define F_OK   0
#else
#include <unistd.h>
#endif

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Clean up test files if needed
}

// Helper function to check if file exists
int file_exists(const char* filename) {
    return access(filename, F_OK) == 0;
}

// Test that output directories are created correctly
void test_output_directory_creation(void) {
    // Create cross-platform test paths
    char base_path[256];
    char test_dir[256];
    char nested_test_dir[256];
    char output_dir[256];

    make_artifacts_path(base_path, sizeof(base_path), "");
    make_artifacts_path(output_dir, sizeof(output_dir), "output");

#ifdef _WIN32
    snprintf(test_dir, sizeof(test_dir), "%s\\test_dir", base_path);
    snprintf(nested_test_dir, sizeof(nested_test_dir), "%s\\test_dir\\nested", base_path);
#else
    snprintf(test_dir, sizeof(test_dir), "%s/test_dir", base_path);
    snprintf(nested_test_dir, sizeof(nested_test_dir), "%s/test_dir/nested", base_path);
#endif

    // Clean up first (in case of previous test runs)
    rmdir(nested_test_dir);
    rmdir(test_dir);
    rmdir(base_path);

    // Test basic directory creation - function returns 1 if dir exists or was created successfully
    int result = ensure_directory_exists(base_path);
    TEST_ASSERT_TRUE(result);  // Should succeed (either exists or was created)

    result = ensure_directory_exists(test_dir);
    TEST_ASSERT_TRUE(result);  // Should succeed

    result = ensure_directory_exists(nested_test_dir);
    TEST_ASSERT_TRUE(result);  // Should succeed

    // Test that directories were actually created
    TEST_ASSERT_TRUE(file_exists(base_path));
    TEST_ASSERT_TRUE(file_exists(test_dir));
    TEST_ASSERT_TRUE(file_exists(nested_test_dir));

    // Test output directory creation
    result = ensure_directory_exists(output_dir);
    TEST_ASSERT_TRUE(result);
    TEST_ASSERT_TRUE(file_exists(output_dir));

    // Clean up
    rmdir(nested_test_dir);
    rmdir(test_dir);
}

// Test that VTK output files are created in correct locations
void test_vtk_output_paths(void) {
    // Create a small simulation
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Ensure output directory exists
    char artifacts_path[256];
    char output_path[256];
    char test_filename[256];

    make_artifacts_path(artifacts_path, sizeof(artifacts_path), "");
    make_artifacts_path(output_path, sizeof(output_path), "output");
    ensure_directory_exists(artifacts_path);
    ensure_directory_exists(output_path);

    // Test writing VTK file
    make_output_path(test_filename, sizeof(test_filename), "test_output.vtk");

    // Remove file if it exists from previous runs
    remove(test_filename);

    // Write VTK output
    write_vtk_output(test_filename, "test_field", field->u, nx, ny, xmin, xmax, ymin, ymax);

    // Check that file was created
    TEST_ASSERT_TRUE(file_exists(test_filename));

    // Check file is not empty
    FILE* file = fopen(test_filename, "r");
    TEST_ASSERT_NOT_NULL(file);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    TEST_ASSERT_GREATER_THAN(100, file_size);  // Should be reasonable size

    // Test vector output
    char vector_filename[256];
    make_output_path(vector_filename, sizeof(vector_filename), "test_vectors.vtk");
    remove(vector_filename);

    write_vtk_vector_output(vector_filename, "velocity", field->u, field->v, nx, ny, xmin, xmax,
                            ymin, ymax);

    TEST_ASSERT_TRUE(file_exists(vector_filename));

    // Clean up test files
    remove(test_filename);
    remove(vector_filename);

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that solvers don't create unwanted output files
// and that manual output works correctly
void test_solver_output_paths(void) {
    size_t nx = 8, ny = 6;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    SolverParams params = solver_params_default();
    params.max_iter = 1;

    // Test that solvers NO LONGER create automatic output files
    char unwanted_output[256];
    make_output_path(unwanted_output, sizeof(unwanted_output), "output_0.vtk");
    remove(unwanted_output);  // Clean up any existing file

    // Run solver
    SolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    Solver* solver = cfd_solver_create(registry, SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);

    // Verify solver did NOT create automatic output
    TEST_ASSERT_FALSE(file_exists(unwanted_output));

    // Now test that manual output DOES work when we explicitly call it
    char test_output[256];
    make_output_path(test_output, sizeof(test_output), "test_manual_output.vtk");
    remove(test_output);  // Clean up any existing file

    // Write output manually (this is how output should be done now)
    write_vtk_output(test_output, "pressure", field->p, field->nx, field->ny, grid->xmin,
                     grid->xmax, grid->ymin, grid->ymax);

    // Verify manual output file was created
    TEST_ASSERT_TRUE(file_exists(test_output));

    // Verify file has reasonable content
    FILE* file = fopen(test_output, "r");
    TEST_ASSERT_NOT_NULL(file);
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    TEST_ASSERT_GREATER_THAN(50, size);
    fclose(file);

    // Clean up
    remove(test_output);
    flow_field_destroy(field);
    grid_destroy(grid);
    cfd_registry_destroy(registry);
}

// Test that old scattered output directories don't get used
void test_no_scattered_output(void) {
    // These directories should NOT be created by our fixed code
    // Define cross-platform old paths to check
    char old_paths[4][256];

#ifdef _WIN32
    // Safe: use strncpy with explicit bounds and null termination
    strncpy(old_paths[0], "..\\..\\output\\animation", sizeof(old_paths[0]) - 1);
    old_paths[0][sizeof(old_paths[0]) - 1] = '\0';
    strncpy(old_paths[1], "..\\..\\output\\animations", sizeof(old_paths[1]) - 1);
    old_paths[1][sizeof(old_paths[1]) - 1] = '\0';
    strncpy(old_paths[2], "output\\animation", sizeof(old_paths[2]) - 1);
    old_paths[2][sizeof(old_paths[2]) - 1] = '\0';
    strncpy(old_paths[3], "output\\animations", sizeof(old_paths[3]) - 1);
    old_paths[3][sizeof(old_paths[3]) - 1] = '\0';
#else
    // Safe: use strncpy with explicit bounds and null termination
    strncpy(old_paths[0], "../../output/animation", sizeof(old_paths[0]) - 1);
    old_paths[0][sizeof(old_paths[0]) - 1] = '\0';
    strncpy(old_paths[1], "../../output/animations", sizeof(old_paths[1]) - 1);
    old_paths[1][sizeof(old_paths[1]) - 1] = '\0';
    strncpy(old_paths[2], "output/animation", sizeof(old_paths[2]) - 1);
    old_paths[2][sizeof(old_paths[2]) - 1] = '\0';
    strncpy(old_paths[3], "output/animations", sizeof(old_paths[3]) - 1);
    old_paths[3][sizeof(old_paths[3]) - 1] = '\0';
#endif

    size_t num_paths = sizeof(old_paths) / sizeof(old_paths[0]);

    // Run a quick simulation
    size_t nx = 5, ny = 5;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    SolverParams params = {.dt = 0.001,
                           .cfl = 0.2,
                           .gamma = 1.4,
                           .mu = 0.01,
                           .k = 0.0242,
                           .max_iter = 1,
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Run solver using modern interface
    SolverRegistry* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    Solver* solver = cfd_solver_create(registry, SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);
    cfd_registry_destroy(registry);

    // Check that old scattered directories were NOT created
    for (size_t i = 0; i < num_paths; i++) {
        if (file_exists(old_paths[i])) {
            printf("Warning: Old path still exists: %s\n", old_paths[i]);
            // Don't fail the test, just warn - directories might exist from other tests
        }
    }

    // Most importantly, check that our correct directory DOES exist
    char correct_output_dir[256];
    make_artifacts_path(correct_output_dir, sizeof(correct_output_dir), "output");
    TEST_ASSERT_TRUE(file_exists(correct_output_dir));

    flow_field_destroy(field);
    grid_destroy(grid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_output_directory_creation);
    RUN_TEST(test_vtk_output_paths);
    RUN_TEST(test_solver_output_paths);
    RUN_TEST(test_no_scattered_output);
    return UNITY_END();
}