#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"


#include "cfd/io/vtk_output.h"
#include "cfd/solvers/solver_interface.h"
#include "unity.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#define F_OK   0
#else
#include <unistd.h>
#endif

// Test fixtures
static grid* test_grid = NULL;
static flow_field* test_field = NULL;
static char test_output_dir[256];

void setUp(void) {
    // Create test grid and field
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    test_grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(test_grid);

    test_field = flow_field_create(nx, ny);
    initialize_flow_field(test_field, test_grid);

    // Set some known values for testing
    for (size_t i = 0; i < nx * ny; i++) {
        test_field->u[i] = 0.5 + (0.1 * sin((double)i));
        test_field->v[i] = 0.3 + (0.05 * cos((double)i));
        test_field->p[i] = 100000.0 + 1000.0 * sin((double)i);
        test_field->rho[i] = 1.2 + (0.01 * cos((double)i));
        test_field->T[i] = 300.0 + (10.0 * sin((double)i));
    }

    // Set up output directory
    make_artifacts_path(test_output_dir, sizeof(test_output_dir), "output");
    ensure_directory_exists(test_output_dir);
}

void tearDown(void) {
    if (test_field) {
        flow_field_destroy(test_field);
        test_field = NULL;
    }
    if (test_grid) {
        grid_destroy(test_grid);
        test_grid = NULL;
    }
}

// Helper function to check if file exists
static int file_exists(const char* filename) {
    return access(filename, F_OK) == 0;
}

// Helper to check if file contains a string
static int file_contains(const char* filename, const char* str) {
    if (filename == NULL) {
        return 0;
    }
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        return 0;
    }

    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (strstr(buffer, str)) {
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);
    return 0;
}

//=============================================================================
// VTK SCALAR OUTPUT TESTS
//=============================================================================

void test_vtk_output_creates_file(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_scalar.vtk");
    remove(filename);

    write_vtk_output(filename, "pressure", test_field->p, test_grid->nx, test_grid->ny,
                     test_grid->xmin, test_grid->xmax, test_grid->ymin, test_grid->ymax);

    TEST_ASSERT_TRUE(file_exists(filename));
    remove(filename);
}

void test_vtk_output_has_header(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_header.vtk");
    remove(filename);

    write_vtk_output(filename, "velocity_u", test_field->u, test_grid->nx, test_grid->ny,
                     test_grid->xmin, test_grid->xmax, test_grid->ymin, test_grid->ymax);

    TEST_ASSERT_TRUE(file_contains(filename, "# vtk DataFile Version"));
    TEST_ASSERT_TRUE(file_contains(filename, "STRUCTURED_POINTS"));
    TEST_ASSERT_TRUE(file_contains(filename, "DIMENSIONS"));
    TEST_ASSERT_TRUE(file_contains(filename, "SCALARS velocity_u"));

    remove(filename);
}

void test_vtk_output_file_size(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_size.vtk");
    remove(filename);

    write_vtk_output(filename, "temperature", test_field->T, test_grid->nx, test_grid->ny,
                     test_grid->xmin, test_grid->xmax, test_grid->ymin, test_grid->ymax);

    FILE* fp = fopen(filename, "r");
    TEST_ASSERT_NOT_NULL(fp);

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    // VTK file should have reasonable size (header + data)
    TEST_ASSERT_GREATER_THAN(100, file_size);

    remove(filename);
}

//=============================================================================
// VTK VECTOR OUTPUT TESTS
//=============================================================================

void test_vtk_vector_output_creates_file(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_vector.vtk");
    remove(filename);

    write_vtk_vector_output(filename, "velocity", test_field->u, test_field->v, test_grid->nx,
                            test_grid->ny, test_grid->xmin, test_grid->xmax, test_grid->ymin,
                            test_grid->ymax);

    TEST_ASSERT_TRUE(file_exists(filename));
    remove(filename);
}

void test_vtk_vector_output_has_header(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_vector_header.vtk");
    remove(filename);

    write_vtk_vector_output(filename, "velocity", test_field->u, test_field->v, test_grid->nx,
                            test_grid->ny, test_grid->xmin, test_grid->xmax, test_grid->ymin,
                            test_grid->ymax);

    TEST_ASSERT_TRUE(file_contains(filename, "# vtk DataFile Version"));
    TEST_ASSERT_TRUE(file_contains(filename, "VECTORS velocity"));

    remove(filename);
}

//=============================================================================
// VTK FLOW FIELD OUTPUT TESTS
//=============================================================================

void test_vtk_flow_field_creates_file(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_flow.vtk");
    remove(filename);

    write_vtk_flow_field(filename, test_field, test_grid->nx, test_grid->ny, test_grid->xmin,
                         test_grid->xmax, test_grid->ymin, test_grid->ymax);

    TEST_ASSERT_TRUE(file_exists(filename));
    remove(filename);
}

void test_vtk_flow_field_has_all_fields(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_flow_fields.vtk");
    remove(filename);

    write_vtk_flow_field(filename, test_field, test_grid->nx, test_grid->ny, test_grid->xmin,
                         test_grid->xmax, test_grid->ymin, test_grid->ymax);

    // Should contain velocity, pressure, density, temperature
    TEST_ASSERT_TRUE(file_contains(filename, "VECTORS velocity"));
    TEST_ASSERT_TRUE(file_contains(filename, "SCALARS pressure"));
    TEST_ASSERT_TRUE(file_contains(filename, "SCALARS density"));
    TEST_ASSERT_TRUE(file_contains(filename, "SCALARS temperature"));

    remove(filename);
}

//=============================================================================
// NULL SAFETY TESTS
//=============================================================================

void test_vtk_output_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_null.vtk");
    remove(filename);

    // These should not crash
    write_vtk_output(NULL, "test", test_field->u, 10, 10, 0, 1, 0, 1);
    write_vtk_output(filename, NULL, test_field->u, 10, 10, 0, 1, 0, 1);
    write_vtk_output(filename, "test", NULL, 10, 10, 0, 1, 0, 1);

    // File should not exist since all calls had NULL params
    TEST_ASSERT_FALSE(file_exists(filename));
}

void test_vtk_vector_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_vec_null.vtk");
    remove(filename);

    // These should not crash
    write_vtk_vector_output(NULL, "vel", test_field->u, test_field->v, 10, 10, 0, 1, 0, 1);
    write_vtk_vector_output(filename, NULL, test_field->u, test_field->v, 10, 10, 0, 1, 0, 1);
    write_vtk_vector_output(filename, "vel", NULL, test_field->v, 10, 10, 0, 1, 0, 1);
    write_vtk_vector_output(filename, "vel", test_field->u, NULL, 10, 10, 0, 1, 0, 1);

    TEST_ASSERT_FALSE(file_exists(filename));
}

void test_vtk_flow_field_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_flow_null.vtk");
    remove(filename);

    // These should not crash
    write_vtk_flow_field(NULL, test_field, 10, 10, 0, 1, 0, 1);
    write_vtk_flow_field(filename, NULL, 10, 10, 0, 1, 0, 1);

    TEST_ASSERT_FALSE(file_exists(filename));
}

//=============================================================================
// EDGE CASE TESTS
//=============================================================================

void test_vtk_output_small_grid(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_small.vtk");
    remove(filename);

    // Minimum grid size
    size_t nx = 2, ny = 2;
    double* data = (double*)malloc(nx * ny * sizeof(double));
    if (data == NULL) {
        TEST_FAIL_MESSAGE("Memory allocation failed");
    }
    for (size_t i = 0; i < nx * ny; i++) {
        data[i] = (double)i;
    }

    write_vtk_output(filename, "small", data, nx, ny, 0, 1, 0, 1);

    TEST_ASSERT_TRUE(file_exists(filename));

    free(data);
    remove(filename);
}

void test_vtk_output_large_values(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_vtk_large.vtk");
    remove(filename);

    // Create data with large/small values
    size_t nx = 5, ny = 5;
    double* data = (double*)malloc(nx * ny * sizeof(double));
    if (data == NULL) {
        TEST_FAIL_MESSAGE("Memory allocation failed");
    }
    data[0] = 1e10;
    data[1] = -1e10;
    data[2] = 1e-10;
    data[3] = -1e-10;
    for (size_t i = 4; i < nx * ny; i++) {
        data[i] = 0.0;
    }

    write_vtk_output(filename, "extreme", data, nx, ny, 0, 1, 0, 1);

    TEST_ASSERT_TRUE(file_exists(filename));

    // Verify file has reasonable content
    FILE* fp = fopen(filename, "r");
    TEST_ASSERT_NOT_NULL(fp);
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    TEST_ASSERT_GREATER_THAN(50, size);
    fclose(fp);

    free(data);
    remove(filename);
}

int main(void) {
    UNITY_BEGIN();

    // Scalar output tests
    RUN_TEST(test_vtk_output_creates_file);
    RUN_TEST(test_vtk_output_has_header);
    RUN_TEST(test_vtk_output_file_size);

    // Vector output tests
    RUN_TEST(test_vtk_vector_output_creates_file);
    RUN_TEST(test_vtk_vector_output_has_header);

    // Flow field output tests
    RUN_TEST(test_vtk_flow_field_creates_file);
    RUN_TEST(test_vtk_flow_field_has_all_fields);

    // Null safety tests
    RUN_TEST(test_vtk_output_null_safety);
    RUN_TEST(test_vtk_vector_null_safety);
    RUN_TEST(test_vtk_flow_field_null_safety);

    // Edge case tests
    RUN_TEST(test_vtk_output_small_grid);
    RUN_TEST(test_vtk_output_large_values);

    return UNITY_END();
}
