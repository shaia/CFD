#include "csv_output.h"
#include "derived_fields.h"
#include "grid.h"
#include "solver_interface.h"
#include "unity.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
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
static Grid* test_grid = NULL;
static FlowField* test_field = NULL;
static DerivedFields* test_derived = NULL;
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
        test_field->u[i] = 0.5 + 0.1 * sin((double)i);
        test_field->v[i] = 0.3 + 0.05 * cos((double)i);
        test_field->p[i] = 100000.0 + 1000.0 * sin((double)i);
        test_field->rho[i] = 1.2 + 0.01 * cos((double)i);
        test_field->T[i] = 300.0 + 10.0 * sin((double)i);
    }

    // Create derived fields and compute statistics
    test_derived = derived_fields_create(nx, ny);
    derived_fields_compute_velocity_magnitude(test_derived, test_field);
    derived_fields_compute_statistics(test_derived, test_field);

    // Set up output directory
    make_artifacts_path(test_output_dir, sizeof(test_output_dir), "output");
    ensure_directory_exists(test_output_dir);
}

void tearDown(void) {
    if (test_derived) {
        derived_fields_destroy(test_derived);
        test_derived = NULL;
    }
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

// Helper to count lines in a file
static int count_file_lines(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;

    int lines = 0;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), fp)) {
        lines++;
    }
    fclose(fp);
    return lines;
}

// Helper to check if file contains a string
static int file_contains(const char* filename, const char* str) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return 0;

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
// TIMESERIES TESTS
//=============================================================================

void test_csv_timeseries_creates_file(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_timeseries.csv");
    remove(filename);

    SolverParams params = solver_params_default();
    SolverStats stats = solver_stats_default();
    stats.iterations = 10;
    stats.residual = 1e-5;
    stats.elapsed_time_ms = 12.5;

    write_csv_timeseries(filename, 0, 0.0, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 1);

    TEST_ASSERT_TRUE(file_exists(filename));
    remove(filename);
}

void test_csv_timeseries_has_header(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_timeseries_header.csv");
    remove(filename);

    SolverParams params = solver_params_default();
    SolverStats stats = solver_stats_default();

    write_csv_timeseries(filename, 0, 0.0, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 1);

    TEST_ASSERT_TRUE(file_contains(filename, "step,time,dt"));
    TEST_ASSERT_TRUE(file_contains(filename, "max_u,max_v,max_p"));
    TEST_ASSERT_TRUE(file_contains(filename, "iterations,residual"));

    remove(filename);
}

void test_csv_timeseries_appends_data(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_timeseries_append.csv");
    remove(filename);

    SolverParams params = solver_params_default();
    SolverStats stats = solver_stats_default();

    // Write first entry (creates file with header)
    write_csv_timeseries(filename, 0, 0.0, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 1);

    // Write second entry (appends)
    write_csv_timeseries(filename, 1, 0.001, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 0);

    // Write third entry (appends)
    write_csv_timeseries(filename, 2, 0.002, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 0);

    // Should have 1 header + 3 data lines = 4 lines
    int lines = count_file_lines(filename);
    TEST_ASSERT_EQUAL_INT(4, lines);

    remove(filename);
}

void test_csv_timeseries_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_null.csv");
    remove(filename);

    SolverParams params = solver_params_default();
    SolverStats stats = solver_stats_default();

    // These should not crash - derived is required for stats
    write_csv_timeseries(NULL, 0, 0.0, test_field, test_derived, &params, &stats, 10, 10, 1);
    write_csv_timeseries(filename, 0, 0.0, test_field, NULL, &params, &stats, 10, 10, 1);
    write_csv_timeseries(filename, 0, 0.0, test_field, test_derived, NULL, &stats, 10, 10, 1);
    write_csv_timeseries(filename, 0, 0.0, test_field, test_derived, &params, NULL, 10, 10, 1);

    // File should not exist since all calls had NULL required params
    TEST_ASSERT_FALSE(file_exists(filename));
}

//=============================================================================
// CENTERLINE TESTS
//=============================================================================

void test_csv_centerline_horizontal(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_centerline_h.csv");
    remove(filename);

    write_csv_centerline(filename, test_field, test_derived, test_grid->x, test_grid->y,
                         test_grid->nx, test_grid->ny, PROFILE_HORIZONTAL);

    TEST_ASSERT_TRUE(file_exists(filename));
    TEST_ASSERT_TRUE(file_contains(filename, "x,u,v,p,rho,T"));

    // Should have header + nx data points
    int lines = count_file_lines(filename);
    TEST_ASSERT_EQUAL_INT((int)test_grid->nx + 1, lines);

    remove(filename);
}

void test_csv_centerline_vertical(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_centerline_v.csv");
    remove(filename);

    write_csv_centerline(filename, test_field, test_derived, test_grid->x, test_grid->y,
                         test_grid->nx, test_grid->ny, PROFILE_VERTICAL);

    TEST_ASSERT_TRUE(file_exists(filename));
    TEST_ASSERT_TRUE(file_contains(filename, "y,u,v,p,rho,T"));

    // Should have header + ny data points
    int lines = count_file_lines(filename);
    TEST_ASSERT_EQUAL_INT((int)test_grid->ny + 1, lines);

    remove(filename);
}

void test_csv_centerline_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_centerline_null.csv");
    remove(filename);

    // These should not crash
    write_csv_centerline(NULL, test_field, test_derived, test_grid->x, test_grid->y, 10, 10,
                         PROFILE_HORIZONTAL);
    write_csv_centerline(filename, NULL, test_derived, test_grid->x, test_grid->y, 10, 10,
                         PROFILE_HORIZONTAL);
    write_csv_centerline(filename, test_field, test_derived, NULL, test_grid->y, 10, 10,
                         PROFILE_HORIZONTAL);
    write_csv_centerline(filename, test_field, test_derived, test_grid->x, NULL, 10, 10,
                         PROFILE_HORIZONTAL);

    TEST_ASSERT_FALSE(file_exists(filename));
}

//=============================================================================
// STATISTICS TESTS
//=============================================================================

void test_csv_statistics_creates_file(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_statistics.csv");
    remove(filename);

    write_csv_statistics(filename, 0, 0.0, test_field, test_derived,
                         test_grid->nx, test_grid->ny, 1);

    TEST_ASSERT_TRUE(file_exists(filename));
    remove(filename);
}

void test_csv_statistics_has_header(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_statistics_header.csv");
    remove(filename);

    write_csv_statistics(filename, 0, 0.0, test_field, test_derived,
                         test_grid->nx, test_grid->ny, 1);

    TEST_ASSERT_TRUE(file_contains(filename, "step,time"));
    TEST_ASSERT_TRUE(file_contains(filename, "min_u,max_u,avg_u"));
    TEST_ASSERT_TRUE(file_contains(filename, "min_v,max_v,avg_v"));
    TEST_ASSERT_TRUE(file_contains(filename, "min_p,max_p,avg_p"));
    TEST_ASSERT_TRUE(file_contains(filename, "min_rho,max_rho,avg_rho"));
    TEST_ASSERT_TRUE(file_contains(filename, "min_T,max_T,avg_T"));

    remove(filename);
}

void test_csv_statistics_appends_data(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_statistics_append.csv");
    remove(filename);

    // Write multiple entries
    write_csv_statistics(filename, 0, 0.0, test_field, test_derived, test_grid->nx, test_grid->ny,
                         1);
    write_csv_statistics(filename, 1, 0.001, test_field, test_derived, test_grid->nx, test_grid->ny,
                         0);
    write_csv_statistics(filename, 2, 0.002, test_field, test_derived, test_grid->nx, test_grid->ny,
                         0);
    write_csv_statistics(filename, 3, 0.003, test_field, test_derived, test_grid->nx, test_grid->ny,
                         0);

    // Should have 1 header + 4 data lines = 5 lines
    int lines = count_file_lines(filename);
    TEST_ASSERT_EQUAL_INT(5, lines);

    remove(filename);
}

void test_csv_statistics_null_safety(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_stats_null.csv");
    remove(filename);

    // These should not crash - derived is required for stats
    write_csv_statistics(NULL, 0, 0.0, test_field, test_derived, 10, 10, 1);
    write_csv_statistics(filename, 0, 0.0, test_field, NULL, 10, 10, 1);

    TEST_ASSERT_FALSE(file_exists(filename));
}

//=============================================================================
// DATA CORRECTNESS TESTS
//=============================================================================

void test_csv_timeseries_data_values(void) {
    char filename[256];
    make_output_path(filename, sizeof(filename), "test_timeseries_values.csv");
    remove(filename);

    SolverParams params = solver_params_default();
    params.dt = 0.001;

    SolverStats stats = solver_stats_default();
    stats.iterations = 42;
    stats.residual = 1.5e-6;
    stats.elapsed_time_ms = 25.5;

    write_csv_timeseries(filename, 5, 0.005, test_field, test_derived, &params, &stats,
                         test_grid->nx, test_grid->ny, 1);

    // Check that specific values appear in the file
    FILE* fp = fopen(filename, "r");
    TEST_ASSERT_NOT_NULL(fp);

    char line[1024];
    fgets(line, sizeof(line), fp);  // Skip header
    fgets(line, sizeof(line), fp);  // Data line

    int step;
    double time, dt;
    sscanf(line, "%d,%lf,%lf", &step, &time, &dt);

    TEST_ASSERT_EQUAL_INT(5, step);
    // Use float comparison since Unity double is disabled
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.005f, (float)time);
    TEST_ASSERT_FLOAT_WITHIN(1e-5f, 0.001f, (float)dt);

    fclose(fp);
    remove(filename);
}

int main(void) {
    UNITY_BEGIN();

    // Timeseries tests
    RUN_TEST(test_csv_timeseries_creates_file);
    RUN_TEST(test_csv_timeseries_has_header);
    RUN_TEST(test_csv_timeseries_appends_data);
    RUN_TEST(test_csv_timeseries_null_safety);

    // Centerline tests
    RUN_TEST(test_csv_centerline_horizontal);
    RUN_TEST(test_csv_centerline_vertical);
    RUN_TEST(test_csv_centerline_null_safety);

    // Statistics tests
    RUN_TEST(test_csv_statistics_creates_file);
    RUN_TEST(test_csv_statistics_has_header);
    RUN_TEST(test_csv_statistics_appends_data);
    RUN_TEST(test_csv_statistics_null_safety);

    // Data correctness tests
    RUN_TEST(test_csv_timeseries_data_values);

    return UNITY_END();
}
