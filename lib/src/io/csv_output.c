#include "csv_output.h"
#include "csv_output_internal.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <sys/stat.h>
#include <string.h>

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

// Helper to build filepath
static void build_filepath(char* filepath, size_t filepath_size,
                           const char* run_dir, const char* filename) {
#ifdef _WIN32
    snprintf(filepath, filepath_size, "%s\\%s", run_dir, filename);
#else
    snprintf(filepath, filepath_size, "%s/%s", run_dir, filename);
#endif
}

// Check if file exists
static int file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

//=============================================================================
// CSV OUTPUT DISPATCH
//=============================================================================

// Function pointer type for CSV output handlers
typedef void (*CsvOutputHandler)(const char* run_dir, const char* prefix, int step,
                                 double current_time, const FlowField* field,
                                 const Grid* grid, const SolverParams* params,
                                 const SolverStats* stats);

// CSV output handler functions
static void write_timeseries_csv(const char* run_dir, const char* prefix, int step,
                                 double current_time, const FlowField* field,
                                 const Grid* grid, const SolverParams* params,
                                 const SolverStats* stats) {
    const char* name = prefix ? prefix : "timeseries";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s.csv", name);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    int create_new = (step == 0);
    write_csv_timeseries(filepath, step, current_time, field, params, stats,
                       grid->nx, grid->ny, create_new);
}

static void write_centerline_csv(const char* run_dir, const char* prefix, int step,
                                  double current_time, const FlowField* field,
                                  const Grid* grid, const SolverParams* params,
                                  const SolverStats* stats) {
    (void)current_time; (void)params; (void)stats; // Unused for centerline
    const char* name = prefix ? prefix : "centerline";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s_%03d.csv", name, step);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    write_csv_centerline(filepath, field, grid->x, grid->y,
                       grid->nx, grid->ny, PROFILE_HORIZONTAL);
}

static void write_statistics_csv(const char* run_dir, const char* prefix, int step,
                                  double current_time, const FlowField* field,
                                  const Grid* grid, const SolverParams* params,
                                  const SolverStats* stats) {
    (void)params; (void)stats; // Unused for statistics
    const char* name = prefix ? prefix : "statistics";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s.csv", name);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    int create_new = (step == 0);
    write_csv_statistics(filepath, step, current_time, field,
                       grid->nx, grid->ny, create_new);
}

// CSV output handler table - indexed by CsvOutputType
static const CsvOutputHandler csv_output_table[] = {
    write_timeseries_csv,    // CSV_OUTPUT_TIMESERIES = 0
    write_centerline_csv,    // CSV_OUTPUT_CENTERLINE = 1
    write_statistics_csv     // CSV_OUTPUT_STATISTICS = 2
};

#define CSV_OUTPUT_TABLE_SIZE (sizeof(csv_output_table) / sizeof(csv_output_table[0]))

void csv_dispatch_output(CsvOutputType csv_type,
                         const char* run_dir,
                         const char* prefix,
                         int step,
                         double current_time,
                         const FlowField* field,
                         const Grid* grid,
                         const SolverParams* params,
                         const SolverStats* stats) {
    // Bounds check and dispatch via function table
    if (csv_type < CSV_OUTPUT_TABLE_SIZE) {
        csv_output_table[csv_type](run_dir, prefix, step, current_time,
                                   field, grid, params, stats);
    } else {
        cfd_warning("Unknown CSV output type");
    }
}

//=============================================================================
// CSV TIMESERIES OUTPUT
//=============================================================================

void write_csv_timeseries(const char* filename,
                          int step,
                          double time,
                          const FlowField* field,
                          const SolverParams* params,
                          const SolverStats* stats,
                          size_t nx, size_t ny,
                          int create_new) {
    if (!filename || !field || !params || !stats) return;

    size_t count = nx * ny;
    int write_header = create_new || !file_exists(filename);
    const char* mode = write_header ? "w" : "a";

    FILE* fp = fopen(filename, mode);
    if (!fp) {
        cfd_warning("Failed to open CSV timeseries file for writing");
        return;
    }

    // Write header if new file
    if (write_header) {
        fprintf(fp, "step,time,dt,max_u,max_v,max_p,avg_u,avg_v,avg_p,iterations,residual,elapsed_ms\n");
    }

    // Calculate statistics
    FieldStats u_stats = calculate_field_statistics(field->u, count);
    FieldStats v_stats = calculate_field_statistics(field->v, count);
    FieldStats p_stats = calculate_field_statistics(field->p, count);

    // Write data row
    fprintf(fp, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e,%.2f\n",
            step,
            time,
            params->dt,
            u_stats.max_val,
            v_stats.max_val,
            p_stats.max_val,
            u_stats.avg_val,
            v_stats.avg_val,
            p_stats.avg_val,
            stats->iterations,
            stats->residual,
            stats->elapsed_time_ms);

    fclose(fp);
}

//=============================================================================
// CSV CENTERLINE PROFILE OUTPUT
//=============================================================================

void write_csv_centerline(const char* filename,
                          const FlowField* field,
                          const double* x_coords,
                          const double* y_coords,
                          size_t nx, size_t ny,
                          ProfileDirection direction) {
    if (!filename || !field || !x_coords || !y_coords) return;

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        cfd_warning("Failed to open CSV centerline file for writing");
        return;
    }

    if (direction == PROFILE_HORIZONTAL) {
        // Horizontal centerline: along x at y = ny/2
        size_t j_mid = ny / 2;

        fprintf(fp, "x,u,v,p,rho,T\n");

        for (size_t i = 0; i < nx; i++) {
            size_t idx = j_mid * nx + i;
            fprintf(fp, "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    x_coords[i],
                    field->u[idx],
                    field->v[idx],
                    field->p[idx],
                    field->rho[idx],
                    field->T[idx]);
        }
    } else {
        // Vertical centerline: along y at x = nx/2
        size_t i_mid = nx / 2;

        fprintf(fp, "y,u,v,p,rho,T\n");

        for (size_t j = 0; j < ny; j++) {
            size_t idx = j * nx + i_mid;
            fprintf(fp, "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    y_coords[j],
                    field->u[idx],
                    field->v[idx],
                    field->p[idx],
                    field->rho[idx],
                    field->T[idx]);
        }
    }

    fclose(fp);
}

//=============================================================================
// CSV STATISTICS OUTPUT
//=============================================================================

void write_csv_statistics(const char* filename,
                          int step,
                          double time,
                          const FlowField* field,
                          size_t nx, size_t ny,
                          int create_new) {
    if (!filename || !field) return;

    size_t count = nx * ny;
    int write_header = create_new || !file_exists(filename);
    const char* mode = write_header ? "w" : "a";

    FILE* fp = fopen(filename, mode);
    if (!fp) {
        cfd_warning("Failed to open CSV statistics file for writing");
        return;
    }

    // Write header if new file
    if (write_header) {
        fprintf(fp, "step,time,min_u,max_u,avg_u,min_v,max_v,avg_v,min_p,max_p,avg_p,min_rho,max_rho,avg_rho,min_T,max_T,avg_T\n");
    }

    // Calculate statistics for all fields
    FieldStats u_stats = calculate_field_statistics(field->u, count);
    FieldStats v_stats = calculate_field_statistics(field->v, count);
    FieldStats p_stats = calculate_field_statistics(field->p, count);
    FieldStats rho_stats = calculate_field_statistics(field->rho, count);
    FieldStats T_stats = calculate_field_statistics(field->T, count);

    // Write data row
    fprintf(fp, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
            step,
            time,
            u_stats.min_val, u_stats.max_val, u_stats.avg_val,
            v_stats.min_val, v_stats.max_val, v_stats.avg_val,
            p_stats.min_val, p_stats.max_val, p_stats.avg_val,
            rho_stats.min_val, rho_stats.max_val, rho_stats.avg_val,
            T_stats.min_val, T_stats.max_val, T_stats.avg_val);

    fclose(fp);
}
