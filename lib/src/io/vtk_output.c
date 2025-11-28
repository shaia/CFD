#include "vtk_output.h"
#include "vtk_output_internal.h"
#include "utils.h"
#include "solver_interface.h"
#include "grid.h"
#include <stdio.h>
#include <stdlib.h>
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

//=============================================================================
// VTK OUTPUT DISPATCH
//=============================================================================

// Function pointer type for VTK output handlers
typedef void (*VtkOutputHandler)(const char* run_dir, const char* prefix, int step,
                                 const FlowField* field, const Grid* grid);

// VTK output handler functions
static void write_pressure_vtk(const char* run_dir, const char* prefix, int step,
                               const FlowField* field, const Grid* grid) {
    const char* name = prefix ? prefix : "pressure";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s_%03d.vtk", name, step);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    double* velocity_magnitude = calculate_velocity_magnitude(field, grid->nx, grid->ny);
    write_vtk_output(filepath, "velocity_magnitude", velocity_magnitude,
                   grid->nx, grid->ny, grid->xmin, grid->xmax,
                   grid->ymin, grid->ymax);
    cfd_free(velocity_magnitude);
}

static void write_velocity_vtk(const char* run_dir, const char* prefix, int step,
                                const FlowField* field, const Grid* grid) {
    const char* name = prefix ? prefix : "velocity";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s_%03d.vtk", name, step);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    write_vtk_vector_output(filepath, "velocity", field->u, field->v,
                          grid->nx, grid->ny, grid->xmin, grid->xmax,
                          grid->ymin, grid->ymax);
}

static void write_full_field_vtk(const char* run_dir, const char* prefix, int step,
                                  const FlowField* field, const Grid* grid) {
    const char* name = prefix ? prefix : "flow_field";
    char filename[256], filepath[512];

    snprintf(filename, sizeof(filename), "%s_%03d.vtk", name, step);
    build_filepath(filepath, sizeof(filepath), run_dir, filename);

    write_vtk_flow_field(filepath, field, grid->nx, grid->ny,
                       grid->xmin, grid->xmax, grid->ymin, grid->ymax);
}

// VTK output handler table - indexed by VtkOutputType
static const VtkOutputHandler vtk_output_table[] = {
    write_pressure_vtk,      // VTK_OUTPUT_PRESSURE = 0
    write_velocity_vtk,      // VTK_OUTPUT_VELOCITY = 1
    write_full_field_vtk     // VTK_OUTPUT_FULL_FIELD = 2
};

#define VTK_OUTPUT_TABLE_SIZE (sizeof(vtk_output_table) / sizeof(vtk_output_table[0]))

void vtk_dispatch_output(VtkOutputType vtk_type,
                         const char* run_dir,
                         const char* prefix,
                         int step,
                         const FlowField* field,
                         const Grid* grid) {
    // Bounds check and dispatch via function table
    if (vtk_type < VTK_OUTPUT_TABLE_SIZE) {
        vtk_output_table[vtk_type](run_dir, prefix, step, field, grid);
    } else {
        cfd_warning("Unknown VTK output type");
    }
}

//=============================================================================
// VTK LEGACY API
//=============================================================================

void write_vtk_output(const char* filename, const char* field_name, 
                     const double* data, size_t nx, size_t ny,
                     double xmin, double xmax, double ymin, double ymax) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        cfd_error("Failed to open VTK output file");
    }

    // Write VTK header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "CFD Framework Output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %zu %zu 1\n", nx, ny);
    fprintf(fp, "ORIGIN %f %f 0.0\n", xmin, ymin);
    fprintf(fp, "SPACING %f %f 1.0\n", (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1));


    // Write field data
    fprintf(fp, "\nPOINT_DATA %zu\n", nx * ny);
    fprintf(fp, "SCALARS %s float 1\n", field_name);
    fprintf(fp, "LOOKUP_TABLE default\n");

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f\n", data[idx]);
        }
    }

    fclose(fp);
}

void write_vtk_vector_output(const char* filename, const char* field_name,
                           const double* u_data, const double* v_data,
                           size_t nx, size_t ny,
                           double xmin, double xmax, double ymin, double ymax) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        cfd_error("Failed to open VTK vector output file");
    }

    // Write VTK header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "CFD Framework Vector Output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %zu %zu 1\n", nx, ny);
    fprintf(fp, "ORIGIN %f %f 0.0\n", xmin, ymin);
    fprintf(fp, "SPACING %f %f 1.0\n", (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1));

    // Write vector field data
    fprintf(fp, "\nPOINT_DATA %zu\n", nx * ny);
    fprintf(fp, "VECTORS %s float\n", field_name);

    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f %f 0.0\n", u_data[idx], v_data[idx]);
        }
    }

    fclose(fp);
}

void write_vtk_flow_field(const char* filename,
                         const FlowField* field,
                         size_t nx, size_t ny,
                         double xmin, double xmax, double ymin, double ymax) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        cfd_error("Failed to open VTK flow field output file");
    }

    // Write VTK header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "CFD Framework Flow Field Output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %zu %zu 1\n", nx, ny);
    fprintf(fp, "ORIGIN %f %f 0.0\n", xmin, ymin);
    fprintf(fp, "SPACING %f %f 1.0\n", (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1));

    // Write multiple field data
    fprintf(fp, "\nPOINT_DATA %zu\n", nx * ny);

    // Velocity vector field
    fprintf(fp, "VECTORS velocity float\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f %f 0.0\n", field->u[idx], field->v[idx]);
        }
    }

    // Pressure scalar field
    fprintf(fp, "\nSCALARS pressure float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f\n", field->p[idx]);
        }
    }

    // Density scalar field
    fprintf(fp, "\nSCALARS density float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f\n", field->rho[idx]);
        }
    }

    // Temperature scalar field
    fprintf(fp, "\nSCALARS temperature float 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f\n", field->T[idx]);
        }
    }

    fclose(fp);
}

// New run-based output functions
static void get_run_filepath(char* buffer, size_t buffer_size, const char* filename) {
    const char* run_dir = cfd_get_run_directory();

    // If no run directory exists yet, create one
    if (run_dir == NULL) {
        char new_run_dir[512];
        cfd_create_run_directory(new_run_dir, sizeof(new_run_dir));
        run_dir = cfd_get_run_directory();
    }

    // Build full path
#ifdef _WIN32
    snprintf(buffer, buffer_size, "%s\\%s", run_dir, filename);
#else
    snprintf(buffer, buffer_size, "%s/%s", run_dir, filename);
#endif
}

void write_vtk_output_run(const char* filename, const char* field_name,
                         const double* data, size_t nx, size_t ny,
                         double xmin, double xmax, double ymin, double ymax) {
    char filepath[1024];
    get_run_filepath(filepath, sizeof(filepath), filename);
    write_vtk_output(filepath, field_name, data, nx, ny, xmin, xmax, ymin, ymax);
}

void write_vtk_vector_output_run(const char* filename, const char* field_name,
                                const double* u_data, const double* v_data,
                                size_t nx, size_t ny,
                                double xmin, double xmax, double ymin, double ymax) {
    char filepath[1024];
    get_run_filepath(filepath, sizeof(filepath), filename);
    write_vtk_vector_output(filepath, field_name, u_data, v_data, nx, ny, xmin, xmax, ymin, ymax);
}

void write_vtk_flow_field_run(const char* filename,
                             const FlowField* field,
                             size_t nx, size_t ny,
                             double xmin, double xmax, double ymin, double ymax) {
    char filepath[1024];
    get_run_filepath(filepath, sizeof(filepath), filename);
    write_vtk_flow_field(filepath, field, nx, ny, xmin, xmax, ymin, ymax);
}

// Calculate velocity magnitude field for visualization
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny) {
    double* velocity_magnitude = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude[i] = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }

    return velocity_magnitude;
}