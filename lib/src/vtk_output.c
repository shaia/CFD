#include "vtk_output.h"
#include "utils.h"
#include "solver.h"
#include <stdio.h>
#include <stdlib.h>

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