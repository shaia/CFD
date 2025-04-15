#include "vtk_output.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdio.h> // For debug prints

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

    // Debug: Print grid points
    printf("Grid Points:\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            printf("(%f, %f)\n", xmin + i * ((xmax - xmin) / (nx - 1)), ymin + j * ((ymax - ymin) / (ny - 1)));
        }
    }

    // Write field data
    fprintf(fp, "\nPOINT_DATA %zu\n", nx * ny);
    fprintf(fp, "SCALARS %s float 1\n", field_name);
    fprintf(fp, "LOOKUP_TABLE default\n");

    // Debug: Print data values
    printf("Data Values:\n");
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            fprintf(fp, "%f\n", data[idx]);
        }
    }

    fclose(fp);
}