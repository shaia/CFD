#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include <stddef.h>

void write_vtk_output(const char* filename, const char* field_name, 
                     const double* data, size_t nx, size_t ny,
                     double xmin, double xmax, double ymin, double ymax);

#endif // VTK_OUTPUT_H