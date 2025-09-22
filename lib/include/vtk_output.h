#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include <stddef.h>
#include "solver.h"

void write_vtk_output(const char* filename, const char* field_name,
                     const double* data, size_t nx, size_t ny,
                     double xmin, double xmax, double ymin, double ymax);

void write_vtk_vector_output(const char* filename, const char* field_name,
                           const double* u_data, const double* v_data,
                           size_t nx, size_t ny,
                           double xmin, double xmax, double ymin, double ymax);

void write_vtk_flow_field(const char* filename,
                         const FlowField* field,
                         size_t nx, size_t ny,
                         double xmin, double xmax, double ymin, double ymax);

#endif // VTK_OUTPUT_H