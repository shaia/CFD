#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "solver_interface.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * VTK Output API - Low-Level Direct Output Functions
 *
 * WHEN TO USE THIS HEADER:
 * - You are writing a custom solver and need manual output control
 * - You are writing test code that needs to verify specific outputs
 * - You need fine-grained control over output filenames and timing
 * - You are prototyping and want simple, direct output
 *
 * WHEN NOT TO USE THIS HEADER:
 * - You are using simulation_api.h for simulations
 *   → Use output_registry via simulation_register_output() instead
 * - You want automatic timestamped outputs at regular intervals
 *   → Use the output registry system for automated output management
 *
 * The output registry provides higher-level features like:
 * - Automatic run directory creation with timestamps
 * - Configurable output intervals (every N steps)
 * - Multiple output types registered once
 * - Consistent naming across simulation runs
 */

//=============================================================================
// VTK OUTPUT API
//=============================================================================

// Write scalar field to VTK file (full path required)
void write_vtk_output(const char* filename, const char* field_name, const double* data, size_t nx,
                      size_t ny, double xmin, double xmax, double ymin, double ymax);

void write_vtk_vector_output(const char* filename, const char* field_name, const double* u_data,
                             const double* v_data, size_t nx, size_t ny, double xmin, double xmax,
                             double ymin, double ymax);

void write_vtk_flow_field(const char* filename, const FlowField* field, size_t nx, size_t ny,
                          double xmin, double xmax, double ymin, double ymax);

// New functions - automatically create timestamped run directory
// These functions create a new directory for each run and write files there
void write_vtk_output_run(const char* filename, const char* field_name, const double* data,
                          size_t nx, size_t ny, double xmin, double xmax, double ymin, double ymax);

void write_vtk_vector_output_run(const char* filename, const char* field_name, const double* u_data,
                                 const double* v_data, size_t nx, size_t ny, double xmin,
                                 double xmax, double ymin, double ymax);

void write_vtk_flow_field_run(const char* filename, const FlowField* field, size_t nx, size_t ny,
                              double xmin, double xmax, double ymin, double ymax);

#ifdef __cplusplus
}
#endif

#endif  // VTK_OUTPUT_H