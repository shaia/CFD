#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include "cfd/cfd_export.h"

#include "cfd/solvers/navier_stokes_solver.h"
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
 *   -> Use output_registry via simulation_register_output() instead
 * - You want automatic timestamped outputs at regular intervals
 *   -> Use the output registry system for automated output management
 *
 * The output registry provides higher-level features like:
 * - Automatic run directory creation with timestamps
 * - Configurable output intervals (every N steps)
 * - Multiple output types registered once
 * - Consistent naming across simulation runs
 *
 * 3D SUPPORT:
 * All functions accept nz, zmin, zmax parameters. For 2D output, pass
 * nz=1, zmin=0.0, zmax=0.0 (produces identical output to pre-3D behavior).
 */

//=============================================================================
// VTK OUTPUT API
//=============================================================================

// Write scalar field to VTK file (full path required)
CFD_LIBRARY_EXPORT void write_vtk_output(const char* filename, const char* field_name,
                                         const double* data, size_t nx, size_t ny, size_t nz,
                                         double xmin, double xmax, double ymin, double ymax,
                                         double zmin, double zmax);

// Write vector field to VTK file. w_data can be NULL (writes 0.0 for z-component).
CFD_LIBRARY_EXPORT void write_vtk_vector_output(const char* filename, const char* field_name,
                                                const double* u_data, const double* v_data,
                                                const double* w_data, size_t nx, size_t ny,
                                                size_t nz, double xmin, double xmax,
                                                double ymin, double ymax, double zmin,
                                                double zmax);

// Write complete flow field (velocity, pressure, density, temperature) to VTK file
CFD_LIBRARY_EXPORT void write_vtk_flow_field(const char* filename, const flow_field* field,
                                             size_t nx, size_t ny, size_t nz, double xmin,
                                             double xmax, double ymin, double ymax, double zmin,
                                             double zmax);

// Run-directory variants - automatically create timestamped run directory
CFD_LIBRARY_EXPORT void write_vtk_output_run(const char* filename, const char* field_name,
                                             const double* data, size_t nx, size_t ny, size_t nz,
                                             double xmin, double xmax, double ymin, double ymax,
                                             double zmin, double zmax);

CFD_LIBRARY_EXPORT void write_vtk_vector_output_run(const char* filename, const char* field_name,
                                                    const double* u_data, const double* v_data,
                                                    const double* w_data, size_t nx, size_t ny,
                                                    size_t nz, double xmin, double xmax,
                                                    double ymin, double ymax, double zmin,
                                                    double zmax);

CFD_LIBRARY_EXPORT void write_vtk_flow_field_run(const char* filename, const flow_field* field,
                                                 size_t nx, size_t ny, size_t nz, double xmin,
                                                 double xmax, double ymin, double ymax,
                                                 double zmin, double zmax);

#ifdef __cplusplus
}
#endif

#endif  // VTK_OUTPUT_H
