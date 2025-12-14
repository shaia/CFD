#ifndef VTK_OUTPUT_INTERNAL_H
#define VTK_OUTPUT_INTERNAL_H

#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// INTERNAL VTK OUTPUT DISPATCH (used by output_registry.c)
//=============================================================================

// VTK output type for dispatch
typedef enum {
    VTK_OUTPUT_VELOCITY_MAGNITUDE,  // Derived field - computed at registry level
    VTK_OUTPUT_VELOCITY,
    VTK_OUTPUT_FULL_FIELD
} vtk_output_type;

// Dispatch VTK output based on type
// This is an internal function called by output_registry.c
void vtk_dispatch_output(vtk_output_type vtk_type, const char* run_dir, const char* prefix,
                         int step, const flow_field* field, const grid* grid);

// Write pre-computed scalar field to VTK (used for velocity magnitude)
// The caller is responsible for computing the data before calling this function
void vtk_write_scalar_field(const char* run_dir, const char* prefix, int step,
                            const char* field_name, const double* data, const grid* grid);

#ifdef __cplusplus
}
#endif

#endif  // VTK_OUTPUT_INTERNAL_H
