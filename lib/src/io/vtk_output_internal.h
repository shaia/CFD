#ifndef VTK_OUTPUT_INTERNAL_H
#define VTK_OUTPUT_INTERNAL_H

#include "vtk_output.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// INTERNAL VTK OUTPUT DISPATCH (used by output_registry.c)
//=============================================================================

// VTK output type for dispatch
typedef enum { VTK_OUTPUT_PRESSURE, VTK_OUTPUT_VELOCITY, VTK_OUTPUT_FULL_FIELD } VtkOutputType;

// Dispatch VTK output based on type
// This is an internal function called by output_registry.c
void vtk_dispatch_output(VtkOutputType vtk_type, const char* run_dir, const char* prefix, int step,
                         const FlowField* field, const Grid* grid);

#ifdef __cplusplus
}
#endif

#endif  // VTK_OUTPUT_INTERNAL_H
