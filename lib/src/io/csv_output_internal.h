#ifndef CSV_OUTPUT_INTERNAL_H
#define CSV_OUTPUT_INTERNAL_H

#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// INTERNAL CSV OUTPUT DISPATCH (used by output_registry.c)
//=============================================================================

// CSV output type for dispatch
typedef enum {
    CSV_OUTPUT_TIMESERIES,
    CSV_OUTPUT_CENTERLINE,
    CSV_OUTPUT_STATISTICS
} csv_output_type;

// Dispatch CSV output based on type
// This is an internal function called by output_registry.c
// The derived parameter contains pre-computed fields like velocity magnitude (can be NULL)
void csv_dispatch_output(csv_output_type csv_type, const char* run_dir, const char* prefix,
                         int step, double current_time, const flow_field* field,
                         const derived_fields* derived, const grid* grid,
                         const ns_solver_params_t* params, const ns_solver_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif  // CSV_OUTPUT_INTERNAL_H
