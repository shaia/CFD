#ifndef CSV_OUTPUT_INTERNAL_H
#define CSV_OUTPUT_INTERNAL_H

#include "csv_output.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// INTERNAL CSV OUTPUT DISPATCH (used by output_registry.c)
//=============================================================================

// CSV output type for dispatch
typedef enum { CSV_OUTPUT_TIMESERIES, CSV_OUTPUT_CENTERLINE, CSV_OUTPUT_STATISTICS } CsvOutputType;

// Dispatch CSV output based on type
// This is an internal function called by output_registry.c
void csv_dispatch_output(CsvOutputType csv_type, const char* run_dir, const char* prefix, int step,
                         double current_time, const FlowField* field, const Grid* grid,
                         const SolverParams* params, const SolverStats* stats);

#ifdef __cplusplus
}
#endif

#endif  // CSV_OUTPUT_INTERNAL_H
