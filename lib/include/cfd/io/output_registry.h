#ifndef OUTPUT_REGISTRY_H
#define OUTPUT_REGISTRY_H

#include "cfd/cfd_export.h"

#include "cfd/api/simulation_api.h"
#include "cfd/core/derived_fields.h"


#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// OUTPUT REGISTRY
//=============================================================================

// Opaque output registry type (internal structure defined in .c file)
typedef struct OutputRegistry OutputRegistry;

// Create output registry
CFD_LIBRARY_EXPORT OutputRegistry* output_registry_create(void);

// Destroy output registry
CFD_LIBRARY_EXPORT void output_registry_destroy(OutputRegistry* reg);

// Register output configuration
CFD_LIBRARY_EXPORT void output_registry_add(OutputRegistry* reg, OutputFieldType field_type,
                                            int interval, const char* prefix);

// Clear all registered outputs
CFD_LIBRARY_EXPORT void output_registry_clear(OutputRegistry* reg);

// Get count of registered outputs
CFD_LIBRARY_EXPORT int output_registry_count(const OutputRegistry* reg);

// Get run directory (creates it if needed)
CFD_LIBRARY_EXPORT const char* output_registry_get_run_dir(OutputRegistry* reg,
                                                           const char* base_dir,
                                                           const char* run_prefix, size_t nx,
                                                           size_t ny);

// Check if any output of given type is registered
CFD_LIBRARY_EXPORT int output_registry_has_type(const OutputRegistry* reg,
                                                OutputFieldType field_type);

// Process all registered outputs for current step (with pre-computed derived fields)
CFD_LIBRARY_EXPORT void output_registry_write_outputs(OutputRegistry* reg, const char* run_dir,
                                                      int step, double current_time,
                                                      const FlowField* field,
                                                      const DerivedFields* derived,
                                                      const Grid* grid, const SolverParams* params,
                                                      const SolverStats* stats);

#ifdef __cplusplus
}
#endif

#endif  // OUTPUT_REGISTRY_H
