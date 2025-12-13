#ifndef OUTPUT_REGISTRY_H
#define OUTPUT_REGISTRY_H

#include "cfd/cfd_export.h"

#include "cfd/api/simulation_api.h"
#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// OUTPUT REGISTRY
//=============================================================================

// Opaque output registry type (internal structure defined in .c file)
typedef struct OutputRegistry output_registry;

// Create output registry
CFD_LIBRARY_EXPORT output_registry* output_registry_create(void);

// Destroy output registry
CFD_LIBRARY_EXPORT void output_registry_destroy(output_registry* reg);

// Register output configuration
CFD_LIBRARY_EXPORT void output_registry_add(output_registry* reg, output_field_type field_type,
                                            int interval, const char* prefix);

// Clear all registered outputs
CFD_LIBRARY_EXPORT void output_registry_clear(output_registry* reg);

// Get count of registered outputs
CFD_LIBRARY_EXPORT int output_registry_count(const output_registry* reg);

// Get run directory (creates it if needed)
CFD_LIBRARY_EXPORT const char* output_registry_get_run_dir(output_registry* reg,
                                                           const char* base_dir,
                                                           const char* run_prefix, size_t nx,
                                                           size_t ny);

// Check if any output of given type is registered
CFD_LIBRARY_EXPORT int output_registry_has_type(const output_registry* reg,
                                                output_field_type field_type);

// Process all registered outputs for current step (with pre-computed derived fields)
CFD_LIBRARY_EXPORT void output_registry_write_outputs(output_registry* reg, const char* run_dir,
                                                      int step, double current_time,
                                                      const flow_field* field,
                                                      const derived_fields* derived,
                                                      const grid* grid, const solver_params* params,
                                                      const solver_stats* stats);

#ifdef __cplusplus
}
#endif

#endif  // OUTPUT_REGISTRY_H
