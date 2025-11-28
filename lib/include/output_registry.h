#ifndef OUTPUT_REGISTRY_H
#define OUTPUT_REGISTRY_H

#include "simulation_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// OUTPUT REGISTRY
//=============================================================================

// Opaque output registry type (internal structure defined in .c file)
typedef struct OutputRegistry OutputRegistry;

// Create output registry
OutputRegistry* output_registry_create(void);

// Destroy output registry
void output_registry_destroy(OutputRegistry* reg);

// Register output configuration
void output_registry_add(OutputRegistry* reg,
                         OutputFieldType field_type,
                         int interval,
                         const char* prefix);

// Clear all registered outputs
void output_registry_clear(OutputRegistry* reg);

// Get count of registered outputs
int output_registry_count(const OutputRegistry* reg);

// Get run directory (creates it if needed)
const char* output_registry_get_run_dir(OutputRegistry* reg,
                                         const char* base_dir,
                                         const char* run_prefix,
                                         size_t nx, size_t ny);

// Process all registered outputs for current step
void output_registry_write_outputs(OutputRegistry* reg,
                                    const char* run_dir,
                                    int step,
                                    double current_time,
                                    const FlowField* field,
                                    const Grid* grid,
                                    const SolverParams* params,
                                    const SolverStats* stats);

#ifdef __cplusplus
}
#endif

#endif // OUTPUT_REGISTRY_H
