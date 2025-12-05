#include "output_registry.h"
#include "../io/csv_output_internal.h"
#include "../io/vtk_output_internal.h"
#include "utils.h"
#include <string.h>

//=============================================================================
// OUTPUT REGISTRY STRUCTURE
//=============================================================================

#define MAX_OUTPUT_CONFIGS 16

struct OutputRegistry {
    OutputConfig configs[MAX_OUTPUT_CONFIGS];
    int count;
    char run_dir[512];    // Cached run directory path
    int run_dir_created;  // Flag to track if run directory was created
};

//=============================================================================
// OUTPUT REGISTRY LIFECYCLE
//=============================================================================

OutputRegistry* output_registry_create(void) {
    OutputRegistry* reg = (OutputRegistry*)cfd_calloc(1, sizeof(OutputRegistry));
    if (reg) {
        reg->count = 0;
        reg->run_dir[0] = '\0';
        reg->run_dir_created = 0;
    }
    return reg;
}

void output_registry_destroy(OutputRegistry* reg) {
    if (reg) {
        cfd_free(reg);
    }
}

//=============================================================================
// OUTPUT REGISTRY CONFIGURATION
//=============================================================================

void output_registry_add(OutputRegistry* reg, OutputFieldType field_type, int interval,
                         const char* prefix) {
    if (!reg)
        return;

    // Check if we have space
    if (reg->count >= MAX_OUTPUT_CONFIGS) {
        cfd_warning("Maximum number of output configurations reached");
        return;
    }

    // Add new configuration
    reg->configs[reg->count].field_type = field_type;
    reg->configs[reg->count].interval = interval;
    reg->configs[reg->count].prefix = prefix;
    reg->count++;
}

void output_registry_clear(OutputRegistry* reg) {
    if (reg) {
        reg->count = 0;
    }
}

int output_registry_count(const OutputRegistry* reg) {
    return reg ? reg->count : 0;
}

//=============================================================================
// RUN DIRECTORY MANAGEMENT
//=============================================================================

const char* output_registry_get_run_dir(OutputRegistry* reg, const char* base_dir,
                                        const char* run_prefix, size_t nx, size_t ny) {
    if (!reg)
        return NULL;

    // Only create once
    if (reg->run_dir_created) {
        return reg->run_dir;
    }

    // Set base output directory
    if (base_dir) {
        cfd_set_output_base_dir(base_dir);
    }

    // Create run directory with prefix
    const char* prefix = run_prefix ? run_prefix : "sim";
    cfd_create_run_directory_ex(reg->run_dir, sizeof(reg->run_dir), prefix, nx, ny);

    reg->run_dir_created = 1;
    return reg->run_dir;
}

//=============================================================================
// OUTPUT DISPATCH
//=============================================================================

// Function pointer type for output dispatchers
typedef void (*OutputDispatchFunc)(const char* run_dir, const char* prefix, int step,
                                   double current_time, const FlowField* field, const Grid* grid,
                                   const SolverParams* params, const SolverStats* stats);

// VTK output wrappers
static void dispatch_vtk_pressure(const char* run_dir, const char* prefix, int step,
                                  double current_time, const FlowField* field, const Grid* grid,
                                  const SolverParams* params, const SolverStats* stats) {
    (void)current_time;
    (void)params;
    (void)stats;  // Unused for VTK
    vtk_dispatch_output(VTK_OUTPUT_PRESSURE, run_dir, prefix, step, field, grid);
}

static void dispatch_vtk_velocity(const char* run_dir, const char* prefix, int step,
                                  double current_time, const FlowField* field, const Grid* grid,
                                  const SolverParams* params, const SolverStats* stats) {
    (void)current_time;
    (void)params;
    (void)stats;  // Unused for VTK
    vtk_dispatch_output(VTK_OUTPUT_VELOCITY, run_dir, prefix, step, field, grid);
}

static void dispatch_vtk_full_field(const char* run_dir, const char* prefix, int step,
                                    double current_time, const FlowField* field, const Grid* grid,
                                    const SolverParams* params, const SolverStats* stats) {
    (void)current_time;
    (void)params;
    (void)stats;  // Unused for VTK
    vtk_dispatch_output(VTK_OUTPUT_FULL_FIELD, run_dir, prefix, step, field, grid);
}

// CSV output wrappers
static void dispatch_csv_timeseries(const char* run_dir, const char* prefix, int step,
                                    double current_time, const FlowField* field, const Grid* grid,
                                    const SolverParams* params, const SolverStats* stats) {
    csv_dispatch_output(CSV_OUTPUT_TIMESERIES, run_dir, prefix, step, current_time, field, grid,
                        params, stats);
}

static void dispatch_csv_centerline(const char* run_dir, const char* prefix, int step,
                                    double current_time, const FlowField* field, const Grid* grid,
                                    const SolverParams* params, const SolverStats* stats) {
    csv_dispatch_output(CSV_OUTPUT_CENTERLINE, run_dir, prefix, step, current_time, field, grid,
                        params, stats);
}

static void dispatch_csv_statistics(const char* run_dir, const char* prefix, int step,
                                    double current_time, const FlowField* field, const Grid* grid,
                                    const SolverParams* params, const SolverStats* stats) {
    csv_dispatch_output(CSV_OUTPUT_STATISTICS, run_dir, prefix, step, current_time, field, grid,
                        params, stats);
}

// Output dispatch table - indexed by OutputFieldType
// This provides O(1) lookup with no branch prediction issues
static const OutputDispatchFunc output_dispatch_table[] = {
    dispatch_vtk_pressure,    // OUTPUT_PRESSURE = 0
    dispatch_vtk_velocity,    // OUTPUT_VELOCITY = 1
    dispatch_vtk_full_field,  // OUTPUT_FULL_FIELD = 2
    dispatch_csv_timeseries,  // OUTPUT_CSV_TIMESERIES = 3
    dispatch_csv_centerline,  // OUTPUT_CSV_CENTERLINE = 4
    dispatch_csv_statistics   // OUTPUT_CSV_STATISTICS = 5
};

#define OUTPUT_DISPATCH_TABLE_SIZE \
    (sizeof(output_dispatch_table) / sizeof(output_dispatch_table[0]))

//=============================================================================
// OUTPUT WRITING
//=============================================================================

void output_registry_write_outputs(OutputRegistry* reg, const char* run_dir, int step,
                                   double current_time, const FlowField* field, const Grid* grid,
                                   const SolverParams* params, const SolverStats* stats) {
    if (!reg || !run_dir)
        return;

    // Process each registered output
    for (int i = 0; i < reg->count; i++) {
        OutputConfig* config = &reg->configs[i];

        // Skip if not time to output
        if (config->interval <= 0 || step % config->interval != 0) {
            continue;
        }

        // Bounds check and dispatch via function table
        // This is cacheable and avoids branch prediction issues
        if (config->field_type < OUTPUT_DISPATCH_TABLE_SIZE) {
            output_dispatch_table[config->field_type](run_dir, config->prefix, step, current_time,
                                                      field, grid, params, stats);
        } else {
            cfd_warning("Unknown output type, skipping");
        }
    }
}
