#include "cfd/io/output_registry.h"
#include "../io/csv_output_internal.h"
#include "../io/vtk_output_internal.h"
#include "cfd/api/simulation_api.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/logging.h"
#include "cfd/core/memory.h"


#include <string.h>

//=============================================================================
// OUTPUT REGISTRY STRUCTURE
//=============================================================================

#define MAX_OUTPUT_CONFIGS 16

struct OutputRegistry {
    output_config configs[MAX_OUTPUT_CONFIGS];
    int count;
    char run_dir[512];    // Cached run directory path
    int run_dir_created;  // Flag to track if run directory was created
};

//=============================================================================
// OUTPUT REGISTRY LIFECYCLE
//=============================================================================

output_registry* output_registry_create(void) {
    output_registry* reg = (output_registry*)cfd_calloc(1, sizeof(struct OutputRegistry));
    if (reg) {
        reg->count = 0;
        reg->run_dir[0] = '\0';
        reg->run_dir_created = 0;
    }
    return reg;
}

void output_registry_destroy(output_registry* reg) {
    if (reg) {
        cfd_free(reg);
    }
}

//=============================================================================
// OUTPUT REGISTRY CONFIGURATION
//=============================================================================

void output_registry_add(output_registry* reg, output_field_type field_type, int interval,
                         const char* prefix) {
    if (!reg) {
        return;
    }

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

void output_registry_clear(output_registry* reg) {
    if (reg) {
        reg->count = 0;
    }
}

int output_registry_count(const output_registry* reg) {
    return reg ? reg->count : 0;
}

int output_registry_has_type(const output_registry* reg, output_field_type field_type) {
    if (!reg) {
        return 0;
    }

    for (int i = 0; i < reg->count; i++) {
        if (reg->configs[i].field_type == field_type) {
            return 1;
        }
    }
    return 0;
}

//=============================================================================
// RUN DIRECTORY MANAGEMENT
//=============================================================================

const char* output_registry_get_run_dir(output_registry* reg, const char* base_dir,
                                        const char* run_prefix, size_t nx, size_t ny) {
    if (!reg) {
        return NULL;
    }

    // Only create once
    if (reg->run_dir_created) {
        return reg->run_dir;
    }

    // Create run directory with prefix using custom base directory (re-entrant)
    const char* prefix = run_prefix ? run_prefix : "sim";
    cfd_create_run_directory_ex_with_base(reg->run_dir, sizeof(reg->run_dir), base_dir, prefix, nx,
                                          ny);

    reg->run_dir_created = 1;
    return reg->run_dir;
}

//=============================================================================
// OUTPUT DISPATCH
//=============================================================================

// Function pointer type for output dispatchers (with derived fields)
typedef void (*output_dispatch_func)(const char* run_dir, const char* prefix, int step,
                                     double current_time, const flow_field* field,
                                     const derived_fields* derived, const grid* grid,
                                     const solver_params* params, const solver_stats* stats);

// VTK output wrappers
static void dispatch_vtk_velocity_magnitude(const char* run_dir, const char* prefix, int step,
                                            double current_time, const flow_field* field,
                                            const derived_fields* derived, const grid* grid,
                                            const solver_params* params,
                                            const solver_stats* stats) {
    (void)current_time;
    (void)field;
    (void)params;
    (void)stats;
    // Use pre-computed velocity magnitude from derived fields
    if (derived && derived->velocity_magnitude) {
        vtk_write_scalar_field(run_dir, prefix, step, "velocity_magnitude",
                               derived->velocity_magnitude, grid);
    }
}

static void dispatch_vtk_velocity(const char* run_dir, const char* prefix, int step,
                                  double current_time, const flow_field* field,
                                  const derived_fields* derived, const grid* grid,
                                  const solver_params* params, const solver_stats* stats) {
    (void)current_time;
    (void)derived;
    (void)params;
    (void)stats;
    vtk_dispatch_output(VTK_OUTPUT_VELOCITY, run_dir, prefix, step, field, grid);
}

static void dispatch_vtk_full_field(const char* run_dir, const char* prefix, int step,
                                    double current_time, const flow_field* field,
                                    const derived_fields* derived, const grid* grid,
                                    const solver_params* params, const solver_stats* stats) {
    (void)current_time;
    (void)derived;
    (void)params;
    (void)stats;
    vtk_dispatch_output(VTK_OUTPUT_FULL_FIELD, run_dir, prefix, step, field, grid);
}

// CSV output wrappers (pass derived fields)
static void dispatch_csv_timeseries(const char* run_dir, const char* prefix, int step,
                                    double current_time, const flow_field* field,
                                    const derived_fields* derived, const grid* grid,
                                    const solver_params* params, const solver_stats* stats) {
    csv_dispatch_output(CSV_OUTPUT_TIMESERIES, run_dir, prefix, step, current_time, field, derived,
                        grid, params, stats);
}

static void dispatch_csv_centerline(const char* run_dir, const char* prefix, int step,
                                    double current_time, const flow_field* field,
                                    const derived_fields* derived, const grid* grid,
                                    const solver_params* params, const solver_stats* stats) {
    csv_dispatch_output(CSV_OUTPUT_CENTERLINE, run_dir, prefix, step, current_time, field, derived,
                        grid, params, stats);
}

static void dispatch_csv_statistics(const char* run_dir, const char* prefix, int step,
                                    double current_time, const flow_field* field,
                                    const derived_fields* derived, const grid* grid,
                                    const solver_params* params, const solver_stats* stats) {
    csv_dispatch_output(CSV_OUTPUT_STATISTICS, run_dir, prefix, step, current_time, field, derived,
                        grid, params, stats);
}

// Output dispatch table - indexed by output_field_type
// This provides O(1) lookup with no branch prediction issues
static const output_dispatch_func output_dispatch_table[] = {
    dispatch_vtk_velocity_magnitude,  // OUTPUT_VELOCITY_MAGNITUDE = 0
    dispatch_vtk_velocity,            // OUTPUT_VELOCITY = 1
    dispatch_vtk_full_field,          // OUTPUT_FULL_FIELD = 2
    dispatch_csv_timeseries,          // OUTPUT_CSV_TIMESERIES = 3
    dispatch_csv_centerline,          // OUTPUT_CSV_CENTERLINE = 4
    dispatch_csv_statistics           // OUTPUT_CSV_STATISTICS = 5
};

#define OUTPUT_DISPATCH_TABLE_SIZE \
    (sizeof(output_dispatch_table) / sizeof(output_dispatch_table[0]))

//=============================================================================
// OUTPUT WRITING
//=============================================================================

void output_registry_write_outputs(output_registry* reg, const char* run_dir, int step,
                                   double current_time, const flow_field* field,
                                   const derived_fields* derived, const grid* grid,
                                   const solver_params* params, const solver_stats* stats) {
    if (!reg || !run_dir) {
        return;
    }

    // Process each registered output
    for (int i = 0; i < reg->count; i++) {
        output_config* config = &reg->configs[i];

        // Skip if not time to output
        if (config->interval <= 0 || step % config->interval != 0) {
            continue;
        }

        // Bounds check and dispatch via function table
        // This is cacheable and avoids branch prediction issues
        if (config->field_type < OUTPUT_DISPATCH_TABLE_SIZE) {
            output_dispatch_table[config->field_type](run_dir, config->prefix, step, current_time,
                                                      field, derived, grid, params, stats);
        } else {
            cfd_warning("Unknown output type, skipping");
        }
    }
}
