#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/derived_fields.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/io/checkpoint.h"
#include "cfd/io/output_registry.h"
#include "cfd/solvers/navier_stokes_solver.h"


#include "cfd/core/cfd_init.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// simulation_data struct is defined in simulation_api.h

// Default solver type used when none is specified
#define DEFAULT_SOLVER_TYPE NS_SOLVER_TYPE_EXPLICIT_EULER

// Ensure registry is initialized
// Internal helper to create simulation with a specific solver
static simulation_data* create_simulation_with_solver(size_t nx, size_t ny, size_t nz,
                                                       double xmin, double xmax,
                                                       double ymin, double ymax,
                                                       double zmin, double zmax,
                                                       const char* solver_type) {
    // Lazy initialization of library
    if (!cfd_is_initialized()) {
        if (cfd_init() != CFD_SUCCESS) {
            cfd_set_error(CFD_ERROR, "Failed to initialize CFD library");
            return NULL;
        }
    }

    if (nx == 0 || ny == 0 || nz == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "Simulation grid dimensions must be positive");
        return NULL;
    }
    if (xmax <= xmin || ymax <= ymin || (nz > 1 && zmax <= zmin)) {
        cfd_set_error(CFD_ERROR_INVALID, "Simulation bounds invalid");
        return NULL;
    }

    simulation_data* sim_data = (simulation_data*)cfd_malloc(sizeof(simulation_data));
    if (!sim_data) {
        return NULL;
    }

    // Initialize base output directory
    snprintf(sim_data->output_base_dir, sizeof(sim_data->output_base_dir), "../../artifacts");

    // Create and initialize grid
    sim_data->grid = grid_create(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax);
    if (!sim_data->grid) {
        cfd_free(sim_data);
        return NULL;
    }
    grid_initialize_uniform(sim_data->grid);

    // Create flow_field
    sim_data->field = flow_field_create(nx, ny, nz);
    if (!sim_data->field) {
        grid_destroy(sim_data->grid);
        cfd_free(sim_data);
        return NULL;
    }
    initialize_flow_field(sim_data->field, sim_data->grid);

    // Initialize solver parameters with defaults
    sim_data->params = ns_solver_params_default();
    sim_data->params.dt = 0.001;
    sim_data->params.cfl = 0.2;
    sim_data->params.mu = 0.01;
    sim_data->params.max_iter = 1;

    // Initialize stats
    sim_data->last_stats = ns_solver_stats_default();

    // Create output registry
    sim_data->outputs = output_registry_create();
    if (!sim_data->outputs) {
        flow_field_destroy(sim_data->field);
        grid_destroy(sim_data->grid);
        cfd_free(sim_data);
        return NULL;
    }
    sim_data->run_prefix = NULL;
    sim_data->current_time = 0.0;

    // Initialize NSSolver Registry
    sim_data->registry = cfd_registry_create();
    if (!sim_data->registry) {
        output_registry_destroy(sim_data->outputs);
        flow_field_destroy(sim_data->field);
        grid_destroy(sim_data->grid);
        cfd_free(sim_data);
        return NULL;
    }
    cfd_registry_register_defaults(sim_data->registry);

    // Create and initialize the solver
    sim_data->solver = cfd_solver_create(sim_data->registry, solver_type);
    if (!sim_data->solver) {
        // Failed to create solver - cleanup and return NULL
        cfd_registry_destroy(sim_data->registry);
        output_registry_destroy(sim_data->outputs);
        flow_field_destroy(sim_data->field);
        grid_destroy(sim_data->grid);
        cfd_free(sim_data);
        return NULL;
    }

    solver_init(sim_data->solver, sim_data->grid, &sim_data->params);

    return sim_data;
}

// Initialize simulation with default solver
simulation_data* init_simulation(size_t nx, size_t ny, size_t nz,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  double zmin, double zmax) {
    return create_simulation_with_solver(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax,
                                          DEFAULT_SOLVER_TYPE);
}

// Initialize simulation with a specific solver type
simulation_data* init_simulation_with_solver(size_t nx, size_t ny, size_t nz,
                                              double xmin, double xmax,
                                              double ymin, double ymax,
                                              double zmin, double zmax,
                                              const char* solver_type) {
    if (!solver_type) {
        solver_type = DEFAULT_SOLVER_TYPE;
    }
    return create_simulation_with_solver(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax,
                                          solver_type);
}

// Set the solver for an existing simulation
void simulation_set_solver(simulation_data* sim_data, struct NSSolver* solver) {
    if (!sim_data || !solver) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for simulation_set_solver");
        return;
    }

    // Destroy existing solver
    if (sim_data->solver) {
        solver_destroy(sim_data->solver);
    }

    sim_data->solver = solver;
    solver_init(solver, sim_data->grid, &sim_data->params);
}

// Set the solver by type name
int simulation_set_solver_by_name(simulation_data* sim_data, const char* solver_type) {
    if (!sim_data || !solver_type) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for simulation solver");
        return -1;
    }

    struct NSSolver* solver = cfd_solver_create(sim_data->registry, solver_type);
    if (!solver) {
        return -1;
    }

    simulation_set_solver(sim_data, solver);
    return 0;
}

// Get the current solver
struct NSSolver* simulation_get_solver(simulation_data* sim_data) {
    return sim_data ? sim_data->solver : NULL;
}

// Get statistics from the last solve
const ns_solver_stats_t* simulation_get_stats(const simulation_data* sim_data) {
    return sim_data ? &sim_data->last_stats : NULL;
}

// Run simulation step
cfd_status_t run_simulation_step(simulation_data* sim_data) {
    if (!sim_data || !sim_data->solver) {
        return CFD_ERROR_INVALID;
    }

    // Use fixed time step for animation stability
    sim_data->params.dt = 0.005;

    cfd_status_t status = solver_step(sim_data->solver, sim_data->field, sim_data->grid,
                                       &sim_data->params, &sim_data->last_stats);
    if (status != CFD_SUCCESS) {
        return status;
    }

    // Accumulate simulation time
    sim_data->current_time += sim_data->params.dt;
    return CFD_SUCCESS;
}

cfd_status_t run_simulation_solve(simulation_data* sim_data) {
    if (!sim_data || !sim_data->solver) {
        return CFD_ERROR_INVALID;
    }

    // Use fixed time step for animation stability
    sim_data->params.dt = 0.005;

    cfd_status_t status = solver_solve(sim_data->solver, sim_data->field, sim_data->grid,
                                        &sim_data->params, &sim_data->last_stats);

    // Accumulate simulation time based on iterations performed
    sim_data->current_time += sim_data->params.dt * sim_data->last_stats.iterations;

    return status;
}

// Free simulation data
// Free simulation data
void free_simulation(simulation_data* sim_data) {
    if (!sim_data) {
        return;
    }

    if (sim_data->solver) {
        solver_destroy(sim_data->solver);
        sim_data->solver = NULL;
    }

    if (sim_data->registry) {
        cfd_registry_destroy(sim_data->registry);
        sim_data->registry = NULL;
    }

    if (sim_data->outputs) {
        output_registry_destroy(sim_data->outputs);
        sim_data->outputs = NULL;
    }

    if (sim_data->run_prefix) {
        cfd_free(sim_data->run_prefix);
        sim_data->run_prefix = NULL;
    }

    flow_field_destroy(sim_data->field);
    grid_destroy(sim_data->grid);
    cfd_free(sim_data);
}

//=============================================================================
// RESTART / CHECKPOINT
//=============================================================================

cfd_status_t save_simulation_checkpoint(const simulation_data* sim_data, const char* path) {
    if (!sim_data || !path) {
        cfd_set_error(CFD_ERROR_INVALID, "save_simulation_checkpoint: NULL argument");
        return CFD_ERROR_INVALID;
    }
    if (!sim_data->grid || !sim_data->field || !sim_data->solver) {
        cfd_set_error(CFD_ERROR_INVALID, "save_simulation_checkpoint: simulation not initialized");
        return CFD_ERROR_INVALID;
    }

    const char* solver_name = sim_data->solver->name ? sim_data->solver->name : "";
    return cfd_checkpoint_write(path, sim_data->grid, sim_data->field, &sim_data->params,
                                sim_data->current_time, solver_name, sim_data->run_prefix,
                                sim_data->output_base_dir);
}

simulation_data* load_simulation_from_checkpoint(const char* path) {
    if (!path) {
        cfd_set_error(CFD_ERROR_INVALID, "load_simulation_from_checkpoint: NULL path");
        return NULL;
    }
    if (!cfd_is_initialized()) {
        if (cfd_init() != CFD_SUCCESS) {
            cfd_set_error(CFD_ERROR, "Failed to initialize CFD library");
            return NULL;
        }
    }

    grid* g = NULL;
    flow_field* f = NULL;
    ns_solver_params_t params;
    double current_time = 0.0;
    char solver_name[128] = {0};
    char run_prefix[256] = {0};
    char output_base_dir[512] = {0};

    cfd_status_t st = cfd_checkpoint_read(path, &g, &f, &params, &current_time, solver_name,
                                          sizeof(solver_name), run_prefix, sizeof(run_prefix),
                                          output_base_dir, sizeof(output_base_dir));
    if (st != CFD_SUCCESS) {
        return NULL;
    }

    simulation_data* sim_data = (simulation_data*)cfd_malloc(sizeof(simulation_data));
    if (!sim_data) {
        grid_destroy(g);
        flow_field_destroy(f);
        return NULL;
    }

    // Initialize every owned pointer up front so free_simulation() is a safe
    // cleanup from any failure point below.
    sim_data->grid = g;
    sim_data->field = f;
    sim_data->solver = NULL;
    sim_data->registry = NULL;
    sim_data->outputs = NULL;
    sim_data->run_prefix = NULL;
    sim_data->params = params;  // custom callbacks remain NULL (documented contract)
    sim_data->last_stats = ns_solver_stats_default();
    sim_data->current_time = current_time;
    snprintf(sim_data->output_base_dir, sizeof(sim_data->output_base_dir), "%s",
             output_base_dir[0] ? output_base_dir : "../../artifacts");

    sim_data->outputs = output_registry_create();
    if (!sim_data->outputs) {
        free_simulation(sim_data);
        return NULL;
    }

    sim_data->registry = cfd_registry_create();
    if (!sim_data->registry) {
        free_simulation(sim_data);
        return NULL;
    }
    cfd_registry_register_defaults(sim_data->registry);

    sim_data->solver = cfd_solver_create(sim_data->registry, solver_name);
    if (!sim_data->solver) {
        cfd_set_error(CFD_ERROR_NOT_FOUND, "load_simulation_from_checkpoint: solver not registered");
        free_simulation(sim_data);
        return NULL;
    }

    if (solver_init(sim_data->solver, sim_data->grid, &sim_data->params) != CFD_SUCCESS) {
        free_simulation(sim_data);
        return NULL;
    }

    if (run_prefix[0] != '\0') {
        size_t len = strlen(run_prefix) + 1;
        sim_data->run_prefix = (char*)cfd_malloc(len);
        if (!sim_data->run_prefix) {
            // Dropping the prefix would silently change output paths relative to
            // the checkpoint; fail hard rather than return a partially-restored sim.
            free_simulation(sim_data);
            return NULL;
        }
        snprintf(sim_data->run_prefix, len, "%s", run_prefix);
    }

    return sim_data;
}

cfd_status_t restore_simulation_checkpoint(simulation_data* sim_data, const char* path) {
    if (!sim_data || !path) {
        cfd_set_error(CFD_ERROR_INVALID, "restore_simulation_checkpoint: NULL argument");
        return CFD_ERROR_INVALID;
    }
    if (!sim_data->registry) {
        cfd_set_error(CFD_ERROR_INVALID, "restore_simulation_checkpoint: simulation not initialized");
        return CFD_ERROR_INVALID;
    }

    grid* new_grid = NULL;
    flow_field* new_field = NULL;
    ns_solver_params_t new_params;
    double current_time = 0.0;
    char solver_name[128] = {0};
    char run_prefix[256] = {0};
    char output_base_dir[512] = {0};

    cfd_status_t st = cfd_checkpoint_read(path, &new_grid, &new_field, &new_params, &current_time,
                                          solver_name, sizeof(solver_name), run_prefix,
                                          sizeof(run_prefix), output_base_dir,
                                          sizeof(output_base_dir));
    if (st != CFD_SUCCESS) {
        return st;
    }

    // Recreate the solver by name before tearing down old state, so a bad name
    // leaves the existing simulation untouched.
    ns_solver_t* new_solver = cfd_solver_create(sim_data->registry, solver_name);
    if (!new_solver) {
        grid_destroy(new_grid);
        flow_field_destroy(new_field);
        cfd_set_error(CFD_ERROR_NOT_FOUND, "restore_simulation_checkpoint: solver not registered");
        return CFD_ERROR_NOT_FOUND;
    }

    // Preserve the caller's custom callbacks; only scalar parameters are restored.
    new_params.source_func = sim_data->params.source_func;
    new_params.source_context = sim_data->params.source_context;
    new_params.heat_source_func = sim_data->params.heat_source_func;
    new_params.heat_source_context = sim_data->params.heat_source_context;

    // Initialize the new solver against the new grid/params *before* touching the
    // old state, so a failed init leaves the existing simulation untouched and
    // only frees the freshly allocated state.
    cfd_status_t init_st = solver_init(new_solver, new_grid, &new_params);
    if (init_st != CFD_SUCCESS) {
        solver_destroy(new_solver);
        grid_destroy(new_grid);
        flow_field_destroy(new_field);
        return init_st;
    }

    // Init succeeded — swap in new state and release the old.
    if (sim_data->solver) {
        solver_destroy(sim_data->solver);
    }
    grid_destroy(sim_data->grid);
    flow_field_destroy(sim_data->field);
    sim_data->grid = new_grid;
    sim_data->field = new_field;
    sim_data->solver = new_solver;
    sim_data->params = new_params;
    sim_data->current_time = current_time;

    // Allocate the new prefix before releasing the old one, so an allocation
    // failure leaves the existing prefix (and output behavior) intact.
    if (run_prefix[0] != '\0') {
        size_t len = strlen(run_prefix) + 1;
        char* new_prefix = (char*)cfd_malloc(len);
        if (new_prefix) {
            snprintf(new_prefix, len, "%s", run_prefix);
            cfd_free(sim_data->run_prefix);
            sim_data->run_prefix = new_prefix;
        }
    } else if (sim_data->run_prefix) {
        cfd_free(sim_data->run_prefix);
        sim_data->run_prefix = NULL;
    }
    snprintf(sim_data->output_base_dir, sizeof(sim_data->output_base_dir), "%s",
             output_base_dir[0] ? output_base_dir : "../../artifacts");

    return CFD_SUCCESS;
}

// Static list of available solver names (const strings have static lifetime)
// This avoids use-after-free issues with dynamically created registries
static const char* const s_solver_names[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER,
    NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
    NS_SOLVER_TYPE_PROJECTION,
    NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,
    NS_SOLVER_TYPE_PROJECTION_GPU,
#ifdef CFD_ENABLE_OPENMP
    NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
    NS_SOLVER_TYPE_PROJECTION_OMP,
#endif
};

static const int s_solver_count = (int)(sizeof(s_solver_names) / sizeof(s_solver_names[0]));

// NSSolver discovery and listing
int simulation_list_solvers(const char** names, int max_count) {
    if (names && max_count > 0) {
        int fill_count = (s_solver_count < max_count) ? s_solver_count : max_count;
        for (int i = 0; i < fill_count; i++) {
            names[i] = s_solver_names[i];
        }
    }
    return s_solver_count;
}

int simulation_has_solver(const char* solver_type) {
    if (!solver_type) {
        return 0;
    }
    for (int i = 0; i < s_solver_count; i++) {
        if (strcmp(s_solver_names[i], solver_type) == 0) {
            return 1;
        }
    }
    return 0;
}


//=============================================================================
// OUTPUT REGISTRY API
//=============================================================================

// Register output for automatic generation
void simulation_register_output(simulation_data* sim_data, output_field_type field_type,
                                int interval, const char* prefix) {
    if (!sim_data || !sim_data->outputs) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid simulation data");
        return;
    }
    output_registry_add(sim_data->outputs, field_type, interval, prefix);
}

// Clear all registered outputs
void simulation_clear_outputs(simulation_data* sim_data) {
    if (!sim_data || !sim_data->outputs) {
        return;
    }
    output_registry_clear(sim_data->outputs);
}

// Set base output directory
void simulation_set_output_dir(simulation_data* sim_data, const char* base_dir) {
    if (sim_data && base_dir && strlen(base_dir) > 0) {
        snprintf(sim_data->output_base_dir, sizeof(sim_data->output_base_dir), "%s", base_dir);
    }
}

// Set run name prefix
void simulation_set_run_prefix(simulation_data* sim_data, const char* prefix) {
    if (!sim_data) {
        return;
    }

    // Free existing prefix
    if (sim_data->run_prefix) {
        cfd_free(sim_data->run_prefix);
        sim_data->run_prefix = NULL;
    }

    // Set new prefix
    if (prefix) {
        size_t len = strlen(prefix) + 1;
        sim_data->run_prefix = (char*)cfd_malloc(len);
        if (sim_data->run_prefix) {
            snprintf(sim_data->run_prefix, len, "%s", prefix);
        }
    }
}

//=============================================================================
// AUTOMATIC OUTPUT GENERATION
//=============================================================================

// Check if any output type that uses velocity magnitude is registered
static int needs_velocity_magnitude(const output_registry* outputs) {
    return output_registry_has_type(outputs, OUTPUT_VELOCITY_MAGNITUDE) ||
           output_registry_has_type(outputs, OUTPUT_CSV_TIMESERIES) ||
           output_registry_has_type(outputs, OUTPUT_CSV_CENTERLINE) ||
           output_registry_has_type(outputs, OUTPUT_CSV_STATISTICS);
}

// Check if any output type that uses statistics is registered
static int needs_statistics(const output_registry* outputs) {
    return output_registry_has_type(outputs, OUTPUT_CSV_TIMESERIES) ||
           output_registry_has_type(outputs, OUTPUT_CSV_STATISTICS);
}

// Automatically write all registered outputs for current step
void simulation_write_outputs(simulation_data* sim_data, int step) {
    if (!sim_data || !sim_data->outputs) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for simulation_write_outputs");
        return;
    }

    // Get run directory (creates it if needed)
    const char* run_dir =
        output_registry_get_run_dir(sim_data->outputs, sim_data->output_base_dir,
                                    sim_data->run_prefix, sim_data->grid->nx, sim_data->grid->ny);

    // Compute derived fields only when needed
    derived_fields* derived = NULL;
    int compute_vel_mag = needs_velocity_magnitude(sim_data->outputs);
    int compute_stats = needs_statistics(sim_data->outputs);

    if (compute_vel_mag || compute_stats) {
        derived = derived_fields_create(sim_data->grid->nx, sim_data->grid->ny,
                                            sim_data->grid->nz);
        if (derived) {
            // Compute velocity magnitude if needed (CSV outputs)
            if (compute_vel_mag) {
                derived_fields_compute_velocity_magnitude(derived, sim_data->field);
            }
            // Compute statistics if needed (CSV timeseries or statistics outputs)
            if (compute_stats) {
                derived_fields_compute_statistics(derived, sim_data->field);
            }
        }
    }

    // Write all registered outputs with pre-computed derived fields
    output_registry_write_outputs(sim_data->outputs, run_dir, step, sim_data->current_time,
                                  sim_data->field, derived, sim_data->grid, &sim_data->params,
                                  &sim_data->last_stats);

    // Clean up derived fields
    if (derived) {
        derived_fields_destroy(derived);
    }
}
