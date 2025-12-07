#include "simulation_api.h"
#include "derived_fields.h"
#include "grid.h"
#include "output_registry.h"
#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// SimulationData struct is defined in simulation_api.h

// Default solver type used when none is specified
#define DEFAULT_SOLVER_TYPE SOLVER_TYPE_EXPLICIT_EULER

// Global base output directory
static char s_base_output_dir[512] = "../../artifacts";

// Static flag to track if registry is initialized
static int s_registry_initialized = 0;

// Ensure registry is initialized
static void ensure_registry_initialized(void) {
    if (!s_registry_initialized) {
        solver_registry_init();
        s_registry_initialized = 1;
    }
}

// Internal helper to create simulation with a specific solver
static SimulationData* create_simulation_with_solver(size_t nx, size_t ny, double xmin, double xmax,
                                                     double ymin, double ymax,
                                                     const char* solver_type) {
    ensure_registry_initialized();

    SimulationData* sim_data = (SimulationData*)cfd_malloc(sizeof(SimulationData));
    if (!sim_data)
        return NULL;

    // Create and initialize grid
    sim_data->grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(sim_data->grid);

    // Create flow field
    sim_data->field = flow_field_create(nx, ny);
    initialize_flow_field(sim_data->field, sim_data->grid);

    // Initialize solver parameters with defaults
    sim_data->params = solver_params_default();
    sim_data->params.dt = 0.001;
    sim_data->params.cfl = 0.2;
    sim_data->params.mu = 0.01;
    sim_data->params.max_iter = 1;

    // Initialize stats
    sim_data->last_stats = solver_stats_default();

    // Create output registry
    sim_data->outputs = output_registry_create();
    sim_data->run_prefix = NULL;
    sim_data->current_time = 0.0;

    // Create and initialize the solver
    sim_data->solver = solver_create(solver_type);
    if (!sim_data->solver) {
        // Failed to create solver - cleanup and return NULL
        output_registry_destroy(sim_data->outputs);
        flow_field_destroy(sim_data->field);
        grid_destroy(sim_data->grid);
        cfd_free(sim_data);
        return NULL;
    }

    solver_init(sim_data->solver, sim_data->grid, &sim_data->params);

    return sim_data;
}

// Initialize simulation data with default solver
SimulationData* init_simulation(size_t nx, size_t ny, double xmin, double xmax, double ymin,
                                double ymax) {
    return create_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, DEFAULT_SOLVER_TYPE);
}

// Initialize simulation with a specific solver type
SimulationData* init_simulation_with_solver(size_t nx, size_t ny, double xmin, double xmax,
                                            double ymin, double ymax, const char* solver_type) {
    if (!solver_type) {
        solver_type = DEFAULT_SOLVER_TYPE;
    }
    return create_simulation_with_solver(nx, ny, xmin, xmax, ymin, ymax, solver_type);
}

// Set the solver for an existing simulation
void simulation_set_solver(SimulationData* sim_data, Solver* solver) {
    if (!sim_data || !solver)
        return;

    // Destroy existing solver
    if (sim_data->solver) {
        solver_destroy(sim_data->solver);
    }

    sim_data->solver = solver;
    solver_init(solver, sim_data->grid, &sim_data->params);
}

// Set the solver by type name
int simulation_set_solver_by_name(SimulationData* sim_data, const char* solver_type) {
    if (!sim_data || !solver_type)
        return -1;

    ensure_registry_initialized();

    Solver* solver = solver_create(solver_type);
    if (!solver)
        return -1;

    simulation_set_solver(sim_data, solver);
    return 0;
}

// Get the current solver
Solver* simulation_get_solver(SimulationData* sim_data) {
    return sim_data ? sim_data->solver : NULL;
}

// Get statistics from the last solve
const SolverStats* simulation_get_stats(const SimulationData* sim_data) {
    return sim_data ? &sim_data->last_stats : NULL;
}

// Run simulation step
void run_simulation_step(SimulationData* sim_data) {
    if (!sim_data || !sim_data->solver)
        return;

    // Use fixed time step for animation stability
    sim_data->params.dt = 0.005;

    solver_step(sim_data->solver, sim_data->field, sim_data->grid, &sim_data->params,
                &sim_data->last_stats);

    // Accumulate simulation time
    sim_data->current_time += sim_data->params.dt;
}

void run_simulation_solve(SimulationData* sim_data) {
    if (!sim_data || !sim_data->solver)
        return;

    // Use fixed time step for animation stability
    sim_data->params.dt = 0.005;

    solver_solve(sim_data->solver, sim_data->field, sim_data->grid, &sim_data->params,
                 &sim_data->last_stats);

    // Accumulate simulation time based on iterations performed
    sim_data->current_time += sim_data->params.dt * sim_data->last_stats.iterations;
}

// Free simulation data
void free_simulation(SimulationData* sim_data) {
    if (!sim_data)
        return;

    if (sim_data->solver) {
        solver_destroy(sim_data->solver);
        sim_data->solver = NULL;
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

// Solver discovery and listing
int simulation_list_solvers(const char** names, int max_count) {
    ensure_registry_initialized();
    return solver_registry_list(names, max_count);
}

int simulation_has_solver(const char* solver_type) {
    ensure_registry_initialized();
    return solver_registry_has(solver_type);
}

//=============================================================================
// OUTPUT REGISTRY API
//=============================================================================

// Register output for automatic generation
void simulation_register_output(SimulationData* sim_data, OutputFieldType field_type, int interval,
                                const char* prefix) {
    if (!sim_data || !sim_data->outputs)
        return;
    output_registry_add(sim_data->outputs, field_type, interval, prefix);
}

// Clear all registered outputs
void simulation_clear_outputs(SimulationData* sim_data) {
    if (!sim_data || !sim_data->outputs)
        return;
    output_registry_clear(sim_data->outputs);
}

// Set base output directory
void simulation_set_output_dir(const char* base_dir) {
    if (base_dir && strlen(base_dir) > 0) {
        strncpy(s_base_output_dir, base_dir, sizeof(s_base_output_dir) - 1);
        s_base_output_dir[sizeof(s_base_output_dir) - 1] = '\0';
    }
}

// Set run name prefix
void simulation_set_run_prefix(SimulationData* sim_data, const char* prefix) {
    if (!sim_data)
        return;

    // Free existing prefix
    if (sim_data->run_prefix) {
        cfd_free(sim_data->run_prefix);
        sim_data->run_prefix = NULL;
    }

    // Set new prefix
    if (prefix) {
        size_t len = strlen(prefix) + 1;
        sim_data->run_prefix = (char*)cfd_malloc(len);
        strncpy(sim_data->run_prefix, prefix, len);
    }
}

//=============================================================================
// AUTOMATIC OUTPUT GENERATION
//=============================================================================

// Automatically write all registered outputs for current step
void simulation_write_outputs(SimulationData* sim_data, int step) {
    if (!sim_data || !sim_data->outputs)
        return;

    // Get run directory (creates it if needed)
    const char* run_dir =
        output_registry_get_run_dir(sim_data->outputs, s_base_output_dir, sim_data->run_prefix,
                                    sim_data->grid->nx, sim_data->grid->ny);

    // Compute derived fields (e.g., velocity magnitude) before output
    // This is the solver level - computation happens here, not in output functions
    DerivedFields* derived = derived_fields_create(sim_data->grid->nx, sim_data->grid->ny);
    if (derived) {
        derived_fields_compute_velocity_magnitude(derived, sim_data->field);
    }

    // Write all registered outputs with pre-computed derived fields
    output_registry_write_outputs(sim_data->outputs, run_dir, step, sim_data->current_time,
                                  sim_data->field, derived, sim_data->grid, &sim_data->params,
                                  &sim_data->last_stats);

    // Clean up derived fields
    derived_fields_destroy(derived);
}
