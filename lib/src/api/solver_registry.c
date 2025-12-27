#include "cfd/core/cfd_status.h"
#include "cfd/core/cpu_features.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Forward declarations for internal solver implementations
// These are not part of the public API
cfd_status_t explicit_euler_impl(flow_field* field, const grid* grid, const ns_solver_params_t* params);
void explicit_euler_optimized_impl(flow_field* field, const grid* grid,
                                   const ns_solver_params_t* params);
#ifdef CFD_ENABLE_OPENMP
cfd_status_t explicit_euler_omp_impl(flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params);
cfd_status_t solve_projection_method_omp(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params);
#endif

// GPU solver functions
cfd_status_t solve_projection_method_gpu(flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params,
                                         const gpu_config_t* config);
int gpu_is_available(void);

// SIMD solver functions

cfd_status_t explicit_euler_simd_init(struct NSSolver* solver, const grid* grid,
                                      const ns_solver_params_t* params);
void explicit_euler_simd_destroy(struct NSSolver* solver);

cfd_status_t explicit_euler_simd_step(struct NSSolver* solver, flow_field* field, const grid* grid,
                                      const ns_solver_params_t* params, ns_solver_stats_t* stats);


cfd_status_t projection_simd_init(ns_solver_t* solver, const grid* grid, const ns_solver_params_t* params);
void projection_simd_destroy(ns_solver_t* solver);

cfd_status_t projection_simd_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params, ns_solver_stats_t* stats);

#ifdef _WIN32
#else
#include <sys/time.h>
#endif

// Maximum number of registered solver types
#define MAX_REGISTERED_SOLVERS 32

// Registry entry
typedef struct {
    char name[64];
    ns_solver_factory_func factory;
    const char* description;
    ns_solver_backend_t backend;  // Backend type for efficient filtering
} solver_registry_entry;

// ns_solver_registry_t structure
struct NSSolverRegistry {
    solver_registry_entry entries[MAX_REGISTERED_SOLVERS];
    int count;
};

// Forward declarations for built-in solver factories
static ns_solver_t* create_explicit_euler_solver(void);
static ns_solver_t* create_explicit_euler_optimized_solver(void);
static ns_solver_t* create_projection_solver(void);
static ns_solver_t* create_projection_optimized_solver(void);
#ifdef CFD_HAS_CUDA
static ns_solver_t* create_explicit_euler_gpu_solver(void);
static ns_solver_t* create_projection_gpu_solver(void);
#endif
#ifdef CFD_ENABLE_OPENMP
static ns_solver_t* create_explicit_euler_omp_solver(void);
static ns_solver_t* create_projection_omp_solver(void);
#endif

// External projection method solver functions
extern cfd_status_t solve_projection_method(flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params);
extern void solve_projection_method_optimized(flow_field* field, const grid* grid,
                                              const ns_solver_params_t* params);

// External GPU solver functions (from solver_gpu.cu or solver_gpu_stub.c)
#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"


// Helper to get current time in milliseconds
static double get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

/**
 * solver Registry Implementation
 */

ns_solver_registry_t* cfd_registry_create(void) {
    ns_solver_registry_t* registry = (ns_solver_registry_t*)cfd_calloc(1, sizeof(ns_solver_registry_t));
    return registry;
}

void cfd_registry_destroy(ns_solver_registry_t* registry) {
    if (registry) {
        cfd_free(registry);
    }
}

void cfd_registry_register_defaults(ns_solver_registry_t* registry) {
    if (!registry) {
        return;
    }

    // Register built-in solvers
    cfd_registry_register(registry, NS_SOLVER_TYPE_EXPLICIT_EULER, create_explicit_euler_solver);
    cfd_registry_register(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
                          create_explicit_euler_optimized_solver);

    // Register projection method solvers
    cfd_registry_register(registry, NS_SOLVER_TYPE_PROJECTION, create_projection_solver);
    cfd_registry_register(registry, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
                          create_projection_optimized_solver);

    // Register GPU solvers (requires CUDA)
#ifdef CFD_HAS_CUDA
    cfd_registry_register(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,
                          create_explicit_euler_gpu_solver);
    cfd_registry_register(registry, NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
                          create_projection_gpu_solver);
#endif

    // Register OpenMP solvers
#ifdef CFD_ENABLE_OPENMP
    cfd_registry_register(registry, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
                          create_explicit_euler_omp_solver);
    cfd_registry_register(registry, NS_SOLVER_TYPE_PROJECTION_OMP, create_projection_omp_solver);
#endif
}

/**
 * Infer backend from solver type name based on naming convention.
 * Returns the likely backend, or NS_SOLVER_BACKEND_SCALAR as default.
 * Note: Defined here (before first use) and also used by cfd_solver_create_checked.
 */
static ns_solver_backend_t infer_backend_from_type(const char* type_name) {
    if (!type_name) {
        return NS_SOLVER_BACKEND_SCALAR;
    }

    /* Check for GPU suffix first (most specific) */
    if (strstr(type_name, "_gpu") != NULL) {
        return NS_SOLVER_BACKEND_CUDA;
    }

    /* Check for OMP suffix */
    if (strstr(type_name, "_omp") != NULL) {
        return NS_SOLVER_BACKEND_OMP;
    }

    /* Check for optimized suffix (SIMD) */
    if (strstr(type_name, "_optimized") != NULL) {
        return NS_SOLVER_BACKEND_SIMD;
    }

    /* Default to scalar */
    return NS_SOLVER_BACKEND_SCALAR;
}

int cfd_registry_register(ns_solver_registry_t* registry, const char* type_name,
                          ns_solver_factory_func factory) {
    if (!registry || !type_name || !factory) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for solver registration");
        return -1;
    }
    if (strlen(type_name) == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "solver type name cannot be empty");
        return -1;
    }
    if (registry->count >= MAX_REGISTERED_SOLVERS) {
        cfd_set_error(CFD_ERROR_LIMIT_EXCEEDED, "Max registered solvers limit reached");
        return -1;
    }

    /* Infer backend from type name for efficient filtering later */
    ns_solver_backend_t backend = infer_backend_from_type(type_name);

    // Check if already registered
    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            // Update existing entry
            registry->entries[i].factory = factory;
            registry->entries[i].backend = backend;
            return 0;
        }
    }

    // Add new entry
    snprintf(registry->entries[registry->count].name,
             sizeof(registry->entries[registry->count].name), "%s", type_name);
    registry->entries[registry->count].factory = factory;
    registry->entries[registry->count].backend = backend;
    registry->count++;

    return 0;
}

int cfd_registry_unregister(ns_solver_registry_t* registry, const char* type_name) {
    if (!registry || !type_name) {
        return -1;
    }

    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            // Shift remaining entries
            for (int j = i; j < registry->count - 1; j++) {
                registry->entries[j] = registry->entries[j + 1];
            }
            registry->count--;
            return 0;
        }
    }
    return -1;
}

int cfd_registry_list(ns_solver_registry_t* registry, const char** names, int max_count) {
    if (!registry) {
        return 0;
    }

    int count = (registry->count < max_count) ? registry->count : max_count;
    if (names) {
        for (int i = 0; i < count; i++) {
            names[i] = registry->entries[i].name;
        }
    }
    return registry->count;
}

int cfd_registry_has(ns_solver_registry_t* registry, const char* type_name) {
    if (!registry || !type_name) {
        return 0;
    }

    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            return 1;
        }
    }
    return 0;
}

const char* cfd_registry_get_description(ns_solver_registry_t* registry, const char* type_name) {
    if (!registry || !type_name) {
        return NULL;
    }

    // Create a temporary solver to get its description
    ns_solver_t* solver = cfd_solver_create(registry, type_name);
    if (solver) {
        const char* desc = solver->description;
        solver_destroy(solver);
        return desc;
    }
    return NULL;
}

/**
 * solver Creation and Management
 */

ns_solver_t* cfd_solver_create(ns_solver_registry_t* registry, const char* type_name) {
    if (!registry || !type_name) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for solver creation");
        return NULL;
    }

    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            ns_solver_t* solver = registry->entries[i].factory();
            if (!solver) {
                /* Factory returned NULL - check if error was already set by the factory.
                 * If not (e.g., out of memory), set a generic error.
                 * GPU factories set CFD_ERROR_UNSUPPORTED when GPU is not available. */
                if (cfd_get_last_status() == CFD_SUCCESS) {
                    cfd_set_error(CFD_ERROR, "Failed to create solver");
                }
            }
            return solver;
        }
    }

    /* Solver type not found in registry */
    cfd_set_error(CFD_ERROR_INVALID, "Solver type not registered");
    return NULL;
}

void solver_destroy(ns_solver_t* solver) {
    if (!solver) {
        return;
    }

    if (solver->destroy) {
        solver->destroy(solver);
    }
    cfd_free(solver);
}


cfd_status_t solver_init(ns_solver_t* solver, const grid* grid, const ns_solver_params_t* params) {
    if (!solver) {
        return CFD_ERROR_INVALID;
    }
    if (!solver->init) {
        return CFD_SUCCESS;  // Optional
    }

    return solver->init(solver, grid, params);
}


cfd_status_t solver_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                         const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }
    if (!solver->step) {
        return CFD_ERROR;
    }

    double start_time = get_time_ms();

    cfd_status_t status = solver->step(solver, field, grid, params, stats);
    double end_time = get_time_ms();

    if (stats) {
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = status;
    }

    return status;
}


cfd_status_t solver_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                          const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }
    if (!solver->solve) {
        return CFD_ERROR;
    }

    double start_time = get_time_ms();

    cfd_status_t status = solver->solve(solver, field, grid, params, stats);
    double end_time = get_time_ms();

    if (stats) {
        stats->elapsed_time_ms = end_time - start_time;
        stats->status = status;
    }

    return status;
}

void solver_apply_boundary(ns_solver_t* solver, flow_field* field, const grid* grid) {
    if (!solver || !field || !grid) {
        return;
    }

    if (solver->apply_boundary) {
        solver->apply_boundary(solver, field, grid);
    } else {
        // Fall back to default boundary conditions
        apply_boundary_conditions(field, grid);
    }
}

double solver_compute_dt(ns_solver_t* solver, const flow_field* field, const grid* grid,
                         const ns_solver_params_t* params) {
    if (!solver || !field || !grid || !params) {
        return 0.0;
    }

    if (solver->compute_dt) {
        return solver->compute_dt(solver, field, grid, params);
    }

    // Default implementation
    double max_vel = 0.0;
    double min_dx = grid->dx[0];
    double min_dy = grid->dy[0];

    for (size_t i = 0; i < grid->nx - 1; i++) {
        if (grid->dx[i] < min_dx) {
            min_dx = grid->dx[i];
        }
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        if (grid->dy[j] < min_dy) {
            min_dy = grid->dy[j];
        }
    }

    for (size_t i = 0; i < field->nx * field->ny; i++) {
        double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
        if (vel > max_vel) {
            max_vel = vel;
        }
    }

    if (max_vel < 1e-10) {
        max_vel = 1.0;
    }

    double dt = params->cfl * fmin(min_dx, min_dy) / max_vel;
    return fmin(fmax(dt, 1e-6), 0.01);
}

/**
 * Built-in solver: Explicit Euler
 * Wraps the existing solve_navier_stokes function
 */

typedef struct {
    int initialized;
} explicit_euler_context;

static cfd_status_t explicit_euler_init(ns_solver_t* solver, const grid* grid,
                                        const ns_solver_params_t* params) {
    (void)grid;
    (void)params;

    explicit_euler_context* ctx =
        (explicit_euler_context*)cfd_malloc(sizeof(explicit_euler_context));
    if (!ctx) {
        return CFD_ERROR;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void explicit_euler_destroy(ns_solver_t* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t explicit_euler_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                        const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    // Create params with single iteration
    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    explicit_euler_impl(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;

        // Compute max velocity
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static cfd_status_t explicit_euler_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    explicit_euler_impl(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;

        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static ns_solver_t* create_explicit_euler_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_EXPLICIT_EULER;
    s->description = "Basic explicit Euler finite difference solver for 2D Navier-Stokes";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT;
    s->backend = NS_SOLVER_BACKEND_SCALAR;

    s->init = explicit_euler_init;
    s->destroy = explicit_euler_destroy;
    s->step = explicit_euler_step;
    s->solve = explicit_euler_solve;
    s->apply_boundary = NULL;  // Use default
    s->compute_dt = NULL;      // Use default

    return s;
}


static cfd_status_t explicit_euler_simd_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                              const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }

    for (int i = 0; i < params->max_iter; i++) {
        cfd_status_t status = explicit_euler_simd_step(solver, field, grid, params, NULL);
        if (status != CFD_SUCCESS) {
            return status;
        }
    }

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static ns_solver_t* create_explicit_euler_optimized_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED;
    s->description = "SIMD-optimized explicit Euler solver (AVX2)";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_SIMD;
    s->backend = NS_SOLVER_BACKEND_SIMD;

    s->init = explicit_euler_simd_init;
    s->destroy = explicit_euler_simd_destroy;
    s->step = explicit_euler_simd_step;
    s->solve = explicit_euler_simd_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

/**
 * Built-in solver: Projection Method (Chorin's Method)
 */

typedef struct {
    int initialized;
} projection_context;

static cfd_status_t projection_init(ns_solver_t* solver, const grid* grid, const ns_solver_params_t* params) {
    (void)grid;
    (void)params;
    projection_context* ctx = (projection_context*)cfd_malloc(sizeof(projection_context));
    if (!ctx) {
        return CFD_ERROR;
    }
    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void projection_destroy(ns_solver_t* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t projection_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                    const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    solve_projection_method(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t projection_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                     const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solve_projection_method(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static ns_solver_t* create_projection_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_PROJECTION;
    s->description = "Projection method (Chorin's method)";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT;
    s->backend = NS_SOLVER_BACKEND_SCALAR;

    s->init = projection_init;
    s->destroy = projection_destroy;
    s->step = projection_step;
    s->solve = projection_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

static cfd_status_t projection_simd_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                          const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    if (!solver || !field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }

    // Use the step function which utilizes the persistent context
    for (int i = 0; i < params->max_iter; i++) {
        cfd_status_t status = projection_simd_step(solver, field, grid, params,
                                                   NULL);  // Pass NULL stats for individual steps
        if (status != CFD_SUCCESS) {
            return status;
        }
    }

    if (stats) {
        stats->iterations = params->max_iter;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static ns_solver_t* create_projection_optimized_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_PROJECTION_OPTIMIZED;
    s->description = "SIMD-optimized Projection solver (AVX2)";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_SIMD;
    s->backend = NS_SOLVER_BACKEND_SIMD;

    s->init = projection_simd_init;
    s->destroy = projection_simd_destroy;
    s->step = projection_simd_step;
    s->solve = projection_simd_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

#ifdef CFD_HAS_CUDA
/**
 * Built-in solver: GPU-Accelerated Explicit Euler
 * Uses CUDA for GPU acceleration (requires CUDA support)
 */

typedef struct {
    gpu_solver_context_t* gpu_ctx;
    gpu_config_t gpu_config_t;
    int use_gpu;
} gpu_solver_wrapper_context;

static cfd_status_t gpu_euler_init(ns_solver_t* solver, const grid* grid, const ns_solver_params_t* params) {
    gpu_solver_wrapper_context* ctx =
        (gpu_solver_wrapper_context*)cfd_malloc(sizeof(gpu_solver_wrapper_context));
    if (!ctx) {
        return CFD_ERROR;
    }

    ctx->gpu_config_t = gpu_config_default();
    ctx->use_gpu = gpu_should_use(&ctx->gpu_config_t, grid->nx, grid->ny, params->max_iter);
    ctx->gpu_ctx = NULL;

    if (ctx->use_gpu) {
        ctx->gpu_ctx = gpu_solver_create(grid->nx, grid->ny, &ctx->gpu_config_t);
        if (!ctx->gpu_ctx) {
            cfd_free(ctx);
            fprintf(stderr, "GPU Euler init: Failed to create GPU context\n");
            return CFD_ERROR_UNSUPPORTED;
        }
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void gpu_euler_destroy(ns_solver_t* solver) {
    if (solver->context) {
        gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;
        if (ctx->gpu_ctx) {
            gpu_solver_destroy(ctx->gpu_ctx);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t gpu_euler_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                   const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Upload, step, download
        if (gpu_solver_upload(ctx->gpu_ctx, field) == 0) {
            gpu_solver_stats_t gpu_stats;
            if (gpu_solver_step(ctx->gpu_ctx, grid, &step_params, &gpu_stats) == 0) {
                gpu_solver_download(ctx->gpu_ctx, field);

                if (stats) {
                    stats->iterations = 1;
                    stats->elapsed_time_ms = gpu_stats.kernel_time_ms;
                    // Compute max velocity/pressure
                    double max_vel = 0.0, max_p = 0.0;
                    for (size_t i = 0; i < field->nx * field->ny; i++) {
                        double vel =
                            sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
                        if (vel > max_vel) {
                            max_vel = vel;
                        }
                        if (fabs(field->p[i]) > max_p) {
                            max_p = fabs(field->p[i]);
                        }
                    }
                    stats->max_velocity = max_vel;
                    stats->max_pressure = max_p;
                }
                return CFD_SUCCESS;
            }
        }
        // GPU operation failed
        fprintf(stderr, "GPU Euler step: GPU operation failed\n");
        return CFD_ERROR_INVALID;
    }

    // GPU not available - could be: CUDA not compiled in, gpu_should_use() returned false,
    // or GPU initialization failed
    fprintf(stderr, "GPU Euler step: GPU solver not initialized\n");
    return CFD_ERROR_UNSUPPORTED;
}

static cfd_status_t gpu_euler_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                    const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Use full GPU solver
        solve_navier_stokes_gpu(field, grid, params, &ctx->gpu_config_t);

        if (stats) {
            stats->iterations = params->max_iter;
            gpu_solver_stats_t gpu_stats = gpu_solver_get_stats(ctx->gpu_ctx);
            stats->elapsed_time_ms = gpu_stats.kernel_time_ms;

            double max_vel = 0.0, max_p = 0.0;
            for (size_t i = 0; i < field->nx * field->ny; i++) {
                double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
                if (vel > max_vel) {
                    max_vel = vel;
                }
                if (fabs(field->p[i]) > max_p) {
                    max_p = fabs(field->p[i]);
                }
            }
            stats->max_velocity = max_vel;
            stats->max_pressure = max_p;
        }
        return CFD_SUCCESS;
    }

    // GPU not available - could be: CUDA not compiled in, gpu_should_use() returned false,
    // or GPU initialization failed
    fprintf(stderr, "GPU Euler solve: GPU solver not initialized\n");
    return CFD_ERROR_UNSUPPORTED;
}

static ns_solver_t* create_explicit_euler_gpu_solver(void) {
    /* Check if GPU is available at runtime before creating the solver */
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return NULL;
    }

    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_EXPLICIT_EULER_GPU;
    s->description = "GPU-accelerated explicit Euler solver (CUDA)";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_GPU;
    s->backend = NS_SOLVER_BACKEND_CUDA;

    s->init = gpu_euler_init;
    s->destroy = gpu_euler_destroy;
    s->step = gpu_euler_step;
    s->solve = gpu_euler_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

/**
 * Built-in solver: GPU-Accelerated Projection Method
 */

static cfd_status_t gpu_projection_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                        const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    (void)stats;
    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;
    /* Override thresholds to allow single-step GPU execution on small grids */
    gpu_config_t cfg = gpu_config_default();
    cfg.min_grid_size = 1;
    cfg.min_steps = 1;
    return solve_projection_method_gpu(field, grid, &step_params, &cfg);
}

static cfd_status_t gpu_projection_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    (void)stats;
    /* Override thresholds to allow GPU execution on small grids */
    gpu_config_t cfg = gpu_config_default();
    cfg.min_grid_size = 1;
    cfg.min_steps = 1;
    return solve_projection_method_gpu(field, grid, params, &cfg);
}

static ns_solver_t* create_projection_gpu_solver(void) {
    /* Check if GPU is available at runtime before creating the solver */
    if (!gpu_is_available()) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED, "CUDA GPU not available at runtime");
        return NULL;
    }

    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU;
    s->description = "GPU-accelerated projection method with Jacobi iteration (CUDA)";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_GPU;
    s->backend = NS_SOLVER_BACKEND_CUDA;

    s->init = NULL;
    s->destroy = NULL;
    s->step = gpu_projection_step;
    s->solve = gpu_projection_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}
#endif /* CFD_HAS_CUDA */

#ifdef CFD_ENABLE_OPENMP
/**
 * Built-in solver: Explicit Euler OpenMP
 */

static cfd_status_t explicit_euler_omp_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                            const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    explicit_euler_omp_impl(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t explicit_euler_omp_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    explicit_euler_omp_impl(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static ns_solver_t* create_explicit_euler_omp_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_EXPLICIT_EULER_OMP;
    s->description = "OpenMP-parallelized explicit Euler solver";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_PARALLEL;
    s->backend = NS_SOLVER_BACKEND_OMP;

    s->init = explicit_euler_init;        // Can reuse existing init
    s->destroy = explicit_euler_destroy;  // Can reuse existing destroy
    s->step = explicit_euler_omp_step;
    s->solve = explicit_euler_omp_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

/**
 * Built-in NSSolver: Projection OpenMP
 */

static cfd_status_t projection_omp_step(ns_solver_t* solver, flow_field* field, const grid* grid,
                                        const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    ns_solver_params_t step_params = *params;
    step_params.max_iter = 1;

    solve_projection_method_omp(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t projection_omp_solve(ns_solver_t* solver, flow_field* field, const grid* grid,
                                         const ns_solver_params_t* params, ns_solver_stats_t* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solve_projection_method_omp(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
            if (vel > max_vel) {
                max_vel = vel;
            }
            if (fabs(field->p[i]) > max_p) {
                max_p = fabs(field->p[i]);
            }
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static ns_solver_t* create_projection_omp_solver(void) {
    ns_solver_t* s = (ns_solver_t*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = NS_SOLVER_TYPE_PROJECTION_OMP;
    s->description = "OpenMP-parallelized Projection solver";
    s->version = "1.0.0";
    s->capabilities = NS_SOLVER_CAP_INCOMPRESSIBLE | NS_SOLVER_CAP_TRANSIENT | NS_SOLVER_CAP_PARALLEL;

    s->init = projection_init;        // Can reuse existing init
    s->destroy = projection_destroy;  // Can reuse existing destroy
    s->step = projection_omp_step;
    s->solve = projection_omp_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;
    s->backend = NS_SOLVER_BACKEND_OMP;

    return s;
}
#endif

//=============================================================================
// Backend Availability API
//=============================================================================

int cfd_backend_is_available(ns_solver_backend_t backend) {
    switch (backend) {
        case NS_SOLVER_BACKEND_SCALAR:
            return 1;  // Always available

        case NS_SOLVER_BACKEND_SIMD:
            return cfd_has_simd();

        case NS_SOLVER_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            return 1;
#else
            return 0;
#endif

        case NS_SOLVER_BACKEND_CUDA:
            return gpu_is_available();

        default:
            return 0;
    }
}

const char* cfd_backend_get_name(ns_solver_backend_t backend) {
    switch (backend) {
        case NS_SOLVER_BACKEND_SCALAR:
            return "scalar";
        case NS_SOLVER_BACKEND_SIMD:
            return "simd";
        case NS_SOLVER_BACKEND_OMP:
            return "openmp";
        case NS_SOLVER_BACKEND_CUDA:
            return "cuda";
        default:
            return "unknown";
    }
}

int cfd_registry_list_by_backend(ns_solver_registry_t* registry, ns_solver_backend_t backend,
                                  const char** names, int max_count) {
    if (!registry) {
        return 0;
    }

    int count = 0;

    for (int i = 0; i < registry->count && count < max_count; i++) {
        /* Use the stored backend instead of creating temporary solvers.
         * This is much more efficient and avoids side effects from factory calls. */
        if (registry->entries[i].backend == backend) {
            if (names) {
                names[count] = registry->entries[i].name;
            }
            count++;
        }
    }

    return count;
}

ns_solver_t* cfd_solver_create_checked(ns_solver_registry_t* registry, const char* type_name) {
    if (!registry || !type_name) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for solver creation");
        return NULL;
    }

    /* Check backend availability BEFORE creating the solver */
    ns_solver_backend_t expected_backend = infer_backend_from_type(type_name);
    if (!cfd_backend_is_available(expected_backend)) {
        const char* backend_name = cfd_backend_get_name(expected_backend);
        char error_msg[128];
        snprintf(error_msg, sizeof(error_msg),
                 "Backend '%s' is not available on this system", backend_name);
        cfd_set_error(CFD_ERROR_UNSUPPORTED, error_msg);
        return NULL;
    }

    /* Now create the solver - backend is available */
    ns_solver_t* solver = cfd_solver_create(registry, type_name);
    if (!solver) {
        /* Error already set by cfd_solver_create */
        return NULL;
    }

    return solver;
}
