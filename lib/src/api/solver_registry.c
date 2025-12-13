#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"


#define WIN32_LEAN_AND_MEAN
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>


// Forward declarations for internal solver implementations
// These are not part of the public API
cfd_status_t explicit_euler_impl(flow_field* field, const grid* grid, const solver_params* params);
void explicit_euler_optimized_impl(flow_field* field, const grid* grid,
                                   const solver_params* params);
#ifdef CFD_ENABLE_OPENMP
cfd_status_t explicit_euler_omp_impl(flow_field* field, const grid* grid,
                                     const solver_params* params);
cfd_status_t solve_projection_method_omp(flow_field* field, const grid* grid,
                                         const solver_params* params);
#endif

// SIMD solver functions

cfd_status_t explicit_euler_simd_init(struct Solver* solver, const grid* grid,
                                      const solver_params* params);
void explicit_euler_simd_destroy(struct Solver* solver);

cfd_status_t explicit_euler_simd_step(struct Solver* solver, flow_field* field, const grid* grid,
                                      const solver_params* params, solver_stats* stats);


cfd_status_t projection_simd_init(solver* solver, const grid* grid, const solver_params* params);
void projection_simd_destroy(solver* solver);

cfd_status_t projection_simd_step(solver* solver, flow_field* field, const grid* grid,
                                  const solver_params* params, solver_stats* stats);

#ifdef _WIN32
#else
#include <sys/time.h>
#endif

// Maximum number of registered solver types
#define MAX_REGISTERED_SOLVERS 32

// Registry entry
typedef struct {
    char name[64];
    solver_factory_func factory;
    const char* description;
} solver_registry_entry;

// solver_registry structure
struct SolverRegistry {
    solver_registry_entry entries[MAX_REGISTERED_SOLVERS];
    int count;
};

// Forward declarations for built-in solver factories
static solver* create_explicit_euler_solver(void);
static solver* create_explicit_euler_optimized_solver(void);
static solver* create_projection_solver(void);
static solver* create_projection_optimized_solver(void);
static solver* create_explicit_euler_gpu_solver(void);
static solver* create_projection_gpu_solver(void);
#ifdef CFD_ENABLE_OPENMP
static solver* create_explicit_euler_omp_solver(void);
static solver* create_projection_omp_solver(void);
#endif

// External projection method solver functions
extern cfd_status_t solve_projection_method(flow_field* field, const grid* grid,
                                            const solver_params* params);
extern void solve_projection_method_optimized(flow_field* field, const grid* grid,
                                              const solver_params* params);

// External GPU solver functions (from solver_gpu.cu or solver_gpu_stub.c)
#include "cfd/core/cfd_status.h"
#include "cfd/solvers/solver_gpu.h"


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

solver_registry* cfd_registry_create(void) {
    solver_registry* registry = (solver_registry*)cfd_calloc(1, sizeof(solver_registry));
    return registry;
}

void cfd_registry_destroy(solver_registry* registry) {
    if (registry) {
        cfd_free(registry);
    }
}

void cfd_registry_register_defaults(solver_registry* registry) {
    if (!registry) {
        return;
    }

    // Register built-in solvers
    cfd_registry_register(registry, SOLVER_TYPE_EXPLICIT_EULER, create_explicit_euler_solver);
    cfd_registry_register(registry, SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
                          create_explicit_euler_optimized_solver);

    // Register projection method solvers
    cfd_registry_register(registry, SOLVER_TYPE_PROJECTION, create_projection_solver);
    cfd_registry_register(registry, SOLVER_TYPE_PROJECTION_OPTIMIZED,
                          create_projection_optimized_solver);

    // Register GPU solvers (will use CPU fallback if CUDA not available)
    cfd_registry_register(registry, SOLVER_TYPE_EXPLICIT_EULER_GPU,
                          create_explicit_euler_gpu_solver);
    cfd_registry_register(registry, SOLVER_TYPE_PROJECTION_JACOBI_GPU,
                          create_projection_gpu_solver);

    // Register OpenMP solvers
#ifdef CFD_ENABLE_OPENMP
    cfd_registry_register(registry, SOLVER_TYPE_EXPLICIT_EULER_OMP,
                          create_explicit_euler_omp_solver);
    cfd_registry_register(registry, SOLVER_TYPE_PROJECTION_OMP, create_projection_omp_solver);
#endif
}

int cfd_registry_register(solver_registry* registry, const char* type_name,
                          solver_factory_func factory) {
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

    // Check if already registered
    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            // Update existing entry
            registry->entries[i].factory = factory;
            return 0;
        }
    }

    // Add new entry
    snprintf(registry->entries[registry->count].name,
             sizeof(registry->entries[registry->count].name), "%s", type_name);
    registry->entries[registry->count].factory = factory;
    registry->count++;

    return 0;
}

int cfd_registry_unregister(solver_registry* registry, const char* type_name) {
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

int cfd_registry_list(solver_registry* registry, const char** names, int max_count) {
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

int cfd_registry_has(solver_registry* registry, const char* type_name) {
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

const char* cfd_registry_get_description(solver_registry* registry, const char* type_name) {
    if (!registry || !type_name) {
        return NULL;
    }

    // Create a temporary solver to get its description
    solver* solver = cfd_solver_create(registry, type_name);
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

solver* cfd_solver_create(solver_registry* registry, const char* type_name) {
    if (!registry || !type_name) {
        cfd_set_error(CFD_ERROR_INVALID, "Invalid arguments for solver creation");
        return NULL;
    }

    for (int i = 0; i < registry->count; i++) {
        if (strcmp(registry->entries[i].name, type_name) == 0) {
            return registry->entries[i].factory();
        }
    }
    return NULL;
}

void solver_destroy(solver* solver) {
    if (!solver) {
        return;
    }

    if (solver->destroy) {
        solver->destroy(solver);
    }
    cfd_free(solver);
}


cfd_status_t solver_init(solver* solver, const grid* grid, const solver_params* params) {
    if (!solver) {
        return CFD_ERROR_INVALID;
    }
    if (!solver->init) {
        return CFD_SUCCESS;  // Optional
    }

    return solver->init(solver, grid, params);
}


cfd_status_t solver_step(solver* solver, flow_field* field, const grid* grid,
                         const solver_params* params, solver_stats* stats) {
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


cfd_status_t solver_solve(solver* solver, flow_field* field, const grid* grid,
                          const solver_params* params, solver_stats* stats) {
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

void solver_apply_boundary(solver* solver, flow_field* field, const grid* grid) {
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

double solver_compute_dt(solver* solver, const flow_field* field, const grid* grid,
                         const solver_params* params) {
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

static cfd_status_t explicit_euler_init(solver* solver, const grid* grid,
                                        const solver_params* params) {
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

static void explicit_euler_destroy(solver* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t explicit_euler_step(solver* solver, flow_field* field, const grid* grid,
                                        const solver_params* params, solver_stats* stats) {
    (void)solver;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    // Create params with single iteration
    solver_params step_params = *params;
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

static cfd_status_t explicit_euler_solve(solver* solver, flow_field* field, const grid* grid,
                                         const solver_params* params, solver_stats* stats) {
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

static solver* create_explicit_euler_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_EXPLICIT_EULER;
    s->description = "Basic explicit Euler finite difference solver for 2D Navier-Stokes";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT;

    s->init = explicit_euler_init;
    s->destroy = explicit_euler_destroy;
    s->step = explicit_euler_step;
    s->solve = explicit_euler_solve;
    s->apply_boundary = NULL;  // Use default
    s->compute_dt = NULL;      // Use default

    return s;
}


static cfd_status_t explicit_euler_simd_solve(solver* solver, flow_field* field, const grid* grid,
                                              const solver_params* params, solver_stats* stats) {
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

static solver* create_explicit_euler_optimized_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED;
    s->description = "SIMD-optimized explicit Euler solver (AVX2)";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_SIMD;

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

static cfd_status_t projection_init(solver* solver, const grid* grid, const solver_params* params) {
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

static void projection_destroy(solver* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t projection_step(solver* solver, flow_field* field, const grid* grid,
                                    const solver_params* params, solver_stats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solver_params step_params = *params;
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

static cfd_status_t projection_solve(solver* solver, flow_field* field, const grid* grid,
                                     const solver_params* params, solver_stats* stats) {
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

static solver* create_projection_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_PROJECTION;
    s->description = "Projection method (Chorin's method)";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT;

    s->init = projection_init;
    s->destroy = projection_destroy;
    s->step = projection_step;
    s->solve = projection_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

static cfd_status_t projection_simd_solve(solver* solver, flow_field* field, const grid* grid,
                                          const solver_params* params, solver_stats* stats) {
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

static solver* create_projection_optimized_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_PROJECTION_OPTIMIZED;
    s->description = "SIMD-optimized Projection solver (AVX2)";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_SIMD;

    s->init = projection_simd_init;
    s->destroy = projection_simd_destroy;
    s->step = projection_simd_step;
    s->solve = projection_simd_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

/**
 * Built-in solver: GPU-Accelerated Explicit Euler
 * Uses CUDA for GPU acceleration with automatic fallback
 */

typedef struct {
    gpu_solver_context* gpu_ctx;
    gpu_config gpu_config;
    int use_gpu;
} gpu_solver_wrapper_context;

static cfd_status_t gpu_euler_init(solver* solver, const grid* grid, const solver_params* params) {
    gpu_solver_wrapper_context* ctx =
        (gpu_solver_wrapper_context*)cfd_malloc(sizeof(gpu_solver_wrapper_context));
    if (!ctx) {
        return CFD_ERROR;
    }

    ctx->gpu_config = gpu_config_default();
    ctx->use_gpu = gpu_should_use(&ctx->gpu_config, grid->nx, grid->ny, params->max_iter);
    ctx->gpu_ctx = NULL;

    if (ctx->use_gpu) {
        ctx->gpu_ctx = gpu_solver_create(grid->nx, grid->ny, &ctx->gpu_config);
        if (!ctx->gpu_ctx) {
            ctx->use_gpu = 0;  // Fall back to CPU
        }
    }

    solver->context = ctx;
    return CFD_SUCCESS;
}

static void gpu_euler_destroy(solver* solver) {
    if (solver->context) {
        gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;
        if (ctx->gpu_ctx) {
            gpu_solver_destroy(ctx->gpu_ctx);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t gpu_euler_step(solver* solver, flow_field* field, const grid* grid,
                                   const solver_params* params, solver_stats* stats) {
    gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solver_params step_params = *params;
    step_params.max_iter = 1;

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Upload, step, download
        if (gpu_solver_upload(ctx->gpu_ctx, field) == 0) {
            gpu_solver_stats gpu_stats;
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
        // GPU failed, fall through to CPU
    }

    // CPU fallback
    explicit_euler_impl(field, grid, &step_params);

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

static cfd_status_t gpu_euler_solve(solver* solver, flow_field* field, const grid* grid,
                                    const solver_params* params, solver_stats* stats) {
    gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Use full GPU solver
        solve_navier_stokes_gpu(field, grid, params, &ctx->gpu_config);

        if (stats) {
            stats->iterations = params->max_iter;
            gpu_solver_stats gpu_stats = gpu_solver_get_stats(ctx->gpu_ctx);
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

    // CPU fallback
    explicit_euler_impl(field, grid, params);

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

static solver* create_explicit_euler_gpu_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_EXPLICIT_EULER_GPU;
    s->description = "GPU-accelerated explicit Euler solver (CUDA) with automatic fallback";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_GPU;

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

static cfd_status_t gpu_projection_step(solver* solver, flow_field* field, const grid* grid,
                                        const solver_params* params, solver_stats* stats) {
    // Dummy implementation
    solver_params step_params = *params;
    step_params.max_iter = 1;
    solve_projection_method(field, grid, &step_params);
    return CFD_SUCCESS;
    /*
        gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

        if (field->nx < 3 || field->ny < 3) {
            return CFD_ERROR_INVALID;
        }

        solver_params step_params = *params;
        step_params.max_iter = 1;

        if (ctx && ctx->use_gpu) {
            solve_projection_method_gpu(field, grid, &step_params, (const
       gpu_config*)&ctx->gpu_config); } else { solve_projection_method(field, grid, &step_params);
        }

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
    */
}

static cfd_status_t gpu_projection_solve(solver* solver, flow_field* field, const grid* grid,
                                         const solver_params* params, solver_stats* stats) {
    solve_projection_method(field, grid, params);
    return CFD_SUCCESS;
    /*
        gpu_solver_wrapper_context* ctx = (gpu_solver_wrapper_context*)solver->context;

        if (field->nx < 3 || field->ny < 3) {
            return CFD_ERROR_INVALID;
        }

        if (ctx && ctx->use_gpu) {
            solve_projection_method_gpu(field, grid, params, (const gpu_config*)&ctx->gpu_config);
        } else {
            solve_projection_method(field, grid, params);
        }

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
    */
}

static solver* create_projection_gpu_solver(void) {
    return NULL;
    /*
        solver* s = (solver*)cfd_calloc(1, sizeof(*s));
        if (!s) {
            return NULL;
        }

        s->name = SOLVER_TYPE_PROJECTION_JACOBI_GPU;
        s->description = "GPU-accelerated projection method with Jacobi iteration (CUDA)";
        s->version = "1.0.0";
        s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_GPU;

        s->init = gpu_euler_init;  // Same init handles GPU context
        s->destroy = gpu_euler_destroy;
        s->step = gpu_projection_step;
        s->solve = gpu_projection_solve;
        s->apply_boundary = NULL;
        s->compute_dt = NULL;

        return s;
    */
}

#ifdef CFD_ENABLE_OPENMP
/**
 * Built-in solver: Explicit Euler OpenMP
 */

static cfd_status_t explicit_euler_omp_step(solver* solver, flow_field* field, const grid* grid,
                                            const solver_params* params, solver_stats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solver_params step_params = *params;
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

static cfd_status_t explicit_euler_omp_solve(solver* solver, flow_field* field, const grid* grid,
                                             const solver_params* params, solver_stats* stats) {
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

static solver* create_explicit_euler_omp_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_EXPLICIT_EULER_OMP;
    s->description = "OpenMP-parallelized explicit Euler solver";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_PARALLEL;

    s->init = explicit_euler_init;        // Can reuse existing init
    s->destroy = explicit_euler_destroy;  // Can reuse existing destroy
    s->step = explicit_euler_omp_step;
    s->solve = explicit_euler_omp_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}

/**
 * Built-in Solver: Projection OpenMP
 */

static cfd_status_t projection_omp_step(solver* solver, flow_field* field, const grid* grid,
                                        const solver_params* params, solver_stats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    solver_params step_params = *params;
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

static cfd_status_t projection_omp_solve(solver* solver, flow_field* field, const grid* grid,
                                         const solver_params* params, solver_stats* stats) {
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

static solver* create_projection_omp_solver(void) {
    solver* s = (solver*)cfd_calloc(1, sizeof(*s));
    if (!s) {
        return NULL;
    }

    s->name = SOLVER_TYPE_PROJECTION_OMP;
    s->description = "OpenMP-parallelized Projection solver";
    s->version = "1.0.0";
    s->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_PARALLEL;

    s->init = projection_init;        // Can reuse existing init
    s->destroy = projection_destroy;  // Can reuse existing destroy
    s->step = projection_omp_step;
    s->solve = projection_omp_solve;
    s->apply_boundary = NULL;
    s->compute_dt = NULL;

    return s;
}
#endif
