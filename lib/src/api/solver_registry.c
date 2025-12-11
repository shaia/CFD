#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Forward declarations for internal solver implementations
// These are not part of the public API
cfd_status_t explicit_euler_impl(FlowField* field, const Grid* grid, const SolverParams* params);
void explicit_euler_optimized_impl(FlowField* field, const Grid* grid, const SolverParams* params);
#ifdef CFD_ENABLE_OPENMP
cfd_status_t explicit_euler_omp_impl(FlowField* field, const Grid* grid,
                                     const SolverParams* params);
cfd_status_t solve_projection_method_omp(FlowField* field, const Grid* grid,
                                         const SolverParams* params);
#endif

// SIMD Solver functions

cfd_status_t explicit_euler_simd_init(Solver* solver, const Grid* grid, const SolverParams* params);
void explicit_euler_simd_destroy(Solver* solver);

cfd_status_t explicit_euler_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                      const SolverParams* params, SolverStats* stats);


cfd_status_t projection_simd_init(Solver* solver, const Grid* grid, const SolverParams* params);
void projection_simd_destroy(Solver* solver);

cfd_status_t projection_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                  const SolverParams* params, SolverStats* stats);

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Maximum number of registered solver types
#define MAX_REGISTERED_SOLVERS 32

// Registry entry
typedef struct {
    char name[64];
    SolverFactoryFunc factory;
    const char* description;
} SolverRegistryEntry;

// Global solver registry
static SolverRegistryEntry g_solver_registry[MAX_REGISTERED_SOLVERS];
static int g_solver_registry_count = 0;
static int g_registry_initialized = 0;

// Forward declarations for built-in solver factories
static Solver* create_explicit_euler_solver(void);
static Solver* create_explicit_euler_optimized_solver(void);
static Solver* create_projection_solver(void);
static Solver* create_projection_optimized_solver(void);
static Solver* create_explicit_euler_gpu_solver(void);
static Solver* create_projection_gpu_solver(void);
#ifdef CFD_ENABLE_OPENMP
static Solver* create_explicit_euler_omp_solver(void);
static Solver* create_projection_omp_solver(void);
#endif

// External projection method solver functions
extern cfd_status_t solve_projection_method(FlowField* field, const Grid* grid,
                                            const SolverParams* params);
extern void solve_projection_method_optimized(FlowField* field, const Grid* grid,
                                              const SolverParams* params);

// External GPU solver functions (from solver_gpu.cu or solver_gpu_stub.c)
#include "cfd_status.h"
#include "solver_gpu.h"


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
 * Solver Registry Implementation
 */

void solver_registry_init(void) {
    if (g_registry_initialized)
        return;

    // Clear registry
    memset(g_solver_registry, 0, sizeof(g_solver_registry));
    g_solver_registry_count = 0;

    // Register built-in solvers
    solver_registry_register(SOLVER_TYPE_EXPLICIT_EULER, create_explicit_euler_solver);
    solver_registry_register(SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
                             create_explicit_euler_optimized_solver);

    // Register projection method solvers
    solver_registry_register(SOLVER_TYPE_PROJECTION, create_projection_solver);
    solver_registry_register(SOLVER_TYPE_PROJECTION_OPTIMIZED, create_projection_optimized_solver);

    // Register GPU solvers (will use CPU fallback if CUDA not available)
    solver_registry_register(SOLVER_TYPE_EXPLICIT_EULER_GPU, create_explicit_euler_gpu_solver);
    solver_registry_register(SOLVER_TYPE_PROJECTION_JACOBI_GPU, create_projection_gpu_solver);

    // Register OpenMP solvers
#ifdef CFD_ENABLE_OPENMP
    solver_registry_register(SOLVER_TYPE_EXPLICIT_EULER_OMP, create_explicit_euler_omp_solver);
    solver_registry_register(SOLVER_TYPE_PROJECTION_OMP, create_projection_omp_solver);
#endif

    g_registry_initialized = 1;
}

void solver_registry_cleanup(void) {
    g_solver_registry_count = 0;
    g_registry_initialized = 0;
}

int solver_registry_register(const char* type_name, SolverFactoryFunc factory) {
    if (!type_name || !factory)
        return -1;
    if (g_solver_registry_count >= MAX_REGISTERED_SOLVERS)
        return -1;

    // Check if already registered
    for (int i = 0; i < g_solver_registry_count; i++) {
        if (strcmp(g_solver_registry[i].name, type_name) == 0) {
            // Update existing entry
            g_solver_registry[i].factory = factory;
            return 0;
        }
    }

    // Add new entry
    strncpy(g_solver_registry[g_solver_registry_count].name, type_name, 63);
    g_solver_registry[g_solver_registry_count].name[63] = '\0';
    g_solver_registry[g_solver_registry_count].factory = factory;
    g_solver_registry_count++;

    return 0;
}

int solver_registry_unregister(const char* type_name) {
    if (!type_name)
        return -1;

    for (int i = 0; i < g_solver_registry_count; i++) {
        if (strcmp(g_solver_registry[i].name, type_name) == 0) {
            // Shift remaining entries
            for (int j = i; j < g_solver_registry_count - 1; j++) {
                g_solver_registry[j] = g_solver_registry[j + 1];
            }
            g_solver_registry_count--;
            return 0;
        }
    }
    return -1;
}

int solver_registry_list(const char** names, int max_count) {
    if (!g_registry_initialized) {
        solver_registry_init();
    }

    int count = (g_solver_registry_count < max_count) ? g_solver_registry_count : max_count;
    if (names) {
        for (int i = 0; i < count; i++) {
            names[i] = g_solver_registry[i].name;
        }
    }
    return g_solver_registry_count;
}

int solver_registry_has(const char* type_name) {
    if (!type_name)
        return 0;
    if (!g_registry_initialized) {
        solver_registry_init();
    }

    for (int i = 0; i < g_solver_registry_count; i++) {
        if (strcmp(g_solver_registry[i].name, type_name) == 0) {
            return 1;
        }
    }
    return 0;
}

const char* solver_registry_get_description(const char* type_name) {
    if (!type_name)
        return NULL;
    if (!g_registry_initialized) {
        solver_registry_init();
    }

    // Create a temporary solver to get its description
    Solver* solver = solver_create(type_name);
    if (solver) {
        const char* desc = solver->description;
        solver_destroy(solver);
        return desc;
    }
    return NULL;
}

/**
 * Solver Creation and Management
 */

Solver* solver_create(const char* type_name) {
    if (!type_name)
        return NULL;
    if (!g_registry_initialized) {
        solver_registry_init();
    }

    for (int i = 0; i < g_solver_registry_count; i++) {
        if (strcmp(g_solver_registry[i].name, type_name) == 0) {
            return g_solver_registry[i].factory();
        }
    }
    return NULL;
}

void solver_destroy(Solver* solver) {
    if (!solver)
        return;

    if (solver->destroy) {
        solver->destroy(solver);
    }
    cfd_free(solver);
}


cfd_status_t solver_init(Solver* solver, const Grid* grid, const SolverParams* params) {
    if (!solver)
        return CFD_ERROR_INVALID;
    if (!solver->init)
        return CFD_SUCCESS;  // Optional

    return solver->init(solver, grid, params);
}


cfd_status_t solver_step(Solver* solver, FlowField* field, const Grid* grid,
                         const SolverParams* params, SolverStats* stats) {
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


cfd_status_t solver_solve(Solver* solver, FlowField* field, const Grid* grid,
                          const SolverParams* params, SolverStats* stats) {
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

void solver_apply_boundary(Solver* solver, FlowField* field, const Grid* grid) {
    if (!solver || !field || !grid)
        return;

    if (solver->apply_boundary) {
        solver->apply_boundary(solver, field, grid);
    } else {
        // Fall back to default boundary conditions
        apply_boundary_conditions(field, grid);
    }
}

double solver_compute_dt(Solver* solver, const FlowField* field, const Grid* grid,
                         const SolverParams* params) {
    if (!solver || !field || !grid || !params)
        return 0.0;

    if (solver->compute_dt) {
        return solver->compute_dt(solver, field, grid, params);
    }

    // Default implementation
    double max_vel = 0.0;
    double min_dx = grid->dx[0];
    double min_dy = grid->dy[0];

    for (size_t i = 0; i < grid->nx - 1; i++) {
        if (grid->dx[i] < min_dx)
            min_dx = grid->dx[i];
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        if (grid->dy[j] < min_dy)
            min_dy = grid->dy[j];
    }

    for (size_t i = 0; i < field->nx * field->ny; i++) {
        double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
        if (vel > max_vel)
            max_vel = vel;
    }

    if (max_vel < 1e-10)
        max_vel = 1.0;

    double dt = params->cfl * fmin(min_dx, min_dy) / max_vel;
    return fmin(fmax(dt, 1e-6), 0.01);
}

/**
 * Built-in Solver: Explicit Euler
 * Wraps the existing solve_navier_stokes function
 */

typedef struct {
    int initialized;
} ExplicitEulerContext;

static cfd_status_t explicit_euler_init(Solver* solver, const Grid* grid,
                                        const SolverParams* params) {
    (void)grid;
    (void)params;

    ExplicitEulerContext* ctx = (ExplicitEulerContext*)cfd_malloc(sizeof(ExplicitEulerContext));
    if (!ctx)
        return CFD_ERROR;

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void explicit_euler_destroy(Solver* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t explicit_euler_step(Solver* solver, FlowField* field, const Grid* grid,
                                        const SolverParams* params, SolverStats* stats) {
    (void)solver;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    // Create params with single iteration
    SolverParams step_params = *params;
    step_params.max_iter = 1;

    explicit_euler_impl(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;

        // Compute max velocity
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static cfd_status_t explicit_euler_solve(Solver* solver, FlowField* field, const Grid* grid,
                                         const SolverParams* params, SolverStats* stats) {
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
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static Solver* create_explicit_euler_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_EXPLICIT_EULER;
    solver->description = "Basic explicit Euler finite difference solver for 2D Navier-Stokes";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT;

    solver->init = explicit_euler_init;
    solver->destroy = explicit_euler_destroy;
    solver->step = explicit_euler_step;
    solver->solve = explicit_euler_solve;
    solver->apply_boundary = NULL;  // Use default
    solver->compute_dt = NULL;      // Use default

    return solver;
}


static cfd_status_t explicit_euler_simd_solve(Solver* solver, FlowField* field, const Grid* grid,
                                              const SolverParams* params, SolverStats* stats) {
    if (!solver || !field || !grid || !params)
        return CFD_ERROR_INVALID;

    for (int i = 0; i < params->max_iter; i++) {
        cfd_status_t status = explicit_euler_simd_step(solver, field, grid, params, NULL);
        if (status != CFD_SUCCESS)
            return status;
    }

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static Solver* create_explicit_euler_optimized_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED;
    solver->description = "SIMD-optimized explicit Euler solver (AVX2)";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_SIMD;

    solver->init = explicit_euler_simd_init;
    solver->destroy = explicit_euler_simd_destroy;
    solver->step = explicit_euler_simd_step;
    solver->solve = explicit_euler_simd_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}

/**
 * Built-in Solver: Projection Method (Chorin's Method)
 */

typedef struct {
    int initialized;
} ProjectionContext;

static cfd_status_t projection_init(Solver* solver, const Grid* grid, const SolverParams* params) {
    (void)grid;
    (void)params;
    ProjectionContext* ctx = (ProjectionContext*)cfd_malloc(sizeof(ProjectionContext));
    if (!ctx)
        return CFD_ERROR;
    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

static void projection_destroy(Solver* solver) {
    if (solver->context) {
        cfd_free(solver->context);
        solver->context = NULL;
    }
}

static cfd_status_t projection_step(Solver* solver, FlowField* field, const Grid* grid,
                                    const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    SolverParams step_params = *params;
    step_params.max_iter = 1;

    solve_projection_method(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t projection_solve(Solver* solver, FlowField* field, const Grid* grid,
                                     const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    solve_projection_method(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static Solver* create_projection_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_PROJECTION;
    solver->description = "Projection method (Chorin's method)";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT;

    solver->init = projection_init;
    solver->destroy = projection_destroy;
    solver->step = projection_step;
    solver->solve = projection_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}

static cfd_status_t projection_simd_solve(Solver* solver, FlowField* field, const Grid* grid,
                                          const SolverParams* params, SolverStats* stats) {
    if (!solver || !field || !grid || !params)
        return CFD_ERROR_INVALID;

    // Use the step function which utilizes the persistent context
    for (int i = 0; i < params->max_iter; i++) {
        cfd_status_t status = projection_simd_step(solver, field, grid, params,
                                                   NULL);  // Pass NULL stats for individual steps
        if (status != CFD_SUCCESS)
            return status;
    }

    if (stats) {
        stats->iterations = params->max_iter;
        // Compute max velocity/pressure
        double max_vel = 0.0;
        double max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static Solver* create_projection_optimized_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_PROJECTION_OPTIMIZED;
    solver->description = "SIMD-optimized Projection solver (AVX2)";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_SIMD;

    solver->init = projection_simd_init;
    solver->destroy = projection_simd_destroy;
    solver->step = projection_simd_step;
    solver->solve = projection_simd_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}

/**
 * Built-in Solver: GPU-Accelerated Explicit Euler
 * Uses CUDA for GPU acceleration with automatic fallback
 */

typedef struct {
    GPUSolverContext* gpu_ctx;
    GPUConfig gpu_config;
    int use_gpu;
} GPUSolverWrapperContext;

static cfd_status_t gpu_euler_init(Solver* solver, const Grid* grid, const SolverParams* params) {
    GPUSolverWrapperContext* ctx =
        (GPUSolverWrapperContext*)cfd_malloc(sizeof(GPUSolverWrapperContext));
    if (!ctx)
        return CFD_ERROR;

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

static void gpu_euler_destroy(Solver* solver) {
    if (solver->context) {
        GPUSolverWrapperContext* ctx = (GPUSolverWrapperContext*)solver->context;
        if (ctx->gpu_ctx) {
            gpu_solver_destroy(ctx->gpu_ctx);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

static cfd_status_t gpu_euler_step(Solver* solver, FlowField* field, const Grid* grid,
                                   const SolverParams* params, SolverStats* stats) {
    GPUSolverWrapperContext* ctx = (GPUSolverWrapperContext*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    SolverParams step_params = *params;
    step_params.max_iter = 1;

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Upload, step, download
        if (gpu_solver_upload(ctx->gpu_ctx, field) == 0) {
            GPUSolverStats gpu_stats;
            if (gpu_solver_step(ctx->gpu_ctx, grid, &step_params, &gpu_stats) == 0) {
                gpu_solver_download(ctx->gpu_ctx, field);

                if (stats) {
                    stats->iterations = 1;
                    stats->elapsed_time_ms = gpu_stats.kernel_time_ms;
                    // Compute max velocity/pressure
                    double max_vel = 0.0, max_p = 0.0;
                    for (size_t i = 0; i < field->nx * field->ny; i++) {
                        double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
                        if (vel > max_vel)
                            max_vel = vel;
                        if (fabs(field->p[i]) > max_p)
                            max_p = fabs(field->p[i]);
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
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static cfd_status_t gpu_euler_solve(Solver* solver, FlowField* field, const Grid* grid,
                                    const SolverParams* params, SolverStats* stats) {
    GPUSolverWrapperContext* ctx = (GPUSolverWrapperContext*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (ctx && ctx->use_gpu && ctx->gpu_ctx) {
        // Use full GPU solver
        solve_navier_stokes_gpu(field, grid, params, &ctx->gpu_config);

        if (stats) {
            stats->iterations = params->max_iter;
            GPUSolverStats gpu_stats = gpu_solver_get_stats(ctx->gpu_ctx);
            stats->elapsed_time_ms = gpu_stats.kernel_time_ms;

            double max_vel = 0.0, max_p = 0.0;
            for (size_t i = 0; i < field->nx * field->ny; i++) {
                double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
                if (vel > max_vel)
                    max_vel = vel;
                if (fabs(field->p[i]) > max_p)
                    max_p = fabs(field->p[i]);
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
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static Solver* create_explicit_euler_gpu_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_EXPLICIT_EULER_GPU;
    solver->description = "GPU-accelerated explicit Euler solver (CUDA) with automatic fallback";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_GPU;

    solver->init = gpu_euler_init;
    solver->destroy = gpu_euler_destroy;
    solver->step = gpu_euler_step;
    solver->solve = gpu_euler_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}

/**
 * Built-in Solver: GPU-Accelerated Projection Method
 */

static cfd_status_t gpu_projection_step(Solver* solver, FlowField* field, const Grid* grid,
                                        const SolverParams* params, SolverStats* stats) {
    GPUSolverWrapperContext* ctx = (GPUSolverWrapperContext*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    SolverParams step_params = *params;
    step_params.max_iter = 1;

    if (ctx && ctx->use_gpu) {
        solve_projection_method_gpu(field, grid, &step_params, &ctx->gpu_config);
    } else {
        solve_projection_method(field, grid, &step_params);
    }

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static cfd_status_t gpu_projection_solve(Solver* solver, FlowField* field, const Grid* grid,
                                         const SolverParams* params, SolverStats* stats) {
    GPUSolverWrapperContext* ctx = (GPUSolverWrapperContext*)solver->context;

    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    if (ctx && ctx->use_gpu) {
        solve_projection_method_gpu(field, grid, params, &ctx->gpu_config);
    } else {
        solve_projection_method(field, grid, params);
    }

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }

    return CFD_SUCCESS;
}

static Solver* create_projection_gpu_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_PROJECTION_JACOBI_GPU;
    solver->description = "GPU-accelerated projection method with Jacobi iteration (CUDA)";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_GPU;

    solver->init = gpu_euler_init;  // Same init handles GPU context
    solver->destroy = gpu_euler_destroy;
    solver->step = gpu_projection_step;
    solver->solve = gpu_projection_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}

#ifdef CFD_ENABLE_OPENMP
/**
 * Built-in Solver: Explicit Euler OpenMP
 */

static cfd_status_t explicit_euler_omp_step(Solver* solver, FlowField* field, const Grid* grid,
                                            const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    SolverParams step_params = *params;
    step_params.max_iter = 1;

    explicit_euler_omp_impl(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t explicit_euler_omp_solve(Solver* solver, FlowField* field, const Grid* grid,
                                             const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    explicit_euler_omp_impl(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static Solver* create_explicit_euler_omp_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_EXPLICIT_EULER_OMP;
    solver->description = "OpenMP multi-threaded explicit Euler solver";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_PARALLEL;

    solver->init = explicit_euler_init;        // Reuse basic init
    solver->destroy = explicit_euler_destroy;  // Reuse basic destroy
    solver->step = explicit_euler_omp_step;
    solver->solve = explicit_euler_omp_solve;
    solver->apply_boundary = NULL;  // Use default
    solver->compute_dt = NULL;      // Use default

    return solver;
}

/**
 * Built-in Solver: Projection OpenMP
 */

static cfd_status_t projection_omp_step(Solver* solver, FlowField* field, const Grid* grid,
                                        const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    SolverParams step_params = *params;
    step_params.max_iter = 1;

    solve_projection_method_omp(field, grid, &step_params);

    if (stats) {
        stats->iterations = 1;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static cfd_status_t projection_omp_solve(Solver* solver, FlowField* field, const Grid* grid,
                                         const SolverParams* params, SolverStats* stats) {
    (void)solver;
    if (field->nx < 3 || field->ny < 3)
        return CFD_ERROR_INVALID;

    solve_projection_method_omp(field, grid, params);

    if (stats) {
        stats->iterations = params->max_iter;
        double max_vel = 0.0, max_p = 0.0;
        for (size_t i = 0; i < field->nx * field->ny; i++) {
            double vel = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
            if (vel > max_vel)
                max_vel = vel;
            if (fabs(field->p[i]) > max_p)
                max_p = fabs(field->p[i]);
        }
        stats->max_velocity = max_vel;
        stats->max_pressure = max_p;
    }
    return CFD_SUCCESS;
}

static Solver* create_projection_omp_solver(void) {
    Solver* solver = (Solver*)cfd_calloc(1, sizeof(Solver));
    if (!solver)
        return NULL;

    solver->name = SOLVER_TYPE_PROJECTION_OMP;
    solver->description = "OpenMP multi-threaded Projection solver";
    solver->version = "1.0.0";
    solver->capabilities = SOLVER_CAP_INCOMPRESSIBLE | SOLVER_CAP_TRANSIENT | SOLVER_CAP_PARALLEL;

    solver->init = projection_init;        // Reuse basic init
    solver->destroy = projection_destroy;  // Reuse basic destroy
    solver->step = projection_omp_step;
    solver->solve = projection_omp_solve;
    solver->apply_boundary = NULL;
    solver->compute_dt = NULL;

    return solver;
}
#endif
