# API Reference

Complete API documentation for the CFD Framework.

## Core API

### Initialization

```c
#include "cfd/core/cfd_init.h"

cfd_status_t cfd_init(void);
void cfd_cleanup(void);
```

**cfd_init()**
- Thread-safe initialization (uses `pthread_once` / `InitOnceExecuteOnce`)
- Safe to call multiple times from different threads
- Returns `CFD_SUCCESS` on success

**cfd_cleanup()**
- Cleanup library resources
- Should be called before program termination

**Example:**
```c
int main(void) {
    cfd_status_t status = cfd_init();
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Init failed: %s\n", cfd_get_last_error());
        return 1;
    }

    // Use library...

    cfd_cleanup();
    return 0;
}
```

### Error Handling

```c
#include "cfd/core/cfd_status.h"

// Error codes
typedef enum {
    CFD_SUCCESS = 0,
    CFD_ERROR = -1,
    CFD_ERROR_NOMEM = -2,
    CFD_ERROR_INVALID = -3,
    CFD_ERROR_IO = -4,
    CFD_ERROR_UNSUPPORTED = -5,
    CFD_ERROR_DIVERGED = -6,
    CFD_ERROR_MAX_ITER = -7,
    CFD_ERROR_LIMIT_EXCEEDED = -8,
    CFD_ERROR_NOT_FOUND = -9
} cfd_status_t;

// Error reporting functions
void cfd_set_error(cfd_status_t status, const char* message);
const char* cfd_get_last_error(void);
cfd_status_t cfd_get_last_status(void);
const char* cfd_get_error_string(cfd_status_t status);
void cfd_clear_error(void);
```

**Error Handling Pattern:**
```c
cfd_status_t status = some_function();
if (status != CFD_SUCCESS) {
    const char* err_str = cfd_get_error_string(status);
    const char* detail = cfd_get_last_error();
    fprintf(stderr, "Error: %s (%d) - %s\n", err_str, status, detail);
    return status;
}
```

**Thread Safety:**
- Error messages use thread-local storage
- Each thread has independent error state

## Simulation API

### High-Level Interface

```c
#include "cfd/api/simulation_api.h"

// Create simulation
simulation_data* init_simulation(size_t nx, size_t ny,
                                 double xmin, double xmax,
                                 double ymin, double ymax);

simulation_data* init_simulation_with_solver(size_t nx, size_t ny,
                                             double xmin, double xmax,
                                             double ymin, double ymax,
                                             const char* solver_type);

// Simulate (dt is in sim_data->params.dt)
cfd_status_t run_simulation_step(simulation_data* sim_data);
cfd_status_t run_simulation_solve(simulation_data* sim_data);

// Access fields (fields are accessible directly from sim_data)
// sim_data->field - flow field
// sim_data->grid - computational grid
// sim_data->params - solver parameters

// Output
void simulation_write_outputs(simulation_data* sim_data, int step);

// Cleanup
void free_simulation(simulation_data* sim_data);
```

**Example:**
```c
// Create 100x50 simulation
simulation_data* sim = init_simulation(100, 50, 0.0, 1.0, 0.0, 0.5);
if (!sim) {
    fprintf(stderr, "Failed to create simulation\n");
    return 1;
}

// Configure parameters
sim->params.dt = 0.001;
sim->params.mu = 0.01;  // Viscosity

// Run simulation
for (int step = 0; step < 1000; step++) {
    cfd_status_t status = run_simulation_step(sim);
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Step failed: %s\n", cfd_get_last_error());
        break;
    }

    // Output every 10 steps
    if (step % 10 == 0) {
        simulation_write_outputs(sim, step);
    }
}

free_simulation(sim);
```

## Solver Registry API

### Registry Management

```c
#include "cfd/api/solver_registry.h"

// Create registry
ns_solver_registry_t* cfd_registry_create(void);
void cfd_registry_destroy(ns_solver_registry_t* registry);

// Register defaults
cfd_status_t cfd_registry_register_defaults(ns_solver_registry_t* registry);

// Register custom solver
cfd_status_t cfd_registry_register(
    ns_solver_registry_t* registry,
    const char* name,
    ns_solver_create_fn create_fn,
    ns_solver_method_t method,
    ns_solver_backend_t backend);

// Create solver
ns_solver_t* cfd_solver_create(ns_solver_registry_t* registry, const char* name);
ns_solver_t* cfd_solver_create_checked(ns_solver_registry_t* registry, const char* name);

// Query solvers
int cfd_registry_list(ns_solver_registry_t* registry,
                      const char** names, int max_count);
int cfd_registry_list_by_backend(ns_solver_registry_t* registry,
                                 ns_solver_backend_t backend,
                                 const char** names, int max_count);
int cfd_registry_has(ns_solver_registry_t* registry, const char* name);
```

**Example:**
```c
ns_solver_registry_t* reg = cfd_registry_create();
cfd_registry_register_defaults(reg);

// List all solvers
const char* names[32];
int count = cfd_registry_list(reg, names, 32);
for (int i = 0; i < count; i++) {
    printf("  %s\n", names[i]);
}

// Create specific solver
ns_solver_t* solver = cfd_solver_create(reg, "projection_optimized");
if (!solver) {
    fprintf(stderr, "Failed to create solver: %s\n", cfd_get_last_error());
}

cfd_registry_destroy(reg);
```

### Backend Availability

```c
// Backend types
typedef enum {
    NS_SOLVER_BACKEND_SCALAR = 0,
    NS_SOLVER_BACKEND_SIMD = 1,
    NS_SOLVER_BACKEND_OMP = 2,
    NS_SOLVER_BACKEND_CUDA = 3,
} ns_solver_backend_t;

// Check availability
int cfd_backend_is_available(ns_solver_backend_t backend);
const char* cfd_backend_get_name(ns_solver_backend_t backend);
```

**Example:**
```c
if (cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
    printf("SIMD (AVX2/NEON) available\n");
}

if (cfd_backend_is_available(NS_SOLVER_BACKEND_CUDA)) {
    printf("CUDA GPU available\n");
}
```

## Solver Interface

### Solver Operations

```c
#include "cfd/solvers/navier_stokes_solver.h"

// Initialize solver
cfd_status_t solver_init(ns_solver_t* solver, grid_t* grid, ns_solver_params_t* params);

// Solve one step
cfd_status_t solver_step(ns_solver_t* solver, flow_field* field, grid_t* grid,
                         ns_solver_params_t* params, ns_solver_stats_t* stats);

// Solve to final time
cfd_status_t solver_solve(ns_solver_t* solver, flow_field* field, grid_t* grid,
                          ns_solver_params_t* params, double final_time,
                          ns_solver_stats_t* stats);

// Cleanup
void solver_destroy(ns_solver_t* solver);
```

### Solver Parameters

```c
typedef struct {
    double dt;              // Time step
    double nu;              // Kinematic viscosity
    double rho;             // Density
    int max_iterations;     // Maximum iterations
    double tolerance;       // Convergence tolerance
} ns_solver_params_t;

ns_solver_params_t ns_solver_params_default(void);
```

### Solver Statistics

```c
typedef struct {
    int iterations;         // Total iterations
    double residual;        // Final residual
    double elapsed_time;    // Execution time (seconds)
    cfd_status_t status;    // Solver status
} ns_solver_stats_t;

ns_solver_stats_t ns_solver_stats_default(void);
```

## Grid API

### Grid Creation

```c
#include "cfd/core/grid.h"

// Create uniform grid
grid_t* grid_create_uniform(size_t nx, size_t ny,
                            double xmin, double xmax,
                            double ymin, double ymax);

// Create stretched grid
grid_t* grid_create_stretched(size_t nx, size_t ny,
                              double xmin, double xmax,
                              double ymin, double ymax,
                              double stretch_factor);

// Destroy grid
void grid_destroy(grid_t* grid);
```

**Example:**
```c
// 100x50 uniform grid from [0,1] x [0,0.5]
grid_t* grid = grid_create_uniform(100, 50, 0.0, 1.0, 0.0, 0.5);

printf("Grid: %zu x %zu\n", grid->nx, grid->ny);
printf("dx = %f, dy = %f\n", grid->dx, grid->dy);

grid_destroy(grid);
```

### Grid Structure

```c
typedef struct {
    size_t nx, ny;      // Grid dimensions
    double dx, dy;      // Grid spacing
    double xmin, xmax;  // Domain bounds
    double ymin, ymax;
    double* x;          // x-coordinates [nx]
    double* y;          // y-coordinates [ny]
} grid_t;
```

## Flow Field API

### Flow Field Operations

```c
#include "cfd/core/flow_field.h"

// Create flow field
flow_field* flow_field_create(size_t nx, size_t ny);

// Initialize with values
void flow_field_set_uniform(flow_field* field, double u_val, double v_val, double p_val);

// Destroy flow field
void flow_field_destroy(flow_field* field);
```

### Flow Field Structure

```c
typedef struct {
    size_t nx, ny;      // Grid dimensions
    double* u;          // x-velocity [nx * ny]
    double* v;          // y-velocity [nx * ny]
    double* p;          // Pressure [nx * ny]
} flow_field;
```

**Indexing:**
```c
// Row-major ordering: index = i + j * nx
size_t idx = i + j * field->nx;
double u_val = field->u[idx];
double v_val = field->v[idx];
double p_val = field->p[idx];
```

## Poisson Solver API

### Poisson Solver Creation

```c
#include "cfd/solvers/poisson_solver.h"

// Create solver
poisson_solver_t* poisson_solver_create(poisson_method_t method,
                                        poisson_backend_t backend);

// Initialize solver
cfd_status_t poisson_solver_init(poisson_solver_t* solver,
                                 size_t nx, size_t ny,
                                 double dx, double dy,
                                 poisson_solver_params_t* params);

// Solve Poisson equation
cfd_status_t poisson_solver_solve(poisson_solver_t* solver,
                                  double* x, double* x0, double* rhs,
                                  poisson_solver_stats_t* stats);

// Cleanup
void poisson_solver_destroy(poisson_solver_t* solver);
```

### Poisson Methods

```c
typedef enum {
    POISSON_METHOD_JACOBI = 0,
    POISSON_METHOD_SOR = 1,
    POISSON_METHOD_REDBLACK_SOR = 2,
    POISSON_METHOD_CG = 3,
    POISSON_METHOD_PCG = 4,
    POISSON_METHOD_BICGSTAB = 5,
} poisson_method_t;
```

### Poisson Backends

```c
typedef enum {
    POISSON_BACKEND_SCALAR = 0,
    POISSON_BACKEND_SIMD = 1,
    POISSON_BACKEND_OMP = 2,
} poisson_backend_t;

int poisson_solver_backend_available(poisson_backend_t backend);
```

### Poisson Parameters

```c
typedef struct {
    double tolerance;           // Relative tolerance (default: 1e-6)
    double absolute_tolerance;  // Absolute tolerance (default: 1e-10)
    int max_iterations;         // Max iterations (default: 5000)
    double omega;               // SOR relaxation (default: 1.5)
    int check_interval;         // Convergence check interval (default: 1)
    int verbose;                // Print convergence info (default: 0)
    poisson_precond_t preconditioner;  // Preconditioner (default: NONE)
} poisson_solver_params_t;

poisson_solver_params_t poisson_solver_params_default(void);
```

### Poisson Statistics

```c
typedef struct {
    poisson_solver_status_t status;  // Convergence status
    int iterations;                  // Iterations performed
    double initial_residual;         // Initial residual norm
    double final_residual;           // Final residual norm
    double elapsed_time;             // Execution time (seconds)
} poisson_solver_stats_t;

poisson_solver_stats_t poisson_solver_stats_default(void);
```

## I/O API

### VTK Output

```c
#include "cfd/io/vtk_output.h"

cfd_status_t write_to_vtk(flow_field* field, grid_t* grid, const char* filename);
cfd_status_t write_velocity_vectors_to_vtk(flow_field* field, grid_t* grid,
                                           const char* filename);
```

### CSV Output

```c
#include "cfd/io/csv_output.h"

cfd_status_t write_field_to_csv(double* data, size_t nx, size_t ny,
                                const char* filename);
cfd_status_t write_centerline_to_csv(flow_field* field, grid_t* grid,
                                     const char* filename);
```

## Memory Management

### Aligned Allocation

```c
#include "cfd/core/memory.h"

// Aligned allocation for SIMD (fixed 32-byte alignment)
void* cfd_aligned_calloc(size_t count, size_t size);
void* cfd_aligned_malloc(size_t size);
void cfd_aligned_free(void* ptr);  // MUST use this for aligned memory

// Regular allocation
void* cfd_calloc(size_t count, size_t size);
void* cfd_malloc(size_t size);
void cfd_free(void* ptr);
```

**Example:**
```c
// Allocate SIMD-aligned array (32-byte aligned for AVX2/AVX512)
double* data = cfd_aligned_calloc(nx * ny, sizeof(double));

// Use with SIMD
__m256d vec = _mm256_load_pd(&data[i]);  // Aligned load

// IMPORTANT: Must free with cfd_aligned_free(), not cfd_free()
cfd_aligned_free(data);  // Correct
// cfd_free(data);       // WRONG - undefined behavior!
```

## Solver Type Constants

```c
// Standard Built-in Solver Types (from navier_stokes_solver.h)
#define NS_SOLVER_TYPE_EXPLICIT_EULER           "explicit_euler"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED "explicit_euler_optimized"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_OMP       "explicit_euler_omp"
#define NS_SOLVER_TYPE_EXPLICIT_EULER_GPU       "explicit_euler_gpu"
#define NS_SOLVER_TYPE_PROJECTION               "projection"
#define NS_SOLVER_TYPE_PROJECTION_OPTIMIZED     "projection_optimized"
#define NS_SOLVER_TYPE_PROJECTION_OMP           "projection_omp"
#define NS_SOLVER_TYPE_RK2                      "rk2"
#define NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU    "projection_jacobi_gpu"
```

## Version Information

```c
#include "cfd/core/cfd_version.h"

const char* cfd_get_version(void);
const char* cfd_get_build_info(void);
```

## Next Steps

- [Examples](../guides/examples.md) - See complete usage examples
- [Solvers](solvers.md) - Learn about numerical methods
- [Architecture](../architecture/architecture.md) - Understand design patterns
