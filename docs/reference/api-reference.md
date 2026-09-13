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
simulation_data* init_simulation(size_t nx, size_t ny, size_t nz,
                                  double xmin, double xmax,
                                  double ymin, double ymax,
                                  double zmin, double zmax);

simulation_data* init_simulation_with_solver(size_t nx, size_t ny, size_t nz,
                                              double xmin, double xmax,
                                              double ymin, double ymax,
                                              double zmin, double zmax,
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
simulation_data* sim = init_simulation(100, 50, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
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
    // Turbulence
    turbulence_model_t        turb_model;  // TURB_MODEL_NONE (default), K_EPSILON, SPALART_ALLMARAS
    ns_turbulence_bc_config_t turb_bc;    // Per-face turbulence BC types
    // Pressure Poisson solver (projection solvers)
    ns_pressure_solver_t pressure_solver;  // NS_PRESSURE_SOLVER_DEFAULT (0) = backend's CG
} ns_solver_params_t;

ns_solver_params_t ns_solver_params_default(void);
```

`ns_pressure_solver_t` selects the pressure Poisson solve of the projection
method (scalar `projection` solver only; the multigrid modes require 2^k+1
grid points per active dimension, and other projection backends reject
non-default values with `CFD_ERROR_UNSUPPORTED` at init):

```c
typedef enum {
    NS_PRESSURE_SOLVER_DEFAULT = 0,   // Backend's default CG pressure solve
    NS_PRESSURE_SOLVER_MULTIGRID = 1, // Geometric multigrid V-cycle solver
    NS_PRESSURE_SOLVER_PCG_MG = 2,    // CG preconditioned by one MG V-cycle
} ns_pressure_solver_t;
```

`turbulence_model_t` is defined in `cfd/solvers/navier_stokes_solver.h`:

```c
typedef enum {
    TURB_MODEL_NONE             = 0,  // Laminar (default, zero overhead)
    TURB_MODEL_K_EPSILON        = 1,  // Standard k-epsilon (Launder-Spalding)
    TURB_MODEL_SPALART_ALLMARAS = 2,  // Spalart-Allmaras 1-equation (no-ft2)
} turbulence_model_t;
```

`ns_turbulence_bc_config_t` holds one BC type per domain face.  Available types:

| Constant | Meaning |
|----------|---------|
| `BC_TYPE_PERIODIC` | Periodic (default for all faces) |
| `BC_TYPE_NEUMANN` | Zero-gradient (outlet) |
| `BC_TYPE_DIRICHLET` | Fixed inlet values via per-face `k_values` / `eps_values` / `nu_tilde_values` |
| `BC_TYPE_NOSLIP` | Log-law wall-function treatment |

### Solver Statistics

```c
typedef struct {
    int iterations;         // Total iterations
    double residual;        // Final residual
    double elapsed_time;    // Execution time (seconds)
    cfd_status_t status;    // Solver status
    double max_nu_t;        // Maximum turbulent viscosity (0.0 when laminar)
} ns_solver_stats_t;

ns_solver_stats_t ns_solver_stats_default(void);
```

## Grid API

### Grid Creation

```c
#include "cfd/core/grid.h"

// Create grid (use nz=1 for 2D)
grid* grid_create(size_t nx, size_t ny, size_t nz,
                  double xmin, double xmax,
                  double ymin, double ymax,
                  double zmin, double zmax);

// Initialize with uniform spacing
void grid_initialize_uniform(grid* g);

// Initialize with tanh-stretched spacing (beta controls clustering)
void grid_initialize_stretched(grid* g, double beta);

// Destroy grid
void grid_destroy(grid* g);
```

**Example:**
```c
// 100x50 uniform 2D grid from [0,1] x [0,0.5]
grid* g = grid_create(100, 50, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
grid_initialize_uniform(g);

printf("Grid: %zu x %zu x %zu\n", g->nx, g->ny, g->nz);
printf("dx = %f, dy = %f\n", g->dx[0], g->dy[0]);

grid_destroy(g);
```

### Grid Structure

```c
typedef struct {
    double* x;        // x-coordinates [nx]
    double* y;        // y-coordinates [ny]
    double* dx;       // x-direction cell sizes [nx-1]
    double* dy;       // y-direction cell sizes [ny-1]
    size_t nx, ny;    // Grid dimensions (x, y)
    double xmin, xmax;  // Domain bounds (x)
    double ymin, ymax;  // Domain bounds (y)

    // 3D extension (nz=1 reproduces 2D behavior)
    double* z;        // z-coordinates [nz] (NULL when nz==1)
    double* dz;       // z-direction cell sizes [nz-1] (NULL when nz==1)
    size_t nz;        // Number of z-points (1 for 2D)
    double zmin, zmax;  // Domain bounds (z) (0.0 for 2D)
    size_t stride_z;  // nx*ny when nz>1, 0 when nz==1
    double inv_dz2;   // 1/(dz*dz) when nz>1, 0.0 when nz==1
    size_t k_start;   // 1 when nz>1, 0 when nz==1
    size_t k_end;     // nz-1 when nz>1, 1 when nz==1
} grid;
```

## Flow Field API

### Flow Field Operations

```c
#include "cfd/core/flow_field.h"

// Create flow field
flow_field* flow_field_create(size_t nx, size_t ny, size_t nz);

// Initialize with values
void flow_field_set_uniform(flow_field* field, double u_val, double v_val, double p_val);

// Destroy flow field
void flow_field_destroy(flow_field* field);
```

### Flow Field Structure

```c
typedef struct {
    size_t nx, ny, nz;    // Grid dimensions (nz=1 for 2D)
    double* u;            // x-velocity [nx * ny * nz]
    double* v;            // y-velocity [nx * ny * nz]
    double* w;            // z-velocity [nx * ny * nz] (zero for 2D)
    double* p;            // Pressure [nx * ny * nz]
    double* rho;          // Density [nx * ny * nz]
    double* T;            // Temperature [nx * ny * nz]
    // Turbulence fields (always allocated; zero when TURB_MODEL_NONE)
    double* turb_k;       // Turbulent kinetic energy k [nx * ny * nz]
    double* turb_eps;     // Dissipation rate epsilon   [nx * ny * nz]
    double* turb_nu_tilde;// SA transported variable ν̃  [nx * ny * nz]
    double* nu_t;         // Turbulent viscosity ν_t    [nx * ny * nz]
} flow_field;
```

**Indexing:**
```c
// Row-major ordering
// 2D: index = i + j * nx (via IDX_2D macro)
// 3D: index = k * nx * ny + j * nx + i (via IDX_3D macro)
#include "cfd/core/indexing.h"
size_t idx = IDX_2D(i, j, field->nx);          // 2D
size_t idx3d = IDX_3D(i, j, k, field->nx, field->ny); // 3D
```

## Poisson Solver API

### Poisson Solver Creation

```c
#include "cfd/solvers/poisson_solver.h"

// Create solver
poisson_solver_t* poisson_solver_create(poisson_solver_method_t method,
                                        poisson_solver_backend_t backend);

// Initialize solver (2D: nz = 1, dz = 0.0)
cfd_status_t poisson_solver_init(poisson_solver_t* solver,
                                 size_t nx, size_t ny, size_t nz,
                                 double dx, double dy, double dz,
                                 const poisson_solver_params_t* params);

// Solve Poisson equation
cfd_status_t poisson_solver_solve(poisson_solver_t* solver,
                                  double* x, double* x_temp, const double* rhs,
                                  poisson_solver_stats_t* stats);

// Cleanup
void poisson_solver_destroy(poisson_solver_t* solver);
```

### Poisson Methods

```c
typedef enum {
    POISSON_METHOD_JACOBI,        // Jacobi iteration (fully parallelizable)
    POISSON_METHOD_GAUSS_SEIDEL,  // Gauss-Seidel iteration
    POISSON_METHOD_SOR,           // Successive Over-Relaxation
    POISSON_METHOD_REDBLACK_SOR,  // Red-Black SOR (parallelizable)
    POISSON_METHOD_CG,            // Conjugate Gradient (SPD systems)
    POISSON_METHOD_BICGSTAB,      // BiCGSTAB (non-symmetric systems)
    POISSON_METHOD_GMRES,         // Restarted GMRES(m) (scalar/SIMD/OMP)
    POISSON_METHOD_MULTIGRID      // Geometric multigrid V/W/F cycles (scalar/OMP; dims 2^k+1)
} poisson_solver_method_t;
```

> Preconditioning is a separate `preconditioner` field on `poisson_solver_params_t`
> (see below), not a distinct method — CG becomes PCG when a preconditioner is set.

### Poisson Backends

```c
typedef enum {
    POISSON_BACKEND_AUTO,     // Auto-select: SIMD when detected at runtime, else scalar
    POISSON_BACKEND_SCALAR,   // Scalar CPU implementation
    POISSON_BACKEND_OMP,      // OpenMP parallelized
    POISSON_BACKEND_SIMD,     // SIMD + OpenMP with runtime detection (AVX2/NEON)
    POISSON_BACKEND_GPU       // CUDA GPU
} poisson_solver_backend_t;

bool poisson_solver_backend_available(poisson_solver_backend_t backend);
```

> Availability is reported per backend, not per method: `poisson_solver_create`
> still returns NULL (last status `CFD_ERROR_UNSUPPORTED`) for a method that has no
> implementation on an otherwise available backend.

### Poisson Parameters

```c
typedef struct {
    double tolerance;           // Relative tolerance (default: 1e-6)
    double absolute_tolerance;  // Absolute tolerance (default: 1e-10)
    int max_iterations;         // Max iterations (default: 5000)
    double omega;               // SOR relaxation (default: 0 = auto-optimal)
    int check_interval;         // Convergence check interval (default: 1)
    bool verbose;               // Print convergence info (default: false)
    poisson_precond_type_t preconditioner;  // Preconditioner (default: POISSON_PRECOND_NONE)
    int restart;                // GMRES(m) restart length (default: 0 = auto/30)

    // Multigrid only (all 0-defaults are backward compatible)
    mg_cycle_type_t mg_cycle;       // MG_CYCLE_V (default) / MG_CYCLE_W / MG_CYCLE_F
    mg_smoother_type_t mg_smoother; // MG_SMOOTHER_REDBLACK_GS (default) / MG_SMOOTHER_JACOBI
    mg_bc_type_t mg_bc;             // MG_BC_NEUMANN (default) / MG_BC_DIRICHLET
    int mg_pre_smooth;              // Pre-smoothing sweeps (0 = default 2)
    int mg_post_smooth;             // Post-smoothing sweeps (0 = default 2)
    int mg_coarse_max_iter;         // Coarsest-grid smoother sweeps (0 = default 50)
    int mg_max_levels;              // Max levels (0 = auto)
} poisson_solver_params_t;

poisson_solver_params_t poisson_solver_params_default(void);
```

> Multigrid requires 2^k+1 grid points per active dimension (e.g. 33, 65, 129);
> `poisson_solver_init` returns `CFD_ERROR_INVALID` otherwise. In the default
> `MG_BC_NEUMANN` mode the solution is defined up to an additive constant.
> Multigrid has scalar and OpenMP backends: `POISSON_BACKEND_AUTO` picks scalar
> (AUTO prefers SIMD, which multigrid lacks), so request `POISSON_BACKEND_OMP`
> (or the `POISSON_SOLVER_MG_OMP` preset) explicitly; its results are
> bit-identical to scalar.

`poisson_precond_type_t` values: `POISSON_PRECOND_NONE` (0, default),
`POISSON_PRECOND_JACOBI` (1), `POISSON_PRECOND_MULTIGRID` (2 — one MG V-cycle
per apply; scalar CG only, 2^k+1 dims; other backends return
`CFD_ERROR_UNSUPPORTED`). The convenience API exposes the MG-preconditioned CG
as the `POISSON_SOLVER_PCG_MG_SCALAR` preset for
`poisson_solve()`/`poisson_solve_3d()`.

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

## Turbulence API

```c
#include "cfd/solvers/turbulence_solver.h"
```

### turbulence_init_uniform

```c
cfd_status_t turbulence_init_uniform(flow_field* field,
                                     const ns_solver_params_t* params,
                                     double k0, double eps0, double nu_tilde0);
```

Initialize turbulence fields to spatially uniform values.  **Call this once before
time-stepping whenever `params->turb_model != TURB_MODEL_NONE`.**

- `k0` — initial turbulent kinetic energy (k-ε only; ignored for SA)
- `eps0` — initial dissipation rate (k-ε only; ignored for SA)
- `nu_tilde0` — initial SA transported variable (SA only; ignored for k-ε)

Returns `CFD_SUCCESS`, or `CFD_ERROR_INVALID` if `field` or `params` is NULL.

### turbulence_step_explicit

```c
cfd_status_t turbulence_step_explicit(flow_field* field,
                                      const grid* g,
                                      const ns_solver_params_t* params,
                                      double dt, double time);
```

Advance the turbulence transport equations by one explicit time step.  This function is
**called automatically** by all CPU/OMP/AVX2 NS solvers after the velocity and energy steps
— users normally do not call it directly.

Returns `CFD_SUCCESS` (no-op when `TURB_MODEL_NONE`), `CFD_ERROR_UNSUPPORTED` on a 3D grid
or non-uniform grid spacing, or `CFD_ERROR_INVALID` for NULL arguments.

### turbulence_apply_bcs

```c
cfd_status_t turbulence_apply_bcs(flow_field* field,
                                  const grid* g,
                                  const ns_solver_params_t* params);
```

Apply turbulence boundary conditions (periodic, Neumann, Dirichlet, or wall-function) to all
faces as configured in `params->turb_bc`.  Also called automatically by the NS solvers.

### turbulence_wall_u_tau

```c
double turbulence_wall_u_tau(double u_p, double y_p, double nu);
```

Compute the friction velocity u_τ from the first-node parallel velocity `u_p`, wall-normal
distance `y_p`, and kinematic viscosity `nu` using the log-law (`u+ = ln(y+)/κ + B`) via
Newton iteration.  Below y+ = 11.63 the linear viscous-sublayer law is used instead.  Used
internally by the wall-function BC; exposed for testing and post-processing.

### VTK and CSV turbulence output

When a turbulence model is active, `write_vtk_flow_field()` automatically includes four
additional point-data scalars:

- `turbulent_kinetic_energy` (k)
- `dissipation_rate` (ε)
- `nu_tilde` (ν̃)
- `turbulent_viscosity` (ν_t)

CSV centerline files written by `write_centerline_to_csv()` gain three additional columns:
`turb_k`, `turb_eps`, `nu_t`.

### Minimal usage example

```c
sim->params.mu = 1.0 / 395.0;
sim->params.turb_model = TURB_MODEL_K_EPSILON;
sim->params.turb_bc.bottom = BC_TYPE_NOSLIP;  /* wall-function walls */
sim->params.turb_bc.top    = BC_TYPE_NOSLIP;  /* other faces stay PERIODIC */

turbulence_init_uniform(sim->field, &sim->params, k0, eps0, 0.0);

/* run_simulation_step advances k/eps/nu_t automatically */
for (int step = 0; step < n_steps; step++) {
    cfd_status_t st = run_simulation_step(sim);
    if (st != CFD_SUCCESS) { /* handle */ break; }
}
```

## I/O API

### VTK Output

```c
#include "cfd/io/vtk_output.h"

// Scalar field to VTK (structured points format)
void write_vtk_output(const char* filename, const char* field_name,
                      const double* data, size_t nx, size_t ny, size_t nz,
                      double xmin, double xmax, double ymin, double ymax,
                      double zmin, double zmax);

// Vector field to VTK (w_data can be NULL for 2D)
void write_vtk_vector_output(const char* filename, const char* field_name,
                             const double* u_data, const double* v_data,
                             const double* w_data, size_t nx, size_t ny, size_t nz,
                             double xmin, double xmax, double ymin, double ymax,
                             double zmin, double zmax);

// Full flow field to VTK (velocity, pressure, density, temperature)
void write_vtk_flow_field(const char* filename, const flow_field* field,
                          size_t nx, size_t ny, size_t nz,
                          double xmin, double xmax, double ymin, double ymax,
                          double zmin, double zmax);
```

### CSV Output

```c
#include "cfd/io/csv_output.h"

cfd_status_t write_field_to_csv(double* data, size_t nx, size_t ny,
                                const char* filename);
cfd_status_t write_centerline_to_csv(flow_field* field, grid_t* grid,
                                     const char* filename);
```

### Restart / Checkpoint

Portable, versioned binary save/restore of the complete simulation state. The
`.cfdchk` format stores the grid, flow field, scalar solver parameters, the
accumulated simulation time, and the active solver's registry name. Solver
context buffers (e.g. Runge-Kutta stages) are pure per-step scratch and are
**not** stored — on restore the solver is recreated by name and re-initialized,
giving a bit-exact restart from a step boundary. Custom `source_func` /
`heat_source_func` callbacks cannot be serialized and must be re-supplied after a
restore (the in-place restore preserves an existing simulation's callbacks).

```c
#include "cfd/io/checkpoint.h"

// Low-level (operates on core types; read ALLOCATES grid/field, caller frees)
cfd_status_t cfd_checkpoint_write(const char* path, const grid* g,
                                  const flow_field* field,
                                  const ns_solver_params_t* params,
                                  double current_time, const char* solver_name,
                                  const char* run_prefix,
                                  const char* output_base_dir);
cfd_status_t cfd_checkpoint_read(const char* path, grid** out_grid,
                                 flow_field** out_field,
                                 ns_solver_params_t* out_params,
                                 double* out_current_time,
                                 char* out_solver_name, size_t solver_name_cap,
                                 char* out_run_prefix, size_t run_prefix_cap,
                                 char* out_output_base_dir, size_t output_base_dir_cap);
```

```c
#include "cfd/api/simulation_api.h"

// High-level (operates on simulation_data)
cfd_status_t     save_simulation_checkpoint(const simulation_data* sim, const char* path);
simulation_data* load_simulation_from_checkpoint(const char* path);          // fresh sim
cfd_status_t     restore_simulation_checkpoint(simulation_data* sim, const char* path); // in place
```

**Example:**
```c
// Save mid-run, then resume later in a new process
save_simulation_checkpoint(sim, "run.cfdchk");
// ...
simulation_data* resumed = load_simulation_from_checkpoint("run.cfdchk");
run_simulation_step(resumed);
free_simulation(resumed);
```

Portability is guaranteed by field-by-field little-endian encoding with
fixed-width integers and IEEE-754 doubles (no raw struct dumps); a header
endianness marker rejects foreign-endian files and a trailing CRC32 detects
truncation or corruption. Incompatible magic/version returns
`CFD_ERROR_INVALID` / `CFD_ERROR_UNSUPPORTED`; truncation or CRC mismatch returns
`CFD_ERROR_IO`.

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
#define NS_SOLVER_TYPE_PROJECTION_GPU           "projection_gpu"
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
