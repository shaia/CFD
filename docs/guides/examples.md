# Examples Guide

Complete guide to example programs demonstrating library usage.

## Overview

The `examples/` directory contains programs demonstrating various features of the CFD Framework, from minimal usage to advanced scenarios.

## Building Examples

Examples are built automatically with the main library:

```bash
# Build all examples
cmake --build build --config Release

# Run from build directory
cd build/Release  # Windows
cd build          # Linux/macOS

# Execute examples
./minimal_example
./basic_simulation
./solver_selection
```

## Example Programs

### 1. minimal_example.c

**Purpose:** Simplest possible usage - quick start reference

**What it demonstrates:**
- Library initialization
- Creating a simulation
- Running simulation steps
- Writing VTK output
- Resource cleanup

**Code (~50 lines):**
```c
#include "cfd/api/simulation_api.h"
#include "cfd/io/vtk_output.h"

int main(void) {
    // Initialize library
    cfd_status_t status = cfd_init();

    // Create 100x50 grid from [0,1] x [0,0.5]
    simulation_data* sim = init_simulation(100, 50, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation\n");
        return 1;
    }

    sim->params.dt = 0.001;  // Set time step

    // Run 100 steps
    for (int step = 0; step < 100; step++) {
        status = run_simulation_step(sim);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Step failed: %s\n", cfd_get_last_error());
            break;
        }

        // Output every 10 steps
        if (step % 10 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output/step_%04d.vtk", step);
            write_vtk_flow_field(filename, sim->field,
                                sim->grid->nx, sim->grid->ny, sim->grid->nz,
                                sim->grid->xmin, sim->grid->xmax,
                                sim->grid->ymin, sim->grid->ymax,
                                sim->grid->zmin, sim->grid->zmax);
        }
    }

    // Cleanup
    free_simulation(sim);
    cfd_cleanup();

    return 0;
}
```

**Output:**
- `output/minimal_step_*.vtk` files
- Visualize with ParaView or VisIt

**Run:**
```bash
cd build/Release
./minimal_example
```

---

### 2. minimal_example_3d.c

**Purpose:** Demonstrates 3D simulation on a small grid

**What it demonstrates:**

- 3D grid initialization with `nz > 1`
- Running 3D simulation steps
- 3D VTK output (STRUCTURED_POINTS with z-dimension)
- All solver backends work identically in 3D

**Code (~55 lines):**
```c
#include "cfd/api/simulation_api.h"

int main() {
    // Initialize 3D simulation: 16x16x16 grid on unit cube
    size_t nx = 16, ny = 16, nz = 16;
    simulation_data* sim = init_simulation(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Configure output
    simulation_set_output_dir(sim, "../../artifacts");
    simulation_set_run_prefix(sim, "minimal_3d");
    simulation_register_output(sim, OUTPUT_VELOCITY_MAGNITUDE, 5, "velocity_mag");

    // Run 10 steps
    for (int step = 0; step < 10; step++) {
        run_simulation_step(sim);
        simulation_write_outputs(sim, step);
    }

    free_simulation(sim);
    return 0;
}
```

**Key point:** When `nz=1`, the library produces bit-identical results to 2D. The branch-free `stride_z=0` pattern means all 3D code collapses to 2D with zero overhead.

**Run:**
```bash
cd build/Release
./minimal_example_3d
```

---

### 3. basic_simulation.c

**Purpose:** Complete simulation workflow with proper error handling

**What it demonstrates:**
- Structured simulation setup
- Error handling patterns
- Periodic output with timestamps
- Production-ready code structure

**Key Features:**
- Configurable grid size and domain
- Timestamped output directories
- Comprehensive error checking
- Clean separation of setup/solve/output

**Code Structure:**
```c
// Setup phase
simulation_data* setup_simulation(void) {
    simulation_data* sim = init_simulation(200, 100, 1, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation\n");
        return NULL;
    }
    sim->params.dt = 0.001;  // Set time step
    return sim;
}

// Solve phase
cfd_status_t run_simulation(simulation_data* sim, int max_steps) {
    for (int step = 0; step < max_steps; step++) {
        cfd_status_t status = run_simulation_step(sim);
        if (status != CFD_SUCCESS) {
            return status;
        }

        if (step % output_interval == 0) {
            write_output(sim, step);
        }
    }
    return CFD_SUCCESS;
}
```

**Run:**
```bash
./basic_simulation
# Output in: artifacts/output/simulation_200x100_<timestamp>/
```

---

### 4. solver_selection.c

**Purpose:** Demonstrate solver switching and discovery

**What it demonstrates:**
- Listing available solvers
- Creating simulations with specific solvers
- Runtime solver switching
- Backend availability checking

**Code Examples:**

**List all solvers:**
```c
ns_solver_registry_t* registry = cfd_registry_create();
cfd_registry_register_defaults(registry);

const char* names[32];
int count = cfd_registry_list(registry, names, 32);

printf("Available solvers:\n");
for (int i = 0; i < count; i++) {
    printf("  %d. %s\n", i+1, names[i]);
}
```

**Check backend availability:**
```c
if (cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
    printf("SIMD (AVX2/NEON) available\n");
} else {
    printf("SIMD not available - using scalar\n");
}
```

**Use specific solver:**
```c
simulation_data* sim = init_simulation_with_solver(
    100, 50, 0.0, 1.0, 0.0, 0.5,
    "projection_optimized"
);
```

**Run:**
```bash
./solver_selection
# Lists solvers, runs with each, compares results
```

---

### 5. performance_comparison.c

**Purpose:** Benchmark different solvers and grid sizes

**What it demonstrates:**
- Performance measurement
- Solver comparison
- Grid size scaling
- Timing methodology

**Benchmark Setup:**
```c
typedef struct {
    const char* name;
    size_t nx, ny;
    int steps;
} benchmark_config_t;

benchmark_config_t configs[] = {
    {"Small Grid",  50,  50, 100},
    {"Medium Grid", 100, 100, 100},
    {"Large Grid",  200, 200, 100},
};
```

**Timing Pattern:**
```c
#include <time.h>

clock_t start = clock();

// Run simulation
for (int step = 0; step < max_steps; step++) {
    run_simulation_step(sim);
}

clock_t end = clock();
double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

printf("Time: %.3f seconds (%.3f ms/step)\n",
       elapsed, 1000.0 * elapsed / max_steps);
```

**Run:**
```bash
# IMPORTANT: Use Release build for accurate benchmarks
cmake --build build --config Release
cd build/Release
./performance_comparison
```

**Expected Output:**
```
==================================================
Grid Size: 100x50 (5000 total cells)
==================================================

=== Basic NSSolver Benchmark ===
Grid size: 100x50, Iterations: 100
Execution time: 0.042 seconds
Performance: 11904762 cell-updates/second
Memory usage: 0.19 MB

=== Optimized NSSolver Benchmark ===
Grid size: 100x50, Iterations: 100
Execution time: 0.015 seconds
Performance: 33333333 cell-updates/second
Memory usage: 0.19 MB

=== Projection NSSolver Benchmark ===
Grid size: 100x50, Iterations: 100
Execution time: 0.245 seconds
Performance: 2040816 cell-updates/second
Memory usage: 0.19 MB

=== Projection Optimized Benchmark ===
Grid size: 100x50, Iterations: 100
Execution time: 0.068 seconds
Performance: 7352941 cell-updates/second
Memory usage: 0.19 MB
```

---

### 6. custom_boundary_conditions.c

**Purpose:** Flow around obstacles with complex geometry

**What it demonstrates:**
- Custom boundary condition implementation
- Inlet/outlet conditions
- No-slip walls
- Obstacle handling (flow around cylinder)
- Parabolic inlet profiles

**Boundary Condition Examples:**

**No-slip walls:**
```c
void apply_no_slip_walls(flow_field* field, grid_t* grid) {
    size_t nx = grid->nx;
    size_t ny = grid->ny;

    // Bottom and top walls
    for (size_t i = 0; i < nx; i++) {
        field->u[i + 0*nx] = 0.0;           // Bottom
        field->v[i + 0*nx] = 0.0;
        field->u[i + (ny-1)*nx] = 0.0;      // Top
        field->v[i + (ny-1)*nx] = 0.0;
    }

    // Left and right walls
    for (size_t j = 0; j < ny; j++) {
        field->u[0 + j*nx] = 0.0;           // Left
        field->v[0 + j*nx] = 0.0;
        field->u[(nx-1) + j*nx] = 0.0;      // Right
        field->v[(nx-1) + j*nx] = 0.0;
    }
}
```

**Parabolic inlet:**
```c
void apply_parabolic_inlet(flow_field* field, grid_t* grid, double u_max) {
    size_t nx = grid->nx;
    size_t ny = grid->ny;

    for (size_t j = 0; j < ny; j++) {
        double y = grid->y[j];
        double h = grid->ymax - grid->ymin;

        // Parabolic profile: u(y) = u_max * 4y(h-y)/h^2
        double u_inlet = u_max * 4.0 * y * (h - y) / (h * h);

        field->u[0 + j*nx] = u_inlet;
        field->v[0 + j*nx] = 0.0;
    }
}
```

**Cylinder obstacle:**
```c
void apply_cylinder_boundary(flow_field* field, grid_t* grid,
                             double cx, double cy, double radius) {
    for (size_t j = 0; j < grid->ny; j++) {
        for (size_t i = 0; i < grid->nx; i++) {
            double x = grid->x[i];
            double y = grid->y[j];
            double dist = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));

            // Inside cylinder: set velocity to zero
            if (dist < radius) {
                size_t idx = i + j * grid->nx;
                field->u[idx] = 0.0;
                field->v[idx] = 0.0;
            }
        }
    }
}
```

**Run:**
```bash
./custom_boundary_conditions
# Output: artifacts/output/cylinder_flow_*/
```

---

### 7. lid_driven_cavity.c

**Purpose:** Classic CFD benchmark - lid-driven cavity flow

**What it demonstrates:**
- Benchmark problem setup
- Command-line arguments
- Reynolds number configuration
- Validation against published results (Ghia et al., 1982)

**Usage:**
```bash
./lid_driven_cavity [Reynolds_number]

# Examples:
./lid_driven_cavity 100   # Re=100 (default)
./lid_driven_cavity 400   # Re=400
./lid_driven_cavity 1000  # Re=1000
```

**Problem Setup:**
- Square cavity [0,1] × [0,1]
- Top wall moves with velocity u=1
- Other walls: no-slip (u=v=0)
- Re = ρUL/μ = UL/ν

**Code:**
```c
void setup_cavity_bc(flow_field* field, grid_t* grid, double lid_vel) {
    size_t nx = grid->nx;
    size_t ny = grid->ny;

    // No-slip on all walls
    for (size_t i = 0; i < nx; i++) {
        field->u[i + 0*nx] = 0.0;
        field->v[i + 0*nx] = 0.0;
        field->u[i + (ny-1)*nx] = lid_vel;  // Moving lid
        field->v[i + (ny-1)*nx] = 0.0;
    }

    for (size_t j = 0; j < ny; j++) {
        field->u[0 + j*nx] = 0.0;
        field->v[0 + j*nx] = 0.0;
        field->u[(nx-1) + j*nx] = 0.0;
        field->v[(nx-1) + j*nx] = 0.0;
    }
}
```

**Expected Results:**
For Re=100, centerline velocities should match Ghia et al. within ~1%.

**Output:**
- VTK files in `output/lid_cavity_Re<number>/`
- Compare with published data in [validation/lid-driven-cavity.md](validation/lid-driven-cavity.md)

---

### 8. csv_data_export.c

**Purpose:** Export simulation data for external analysis

**What it demonstrates:**
- CSV output format
- Centerline profiles
- Time series data
- Data extraction patterns

**Examples:**

**Export timeseries:**
```c
#include "cfd/io/csv_output.h"

// Write timeseries data (step, time, max velocities, etc.)
ns_solver_stats_t stats = ns_solver_stats_default();
write_csv_timeseries("timeseries.csv", step, sim->current_time,
                     sim->field, NULL, &sim->params, &stats,
                     sim->grid->nx, sim->grid->ny, (step == 0));
```

**Export centerline:**
```c
// Export horizontal centerline (along x-axis at y=mid)
write_csv_centerline("centerline.csv", sim->field, NULL,
                     sim->grid->x, sim->grid->y,
                     sim->grid->nx, sim->grid->ny,
                     PROFILE_HORIZONTAL);
```

**Statistics export:**
```c
// Export global statistics (min/max/avg for all fields)
for (int step = 0; step < max_steps; step++) {
    cfd_status_t status = run_simulation_step(sim);
    if (status != CFD_SUCCESS) break;

    // Write statistics every step
    write_csv_statistics("statistics.csv", step, sim->current_time,
                        sim->field, NULL,
                        sim->grid->nx, sim->grid->ny, (step == 0));
}
```

---

### 9. velocity_visualization.c

**Purpose:** Generate VTK files optimized for velocity visualization

**What it demonstrates:**
- Velocity vector fields
- Streamline-ready output
- Vorticity computation

**Run:**
```bash
./velocity_visualization
# Open in ParaView → Add Glyph filter → Select u,v as vectors
```

---

### 10. runtime_comparison.c (CUDA)

**Purpose:** Comprehensive CPU vs GPU benchmarking

**What it demonstrates:**
- GPU configuration
- Crossover point analysis
- Grid size scaling
- Iteration count impact

**Benchmark Configurations:**
```c
typedef struct {
    size_t nx, ny;
    int iterations;
} gpu_benchmark_t;

gpu_benchmark_t tests[] = {
    { 50,  50, 100},    // Small - expect CPU faster
    {100, 100, 100},    // Medium - crossover region
    {200, 200, 100},    // Large - GPU starts winning
    {500, 500, 100},    // Very large - GPU dominates
};
```

**Run:**
```bash
# Requires CUDA build
cmake -B build -DCFD_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build/Release
./runtime_comparison
```

**Expected Results:**
```
Grid 50x50, 100 iterations:
  CPU (AVX2): 0.123s
  GPU (CUDA): 0.245s (slower due to transfer overhead)

Grid 500x500, 100 iterations:
  CPU (AVX2): 82.4s
  GPU (CUDA): 6.8s (12x speedup)
```

---

### 11. lid_driven_cavity_direct.c

**Purpose:** Lid-driven cavity using the mid-level solver registry API

**What it demonstrates:**
- Creating grid and flow_field manually
- Using the solver registry to create a projection solver
- Applying Dirichlet BCs explicitly each time step
- Monitoring solver statistics (max velocity, CFL, timing)
- Manual VTK output with `write_vtk_flow_field()`

**Usage:**
```bash
./lid_driven_cavity_direct [Re]
# Default Re=100
```

---

### 12. platform_diagnostics.c

**Purpose:** Query runtime platform capabilities and demonstrate utility APIs

**What it demonstrates:**
- SIMD detection: `cfd_get_simd_name()`, `cfd_has_avx2()`, `cfd_has_neon()`
- Backend availability: `bc_backend_available()`, `poisson_solver_backend_available()`
- Solver enumeration: `cfd_registry_list()` to list all registered solvers
- Derived fields: `derived_fields_create()`, `derived_fields_compute_statistics()`
- Error handling: `cfd_get_last_error()`, `cfd_get_error_string()`, `cfd_clear_error()`

**Sections:**
1. SIMD capabilities (architecture detection)
2. Backend availability (BC and Poisson solver backends)
3. Available NS solvers (enumerate via `cfd_registry_list()`)
4. Derived fields and statistics (compute velocity magnitude and field stats on a small TG vortex)
5. Error handling patterns (request a nonexistent solver, inspect error state)

**Run:**
```bash
./platform_diagnostics
```

**Expected Output:**
```
CFD Platform Diagnostics
========================

1. SIMD Capabilities
   Architecture: avx2
   AVX2:  yes
   NEON:  no
   Any:   yes

2. Backend Availability
   Boundary Conditions:
     Scalar:  available
     SIMD:    available
     OpenMP:  available
   Poisson Solvers:
     Scalar:  available
     SIMD:    available (avx2)
     OpenMP:  available

3. Available NS Solvers
   Found 8 solver(s):
     - explicit_euler
     - projection
     ...

4. Derived Fields & Statistics
   Taylor-Green vortex (32x32):
   u-velocity:  min=-0.0999, max=0.0999, avg=0.0000
   ...

5. Error Handling Patterns
   Requesting 'nonexistent_solver'... NULL (expected)
     Last error:  "Solver type 'nonexistent_solver' not registered"
     Status code: Resource not found (-9)
```

---

### 13. poisson_solver_tuning.c

**Purpose:** Compare Poisson solver methods, backends, and preconditioners

**What it demonstrates:**
- `poisson_solver_create(method, backend)` factory API
- `poisson_solver_params_t` with tolerance, max iterations, omega, preconditioner
- `poisson_solver_init()`, `poisson_solver_solve()`, `poisson_solver_destroy()`
- `poisson_solver_stats_t` for convergence monitoring
- `poisson_solve()` convenience API
- `poisson_solver_backend_available()` for runtime checking
- Error handling for unavailable solvers

**Sections:**
1. Method comparison (Jacobi, SOR, Red-Black SOR, CG, CG+Jacobi PC, BiCGSTAB) on scalar backend
2. Backend comparison (CG on Scalar, SIMD, OMP)
3. Convenience API demo (`poisson_solve()`)
4. Error handling (requesting unavailable multigrid solver)

**Problem:** Solves ∇²p = -2π²sin(πx)sin(πy) on a 64×64 grid using the library's default homogeneous Neumann boundary conditions. The reported L2 error compares methods/backends against a common reference field — not against the Dirichlet analytical solution sin(πx)sin(πy), since BCs differ.

**Run:**
```bash
./poisson_solver_tuning
```

**Expected Output:**
```
--- Method Comparison (Scalar Backend) ---
  Method                Iters     Residual    L2 Error      Time  Status
  Jacobi                10001   res=8.3e+00  L2=4.8e+00  5548 ms  max_iter
  CG                        1   res=5.1e-11  L2=1.1e-04     2 ms  converged
  CG + Jacobi PC            1   res=5.9e-11  L2=1.1e-04     2 ms  converged
  BiCGSTAB                  1   res=5.1e-11  L2=1.1e-04     2 ms  converged
```

---

### 14. poiseuille_stretched_grid.c

**Purpose:** Validate Poiseuille flow against the analytical parabolic velocity profile, comparing uniform vs stretched grids

**What it demonstrates:**
- `grid_initialize_stretched(g, beta)` with multiple beta values
- `bc_inlet_config_parabolic(U_max)` + `bc_inlet_set_edge()` for parabolic inlet
- `bc_outlet_config_zero_gradient()` + `bc_outlet_set_edge()` for outlet
- No-slip walls (manual loop)
- `bc_apply_neumann()` for pressure
- `derived_fields_create()`, `derived_fields_compute_statistics()`
- Direct solver registry API (`cfd_registry_create()`, `cfd_solver_create()`, `solver_step()`)

**Cases:**
1. Uniform grid (beta=0) — baseline
2. Mild stretching (beta=1.5) — 5:1 cell ratio
3. Strong stretching (beta=2.0) — 12:1 cell ratio

**Run:**
```bash
./poiseuille_stretched_grid
```

**Expected Output:**
```
--- Summary ---
  Grid Type                  min(dy)     max(dy)     Ratio    L2 Error
  Uniform (beta=0)           0.03226     0.03226       1.0   1.465e-02
  Mild (beta=1.5)            0.01055     0.05342       5.1   1.263e-01
  Strong (beta=2.0)          0.00537     0.06683      12.5   1.881e-01
```

---

### 15. taylor_green_convergence.c

**Purpose:** Taylor-Green vortex on a periodic domain with solver comparison and grid refinement

**What it demonstrates:**
- Periodic boundary conditions (`bc_apply_periodic` macro)
- Three NS solver types: `projection`, `rk2`, `explicit_euler`
- Analytical solution comparison (velocity decay exp(-2νt))
- Grid refinement showing error reduction with resolution

**Sections:**
1. **Velocity decay tracking** — Run with projection solver, print max|u| and kinetic energy at intervals alongside analytical predictions
2. **Solver comparison** — Same problem with projection, RK2, and explicit Euler at a single resolution
3. **Grid refinement** — Explicit Euler at 16×16, 32×32, 64×64 showing error decreases with resolution

**Run:**
```bash
./taylor_green_convergence
```

**Expected Output:**
```
Part 1: Velocity Decay (Projection, 32x32, dt=5e-04)
  Time        max|u|  Analytical          KE    KE_exact
  t=0.000   0.993592    1.000000    0.249722    0.250000
  t=0.100   0.968325    0.998002    0.222092    0.249002
  ...

Part 2: Solver Comparison (32x32, dt=5e-04, T=0.5)
  Solver                    L2 Error      max|u|
  Projection               6.403e-02    0.894924
  RK2 (Heun)               4.200e-02    0.951252
  Explicit Euler           6.107e-03    0.991708

Part 3: Grid Refinement (Explicit Euler, dt=5e-04, T=0.5)
  Resolution        L2 Error
   16 x 16      9.276e-03
   32 x 32      6.107e-03
   64 x 64      5.030e-03
```

---

### 16. pulsatile_inlet_flow.c

**Purpose:** Demonstrate all time-varying boundary condition types for pulsatile/transient flows

**What it demonstrates:**
- `bc_inlet_config_time_sinusoidal()` for pulsatile flow
- `bc_inlet_config_time_ramp()` for smooth start-up
- `bc_inlet_config_time_step()` for sudden changes
- `BC_TIME_CONTEXT(time, dt)` macro for time context
- `bc_apply_inlet_time()` for time-varying BC application
- `bc_apply_outlet_velocity()` for outlet
- `bc_apply_neumann()` for pressure

**Cases:**
1. **Sinusoidal** — Base velocity (1.0, 0.0) modulated at 2 Hz with 30% amplitude. Inlet u oscillates between 0.7 and 1.3
2. **Ramp start-up** — Velocity ramps from 0 to 1.0 over t=[0, 0.25], then holds at 1.0
3. **Step change** — Velocity jumps from 0.5 to 1.5 at t=0.2

**Run:**
```bash
./pulsatile_inlet_flow
```

**Expected Output:**
```
  Case: Sinusoidal (freq=2Hz, amp=30%)
    t=0.000: inlet u_mid = 1.0000
    t=0.050: inlet u_mid = 1.1763
    t=0.100: inlet u_mid = 1.2853
    ...

  Case: Ramp Start-up (0 -> 1.0 over t=[0, 0.25])
    t=0.000: inlet u_mid = 0.0000
    t=0.050: inlet u_mid = 0.2000
    t=0.200: inlet u_mid = 0.8000
    t=0.250: inlet u_mid = 1.0000
    ...

  Case: Step Change (0.5 -> 1.5 at t=0.2)
    t=0.000: inlet u_mid = 0.5000
    t=0.200: inlet u_mid = 1.5000
    ...
```

---

## Visualization

### VTK Files (ParaView/VisIt)

1. **Open in ParaView:**
   ```bash
   paraview output/simulation_*/result_*.vtk
   ```

2. **Create Visualization:**
   - Add "Glyph" filter for velocity vectors
   - Add "Contour" filter for pressure isocontours
   - Add "Stream Tracer" for streamlines

## Common Patterns

### Error Handling

```c
cfd_status_t status = some_operation();
if (status != CFD_SUCCESS) {
    const char* err_str = cfd_get_error_string(status);
    const char* detail = cfd_get_last_error();
    fprintf(stderr, "Error: %s (%d)\n", err_str, status);
    if (detail && detail[0]) {
        fprintf(stderr, "Details: %s\n", detail);
    }
    return EXIT_FAILURE;
}
```

### Resource Cleanup

```c
simulation_data* sim = NULL;
grid_t* grid = NULL;
flow_field* field = NULL;

// Setup...
sim = init_simulation(100, 50, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
if (!sim) goto cleanup;

grid = grid_create_uniform(...);
if (!grid) goto cleanup;

// Use resources...

cleanup:
    if (sim) free_simulation(sim);
    if (grid) grid_destroy(grid);
    if (field) flow_field_destroy(field);
```

### Output Organization

```c
// Create timestamped output directory
char output_dir[256];
time_t now = time(NULL);
strftime(output_dir, sizeof(output_dir),
         "output/sim_%Y%m%d_%H%M%S", localtime(&now));

mkdir(output_dir);

// Write numbered files
char filename[512];
for (int step = 0; step < max_steps; step++) {
    if (step % output_interval == 0) {
        snprintf(filename, sizeof(filename),
                 "%s/result_%04d.vtk", output_dir, step);
        write_vtk_flow_field(filename, sim->field,
                           sim->grid->nx, sim->grid->ny, sim->grid->nz,
                           sim->grid->xmin, sim->grid->xmax,
                           sim->grid->ymin, sim->grid->ymax,
                           sim->grid->zmin, sim->grid->zmax);
    }
}
```

## Next Steps

- [API Reference](../reference/api-reference.md) - Detailed API documentation
- [Solvers](../reference/solvers.md) - Understanding numerical methods
- [Building](../guides/building.md) - Build configuration options
