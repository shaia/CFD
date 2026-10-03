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
    cfd_finalize();

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

**Benchmark Setup:** each of the six solvers (`explicit_euler`, `explicit_euler_optimized`,
`explicit_euler_omp`, `projection`, `projection_optimized`, `projection_omp`) runs 100 steps
on grids of 50x25, 100x50, 200x100 and 400x200 over a 1.0 x 0.5 domain.

**Timing Pattern:** a solver that refuses init is reported, not timed, and a failed step ends
that solver's run:
```c
grid* grid = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
grid_initialize_uniform(grid);  // grid_create only allocates; spacing is zero until this

cfd_status_t status = solver_init(solver, grid, &params);
if (status != CFD_SUCCESS) {
    printf("Skipped: %s\n", cfd_get_last_error());
    goto cleanup;
}

clock_t start = clock();
for (int i = 0; i < iterations; i++) {
    status = solver_step(solver, field, grid, &params, &stats);
    if (status != CFD_SUCCESS) {
        printf("Failed at step %d: %s\n", i, cfd_get_error_string(status));
        goto cleanup;
    }
}
double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC;
```

`clock()` measures wall time on Windows but CPU time summed over threads on Linux and macOS,
so OpenMP rows there read slower than they run.

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

The Optimized rows need AVX2 compiled in. On a default build (`CFD_ENABLE_AVX2` is OFF) they
print `Skipped: SIMD Navier-Stokes backend unavailable: built without AVX2 (configure with
-DCFD_ENABLE_AVX2=ON)` instead of a timing.

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
- Compare with published data in [validation/lid-driven-cavity.md](../validation/lid-driven-cavity.md)

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
#include "cfd/core/derived_fields.h"
#include "cfd/io/csv_output.h"

// The row's statistics come from derived, which must have them computed --
// write_csv_timeseries() returns without writing if derived is NULL or its
// statistics are missing.
derived_fields* derived = derived_fields_create(sim->grid->nx, sim->grid->ny, sim->grid->nz);
derived_fields_compute_statistics(derived, sim->field);

// Pass the simulation's own last_stats, not a fresh ns_solver_stats_default():
// the dt column reports stats.dt_used, the step the solver actually took, which
// the Euler solvers clamp below params.dt. A default-constructed stats leaves
// dt_used at zero and the column falls back to params.dt -- fine for a direct
// caller, wrong beside a current_time that accumulated the clamped step.
write_csv_timeseries("timeseries.csv", step, sim->current_time,
                     sim->field, derived, &sim->params, &sim->last_stats,
                     sim->grid->nx, sim->grid->ny, (step == 0));

derived_fields_destroy(derived);
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
- Choosing the convection scheme: central differences (default) or first-order upwind
  (`params.convection_scheme`), and the resulting strength of the primary vortex

**Usage:**
```bash
./lid_driven_cavity_direct [Re] [upwind]
# Default Re=100, central differences
./lid_driven_cavity_direct 1000 upwind
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
- `poisson_solver_params_t` with tolerance, max iterations and `krylov.preconditioner`, leaving `sor.omega` at 0 (automatic)
- `poisson_make_rhs_compatible()` — what the default zero-gradient walls require of any RHS
- `poisson_solver_init()`, `poisson_solver_solve()`, `poisson_solver_destroy()`
- `poisson_solver_stats_t` and `poisson_solver_status_string()` for convergence monitoring
- `poisson_solver_config_preset()` + `poisson_solve()` convenience API
- Error handling for unavailable solvers, printing `cfd_get_last_error()` — the sentence naming the fix — not just the status category

**Sections:**
1. Method comparison (Jacobi, SOR, Red-Black SOR, CG, CG+Jacobi PC, CG+Multigrid PC, BiCGSTAB, Multigrid) on scalar backend
2. Backend comparison (CG on Scalar, SIMD, OMP)
3. Convenience API demo (`poisson_solve()` from an edited preset)
4. Error handling (requesting multigrid on an unavailable backend — GPU)

**Problem:** Solves ∇²p = -2π²sin(πx)sin(πy) on a 65×65 grid (2^k+1, so multigrid can build its hierarchy) against the library's default zero-gradient walls. That operator is singular — the constants are its nullspace — so the example calls `poisson_make_rhs_compatible()` to remove the RHS's interior mean; without it the Krylov methods refuse the solve with `POISSON_INCOMPATIBLE_RHS`, since the system has no solution at all.

The reported L2 error compares methods and backends against a common reference field, **not** against the analytical sin(πx)sin(πy): that field satisfies Dirichlet p=0 on all faces while these solves use zero-gradient walls, and the mean subtraction shifts the problem again. Set `params.walls = poisson_walls_uniform(POISSON_WALL_DIRICHLET, 0.0)` to solve the problem the analytical field actually poses — the Krylov methods honour that.

**Run:**
```bash
./poisson_solver_tuning
```

**Expected Output:** (timings are machine-dependent)
```
--- Method Comparison (Scalar Backend) ---
  Method                Iters        Residual      L2 Error     Time       Status
  Jacobi                 7371 iters  res=1.15e-07  L2=4.49e-01   111.3 ms  converged
  SOR                     378 iters  res=1.08e-07  L2=4.51e-01    14.1 ms  converged
  Red-Black SOR           300 iters  res=1.03e-07  L2=4.51e-01     4.6 ms  converged
  CG                      115 iters  res=2.37e-06  L2=4.49e-01     1.7 ms  converged
  CG + Jacobi PC          115 iters  res=2.37e-06  L2=4.49e-01     1.8 ms  converged
  CG + Multigrid PC        20 iters  res=2.12e-06  L2=3.25e-01     1.7 ms  converged
  BiCGSTAB                 79 iters  res=2.96e-06  L2=4.49e-01     2.1 ms  converged
  Multigrid                12 iters  res=7.49e-08  L2=4.53e-01     0.9 ms  converged

--- Backend Comparison (CG Method) ---
  CG Scalar               115 iters  res=2.37e-06  L2=4.49e-01     1.7 ms  converged
  CG SIMD                 115 iters  res=2.37e-06  L2=4.49e-01    67.8 ms  converged
  CG OMP                  115 iters  res=2.37e-06  L2=4.49e-01    64.8 ms  converged

--- Convenience API ---
  poisson_solve(DEFAULT): 100 iterations, L2 error = 4.49e-01
```

> Every method converges here, which is the point of the mean subtraction: on the
> raw strictly-negative RHS the problem has no solution, and each method fails
> differently — Jacobi and SOR run the budget out chasing a drifting constant,
> while CG drives the field away with the nullspace component it is minimising
> over. Earlier versions of this example documented that as ordinary output.
>
> The convenience call takes 100 iterations against CG's 115 because the preset's
> tolerance is 1e-6 and the benchmark rows use 1e-8.

---

### 14. poiseuille_stretched_grid.c

**Purpose:** Validate Poiseuille flow against the analytical parabolic velocity profile, comparing uniform vs stretched grids

**What it demonstrates:**
- `grid_initialize_stretched(g, beta)` with multiple beta values
- `bc_inlet_config_parabolic(U_max)` + `bc_inlet_set_edge()` for parabolic inlet
  (`bc_inlet_set_range()` would restrict it to part of the edge, as in a backward-facing step)
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
- Four NS solver types: `projection`, `rk2`, `rk4`, `explicit_euler`
- Analytical solution comparison (velocity decay exp(-2νt))
- Grid refinement showing error reduction with resolution

**Sections:**
1. **Velocity decay tracking** — Run with projection solver, print max|u| and kinetic energy at intervals alongside analytical predictions
2. **Solver comparison** — Same problem with projection, RK2, RK4 and explicit Euler at a single resolution
3. **Grid refinement** — Explicit Euler at 16×16, 32×32, 64×64 showing error decreases with resolution

**Run:**
```bash
./taylor_green_convergence
```

**Expected Output:**
```
Part 1: Velocity Decay (Projection, 32x32, dt=5e-04)
  Time          max|u|  Analytical          KE    KE_exact
  t=0.000     0.994344    1.000000    0.249756    0.250000
  t=0.050     0.988576    0.999000    0.236618    0.249500
  t=0.100     0.987451    0.998002    0.234571    0.249002
  ...

Part 2: Solver Comparison (32x32, dt=5e-04, T=0.5)
  Solver                    L2 Error      max|u|
  Projection               2.819e-02    0.978457
  RK2 (Heun)               1.904e-02    0.983619
  RK4 (classical)          1.904e-02    0.983619
  Explicit Euler           5.971e-03    0.992435

Part 3: Grid Refinement (Explicit Euler, dt=5e-04, T=0.5)
  Resolution        L2 Error
   16 x 16      8.402e-03
   32 x 32      5.971e-03
   64 x 64      4.999e-03
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

### 17. turbulent_channel.c

**Purpose:** RANS turbulence model demonstration — turbulent channel flow at Re_τ = 395

**What it demonstrates:**

- Enabling k-ε or Spalart-Allmaras turbulence via `params.turb_model`
- Configuring wall-function walls with `BC_TYPE_NOSLIP` faces in `params.turb_bc`
- Calling `turbulence_init_uniform()` before time-stepping
- Choosing the law of the wall with `params.turb_bc.wall_law`: `NS_WALL_LAW_LOG`
  (default) or `NS_WALL_LAW_SPALDING`
- Measuring u_τ by inverting the log law at the first node, the same yardstick for
  either wall law, independently of the wall function
- Comparing the computed u+ profile against the log law
- VTK output with the four turbulence scalar fields (`turbulent_kinetic_energy`,
  `dissipation_rate`, `nu_tilde`, `turbulent_viscosity`)

**Problem setup:**

- Domain: 4 × 2 (channel half-height δ = 1), flow is streamwise-uniform
- Re_τ = 395, ν = 1/395, ρ = 1; constant body force f_x = u_τ²/δ = 1, so the
  exact steady friction velocity is u_τ = 1
- Grid: 16×21 uniform, first-node y+ ≈ 40 (wall-function window 30–100)
- Bottom/top faces: `BC_TYPE_NOSLIP` (wall functions); left/right periodic
- Direct solver interface (registry + `solver_step`) at dt = 1e-3, marched until
  no velocity changes faster than 1e-3 per unit time. At 2e-3 forward Euler with
  central convection is unstable here (dt must stay below about 2ν_eff/|u|²), and an
  asymmetric mode grown from roundoff wrecks the profile by t ≈ 44

**Run:**
```bash
./turbulent_channel              # k-epsilon, log-law wall function (defaults)
./turbulent_channel ke           # k-epsilon explicitly
./turbulent_channel sa           # Spalart-Allmaras
./turbulent_channel ke spalding  # k-epsilon with Spalding's law of the wall
```

**Expected output (k-ε, log law):**
```
  Converged at step 64165 (max |du/dt| 1.00e-03)

Log-law u_tau at the first node = 0.9742 (exact force balance: 1.0000)

        y+        u+   log-law       err
      39.5     14.10     14.17      0.4%
      79.0     16.25     15.86      2.5%
     118.5     17.29     16.85      2.6%
     ...
     395.0     20.07     19.78      1.5%

Wrote turbulent_channel.vtk (open in ParaView to inspect nu_t/k).
```
The SA variant prints the same u_τ = 0.9742, with u+ within 4.7% of the log law: at
steady state the first-node speed is set by the shared wall function, not the closure.

With `spalding` the k-ε run prints u_τ = 0.9427 and u+ up to 4.2% above the log law.
Spalding's law sits below the log law for y+ < ~100, so the same wall shear needs a
lower first-node speed (u_p 13.22 against 13.74). Against Moser-Kim-Mansour DNS, whose
u+ is 14.27 at y+ = 39.5, that is 7.4% low where the log law is 3.7% low, which is why
the log law is the default for first nodes at 30 ≤ y+ ≤ 100.

---

### 18. sor_omega_sweep.c

**Purpose:** Count the sweeps SOR and Red-Black SOR need for every relaxation factor ω across a range, on one grid, and compare the automatic ω

**What it demonstrates:**

- Setting `params.sor.omega` explicitly, and leaving it at `0` for the automatic value
- Replacing the default zero-gradient walls through `solver->apply_bc`, installed before `poisson_solver_init()`, which chooses ω
- Reading `iterations` and `status` from `poisson_solver_stats_t`

**Problem setup:**

- Unit square, `--grid N` points per side (default 33), zero initial guess
- Right-hand side: seeded splitmix64 noise with zero interior mean, so every error mode is present and the zero-gradient problem has a solution
- Tolerance 1e-6 relative to the initial residual, at most 100,000 sweeps
- `--walls dirichlet` holds the boundary at zero instead of copying the interior onto it (the CUDA solvers reject it)

**Run:**
```bash
./sor_omega_sweep                                   # Red-Black SOR, 33x33, omega 1.00 to 1.99
./sor_omega_sweep --grid 65 --method sor --from 1.90 --to 1.99 --step 0.005
./sor_omega_sweep --walls dirichlet --backend simd  # x86: configure with -DCFD_ENABLE_AVX2=ON
./sor_omega_sweep --backend gpu                     # CUDA build; also scalar, simd, omp
```
`--backend simd` needs a SIMD build: on x86, configure with `-DCFD_ENABLE_AVX2=ON` (off by default) and run on
an AVX2 CPU; ARM64 builds get NEON without a flag. Otherwise the example prints
`Solver not available for this method, backend and walls` and exits with status 1.

**Expected output** (`--grid 33 --from 1.80 --to 1.95 --step 0.05`):
```
omega,sweeps,status,final_residual
1.8000,244,converged,9.639475e-07
1.8500,154,converged,9.741168e-07
1.9000,145,converged,9.315211e-07
1.9500,294,converged,6.763749e-07
auto,117,converged,9.625996e-07
```
Over the full range the sweeps form a U whose lowest point moves towards 2 as the grid grows; the
automatic ω sits just below it (1.863 on this grid, where the fewest sweeps, 104, come at 1.866).

---

### 19. natural_convection.c

**Purpose:** Buoyancy-driven flow in a differentially heated cavity, checked against the de Vahl Davis (1983) benchmark

**What it demonstrates:**
- The energy equation (`params.alpha`, thermal diffusivity)
- Boussinesq buoyancy (`params.beta`, `params.T_ref`, `params.gravity`)
- Per-face thermal boundary conditions (`params.thermal_bc`): Dirichlet on the heated walls, Neumann (adiabatic) on the others
- Deriving the diffusivities from the Rayleigh and Prandtl numbers
- Running to steady state on a residual that covers velocity and temperature
- Comparing the peak centerline velocities and the hot-wall Nusselt number with the benchmark

**Run:**
```bash
./natural_convection            # Ra = 1000, 41x41
./natural_convection 10000 81   # Ra = 1e4 on a finer grid
```

**Expected output (Ra = 1000):**
```
Steady after 5835 steps (t* = 0.456, residual 1.00e-04)

  quantity                     computed de Vahl Davis
  u_max (vertical centerline)      3.610
  v_max (horiz. centerline)       3.698
  Nu (hot wall)                   1.121

  Reference at Ra = 1000: u_max 3.649, v_max 3.697, Nu 1.117
  Differences: 1.1%, 0.0%, 0.4%
```
The VTK file it writes includes the temperature field. The example exits with status 1 if the
run does not reach steady state, or if a tabulated Rayleigh number lands more than 5% from
the benchmark.

---

### 20. steady_flow_multigrid.c

**Purpose:** Run a steady flow quickly with the multigrid pressure solve, and stop and resume a long run with checkpoints

**What it demonstrates:**
- Selecting the pressure solve on the projection solver (`params.pressure_solver = NS_PRESSURE_SOLVER_MULTIGRID`) and timing it against the default CG
- Running a lid-driven cavity to steady state on a velocity residual
- `save_simulation_checkpoint()` mid-run and `load_simulation_from_checkpoint()` into a fresh simulation
- Checking that the restarted run finishes where the uninterrupted one does

Multigrid needs 2^k+1 points per side (17, 33, 65, 129, 257, ...).

**Run:**
```bash
./steady_flow_multigrid          # 129x129, Re = 100
./steady_flow_multigrid 65 400   # n, Re
```

**Expected output** (129x129, `OMP_NUM_THREADS=1`; timings are machine-dependent):
```
1. Pressure solve cost (after 50 steps of start-up)
   CG (default):     36.48 ms/step
   Multigrid:         5.08 ms/step   (7.2x)

2. Running to steady state with multigrid...
   checkpoint written at step 7387 (t = 9.02): ...halfway.cfdchk
   steady at step 17903, t = 21.85

3. Restarting from the checkpoint in a fresh simulation...
   loaded: t = 9.02, solver projection_omp, pressure solve multigrid
   after the same 10516 steps: t = 21.85, max |velocity difference| = 0.0e+00, pressure 0.0e+00
   bit-identical to the uninterrupted run

Steady u at the cavity centre: -0.203223
   reference -0.203223 (129x129 grid-convergence study): matches
```
With several threads the restarted run agrees to round-off rather than bitwise, because threaded
reductions do not sum in a fixed order. The multigrid speed-up depends on the grid and on the
threading runtime: with MSVC OpenMP at full thread count the CG solve is slowed far more than
multigrid, so the printed ratio is much larger. The example exits with status 1 if the run does
not converge, if the restarted run's velocity or pressure differs by more than 1e-10, or, in
the default 129x129 Re = 100 case, if the steady centre velocity is more than 1e-4 from the
grid-convergence study's -0.203223. Without OpenMP it uses the scalar
projection, which also implements multigrid.

---

### 21. pressure_driven_channel.c

**Purpose:** Drive a channel flow with a pressure difference alone, and compare explicit and implicit viscous time stepping

**What it demonstrates:**
- Prescribing the pressure on the inlet and outlet faces (`params.pressure_bc` with `POISSON_WALL_DIRICHLET`); no inlet velocity anywhere
- Zero-gradient velocity at the open ends, no-slip on the plates
- Implicit viscous time integration (`params.viscous_scheme = NS_VISCOUS_SCHEME_CRANK_NICOLSON`), which removes the diffusion limit on dt
- Comparing the steady profile with the exact Poiseuille solution

**Run:**
```bash
./pressure_driven_channel       # 129x33
./pressure_driven_channel 65    # ny; the channel is 4x longer
```

**Expected output:**
```
dt limits: diffusive (explicit viscous only) 2.44e-03, advective 7.81e-03

  viscous term        steps        t  wall [s]     u_center     L2 error
  explicit             6391    14.04     14.83      1.00202     2.02e-03
  Crank-Nicolson       1813    14.16      5.63      1.01153     1.24e-02
```
The implicit run reaches steady state in 3.5x fewer steps, but its profile is further from
Poiseuille: the projection method is non-incremental, so its steady state carries a splitting
error that grows with dt. Implicit viscous stepping buys stability, not accuracy. The example
exits with status 1 unless both runs settle, the L2 errors stay below 1e-2 (explicit) and 5e-2
(Crank-Nicolson), and the implicit run takes fewer steps.

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

### Selecting the Convection Scheme

Central differencing (the default) is second-order accurate but oscillates when
convection dominates, for example on a coarse grid at high Reynolds number.
First-order upwind trades accuracy for bounded, wiggle-free advection of both
velocity and temperature:

```c
simulation_data* sim = init_simulation_with_solver(33, 33, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                    NS_SOLVER_TYPE_RK2_OMP);
sim->params.convection_scheme = NS_CONVECTION_SCHEME_UPWIND;

/* GPU solvers return CFD_ERROR_UNSUPPORTED for upwind */
cfd_status_t status = run_simulation_step(sim);
```

See [Convection Scheme](../reference/solvers.md#convection-scheme) for backend
support and stability notes.

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
