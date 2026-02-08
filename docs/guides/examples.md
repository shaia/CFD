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

int main(void) {
    // Initialize library
    cfd_status_t status = cfd_init();

    // Create 100x50 grid from [0,1] x [0,0.5]
    simulation* sim = simulation_create(100, 50, 0.0, 1.0, 0.0, 0.5);

    // Run 100 steps
    for (int step = 0; step < 100; step++) {
        status = run_simulation_step(sim, 0.001);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Step failed: %s\n", cfd_get_last_error());
            break;
        }

        // Output every 10 steps
        if (step % 10 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename),
                     "output/step_%04d.vtk", step);
            write_simulation_to_vtk(sim, filename);
        }
    }

    // Cleanup
    simulation_destroy(sim);
    cfd_cleanup();

    return 0;
}
```

**Output:**
- `output/minimal_step_*.vtk` files
- Visualize with ParaView or `visualization/simple_viz.py`

**Run:**
```bash
cd build/Release
./minimal_example
python ../../visualization/simple_viz.py
```

---

### 2. basic_simulation.c

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
simulation* setup_simulation(void) {
    simulation* sim = simulation_create(200, 100, 0.0, 2.0, 0.0, 1.0);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation\n");
        return NULL;
    }
    return sim;
}

// Solve phase
cfd_status_t run_simulation(simulation* sim, int max_steps) {
    for (int step = 0; step < max_steps; step++) {
        cfd_status_t status = run_simulation_step(sim, 0.001);
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

### 3. solver_selection.c

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

### 4. performance_comparison.c

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
    run_simulation_step(sim, dt);
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
Benchmark: euler_cpu (100x50, 50 steps)
  Time: 0.003 seconds (0.053 ms/step)

Benchmark: euler_avx2 (100x50, 50 steps)
  Time: 0.001 seconds (0.018 ms/step)
  Speedup: 2.9x

Benchmark: projection_optimized (100x50, 50 steps)
  Time: 0.005 seconds (0.106 ms/step)
  Speedup: 3.6x vs projection
```

---

### 5. custom_boundary_conditions.c

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

### 6. lid_driven_cavity.c

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

### 7. csv_data_export.c

**Purpose:** Export simulation data for external analysis

**What it demonstrates:**
- CSV output format
- Centerline profiles
- Time series data
- Data extraction patterns

**Examples:**

**Export full field:**
```c
#include "cfd/io/csv_output.h"

write_field_to_csv(field->u, grid->nx, grid->ny, "u_velocity.csv");
write_field_to_csv(field->v, grid->nx, grid->ny, "v_velocity.csv");
write_field_to_csv(field->p, grid->nx, grid->ny, "pressure.csv");
```

**Export centerline:**
```c
write_centerline_to_csv(field, grid, "centerline.csv");
// Output: x, u(x, y=0.5), v(x, y=0.5), p(x, y=0.5)
```

**Time series:**
```c
FILE* fp = fopen("timeseries.csv", "w");
fprintf(fp, "time,kinetic_energy,max_velocity\n");

for (int step = 0; step < max_steps; step++) {
    run_simulation_step(sim, dt);

    double ke = compute_kinetic_energy(field, grid);
    double u_max = compute_max_velocity(field, grid);

    fprintf(fp, "%.6f,%.6e,%.6e\n", step*dt, ke, u_max);
}

fclose(fp);
```

---

### 8. velocity_visualization.c

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

### 9. runtime_comparison.c (CUDA)

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

### Python Scripts

**Simple Visualization:**
```bash
python visualization/simple_viz.py
```

**Advanced Plotting:**
```bash
python visualization/visualize_cfd.py --input output/result_0100.vtk
```

**Animated Flow:**
```bash
python visualization/enhanced_visualize.py --animate output/simulation_*/*.vtk
```

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
simulation* sim = NULL;
grid_t* grid = NULL;
flow_field* field = NULL;

// Setup...
sim = simulation_create(...);
if (!sim) goto cleanup;

grid = grid_create_uniform(...);
if (!grid) goto cleanup;

// Use resources...

cleanup:
    if (sim) simulation_destroy(sim);
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
        write_simulation_to_vtk(sim, filename);
    }
}
```

## Next Steps

- [API Reference](../reference/api-reference.md) - Detailed API documentation
- [Solvers](../reference/solvers.md) - Understanding numerical methods
- [Building](../guides/building.md) - Build configuration options
