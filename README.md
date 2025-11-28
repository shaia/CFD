# CFD Framework

A computational fluid dynamics (CFD) framework implemented in C. This project provides a foundation for solving fluid dynamics problems using numerical methods, with VTK output for visualization.

## Features

- 2D structured grid generation (uniform and stretched)
- **Pluggable solver architecture** - easily switch between different numerical methods
- Multiple solver implementations:
  - Explicit Euler (basic and SIMD-optimized)
  - Projection Method (Chorin's method with Poisson pressure solver)
  - GPU-accelerated solvers (CUDA) with automatic CPU fallback
- **Output registry system** - flexible CSV and VTK output management
- Support for various boundary conditions
- VTK and CSV output formats for visualization and analysis
- Multiple example programs demonstrating different use cases
- Memory-efficient implementation with function pointer dispatch
- Zero-branch dispatch for optimal CPU performance
- Comprehensive build automation
- Unity testing framework integration

## Available Solvers

| Solver | Type | Description | Best For |
|--------|------|-------------|----------|
| `explicit_euler` | CPU | Basic finite difference solver | Learning, debugging |
| `explicit_euler_optimized` | CPU+SIMD | AVX2/FMA optimized | Large grids on modern CPUs |
| `projection` | CPU | Chorin's projection method | Accurate pressure-velocity coupling |
| `projection_optimized` | CPU+SIMD | SIMD-optimized projection | Production simulations |
| `explicit_euler_gpu` | GPU | CUDA-accelerated Euler solver | Very large grids with GPU |
| `projection_jacobi_gpu` | GPU | CUDA-accelerated projection | High-performance simulations |

## Project Structure

```
.
├── CMakeLists.txt              # Main CMake configuration
├── build.sh                    # Build automation script
├── lib/                        # CFD Library
│   ├── CMakeLists.txt          # Library CMake configuration
│   ├── include/                # Public headers
│   │   ├── grid.h              # Grid structures
│   │   ├── solver_interface.h  # Pluggable solver API (types, interface, utils)
│   │   ├── solver_gpu.h        # GPU solver configuration
│   │   ├── simulation_api.h    # High-level simulation API
│   │   ├── output_registry.h   # Output management system
│   │   ├── vtk_output.h        # VTK output functions
│   │   ├── csv_output.h        # CSV output functions
│   │   └── utils.h             # Utilities and memory management
│   └── src/                    # Library source code (organized by function)
│       ├── core/               # Core utilities and data structures
│       │   ├── grid.c
│       │   └── utils.c
│       ├── io/                 # Input/output systems
│       │   ├── vtk_output.c
│       │   ├── csv_output.c
│       │   ├── vtk_output_internal.h
│       │   └── csv_output_internal.h
│       ├── solvers/            # Numerical solvers
│       │   ├── cpu/            # CPU-optimized solvers
│       │   │   ├── solver_explicit_euler.c           # Basic Euler
│       │   │   └── solver_projection.c               # Projection method
│       │   ├── simd/           # SIMD-optimized solvers
│       │   │   ├── solver_explicit_euler_simd.c      # AVX2 Euler
│       │   │   └── solver_projection_simd.c          # AVX2 projection
│       │   └── gpu/            # GPU/CUDA solvers
│       │       ├── solver_projection_jacobi_gpu.cu   # CUDA implementation
│       │       └── solver_gpu_stub.c                 # CPU fallback
│       └── api/                # Public API implementation
│           ├── simulation_api.c
│           ├── solver_registry.c
│           └── output_registry.c
├── examples/                   # Example programs
│   ├── minimal_example.c       # Simplest usage example
│   ├── basic_simulation.c      # Complete simulation workflow
│   ├── solver_selection.c      # Demonstrates solver switching
│   ├── performance_comparison.c
│   └── custom_boundary_conditions.c
├── tests/                      # Unit tests
│   ├── test_simulation_basic.c
│   └── test_runner.c
├── output/                     # VTK output files
├── visualization/              # Visualization scripts
│   ├── visualize_cfd.py        # Main visualization script
│   ├── enhanced_visualize.py   # Advanced visualizations
│   ├── simple_viz.py           # Simple VTK plotter
│   └── run_visualization.py    # Automated workflow
└── README.md                   # This file
```

## Building the Project

### Prerequisites

- CMake (version 3.10 or higher)
- C compiler (GCC, Clang, or MSVC)
- Make or Ninja build system
- Python 3.x (for visualization)
- CUDA Toolkit (optional, for GPU acceleration)

### Quick Start with Build Script

The project includes a comprehensive build script that handles all common operations:

```bash
# Build the project (CPU only)
./build.sh build

# Build with CUDA GPU support
cmake -DCFD_ENABLE_CUDA=ON -B build
cmake --build build --config Release

# Run all examples
./build.sh run

# Build with tests
./build.sh build-tests

# Run tests
./build.sh test

# Complete clean build with tests
./build.sh full

# Clean build directory
./build.sh clean

# Show project status
./build.sh status

# Show help
./build.sh help
```

### Manual Build Instructions

1. Configure and build:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```

2. With CUDA support:
   ```bash
   mkdir build && cd build
   cmake -DCFD_ENABLE_CUDA=ON ..
   cmake --build .
   ```

3. Run examples:
   ```bash
   cd build/Debug  # or build/ on Linux/macOS
   ./minimal_example.exe
   ./basic_simulation.exe
   ./solver_selection.exe
   ```

## Usage

### Basic Usage

```c
#include "simulation_api.h"

// Initialize simulation with default solver
SimulationData* sim = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

// Run simulation steps
for (int step = 0; step < max_steps; step++) {
    run_simulation_step(sim);
}

// Output results
write_simulation_to_vtk(sim, "output.vtk");

// Cleanup
free_simulation(sim);
```

### Selecting a Specific Solver

```c
#include "simulation_api.h"
#include "solver_interface.h"

// Initialize with a specific solver
SimulationData* sim = init_simulation_with_solver(
    nx, ny, xmin, xmax, ymin, ymax,
    SOLVER_TYPE_PROJECTION_OPTIMIZED  // or any other solver type
);

// Run simulation
for (int step = 0; step < max_steps; step++) {
    run_simulation_step(sim);
}

free_simulation(sim);
```

### Dynamic Solver Switching

```c
#include "simulation_api.h"
#include "solver_interface.h"

SimulationData* sim = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

// Run some steps with default solver
for (int i = 0; i < 100; i++) {
    run_simulation_step(sim);
}

// Switch to a different solver at runtime
simulation_set_solver_by_name(sim, SOLVER_TYPE_PROJECTION);

// Continue with new solver
for (int i = 0; i < 100; i++) {
    run_simulation_step(sim);
}

free_simulation(sim);
```

### Listing Available Solvers

```c
#include "simulation_api.h"

const char* solver_names[10];
int num_solvers = simulation_list_solvers(solver_names, 10);

printf("Available solvers:\n");
for (int i = 0; i < num_solvers; i++) {
    printf("  %d. %s\n", i + 1, solver_names[i]);
}
```

### GPU Solver Configuration

```c
#include "solver_gpu.h"

// Get default GPU configuration
GPUConfig config = gpu_config_default();

// Customize settings
config.enable_gpu = 1;
config.min_grid_size = 10000;   // Minimum grid points for GPU usage
config.min_steps = 10;          // Minimum steps to use GPU
config.poisson_max_iter = 1000; // Poisson solver iterations
config.verbose = 1;             // Print GPU info

// Check if GPU should be used
if (gpu_should_use(&config, nx, ny, num_steps)) {
    printf("Using GPU acceleration\n");
}

// GPU solver will automatically fall back to CPU if:
// - CUDA is not available
// - Grid is too small
// - Too few simulation steps
```

### Example Programs

The `examples/` directory contains several programs demonstrating library usage:

#### 1. **minimal_example.c** - Best starting point
The simplest possible example showing basic library usage:
- Initialize simulation with high-level API
- Run simulation steps
- Output VTK files
- Clean up resources

```bash
cd build/Release && ./minimal_example.exe
```
**Output:** Creates `artifacts/output/minimal_step_*.vtk` files

#### 2. **basic_simulation.c** - Complete workflow
A comprehensive example showing:
- Full simulation setup with parameters
- Iterative solving with periodic output
- Production-ready structure

```bash
cd build/Release && ./basic_simulation.exe
```
**Output:** Creates `artifacts/output/output_optimized_*.vtk` files

#### 3. **solver_selection.c** - Demonstrates solver switching
Shows how to:
- List all available solvers
- Switch between solvers at runtime
- Compare solver performance

```bash
cd build/Release && ./solver_selection.exe
```

#### 4. **performance_comparison.c** - Benchmarking
Compares performance across:
- Different solvers (basic vs optimized)
- Grid sizes
- Execution time measurement
- Memory usage analysis

```bash
cd build/Release && ./performance_comparison.exe
```

#### 5. **runtime_comparison.c** - CPU vs GPU benchmarks
Comprehensive CUDA vs SIMD performance benchmarking:
- Tests multiple grid sizes
- Iteration count scaling
- Crossover analysis for GPU efficiency

```bash
cd build/Release && ./runtime_comparison.exe
```

#### 6. **custom_boundary_conditions.c** - Advanced scenarios
Demonstrates:
- Custom inlet/outlet conditions
- No-slip wall boundaries
- Flow around obstacles (cylinder)
- Complex geometry handling

```bash
cd build/Release && ./custom_boundary_conditions.exe
```
**Output:** Creates `artifacts/output/cylinder_flow_*.vtk` files

### Output and Visualization

All examples generate VTK files in the `output/` directory. These can be visualized using:

#### Python Visualization (Recommended)
```bash
# Install dependencies
pip install matplotlib numpy

# Create visualizations
python visualization/simple_viz.py
python visualization/visualize_cfd.py
```

#### External Tools
- **ParaView** - Professional CFD visualization
- **VisIt** - Scientific visualization
- **Mayavi** - Python-based 3D visualization

## Library Source Code Organization

The library source code in `lib/src/` is organized by functionality for clarity:

### Core Components (`lib/src/core/`)
- **grid.c** - Grid generation and management
- **utils.c** - Memory management, file I/O utilities

### I/O Systems (`lib/src/io/`)

Output file generation and management:

- **vtk_output.c** - VTK file format export for visualization
- **csv_output.c** - CSV data export for analysis
- **vtk_output_internal.h** - Internal VTK utilities
- **csv_output_internal.h** - Internal CSV utilities

### CPU Solvers (`lib/src/solvers/cpu/`)

Basic CPU implementations:

- **solver_explicit_euler.c** - Basic explicit Euler method
- **solver_projection.c** - Projection method (Chorin's algorithm) with SOR

### SIMD Solvers (`lib/src/solvers/simd/`)

SIMD-optimized implementations for modern CPUs:

- **solver_explicit_euler_simd.c** - AVX2/FMA-optimized explicit Euler
- **solver_projection_simd.c** - AVX2/FMA-optimized projection method

### GPU Solvers (`lib/src/solvers/gpu/`)

CUDA-accelerated solvers for NVIDIA GPUs:

- **solver_projection_jacobi_gpu.cu** - CUDA implementation with Jacobi Poisson solver
- **solver_gpu_stub.c** - CPU fallback when CUDA is not available

### API Layer (`lib/src/api/`)

Public API implementations with zero-branch dispatch:

- **simulation_api.c** - High-level simulation API
- **solver_registry.c** - Solver registration and factory functions
- **output_registry.c** - Output format registration and management

This organization makes it easy to:
- Find CPU vs SIMD vs GPU implementations
- Identify optimization strategies by folder
- Understand the separation between core utilities, solvers, and I/O
- Add new solvers in the appropriate category
- Maintain clean public/private API boundaries

## Solver Details

### Explicit Euler Solvers

The explicit Euler solvers solve the incompressible Navier-Stokes equations using finite differences:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
∇·u = 0
```

- **explicit_euler**: Basic implementation, good for learning
- **explicit_euler_optimized**: Uses AVX2/FMA SIMD instructions for ~2-3x speedup

### Projection Method Solvers

The projection method (Chorin's method) properly enforces the incompressibility constraint:

1. **Predictor**: Compute intermediate velocity ignoring pressure
   ```
   u* = uⁿ + dt * (-u·∇u + ν∇²u)
   ```

2. **Pressure Solve**: Solve Poisson equation
   ```
   ∇²p = (ρ/dt) * ∇·u*
   ```

3. **Corrector**: Project velocity to be divergence-free
   ```
   uⁿ⁺¹ = u* - (dt/ρ) * ∇p
   ```

The Poisson equation is solved using Successive Over-Relaxation (SOR) with red-black Gauss-Seidel ordering.

### GPU Solvers

GPU solvers use CUDA for acceleration with:
- Structure of Arrays (SoA) data layout for coalesced memory access
- Shared memory tiling for stencil operations
- Red-black SOR Poisson solver
- Automatic fallback to CPU when GPU is unavailable or inefficient

GPU usage is automatically determined based on:
- Grid size (default: ≥10,000 points)
- Number of simulation steps (default: ≥10 steps)
- CUDA availability

## Performance Benchmarks

The framework includes multiple solver implementations. Performance comparison results (Release mode):

### Solver Performance Comparison (100x50 grid, 50 steps)

| Solver | Time (ms) | Capabilities |
|--------|-----------|--------------|
| explicit_euler | 2.64 | Basic |
| explicit_euler_optimized | 8.90 | SIMD |
| projection | 19.04 | Accurate pressure |
| projection_optimized | 5.34 | SIMD + Accurate |
| explicit_euler_gpu | 8.42 | GPU (fallback) |
| projection_jacobi_gpu | 23.15 | GPU (fallback) |

### Key Insights:
- **Small grids**: CPU solvers are typically faster due to transfer overhead
- **Large grids (≥200x200)**: GPU solvers provide significant speedup
- **Projection method**: More accurate pressure-velocity coupling at higher cost
- **SIMD optimization**: 2-3x speedup on large grids

### Running Benchmarks:
```bash
# Build in Release mode for accurate performance measurement
CMAKE_BUILD_TYPE=Release ./build.sh build

# Run solver comparison
cd build/Release && ./solver_selection.exe

# Run performance comparison
cd build/Release && ./performance_comparison.exe
```

## API Reference

### Simulation API

```c
// Create simulation
SimulationData* init_simulation(size_t nx, size_t ny,
                                double xmin, double xmax,
                                double ymin, double ymax);

SimulationData* init_simulation_with_solver(size_t nx, size_t ny,
                                            double xmin, double xmax,
                                            double ymin, double ymax,
                                            const char* solver_type);

// Run simulation
void run_simulation_step(SimulationData* sim);
void run_simulation_solve(SimulationData* sim);

// Solver management
int simulation_set_solver_by_name(SimulationData* sim, const char* solver_type);
Solver* simulation_get_solver(SimulationData* sim);
const SolverStats* simulation_get_stats(const SimulationData* sim);

// Solver discovery
int simulation_list_solvers(const char** names, int max_count);
int simulation_has_solver(const char* solver_type);

// Output
void write_simulation_to_vtk(SimulationData* sim, const char* filename);
void write_velocity_vectors_to_vtk(SimulationData* sim, const char* filename);
void write_flow_field_to_vtk(SimulationData* sim, const char* filename);

// Cleanup
void free_simulation(SimulationData* sim);
```

### Solver Type Constants

```c
// Explicit Euler family
#define SOLVER_TYPE_EXPLICIT_EULER           "explicit_euler"
#define SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED "explicit_euler_optimized"

// Projection method family
#define SOLVER_TYPE_PROJECTION               "projection"
#define SOLVER_TYPE_PROJECTION_OPTIMIZED     "projection_optimized"

// GPU-accelerated
#define SOLVER_TYPE_EXPLICIT_EULER_GPU       "explicit_euler_gpu"
#define SOLVER_TYPE_PROJECTION_JACOBI_GPU           "projection_jacobi_gpu"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Adding a New Solver

1. Create solver implementation in the appropriate directory:
   - CPU implementation: `lib/src/solvers/cpu/solver_yourname.c`
   - SIMD optimization: `lib/src/solvers/simd/solver_yourname_simd.c`
   - GPU implementation: `lib/src/solvers/gpu/solver_yourname_gpu.cu`
2. Implement the internal function (e.g., `yourname_impl`) with signature:

   ```c
   void yourname_impl(FlowField* field, const Grid* grid, const SolverParams* params);
   ```

3. Add factory function and register in `lib/src/api/solver_registry.c`
4. Add type constant to `lib/include/solver_interface.h`
5. Update `lib/CMakeLists.txt` to include new source file
6. Add tests in `tests/` and update examples as needed
7. Update README.md with solver description and benchmarks

**Note**: Internal solver implementations use descriptive names (e.g., `explicit_euler_impl`, `projection_impl`) and are not exposed in the public API. The public API uses the modern `Solver` interface with function pointers for zero-branch dispatch.
