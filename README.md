# CFD Framework

![CFD Logo](https://raw.githubusercontent.com/shaia/CFD/master/assets/cfd-logo-nbg.png)

A production-grade computational fluid dynamics (CFD) library in C for solving 2D/3D incompressible Navier-Stokes equations.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/shaia/CFD/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.x-orange)](https://github.com/shaia/CFD/releases)

## Features

- 🚀 **Multiple Backends**: CPU (scalar), SIMD (AVX2/NEON), OpenMP, CUDA
- 🔧 **Pluggable Solvers**: Explicit Euler, RK2 (Heun), Projection Method (Chorin's algorithm)
- 📊 **Linear Solvers**: Jacobi, SOR, Red-Black SOR, CG/PCG, BiCGSTAB
- 🌡️ **Heat Transfer**: Energy equation (advection–diffusion) + Boussinesq buoyancy + thermal BCs
- 🎯 **Validated**: Ghia lid-driven cavity, Taylor-Green vortex, Poiseuille flow, natural convection benchmarks
- 📈 **VTK/CSV Output**: Ready for ParaView, VisIt visualization
- ⚡ **Performance**: SIMD-optimized with runtime CPU detection
- 🌐 **3D Support**: Full 3D simulations with nz>1, branch-free 2D compatibility

## Quick Start

### Prerequisites

- CMake 3.10+
- C compiler (GCC, Clang, MSVC)
- CUDA Toolkit (optional, for GPU)

### Build & Run

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run example
cd build/Release
./minimal_example

# Visualize with ParaView or VisIt
# VTK files are written to output/ directory
```

### Windows Quick Build

```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

### Windows CUDA builds

Use the CUDA configure preset, then build and test as usual:

```powershell
cmake --preset windows-msvc-cuda     # configure with CUDA enabled
cmake --build build --config Debug    # or Release
ctest --test-dir build -C Debug --output-on-failure
```

**Troubleshooting — `nvcc` build dies with a swallowed `exit 1` and no diagnostic:**
when CUDA is enabled, `nvcc` runs with `--use-local-env` and spawns a `cmd /c` subprocess to
set up the MSVC host-compiler environment. A very long inherited `PATH` (e.g. ~7000+ chars)
overflows `cmd.exe`'s ~8191-char limit and that subprocess dies silently. If you hit this,
prune your user `PATH` once — dedupe entries and remove non-existent directories — so it stays
well under the limit, then re-run the build.

### Linux/macOS Quick Build

```bash
./build.sh build    # Build project
./build.sh test     # Run tests
./build.sh run      # Run examples
```

## Basic Usage

```c
#include "cfd/api/simulation_api.h"
#include "cfd/io/vtk_output.h"

int main(void) {
    // Initialize library
    cfd_status_t status = cfd_init();
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Init failed: %s\n", cfd_get_last_error());
        return 1;
    }

    // Create 2D simulation (100x50 grid, domain [0,1] x [0,0.5])
    // For 3D: use nz>1 and set z-range, e.g. init_simulation(64, 64, 64, ...)
    simulation_data* sim = init_simulation(100, 50, 1, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0);
    if (!sim) {
        fprintf(stderr, "Failed to create simulation\n");
        return 1;
    }

    // Configure parameters
    sim->params.dt = 0.001;
    sim->params.mu = 0.01;  // Viscosity

    // Run simulation steps
    for (int step = 0; step < 1000; step++) {
        status = run_simulation_step(sim);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Step failed: %s\n", cfd_get_last_error());
            break;
        }
    }

    // Export results to VTK
    grid* g = sim->grid;
    write_vtk_flow_field("output/result.vtk", sim->field,
                         g->nx, g->ny, g->nz,
                         g->xmin, g->xmax, g->ymin, g->ymax, g->zmin, g->zmax);

    // Cleanup
    free_simulation(sim);
    cfd_finalize();

    return 0;
}
```

## Available Solvers

| Solver | Backend | Description |
| ------ | ------- | ----------- |
| `explicit_euler` | Scalar | Basic explicit Euler |
| `explicit_euler_optimized` | SIMD | SIMD-optimized Euler (AVX2/NEON) |
| `explicit_euler_omp` | OpenMP | Multi-threaded Euler |
| `projection` | Scalar | Chorin's projection method |
| `projection_optimized` | SIMD | SIMD-optimized projection (AVX2/NEON) |
| `projection_omp` | OpenMP | Multi-threaded projection |
| `projection_gpu` | GPU | CUDA-accelerated projection (CG pressure solve) |
| `rk2` | Scalar | 2nd-order Runge-Kutta (Heun) |
| `rk2_optimized` | SIMD | SIMD-optimized RK2 (AVX2/NEON) |
| `rk2_omp` | OpenMP | Multi-threaded RK2 |
| `rk2_gpu` | GPU | CUDA-accelerated RK2 |
| `rk4` | Scalar | 4th-order Runge-Kutta (classical) |
| `rk4_optimized` | SIMD | SIMD-optimized RK4 (AVX2/NEON) |
| `rk4_omp` | OpenMP | Multi-threaded RK4 |
| `rk4_gpu` | GPU | CUDA-accelerated RK4 |

## Project Structure

```text
.
├── lib/                    # CFD Library
│   ├── include/cfd/        # Public headers
│   └── src/                # Implementation
│       ├── core/           # Grid, memory, utilities
│       ├── solvers/        # Navier-Stokes solvers
│       │   ├── cpu/        # Scalar implementations
│       │   ├── simd/       # AVX2/NEON optimized
│       │   ├── omp/        # OpenMP parallelized
│       │   └── gpu/        # CUDA kernels
│       ├── linear/         # Poisson/linear solvers
│       └── api/            # Public API
├── tests/                  # Comprehensive test suite
├── examples/               # Example programs
└── docs/                   # Documentation
```

## Documentation

- **[Building & Installation](docs/guides/building.md)** - Detailed build instructions
- **[Architecture](docs/architecture/architecture.md)** - Design principles and patterns
- **[API Reference](docs/reference/api-reference.md)** - Complete API documentation
- **[Solvers](docs/reference/solvers.md)** - Numerical methods and performance
- **[Examples](docs/guides/examples.md)** - Example programs guide
- **[Validation](docs/validation/)** - Benchmark results

## Examples

### 1. Minimal Example

```bash
./build/Release/minimal_example
```

Simplest possible usage - 50 lines showing library basics.

### 2. Minimal 3D Example

```bash
./build/Release/minimal_example_3d
```

3D simulation on a 16×16×16 grid — demonstrates 3D API usage.

### 3. Lid-Driven Cavity

```bash
./build/Release/lid_driven_cavity 100
```

Classic CFD benchmark validated against Ghia et al. (1982).

### 4. Custom Boundary Conditions

```bash
./build/Release/custom_boundary_conditions
```

Flow around cylinder with complex geometry.

See [examples documentation](docs/guides/examples.md) for more details.

## Performance

Typical performance (100x50 grid, 50 steps, Release mode):

| Solver | Time | Speedup |
| ------ | ---- | ------- |
| explicit_euler | 2.6ms | 1.0x |
| explicit_euler_optimized | 0.9ms | 2.9x |
| projection | 19.0ms | 1.0x |
| projection_optimized | 5.3ms | 3.6x |

GPU acceleration shows significant benefits for grids ≥200x200.

## Testing

```bash
# Run all tests
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

# Run specific test category
ctest --test-dir build -C Debug -R "Validation" --output-on-failure
```

60+ tests covering:

- Unit tests for core functionality
- Solver accuracy and convergence
- Physics validation benchmarks
- Cross-architecture consistency

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [development guidelines](.claude/CLAUDE.md) and [ROADMAP](ROADMAP.md) for current priorities.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{cfd_framework,
  title = {CFD Framework: A Modular C Library for Computational Fluid Dynamics},
  author = {Shaia Halevy},
  year = {2026},
  url = {https://github.com/shaia/CFD}
}
```

## Support

For questions, bug reports, or feature requests, please:

- Check existing [documentation](docs/index.md)
- Review [examples](docs/guides/examples.md)
- See [architecture guide](docs/architecture/architecture.md) for design details

## Acknowledgments

- Validated against Ghia et al. (1982) lid-driven cavity benchmark
- Unity testing framework
- Inspiration from Ferziger & Peric's "Computational Methods for Fluid Dynamics"
