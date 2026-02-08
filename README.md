<p align="center">
  <img src="https://raw.githubusercontent.com/shaia/CFD/master/assets/cfd-logo-nbg.png" alt="CFD Logo" width="200">
</p>

# CFD Framework

A production-grade computational fluid dynamics (CFD) library in C for solving 2D incompressible Navier-Stokes equations.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Version](https://img.shields.io/badge/version-0.1.x-orange)]()

## Features

- 🚀 **Multiple Backends**: CPU (scalar), SIMD (AVX2/NEON), OpenMP, CUDA
- 🔧 **Pluggable Solvers**: Explicit Euler, Projection Method (Chorin's algorithm)
- 📊 **Linear Solvers**: Jacobi, SOR, Red-Black SOR, CG/PCG, BiCGSTAB
- 🎯 **Validated**: Ghia lid-driven cavity, Taylor-Green vortex benchmarks
- 📈 **VTK/CSV Output**: Ready for ParaView, VisIt visualization
- ⚡ **Performance**: SIMD-optimized with runtime CPU detection

## Quick Start

### Prerequisites

- CMake 3.10+
- C compiler (GCC, Clang, MSVC)
- CUDA Toolkit (optional, for GPU)

### Build & Run

```bash
# Clone and build
git clone https://github.com/yourusername/cfd.git
cd cfd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run example
cd build/Release
./minimal_example

# Visualize (requires Python + matplotlib)
python ../../visualization/simple_viz.py
```

### Windows Quick Build

```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

### Linux/macOS Quick Build

```bash
./build.sh build    # Build project
./build.sh test     # Run tests
./build.sh run      # Run examples
```

## Basic Usage

```c
#include "cfd/api/simulation_api.h"

int main(void) {
    // Initialize library
    cfd_status_t status = cfd_init();

    // Create simulation (100x50 grid)
    simulation* sim = simulation_create(100, 50, 0.0, 1.0, 0.0, 0.5);

    // Run 1000 steps
    int steps_taken;
    status = run_simulation_solve(sim, 1.0, &steps_taken);
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Error: %s\n", cfd_get_last_error());
        return 1;
    }

    // Export results
    write_simulation_to_vtk(sim, "output/result.vtk");

    // Cleanup
    simulation_destroy(sim);
    cfd_cleanup();

    return 0;
}
```

## Available Solvers

| Solver | Backend | Description |
|--------|---------|-------------|
| `euler_cpu` | Scalar | Basic explicit Euler |
| `euler_avx2` | SIMD | AVX2-optimized Euler |
| `euler_omp` | OpenMP | Multi-threaded Euler |
| `projection_cpu` | Scalar | Chorin's projection method |
| `projection_avx2` | SIMD | SIMD-optimized projection |
| `projection_omp` | OpenMP | Multi-threaded projection |
| `projection_cuda` | GPU | CUDA-accelerated projection |

## Project Structure

```
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
├── docs/                   # Documentation
└── visualization/          # Python plotting scripts
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

### 2. Lid-Driven Cavity
```bash
./build/Release/lid_driven_cavity 100
```
Classic CFD benchmark validated against Ghia et al. (1982).

### 3. Custom Boundary Conditions
```bash
./build/Release/custom_boundary_conditions
```
Flow around cylinder with complex geometry.

See [examples documentation](docs/guides/examples.md) for more details.

## Performance

Typical performance (100x50 grid, 50 steps, Release mode):

| Solver | Time | Speedup |
|--------|------|---------|
| euler_cpu | 2.6ms | 1.0x |
| euler_avx2 | 0.9ms | 2.9x |
| projection_cpu | 19.0ms | 1.0x |
| projection_avx2 | 5.3ms | 3.6x |

GPU acceleration shows significant benefits for grids ≥200x200.

## Testing

```bash
# Run all tests
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure

# Run specific test category
ctest --test-dir build -C Debug -R "Validation" --output-on-failure
```

58 tests covering:
- Unit tests for core functionality
- Solver accuracy and convergence
- Physics validation benchmarks
- Cross-architecture consistency

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{cfd_framework,
  title = {CFD Framework: A Modular C Library for Computational Fluid Dynamics},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cfd}
}
```

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/cfd/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/cfd/discussions)

## Acknowledgments

- Validated against Ghia et al. (1982) lid-driven cavity benchmark
- Unity testing framework
- Inspiration from Ferziger & Peric's "Computational Methods for Fluid Dynamics"
