# CFD Library

A high-performance Computational Fluid Dynamics (CFD) library written in C, featuring both basic and optimized solvers with SIMD acceleration.

## Features

- **Grid Management**: Flexible grid creation and boundary condition handling
- **Multiple Solvers**: Basic and SIMD-optimized CFD solvers
- **VTK Output**: Standard VTK file output for visualization
- **High-Level API**: Simple simulation interface for easy integration
- **Cross-Platform**: Supports Windows, Linux, and macOS
- **Optimized**: AVX2/FMA acceleration when available

## Library Components

### Headers (`include/`)
- `simulation_api.h` - High-level simulation interface (recommended)
- `grid.h` - Grid management and boundary conditions
- `solver.h` - Direct solver access (basic and optimized)
- `vtk_output.h` - VTK file output functionality
- `utils.h` - Utility functions and memory management

### Source Files (`src/`)
- Core implementation files (compiled into static library)

## Usage

### CMake Integration

Add this library to your CMake project:

```cmake
# Add the CFD library subdirectory
add_subdirectory(path/to/cfd/lib)

# Link against the library
target_link_libraries(your_target PRIVATE CFD::Library)
```

### Basic Usage Example

```c
#include "simulation_api.h"

int main() {
    // Initialize simulation (100x50 grid, domain [0,1] x [0,0.5])
    SimulationData* sim = init_simulation(100, 50, 0.0, 1.0, 0.0, 0.5);

    // Run simulation steps
    for (int i = 0; i < 1000; i++) {
        run_simulation_step(sim);

        // Output every 100 steps
        if (i % 100 == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "artifacts/output/result_%d.vtk", i);
            write_simulation_to_vtk(sim, filename);
        }
    }

    // Cleanup
    free_simulation(sim);
    return 0;
}
```

### Advanced Usage

For more control, use the lower-level solver interface:

```c
#include "grid.h"
#include "solver.h"
#include "vtk_output.h"

int main() {
    // Create grid and flow field
    Grid* grid = grid_create(100, 50, 0.0, 1.0, 0.0, 0.5);
    FlowField* field = flow_field_create(100, 50);

    // Initialize solver parameters
    SolverParams params = {
        .max_iter = 1000,
        .dt = 0.001,
        .Re = 100.0
    };

    // Run optimized solver
    solve_flow_optimized(field, grid, &params);

    // Output results
    write_vtk_output("artifacts/output/result.vtk", "pressure", field->p,
                     field->nx, field->ny,
                     grid->xmin, grid->xmax,
                     grid->ymin, grid->ymax);

    // Cleanup
    flow_field_destroy(field);
    grid_destroy(grid);
    return 0;
}
```

## Building the Library

### Standalone Build

```bash
cd lib
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### As Part of Main Project

```bash
# From project root
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Installation

To install the library system-wide:

```bash
cd lib/build
cmake --install . --prefix /usr/local
```

After installation, other projects can find the library using:

```cmake
find_package(cfd_library REQUIRED)
target_link_libraries(your_target PRIVATE CFD::Library)
```

## Performance Notes

- The optimized solver uses AVX2/FMA instructions when available
- Best performance on systems with AVX2 support
- Fallback implementations for older hardware
- Memory-aligned data structures for SIMD efficiency

## Compatibility

- **C Standard**: C11 or later
- **CMake**: 3.10 or later
- **Compilers**: GCC, Clang, MSVC
- **Platforms**: Windows, Linux, macOS

## License

[Add your license information here]