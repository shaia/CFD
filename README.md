# CFD Framework

A computational fluid dynamics (CFD) framework implemented in C. This project provides a foundation for solving fluid dynamics problems using numerical methods, with VTK output for visualization.

## Features

- 2D structured grid generation (uniform and stretched)
- Navier-Stokes equations solver
- Support for various boundary conditions
- VTK output format for visualization
- Multiple example programs demonstrating different use cases
- Memory-efficient implementation
- Thread-safe design
- Comprehensive build automation
- Unity testing framework integration

## Project Structure

```
.
├── CMakeLists.txt              # Main CMake configuration
├── build.sh                   # Build automation script
├── lib/                       # CFD Library
│   ├── CMakeLists.txt         # Library CMake configuration
│   ├── include/               # Public headers
│   │   ├── grid.h
│   │   ├── solver.h
│   │   ├── simulation_api.h
│   │   ├── utils.h
│   │   └── vtk_output.h
│   └── src/                   # Library source code
│       ├── grid.c
│       ├── solver.c
│       ├── solver_optimized.c
│       ├── simulation_api.c
│       ├── utils.c
│       └── vtk_output.c
├── examples/                  # Example programs
│   ├── minimal_example.c      # Simplest usage example
│   ├── basic_simulation.c     # Complete simulation workflow
│   ├── performance_comparison.c
│   └── custom_boundary_conditions.c
├── tests/                     # Unit tests
│   ├── test_simulation_basic.c
│   └── test_runner.c
├── output/                    # VTK output files
├── visualization/             # Visualization scripts
│   ├── visualize_cfd.py      # Main visualization script
│   ├── enhanced_visualize.py # Advanced visualizations
│   ├── simple_viz.py         # Simple VTK plotter
│   └── run_visualization.py  # Automated workflow
└── README.md                 # This file
```

## Building the Project

### Prerequisites

- CMake (version 3.10 or higher)
- C compiler (GCC, Clang, or MSVC)
- Make or Ninja build system
- Python 3.x (for visualization)

### Quick Start with Build Script

The project includes a comprehensive build script that handles all common operations:

```bash
# Build the project
./build.sh build

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

2. Run examples:
   ```bash
   cd build/Debug  # or build/ on Linux/macOS
   ./minimal_example.exe
   ./basic_simulation.exe
   ```

## Usage

### Example Programs

1. **Minimal Example** - Simplest usage of the CFD library
   ```bash
   ./build.sh run  # Runs all examples
   # or manually:
   cd build/Debug && ./minimal_example.exe
   ```

2. **Basic Simulation** - Complete CFD simulation workflow
   ```bash
   cd build/Debug && ./basic_simulation.exe
   ```

3. **Performance Comparison** - Benchmarking different solvers
   ```bash
   cd build/Debug && ./performance_comparison.exe
   ```

4. **Custom Boundary Conditions** - Advanced flow scenarios
   ```bash
   cd build/Debug && ./custom_boundary_conditions.exe
   ```

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

### Library Integration

To use the CFD library in your own project:

```c
#include "simulation_api.h"

// Initialize simulation
SimulationData* sim = init_simulation(nx, ny, xmin, xmax, ymin, ymax);

// Run simulation steps
for (int step = 0; step < max_steps; step++) {
    run_simulation_step(sim);

    // Output results
    char filename[256];
    snprintf(filename, sizeof(filename), "output/result_%d.vtk", step);
    write_simulation_to_vtk(sim, filename);
}

// Cleanup
free_simulation(sim);
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.