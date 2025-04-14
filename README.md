# CFD Framework

A computational fluid dynamics (CFD) framework implemented in C. This project provides a foundation for solving fluid dynamics problems using numerical methods.

## Features

- 2D structured grid generation (uniform and stretched)
- Navier-Stokes equations solver
- Support for various boundary conditions
- Memory-efficient implementation
- Thread-safe design

## Project Structure

```
.
├── CMakeLists.txt          # Main CMake configuration
├── src/                    # Source code directory
│   ├── CMakeLists.txt     # Source-specific CMake configuration
│   ├── main.c             # Main program entry point
│   ├── grid.h             # Grid generation and management
│   ├── grid.c
│   ├── solver.h           # CFD solver implementation
│   ├── solver.c
│   ├── utils.h            # Utility functions
│   └── utils.c
└── README.md              # This file
```

## Building the Project

### Prerequisites

- CMake (version 3.10 or higher)
- C compiler (GCC, Clang, or MSVC)
- Make or Ninja build system

### Build Instructions

1. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

2. Configure the project:
   ```bash
   cmake ..
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

## Usage

The framework can be used to solve various fluid dynamics problems. Example usage will be provided in the documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.