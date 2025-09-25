# CFD Library Examples

This directory contains example programs demonstrating various features of the CFD library. Each example is designed to showcase different aspects and use cases.

## Available Examples

### 1. `minimal_example.c`
**Best starting point for new users**

The simplest possible example showing basic library usage:
- Initialize simulation with high-level API
- Run a few simulation steps
- Output VTK files
- Clean up resources

**Run it:**
```bash
./minimal_example
```

**Output:** Creates `artifacts/output/minimal_step_*.vtk` files

---

### 2. `basic_simulation.c`
**Complete simulation workflow**

A more comprehensive example (original main.c) showing:
- Full simulation setup
- Parameter configuration
- Iterative solving with output
- Production-ready structure

**Run it:**
```bash
./basic_simulation
```

**Output:** Creates `artifacts/output/output_optimized_*.vtk` files every 100 iterations

---

### 3. `performance_comparison.c`
**Benchmarking different solvers**

Demonstrates performance differences between:
- Basic vs. optimized solvers
- Different grid sizes
- Execution time measurement
- Memory usage analysis

**Run it:**
```bash
./performance_comparison
```

**Output:** Console performance metrics for different configurations

---

### 4. `custom_boundary_conditions.c`
**Advanced boundary condition setup**

Shows how to implement:
- Custom inlet/outlet conditions
- No-slip wall boundaries
- Flow around obstacles (cylinder)
- Complex geometry handling

**Run it:**
```bash
./custom_boundary_conditions
```

**Output:** Creates `artifacts/output/cylinder_flow_*.vtk` files showing flow patterns

---

## Building Examples

### Build All Examples (Default)
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Skip Examples
```bash
cmake -DBUILD_EXAMPLES=OFF ..
```

### Build Specific Example
```bash
# Build only minimal_example
cmake --build . --target minimal_example
```

## Running Examples

After building, executables are located in:
- **Windows:** `build/Debug/` or `build/Release/`
- **Linux/macOS:** `build/`

```bash
# Run from project root directory
cd build
./minimal_example
./basic_simulation
./performance_comparison
./custom_boundary_conditions
```

## Example Output

All examples create VTK files in the `output/` directory. These files can be visualized using:

- **ParaView** (recommended): Professional visualization tool
- **VisIt**: Open-source scientific visualization
- **Python + matplotlib**: For quick plots and analysis
- **Project visualization scripts**: Located in `visualization/` directory

## Learning Path

**Recommended order for learning:**

1. **Start with `minimal_example.c`** - Learn basic API
2. **Study `basic_simulation.c`** - Understand full workflow
3. **Try `performance_comparison.c`** - Learn about optimization
4. **Explore `custom_boundary_conditions.c`** - Advanced techniques

## Modifying Examples

Each example is self-contained and can be easily modified:

```c
// Change grid resolution
SimulationData* sim = init_simulation(100, 50, 0.0, 1.0, 0.0, 0.5);
//                                    ↑    ↑   ↑grid size and domain

// Adjust solver parameters
SolverParams params = {
    .max_iter = 1000,  // Number of iterations
    .dt = 0.001,       // Time step
    .Re = 100.0        // Reynolds number
};

// Change output frequency
if (iter % 50 == 0) {  // Output every 50 steps instead of 100
    // ... save VTK file
}
```

## Common Issues

**Build errors:**
- Ensure CFD library built successfully first
- Check that all header files are found
- Verify CMake version ≥ 3.10

**Runtime errors:**
- Create `output/` directory manually if it doesn't exist
- Check file permissions for VTK output
- Ensure sufficient memory for large grids

**Performance issues:**
- Use optimized solver for large grids
- Reduce grid size for initial testing
- Enable compiler optimizations (`-O3` or `/O2`)

## Adding New Examples

To add your own example:

1. Create `examples/my_example.c`
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(my_example examples/my_example.c)
   target_link_libraries(my_example PRIVATE CFD::Library)
   ```
3. Include basic structure:
   ```c
   #include "simulation_api.h"  // or specific headers

   int main() {
       // Your CFD code here
       return 0;
   }
   ```

## Next Steps

After running examples:
- Examine the generated VTK files
- Try the visualization scripts in `visualization/`
- Read the library documentation in `lib/README.md`
- Explore the test cases in `tests/`