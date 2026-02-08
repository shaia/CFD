# Building & Installation

Complete guide to building the CFD Framework on Windows, Linux, and macOS.

## Prerequisites

### Required

- **CMake** 3.10 or higher
- **C Compiler**:
  - Windows: MSVC 2019+ or MinGW-w64
  - Linux: GCC 7+ or Clang 6+
  - macOS: Clang (via Xcode Command Line Tools)

### Optional

- **CUDA Toolkit** 11.0+ (for GPU acceleration)
- **OpenMP** (usually included with compiler)
- **Python 3.7+** (for visualization scripts)
  - matplotlib
  - numpy

## Quick Start

### Windows

```cmd
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --config Debug

# Test
ctest --test-dir build -C Debug --output-on-failure
```

### Linux / macOS

```bash
# Using build script (recommended)
./build.sh build       # Configure and build
./build.sh test        # Run tests
./build.sh run         # Run examples

# Manual build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
ctest --output-on-failure
```

## Build Options

### CMake Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Debug` | Build type: Debug, Release, RelWithDebInfo |
| `BUILD_SHARED_LIBS` | `OFF` | Build shared libraries instead of static |
| `CFD_ENABLE_CUDA` | `OFF` | Enable CUDA GPU support |
| `CFD_CUDA_ARCHITECTURES` | Auto | CUDA compute capabilities to build for |
| `CFD_ENABLE_TESTING` | `ON` | Build test suite |
| `CFD_ENABLE_EXAMPLES` | `ON` | Build example programs |

### Example Configurations

#### Release Build (Optimized)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

#### Debug Build with Tests
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCFD_ENABLE_TESTING=ON
cmake --build build --config Debug
ctest --test-dir build -C Debug
```

#### Shared Libraries
```bash
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build
```

#### CUDA GPU Support
```bash
# Auto-detect GPU architecture
cmake -B build -DCFD_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Specific GPU architectures
cmake -B build -DCFD_ENABLE_CUDA=ON -DCFD_CUDA_ARCHITECTURES="80;86;89"
cmake --build build --config Release
```

**CUDA Architecture Reference:**
| GPU Family | Compute Capability | CMake Value |
|------------|-------------------|-------------|
| Pascal (GTX 10xx) | 6.0, 6.1 | 60;61 |
| Volta (V100) | 7.0 | 70 |
| Turing (RTX 20xx) | 7.5 | 75 |
| Ampere (RTX 30xx, A100) | 8.0, 8.6 | 80;86 |
| Ada (RTX 40xx) | 8.9 | 89 |
| Hopper (H100) | 9.0 | 90 |

## Build Script Usage

The `build.sh` script (Linux/macOS) provides convenient build automation:

```bash
# Build commands
./build.sh build          # Configure and build (Release)
./build.sh debug          # Debug build
./build.sh clean          # Clean build directory
./build.sh rebuild        # Clean and rebuild

# Testing
./build.sh test           # Run all tests
./build.sh build-tests    # Build with tests enabled

# Examples
./build.sh run            # Run all examples
./build.sh examples       # Build examples only

# Combined workflows
./build.sh full           # Clean build with tests

# Information
./build.sh status         # Show build status
./build.sh help           # Show all commands
```

## Platform-Specific Notes

### Windows (MSVC)

1. **Visual Studio Integration:**
   ```cmd
   # Open in Visual Studio
   cmake -B build -G "Visual Studio 17 2022"
   start build\CFD.sln
   ```

2. **Multi-Config Build:**
   ```cmd
   cmake -B build
   cmake --build build --config Debug
   cmake --build build --config Release
   ```

3. **DLL Export:**
   - Static builds (default): No special handling needed
   - Shared builds: `CFD_LIBRARY_EXPORT` macro handles `__declspec(dllexport/dllimport)`

### Linux

1. **Install Dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential cmake git
   sudo apt-get install python3 python3-matplotlib  # For visualization
   ```

2. **Install Dependencies (Fedora/RHEL):**
   ```bash
   sudo dnf install gcc gcc-c++ cmake make git
   sudo dnf install python3 python3-matplotlib
   ```

3. **CUDA Support:**
   ```bash
   # Install CUDA Toolkit
   sudo apt-get install nvidia-cuda-toolkit  # Ubuntu
   # or download from NVIDIA: https://developer.nvidia.com/cuda-downloads

   # Build with CUDA
   cmake -B build -DCFD_ENABLE_CUDA=ON
   ```

### macOS

1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install CMake (via Homebrew):**
   ```bash
   brew install cmake
   ```

3. **SIMD Support:**
   - Intel Macs: AVX2 support via runtime detection
   - Apple Silicon: NEON support (ARM64 SIMD)

4. **Build:**
   ```bash
   ./build.sh build
   ```

## Testing

### Running Tests

```bash
# All tests
ctest --test-dir build -C Debug --output-on-failure

# Specific test
ctest --test-dir build -C Debug -R "PoissonAccuracyTest" --output-on-failure

# Test categories
ctest --test-dir build -C Debug -L validation --output-on-failure
ctest --test-dir build -C Debug -L cross-arch --output-on-failure

# Verbose output
ctest --test-dir build -C Debug -V
```

### Test Categories

| Label | Description | Count |
|-------|-------------|-------|
| `validation` | Physics benchmark tests | 12 |
| `cross-arch` | Backend consistency | 1 |
| (unlabeled) | Unit and integration tests | 45 |

### Quick Iteration

```bash
# Build and test specific component
cmake --build build --config Debug --target test_poisson_accuracy
ctest --test-dir build -C Debug -R "PoissonAccuracyTest" --output-on-failure
```

## Installation

### System-Wide Install

```bash
# Build and install
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --config Release
sudo cmake --install build

# Use in your project
find_package(CFD REQUIRED)
target_link_libraries(my_app PRIVATE CFD::Library)
```

### Local Install (No sudo)

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=~/.local
cmake --build build --config Release
cmake --install build

# Add to CMake search path
export CMAKE_PREFIX_PATH=~/.local:$CMAKE_PREFIX_PATH
```

## Troubleshooting

### CUDA Build Issues

**Problem:** `nvcc fatal : Unsupported gpu architecture 'compute_XX'`

**Solution:** Specify compatible architectures:
```bash
cmake -B build -DCFD_ENABLE_CUDA=ON -DCFD_CUDA_ARCHITECTURES="75;80;86"
```

**Problem:** CUDA runtime not found at runtime

**Solution:** Set library path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH  # Linux
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH  # macOS
```

### OpenMP Not Found

**Problem:** `Could NOT find OpenMP`

**Solution:**
- **Linux:** Install `libomp-dev` (Ubuntu) or `libomp` (Fedora)
- **macOS:** `brew install libomp`, then:
  ```bash
  cmake -B build -DOpenMP_ROOT=$(brew --prefix libomp)
  ```
- **Windows:** Use MSVC 2019+ (built-in support)

### SIMD Dispatch Issues

**Problem:** SIMD solvers not available on supported CPU

**Solution:** Check CPU flags:
```bash
# Linux
cat /proc/cpuinfo | grep avx2

# macOS
sysctl -a | grep machdep.cpu.features

# Windows (PowerShell)
Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty Description
```

Build system automatically detects and enables AVX2/NEON.

### Test Failures

**Problem:** Validation tests fail with timeout

**Solution:** Increase CTest timeout:
```bash
ctest --test-dir build -C Debug --timeout 600
```

**Problem:** Numerical tolerance failures

**Solution:** Check build type (Debug vs Release):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release  # Release for accurate benchmarks
```

## Build Performance

### Parallel Builds

```bash
# Linux/macOS
cmake --build build -- -j$(nproc)

# Windows
cmake --build build -- /m
```

### Incremental Builds

```bash
# Only rebuild changed files
cmake --build build --config Debug

# Rebuild specific target
cmake --build build --config Debug --target cfd_library
```

### Clean Builds

```bash
# Clean build directory
cmake --build build --target clean

# Or delete and reconfigure
rm -rf build
cmake -B build
cmake --build build
```

## Advanced Configuration

### Custom Compiler

```bash
# GCC
cmake -B build -DCMAKE_C_COMPILER=gcc-11

# Clang
cmake -B build -DCMAKE_C_COMPILER=clang-14

# Custom paths
cmake -B build -DCMAKE_C_COMPILER=/opt/gcc-12/bin/gcc
```

### Compiler Flags

```bash
# Add custom flags
cmake -B build -DCMAKE_C_FLAGS="-march=native -mtune=native"

# Debug symbols in Release
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Cross-Compilation

```bash
# ARM64 (from x86-64)
cmake -B build-arm64 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc
```

## Next Steps

- [API Reference](../reference/api-reference.md) - Learn the API
- [Examples](../guides/examples.md) - Run example programs
- [Architecture](../architecture/architecture.md) - Understand design principles
