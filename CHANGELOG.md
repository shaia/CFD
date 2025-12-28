# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2025-12-28

### Added

- **Modular Backend Libraries** - Split library into separate per-backend components:
  - `cfd_core` - Grid, memory, I/O, utilities (base library)
  - `cfd_scalar` - Scalar CPU solvers (baseline implementation)
  - `cfd_simd` - AVX2/NEON optimized solvers
  - `cfd_omp` - OpenMP parallelized solvers (with stubs when OpenMP unavailable)
  - `cfd_cuda` - CUDA GPU solvers (conditional compilation)
  - `cfd_api` - Dispatcher layer and high-level API (links all backends)
  - `cfd_library` - Unified library (all backends, backward compatible)
  - CMake aliases: `CFD::Core`, `CFD::Scalar`, `CFD::SIMD`, `CFD::OMP`, `CFD::CUDA`, `CFD::API`, `CFD::Library`
- **Backend Availability API** for runtime detection of computational backends:
  - `cfd_backend_is_available()` - Check if SCALAR/SIMD/OMP/CUDA backend is available
  - `cfd_backend_get_name()` - Get human-readable backend name
  - `cfd_registry_list_by_backend()` - List solvers for a specific backend
  - `cfd_solver_create_checked()` - Create solver with backend validation
- `ns_solver_backend_t` enum and `backend` field on solver struct
- Runtime GPU availability detection with proper error codes
- Comprehensive test suite for backend API (`test_solver_backend_api.c`)
- Comprehensive test suite for modular libraries (`test_modular_libraries.c`)

### Changed

- Modular library architecture now uses dispatcher pattern with `cfd_api` library
- GNU linker groups resolve circular dependencies on Linux static builds
- Shared library builds recompile sources for proper symbol export
- OpenMP library always built (provides stubs when OpenMP unavailable)
- ROADMAP updated to document linker group solution for circular dependencies

### Fixed

- Removed duplicate `bc_impl_omp` symbol definition causing linker errors
- Fixed `cfd_registry_list_by_backend()` to properly handle discovery mode (`names == NULL`)
- OpenMP source files conditionally compiled based on availability (prevents `<omp.h>` errors)
- CI GPU symbol check now works with INTERFACE libraries (checks `libcfd_api.a` fallback)
- Cross-backend symbol dependencies resolved via `cfd_api` dispatcher library

## [0.1.5] - 2025-12-26

### Added

- **Per-architecture Ghia validation tests** for CPU, AVX2, OpenMP, and GPU backends
- **Conjugate Gradient (CG) solver** with SIMD and OpenMP support
- **Outlet boundary conditions** with zero-gradient and convective types
- **Inlet velocity boundary conditions** with uniform, parabolic, and custom profiles
- **No-slip wall boundary conditions**
- **Dirichlet (fixed value) boundary conditions**
- **Runtime CPU feature detection** with unified SIMD architecture
- **Boundary condition abstraction layer** with runtime backend selection
- CHANGELOG.md following Keep a Changelog format
- GPU solver unit tests for configuration and execution
- Comprehensive simulation API tests
- DerivedFields module with pre-computed statistics for CSV output
- OpenMP parallelization for derived field computations
- Code of Conduct (Contributor Covenant)
- Contributing guidelines (CONTRIBUTING.md)
- CFD logo and branding assets
- GitHub Pages deployment for API documentation and code coverage

### Changed

- Simplified AVX2 internal symbol names (removed redundant suffixes)
- Removed SSE2 support, simplified to AVX2-only SIMD
- Documentation now publishes only on version releases (not every push)
- Renamed tests for better descriptiveness and consistency
- Refactored field statistics computation into DerivedFields module
- Updated macOS CI runner from retired macos-13 to macos-14 (ARM64)
- Reorganized solver tests by architecture (CPU, SIMD, OMP, GPU)
- Removed silent fallbacks from SIMD, GPU, and BC backends

### Fixed

- CI workflow permissions for GitHub Pages deployment
- Heredoc variable interpolation in version-release workflow
- Buffer overflow security issue
- Missing includes in test files (stdlib.h, string.h)
- GPU Poisson solver sign error
- Use-after-free in `simulation_list_solvers`
- CMAKE_CUDA_ARCHITECTURES CMP0104 warning

## [0.1.0] - 2025-12-13

### Added
- **Library Initialization**: Thread-safe `cfd_init()` and `cfd_finalize()` functions.
- **Lazy Initialization**: API functions now automatically initialize the library if needed.
- **Thread Safety**: Core initialization uses atomic operations for safe concurrent access.
- **Threading Abstraction**: Internal cross-platform threading layer (`cfd_threading_internal.h`) supporting Windows and C11 atomics.

### Changed
- Refactored `init_simulation` to perform safe lazy initialization.
- Improved error handling during initialization failures.

### Fixed
- Fixed race conditions during library initialization.

## [0.0.6] - 2024-12-01

### Added
- Push trigger to build workflow

### Changed
- Split build workflow for security: separate build from release
- Enable PIC (Position Independent Code) for static library to support Python bindings

### Removed
- Legacy API functions (refactored to modern solver interface)

## [0.0.5] - 2024-11-28

### Changed
- Implement modular CI/CD workflows with smart artifact management

### Fixed
- Version release workflow permissions and syntax errors

_Note: v0.0.4 was skipped due to release pipeline testing._

## [0.0.3] - 2024-11-25

### Added
- Initial CI/CD pipeline with GitHub Actions
- Cross-platform builds (Windows, Linux, macOS)
- Automated release creation

## [0.0.2] - 2024-11-20

### Added
- Pluggable solver architecture with registry pattern
- **SIMD-optimized solvers** with AVX2 and FMA instruction support for vectorized computations
- **CUDA GPU-accelerated solvers** with automatic CPU fallback for systems without NVIDIA GPUs
- **OpenMP parallel solvers** for multi-threaded CPU execution (Explicit Euler and Projection methods)
- Projection method solver (Chorin's algorithm)
- Output registry system for flexible VTK and CSV output
- Multiple example programs
- Performance comparison example demonstrating scalar vs SIMD vs OpenMP vs CUDA solvers

### Changed
- Refactored solver interface to use function pointers (zero-branch dispatch)

## [0.0.1] - 2024-11-15

### Added
- Initial release
- 2D structured grid generation (uniform and stretched)
- Explicit Euler solver for incompressible Navier-Stokes
- VTK output for visualization
- Basic boundary condition support
- Unity testing framework integration

[Unreleased]: https://github.com/shaia/CFD/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/shaia/CFD/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/shaia/CFD/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/shaia/CFD/compare/v0.0.6...v0.1.0
[0.0.6]: https://github.com/shaia/CFD/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/shaia/CFD/compare/v0.0.3...v0.0.5
[0.0.3]: https://github.com/shaia/CFD/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/shaia/CFD/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/shaia/CFD/releases/tag/v0.0.1
