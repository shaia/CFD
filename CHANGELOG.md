# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Documentation now publishes only on version releases (not every push)
- Renamed tests for better descriptiveness and consistency
- Refactored field statistics computation into DerivedFields module
- Updated macOS CI runner from retired macos-13 to macos-14 (ARM64)

### Fixed
- CI workflow permissions for GitHub Pages deployment
- Heredoc variable interpolation in version-release workflow
- Buffer overflow security issue
- Missing includes in test files (stdlib.h, string.h)

## [0.7.0] - 2025-12-13

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

[Unreleased]: https://github.com/shaia/CFD/compare/v0.0.6...HEAD
[0.0.6]: https://github.com/shaia/CFD/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/shaia/CFD/compare/v0.0.3...v0.0.5
[0.0.3]: https://github.com/shaia/CFD/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/shaia/CFD/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/shaia/CFD/releases/tag/v0.0.1
