# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Geometric multigrid OpenMP backend** — `poisson_solver_create(POISSON_METHOD_MULTIGRID,
  POISSON_BACKEND_OMP)` (`multigrid_omp`) and the `POISSON_SOLVER_MG_OMP`
  convenience preset. The algorithm now lives in one template shared with the
  scalar solver; the OpenMP backend supplies row-parallel smoother, residual,
  restriction/prolongation and boundary primitives with no parallel reductions
  (the Neumann interior mean and the convergence residual stay serial), so its
  solutions are bit-identical to `multigrid_scalar` at any thread count (verified
  over a V/W/F x smoother x BC-mode matrix at 1, 2 and 4 threads). A kernel uses the
  thread team only when its loop covers at least 32,768 points of a plane; smaller
  loops (the coarser levels, small grids) run on the calling thread, where a team
  would cost more than the work.
  `POISSON_BACKEND_AUTO` still resolves multigrid to scalar; request OMP
  explicitly. Both multigrid backends now reject grids whose `nx` or `ny`
  exceeds `INT_MAX` with `CFD_ERROR_LIMIT_EXCEEDED` at init. The multigrid
  pressure solve in `projection_omp` and MG preconditioning for OMP CG remain
  follow-ups (`lib/src/solvers/linear/multigrid_template/linear_solver_multigrid_template.h`,
  `lib/src/solvers/linear/omp/linear_solver_multigrid_omp.c`,
  `lib/src/solvers/linear/cpu/linear_solver_multigrid.c`,
  `tests/math/test_omp_consistency.c`, `tests/math/test_multigrid_convergence.c`,
  `tests/solvers/test_linear_solver.c`).
- **Jacobi and BiCGSTAB OpenMP backends** — completes the OpenMP linear-solver
  tier (previously CG, GMRES, and Red-Black SOR). Both are selectable via
  `poisson_solver_create(method, POISSON_BACKEND_OMP)`; per-element updates are
  identical to the scalar reference, so results match within reduction rounding
  (verified by scalar-vs-OMP consistency tests). Both reject grids whose `nx` or
  `ny` exceeds `INT_MAX`, or whose `nx*ny*nz` overflows `size_t`, with
  `CFD_ERROR_LIMIT_EXCEEDED` at init. Plain lexicographic SOR remains
  scalar/SIMD-only — its parallel form is the existing Red-Black SOR OMP solver
  (`lib/src/solvers/linear/omp/linear_solver_jacobi_omp.c`,
  `lib/src/solvers/linear/omp/linear_solver_bicgstab_omp.c`,
  `tests/math/test_omp_consistency.c`, `tests/solvers/test_linear_solver.c`).
- **Multigrid wired into the projection method** — new
  `ns_solver_params_t.pressure_solver` field (`ns_pressure_solver_t`; 0 =
  existing CG behavior) selects the pressure Poisson solve of the scalar
  `projection` solver: `NS_PRESSURE_SOLVER_MULTIGRID` (multigrid V-cycles,
  RHS interior mean subtracted for Neumann compatibility) or
  `NS_PRESSURE_SOLVER_PCG_MG` (MG-preconditioned CG). Non-2^k+1 grids and
  non-scalar projection backends reject the selection with
  `CFD_ERROR_UNSUPPORTED` at init
  (`lib/src/solvers/navier_stokes/cpu/solver_projection.c`,
  `lib/src/api/solver_registry.c`,
  `tests/solvers/navier_stokes/cpu/test_projection_pressure_solver.c`).
- **Multigrid-preconditioned CG** — `POISSON_PRECOND_MULTIGRID` runs one
  symmetric V(2,2) weighted-Jacobi multigrid cycle in Dirichlet mode per
  preconditioner apply (scalar CG only; other CG/GMRES backends reject it
  with `CFD_ERROR_UNSUPPORTED`). Grid-size-independent convergence: 5 CG
  iterations at 33²–129² (tol 1e-8) vs 50–170 unpreconditioned. Exposed as
  the `POISSON_SOLVER_PCG_MG_SCALAR` convenience preset
  (`lib/src/solvers/linear/cpu/linear_solver_cg.c`,
  `tests/math/test_mg_pcg_convergence.c`).
- **Geometric multigrid Poisson solver** (scalar backend) — V/W/F(FMG) cycles with
  Red-Black Gauss-Seidel or weighted-Jacobi smoothers, full-weighting restriction and
  bilinear/trilinear prolongation, 2D/3D. Two BC modes: Neumann zero-gradient (default,
  matching the other Poisson solvers; nullspace handled via Neumann-folded restriction
  weights and coarse-level mean projection) and Dirichlet (inhomogeneous boundary data
  supported). Grid dims must be 2^k+1 per active dimension. Grid-size-independent
  convergence (< 0.15 residual reduction per V(2,2) cycle, 9²–129² validated); new
  `POISSON_SOLVER_MG_SCALAR` convenience preset
  (`lib/src/solvers/linear/cpu/linear_solver_multigrid.c`,
  `lib/src/solvers/linear/cpu/multigrid_transfer.c`,
  `tests/math/test_multigrid_operators.c`, `tests/math/test_multigrid_convergence.c`).
- **Restarted GMRES(m) Poisson solver** (`POISSON_METHOD_GMRES`) — Arnoldi with modified
  Gram-Schmidt and incremental Givens rotations for non-symmetric systems, 2D/3D, restart
  length via `poisson_solver_params_t.restart` (0 = default 30) and optional right Jacobi
  preconditioning. Scalar, AVX2, NEON and OpenMP backends (`gmres_scalar`, `gmres_simd`,
  `gmres_omp`) share one algorithm template and differ only in their O(n) vector
  primitives. `poisson_solver_init` rejects restart lengths whose Hessenberg matrix cannot
  be indexed with `int` (m ≥ 46341), and grids too large to index (`nx` or `ny` above
  `INT_MAX`, or an `(m+1)`-vector Krylov basis overflowing `size_t` bytes), with
  `CFD_ERROR_LIMIT_EXCEEDED`
  (`lib/src/solvers/linear/gmres_template/linear_solver_gmres_template.h`,
  `tests/math/test_gmres.c`, `tests/math/test_omp_consistency.c`).
- **Restart / checkpoint support** — portable, versioned, CRC-protected binary checkpoint
  format (`.cfdchk`) that saves and restores complete simulation state (grid, flow field,
  scalar params, time, solver name). Little-endian fixed-width encoding with an endianness
  marker and a format-version header that rejects unknown versions
  (`lib/src/io/checkpoint.c`, `lib/include/cfd/io/checkpoint.h`, `tests/io/test_checkpoint.c`).

### Fixed

- `poisson_solve_3d()` now checks `poisson_solver_init()`'s return status: a failed init
  (e.g. multigrid on non-2^k+1 dims) no longer leaves a broken solver in the convenience
  cache; the call returns -1 and later valid calls re-create the solver.
- `cg_omp` and `bicgstab_simd` now reject grids whose `nx` or `ny` exceeds `INT_MAX`, or
  whose `nx*ny*nz` overflows `size_t`, with `CFD_ERROR_LIMIT_EXCEEDED` at init. Their
  primitives loop over `int` bounds, which such grids silently truncated (`cg_omp`) or
  emptied (`bicgstab_simd`, where a solve then reported convergence from a zero residual).
- `poisson_solve()` and `poisson_solve_3d()` are safe to call from several threads. Their
  per-preset solver cache handed the same instance to concurrent callers, and a
  function-static flag guarded the `atexit` cleanup registration. A call now takes the
  cached instance out of an atomic slot for its duration, so a concurrent call builds its
  own (`lib/src/solvers/linear/linear_solver.c`, `lib/src/core/cfd_threading_internal.h`,
  `tests/solvers/test_linear_solver.c`).
- Solvers on the shared solve loop (Jacobi, SOR, Red-Black SOR and multigrid on every CPU
  backend) reported one more iteration than they ran when they exhausted
  `max_iterations`; `stats.iterations` now counts the iterations performed, as CG,
  BiCGSTAB and GMRES already did (`lib/src/solvers/linear/linear_solver.c`,
  `tests/solvers/test_linear_solver.c`, `tests/math/test_omp_consistency.c`).
- SOR and Red-Black SOR ran far from their best omega with the default zero-gradient walls.
  The walls are copied from the interior after each sweep, so a point beside a wall read its own
  previous value through the copy, and the automatic omega came from the Dirichlet formula, well
  below the best one: Red-Black SOR on a 33x33 seeded-noise problem took 361 sweeps at the
  automatic omega and 193 at the best. Wall-adjacent points now relax by
  omega * factor / (factor - wall weight), which makes the iteration SOR on the Neumann matrix
  itself, and the automatic omega is that matrix's optimum, from a Rayleigh quotient of its slowest
  mode. The same problem now takes 117 sweeps, and 65x65 to 257x257 grids about a third of their
  former sweeps (709 to 235 at 65x65). The converged solution is unchanged. A custom `apply_bc`
  keeps the Dirichlet formula and no wall scaling; wall values other than the default copy now
  have to be set through `apply_bc` rather than written between iterations. Applies to every SOR
  and Red-Black SOR backend: scalar, OpenMP, AVX2, NEON and CUDA
  (`lib/src/solvers/linear/linear_solver_internal.h`, `lib/src/solvers/linear/cpu/`,
  `lib/src/solvers/linear/omp/`, `lib/src/solvers/linear/avx2/`, `lib/src/solvers/linear/neon/`,
  `lib/src/solvers/linear/gpu/`, `tests/math/test_optimal_omega.c`,
  `tests/math/test_poisson_accuracy.c`).

## [0.3.0] - 2026-06-23

### Added

- **Energy equation solver** with Boussinesq buoyancy coupling — advection-diffusion
  transport of temperature with buoyancy feedback into the momentum equations and
  thermal boundary conditions. Implemented across CPU, AVX2, OpenMP, and CUDA GPU backends.
- **Thermal boundary conditions** (including adiabatic walls) wired through all backends
- **Natural convection validation** test (`tests/validation/test_natural_convection.c`)
- **SOR SIMD variants** (AVX2 + NEON) using the Block SOR technique
- **Structured logging API** with level filtering and component tags
- **ThreadSanitizer CI job** for data race detection, plus enhanced AddressSanitizer options
- **3D Poiseuille flow validation** test with analytical reference
  (`tests/validation/test_poiseuille_3d.c`, `tests/validation/poiseuille_3d_reference.h`)
- **7 math-subsystem test files** closing identified coverage gaps: CG scaling,
  3D finite differences, non-uniform grid, OMP consistency, optimal omega,
  residual computation, and solver breakdown/robustness

### Changed

- All SOR and Red-Black SOR solvers now auto-compute the optimal relaxation factor
  omega from grid dimensions using the Jacobi spectral radius formula; set
  `params.omega > 0` to override
- ROADMAP reorganized with accurate status and priority-based phasing

### Fixed

- OMP Red-Black SOR Poisson solver convergence failure on larger grids
  (root cause: hard-coded omega=1.5 was suboptimal; resolved by auto-computed omega)
- Grid convergence is now strictly monotonic (enforced via test) after switching
  projection Poisson solves to CG
- Adiabatic boundary condition corner overwrites in the natural convection test

## [0.2.0] - 2026-03-04

### Added

- **Full 3D Support** across all subsystems (8-phase rollout):
  - Core data structures extended (`grid` with z/dz/nz/stride_z, `flow_field` with w-velocity)
  - `IDX_2D`/`IDX_3D` indexing macros replacing inline indexing throughout codebase
  - 3D stencils and scalar CPU linear solvers (Jacobi, SOR, Red-Black SOR, CG, BiCGSTAB)
  - w-momentum equations in all scalar CPU NS solvers (Explicit Euler, Projection, RK2)
  - 3D boundary conditions for z-faces across all backends (CPU, SIMD, OMP, GPU)
  - AVX2/NEON SIMD backends updated for 3D
  - OpenMP backends extended for 3D
  - CUDA GPU backend extended for 3D
  - 3D VTK/CSV I/O support
- **RK2 (Heun's method) time integrator** with O(dt^2) temporal accuracy:
  - Scalar CPU backend (`rk2`)
  - AVX2 SIMD backend (`rk2_optimized`)
  - OpenMP backend (`rk2_omp`)
- **BiCGSTAB linear solver** for non-symmetric systems with AVX2 and NEON SIMD backends
- **Jacobi preconditioner** for CG solver (PCG) improving convergence
- **CG OpenMP Poisson solver** (`CG_OMP`) with fully parallelized primitives
- **Symmetry plane boundary conditions** for all backends
- **Time-varying boundary conditions** support
- **Method of Manufactured Solutions (MMS)** testing framework with source term propagation to all solver backends
- **Negative test suite** for error handling and edge cases
- **TEST_FAIL_PRINTF** macro eliminating snprintf+TEST_FAIL_MESSAGE boilerplate in tests
- **Validation test suite:**
  - Taylor-Green vortex (2D and 3D)
  - Poiseuille flow
  - Finite difference stencil accuracy
  - Poisson equation accuracy
  - Laplacian operator accuracy
  - Linear solver convergence
  - Convergence order verification (self-convergence)
  - Divergence-free constraint
  - Comprehensive multi-backend lid-driven cavity validation
- **5 new example programs:** `poiseuille_stretched_grid`, `taylor_green_convergence`, `pulsatile_inlet_flow`, `poisson_solver_tuning`, `platform_diagnostics` (19 total)
- Rewritten `lid_driven_cavity` example using library solver API

### Changed

- CPU projection solver uses CG Poisson solver instead of Red-Black SOR for reliable convergence
- OMP projection solver uses dedicated CG_OMP Poisson solver (no scalar fallback)
- Parallelized NaN check and statistics computation in OMP projection solver
- Boundary condition subsystem refactored for improved modularity
- CI GPU validation switched to on-demand EC2 instances (g4dn.2xlarge)

### Fixed

- Stretched grid formula producing incorrect spacing
- Silent fallback from OMP/SIMD Poisson solvers to scalar backend removed — now returns `CFD_ERROR_UNSUPPORTED`

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

[Unreleased]: https://github.com/shaia/CFD/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/shaia/CFD/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shaia/CFD/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/shaia/CFD/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/shaia/CFD/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/shaia/CFD/compare/v0.0.6...v0.1.0
[0.0.6]: https://github.com/shaia/CFD/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/shaia/CFD/compare/v0.0.3...v0.0.5
[0.0.3]: https://github.com/shaia/CFD/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/shaia/CFD/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/shaia/CFD/releases/tag/v0.0.1
