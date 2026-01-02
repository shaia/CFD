# CFD Library Roadmap to v1.0

This document outlines the development roadmap for achieving a commercial-grade, open-source CFD library.

## Current State (v0.1.5)

### What We Have

- [x] Pluggable solver architecture (function pointers, registry pattern)
- [x] Multiple solver backends (CPU, SIMD/AVX2, OpenMP, CUDA)
- [x] 2D incompressible Navier-Stokes solver
- [x] Explicit Euler and projection methods
- [x] Cross-platform builds (Windows, Linux, macOS)
- [x] CI/CD with GitHub Actions
- [x] Unity test framework integration
- [x] VTK and CSV output with timestamped directories
- [x] Python bindings infrastructure (cfd-python)
- [x] Visualization library (cfd-visualization)
- [x] Thread-safe library initialization
- [x] SIMD Poisson solvers (Jacobi and Red-Black SOR with AVX2)
- [x] Boundary condition abstraction layer with runtime backend selection
- [x] Neumann and Periodic boundary conditions (all backends)
- [x] GPU boundary condition kernels (CUDA)
- [x] Comprehensive test suite (core, solvers, simulation, I/O)
- [x] 11 example programs demonstrating various features

### Critical Gaps

- [ ] Only 2D (no 3D support)
- [ ] Limited boundary conditions (no outlets, symmetry planes)
- [ ] Only structured grids
- [ ] No turbulence models
- [ ] Limited linear solvers (no BiCGSTAB/multigrid)
- [ ] No restart/checkpoint capability

### Known Issues

#### Stretched Grid Formula Bug (P1)

**File:** `lib/src/core/grid.c:78-91` (`grid_initialize_stretched`)

**Issue:** The hyperbolic cosine stretching formula is mathematically incorrect. Both endpoints (i=0 and i=n-1) map to `xmin`, and the maximum coordinate is at the center index. The grid does not span from `xmin` to `xmax`.

**Current behavior:**
- Grid clusters points toward the **center** of the domain (not boundaries)
- Both endpoints map to `xmin` (grid never reaches `xmax`)
- Maximum coordinate value is at the center index
- Higher beta = more clustering toward center

**Current formula:**
```c
x[i] = xmin + (xmax - xmin) * (1.0 - cosh(beta * (1.0 - 2.0 * xi)) / cosh(beta))
```

**Problem:** When `xi=0` or `xi=1`, `cosh(beta*±1)/cosh(beta) = 1`, so `x = xmin + 0 = xmin` for both endpoints.

**Correct formula (tanh stretching for boundary clustering):**
```c
// Cluster points near both boundaries (good for boundary layers)
x[i] = xmin + (xmax - xmin) * (1.0 + tanh(beta * (2.0 * xi - 1.0)) / tanh(beta)) / 2.0
```

**Expected behavior after fix:**
- Grid spans full domain from `xmin` to `xmax`
- Points cluster near **boundaries** (useful for boundary layer resolution)
- Higher beta = more clustering near boundaries

**TODO:**
- [ ] Fix the stretching formula to span full domain `[xmin, xmax]`
- [ ] Update Python tests in `cfd-python/tests/test_cpu_features.py` to verify correct behavior
- [ ] Document stretching parameter `beta` (higher = more clustering near boundaries)

---

## Phase 0: Architecture & Robustness (P0 - Critical)

**Goal:** Transform the codebase from a research prototype to a production-safe library.

### 0.1 Safe Error Handling

- [x] Replace all `exit()` calls with error code propagation
- [x] Implement `cfd_status_t` and `cfd_get_error_string()`
- [x] Thread-local error context for rich error reporting
- [x] Ensure resource cleanup on error paths (no leaks on failure)

### 0.2 Thread Safety & Global State

- [x] Remove static buffers in `utils.c` (path management)
- [x] Make `SolverRegistry` thread-safe or context-bound
- [x] Ensure `SimulationData` and solvers are re-entrant
- [x] Validate thread-safety with concurrent simulation tests

### 0.3 API Robustness

- [x] Comprehensive input validation (check for NULL, NaN, invalid ranges)
- [x] Configurable logging callback (`cfd_set_log_callback`)
- [x] Version header (`cfd_version.h`) with version macros
- [x] Symbol visibility control (hide private symbols) & export headers
- [x] Library initialization/shutdown functions

---

## Phase 1: Core Solver Improvements

**Goal:** Make the solver practically usable for real problems.

### 1.1 Boundary Conditions (P0 - Critical)

- [x] Boundary condition abstraction layer (`boundary_conditions.h/.c`)
- [x] Runtime backend selection (Scalar, SIMD, OpenMP, GPU)
- [x] Neumann (zero-gradient) boundary conditions
- [x] Periodic boundary conditions
- [x] SIMD-optimized BC application (AVX2)
- [x] OpenMP-parallelized BC application
- [x] CUDA GPU BC kernels
- [x] Dirichlet (fixed value) boundary conditions
- [x] No-slip wall conditions
- [x] Inlet velocity specification (uniform, parabolic, custom profiles)
- [x] Outlet (zero-gradient/convective)
- [ ] Symmetry planes
- [ ] Moving wall boundaries
- [ ] Time-varying boundary conditions

**Implemented files:**

- `lib/include/cfd/boundary/boundary_conditions.h` - Public API with backend selection
- `lib/include/cfd/boundary/boundary_conditions_gpu.cuh` - GPU API declarations
- `lib/src/boundary/boundary_conditions.c` - Public API dispatcher
- `lib/src/boundary/cpu/boundary_conditions_scalar.c` - Scalar implementation
- `lib/src/boundary/cpu/boundary_conditions_outlet_scalar.c` - Scalar outlet BC
- `lib/src/boundary/omp/boundary_conditions_omp.c` - OpenMP parallelization
- `lib/src/boundary/omp/boundary_conditions_outlet_omp.c` - OpenMP outlet BC
- `lib/src/boundary/avx2/boundary_conditions_avx2.c` - AVX2 SIMD optimizations
- `lib/src/boundary/avx2/boundary_conditions_inlet_avx2.c` - AVX2 inlet BC
- `lib/src/boundary/avx2/boundary_conditions_outlet_avx2.c` - AVX2 outlet BC
- `lib/src/boundary/neon/boundary_conditions_neon.c` - NEON SIMD optimizations
- `lib/src/boundary/neon/boundary_conditions_inlet_neon.c` - NEON inlet BC
- `lib/src/boundary/neon/boundary_conditions_outlet_neon.c` - NEON outlet BC
- `lib/src/boundary/simd/boundary_conditions_simd_dispatch.c` - SIMD runtime dispatch
- `lib/src/boundary/gpu/boundary_conditions_gpu.cu` - CUDA kernels
- `lib/src/boundary/boundary_conditions_internal.h` - Internal declarations
- `lib/src/boundary/boundary_conditions_inlet_common.h` - Shared inlet BC helpers
- `lib/src/boundary/boundary_conditions_outlet_common.h` - Shared outlet BC helpers

#### 1.1.1 Boundary Conditions Code Refactoring (P2)

**Current State:** Moderate to high code duplication across backends (CPU, OMP, AVX2, NEON).

| BC Type | Duplication Level | Notes |
|---------|-------------------|-------|
| **Inlet** | **98%** | 4 nearly identical implementations (~42 lines each) - no SIMD benefit for 1D boundaries |
| **Outlet** | **70-95%** | AVX2/NEON have similar edge dispatch logic (~80 lines duplicated) |
| **Neumann/Periodic/Dirichlet** | **60-85%** | OMP just adds pragmas; AVX2/NEON differ only in intrinsics |

**What's Done Well:**
- Common headers (`boundary_conditions_inlet_common.h`, `boundary_conditions_outlet_common.h`) extract shared logic
- Table-driven dispatch design reduces branching
- Clear separation between dispatcher and backend implementations

**Refactoring Opportunities:**

1. **Consolidate Inlet BC Implementations (Priority 1)**
   - [ ] Extract single `bc_apply_inlet_impl()` that all backends use
   - [ ] Inlet BCs operate on 1D boundaries - no SIMD benefit justifies 4 copies
   - **Estimated savings:** ~126 lines of redundant code

2. **Extract Common Outlet Dispatch Logic (Priority 2)**
   - [ ] Create edge-specific helper functions for AVX2/NEON
   - [ ] Unify edge-switch patterns (LEFT, RIGHT, BOTTOM, TOP)
   - **Estimated savings:** ~80 lines

3. **Templatize OMP vs Scalar (Priority 3)**
   - [ ] Use conditional macros for OMP pragmas instead of duplicate functions
   - [ ] Example: `OMP_PRAGMA(pragma omp parallel for)` wrapper
   - **Estimated savings:** ~50-60 lines

4. **Unify AVX2 and NEON (Priority 4)**
   - [ ] Use preprocessor or code generation for SIMD backends
   - [ ] Only differences: intrinsic set and vector width (4 vs 2 doubles)
   - **Benefit:** Easier to add new SIMD backends (e.g., AVX-512)

**Total Estimated Savings:** ~150-200 lines of redundant code

### 1.2 Linear Solvers (P0 - Critical)

**Implemented:**

- [x] SOR (Successive Over-Relaxation) - CPU baseline
- [x] Jacobi SIMD (`poisson_jacobi_simd.c`) - AVX2 vectorized, fully parallelizable
- [x] Red-Black SOR SIMD (`poisson_redblack_simd.c`) - AVX2 with SOR convergence rate
- [x] GPU Jacobi Poisson solver (`solver_projection_jacobi_gpu.cu`)
- [x] Integrate SIMD Poisson into projection solver (`solver_projection_simd.c`)

**Still needed:**

- [x] Solver abstraction interface
- [x] Conjugate Gradient (CG) for SPD systems (scalar, AVX2, NEON backends)
- [ ] BiCGSTAB for non-symmetric systems
- [ ] Preconditioners (Jacobi, ILU)
- [ ] Geometric multigrid
- [ ] Algebraic multigrid (AMG)
- [ ] Improve convergence for non-trivial problems (preconditioning)
- [ ] Performance benchmarking in Release mode

**Note:** Current SIMD Poisson solvers produce valid results but may not converge to strict tolerance (1e-6) on challenging problems like sinusoidal RHS within iteration limits. They converge properly on simpler problems (zero RHS, uniform RHS). See `docs/simd-optimization-analysis.md` for details.

### 1.3 Numerical Schemes (P1)

- [ ] Upwind differencing (1st order) for stability
- [ ] Central differencing with delayed correction
- [ ] High-resolution TVD schemes (Van Leer, Superbee)
- [ ] Gradient limiters (Barth-Jespersen, Venkatakrishnan)

### 1.4 Steady-State Solver (P1)

- [ ] SIMPLE algorithm for incompressible flow
- [ ] SIMPLEC / PISO variants
- [ ] Pseudo-transient continuation
- [ ] Convergence acceleration (relaxation)

**Files to create:**

- `include/cfd/solvers/linear_solvers.h`
- `src/solvers/linear/cg.c`
- `src/solvers/linear/bicgstab.c`
- `src/solvers/linear/multigrid.c`
- `src/solvers/linear/preconditioners.c`

### 1.5 Time Integration (P1)

- [ ] RK2 (Heun's method)
- [ ] RK4 (classical Runge-Kutta)
- [ ] Implicit Euler (backward Euler)
- [ ] Crank-Nicolson (2nd order implicit)
- [ ] BDF2 (backward differentiation)
- [ ] Adaptive time stepping with error control

### 1.6 Restart/Checkpoint (P1)

- [ ] Binary checkpoint format
- [ ] Save/restore complete simulation state
- [ ] Portable across platforms
- [ ] Version compatibility

### 1.7 Nonlinear Solvers (P2)

Solve F(x)=0 where F is nonlinear. Required for steady-state Navier-Stokes.

- [ ] Newton-Raphson iteration
- [ ] Picard iteration (successive substitution)
- [ ] Quasi-Newton methods (BFGS, L-BFGS)
- [ ] Line search and globalization
- [ ] Nonlinear solver abstraction interface

**Files to create:**

- `include/cfd/solvers/nonlinear_solver.h`
- `src/solvers/nonlinear/newton.c`
- `src/solvers/nonlinear/picard.c`

### 1.8 Eigenvalue Solvers (P3)

Find eigenvalues/eigenvectors for stability analysis.

- [ ] Power iteration
- [ ] Inverse iteration
- [ ] Arnoldi iteration
- [ ] Stability analysis framework

**Files to create:**

- `include/cfd/solvers/eigen_solver.h`
- `src/solvers/eigen/power_iteration.c`
- `src/solvers/eigen/arnoldi.c`

---

## Phase 2: Physics Extensions

**Goal:** Support more physical phenomena.

### 2.1 Energy Equation (P1)

- [ ] Temperature advection-diffusion
- [ ] Thermal boundary conditions
- [ ] Buoyancy coupling (Boussinesq approximation)
- [ ] Heat source terms
- [ ] Conjugate heat transfer
- [ ] Variable properties (viscosity/density as function of T)
- [ ] Temperature-dependent thermal conductivity

### 2.2 Turbulence Models (P1)

- [ ] Spalart-Allmaras (1-equation)
- [ ] k-epsilon standard
- [ ] k-epsilon realizable
- [ ] k-omega SST
- [ ] Wall functions
- [ ] Low-Reynolds number treatment

### 2.3 Compressible Flow (P2)

- [ ] Density-based solver
- [ ] Ideal gas equation of state
- [ ] Shock capturing (MUSCL, WENO)
- [ ] Pressure-based compressible (SIMPLE variants)

### 2.4 Species Transport (P2)

- [ ] Multi-species advection-diffusion
- [ ] Variable diffusivity
- [ ] Source terms for reactions
- [ ] Mass fraction constraints

### 2.5 Multiphase Flow (P3)

- [ ] Volume of Fluid (VOF)
- [ ] Level Set method
- [ ] Surface tension
- [ ] Phase change

---

## Phase 3: Geometry & Mesh

**Goal:** Support complex geometries.

### 3.1 3D Support (P0 - Critical)

- [ ] Extend data structures (FlowField, Grid)
- [ ] 3D stencil operations
- [ ] 3D boundary conditions
- [ ] 3D VTK output
- [ ] Update all solvers for 3D

**Estimated effort:** High - touches most of codebase

### 3.2 Unstructured Meshes (P1)

- [ ] Cell-centered finite volume
- [ ] Face-based data structures
- [ ] Gradient reconstruction
- [ ] Cell connectivity
- [ ] Triangle/tetrahedral elements
- [ ] Quadrilateral/hexahedral elements
- [ ] Mixed element support

### 3.3 Mesh I/O (P1)

- [ ] Gmsh format (.msh)
- [ ] VTK unstructured (.vtu)
- [ ] CGNS format
- [ ] OpenFOAM polyMesh
- [ ] Mesh quality metrics
- [ ] Mesh validation

### 3.4 Adaptive Mesh Refinement (P2)

- [ ] Cell-based refinement
- [ ] Refinement criteria (gradient, error)
- [ ] Coarsening
- [ ] Load balancing
- [ ] Hanging nodes treatment

---

## Phase 4: Scalability & Performance

**Goal:** Scale to large problems.

### 4.1 MPI Parallelization (P1)

- [ ] Domain decomposition
- [ ] Ghost cell exchange
- [ ] Parallel I/O
- [ ] Load balancing
- [ ] Hybrid MPI+OpenMP

### 4.2 Modular Backend Libraries (P1) ✅

**Goal:** Split the monolithic library into separate per-backend libraries for flexible deployment.

**Status:** Implemented in v0.1.5

**Rationale:**

- Users only link what they need (smaller binaries)
- CUDA library can be loaded dynamically (plugin pattern)
- Clear ABI boundaries between backends
- Allows mixing compiler toolchains (e.g., nvcc for CUDA only)
- Easier to add new backends (OpenCL, SYCL, etc.)

**Library Structure:**

| Library | CMake Target | Contents | Dependencies |
| ------- | ------------ | -------- | ------------ |
| `cfd_core` | `CFD::Core` | Grid, memory, I/O, status, common utilities | None |
| `cfd_scalar` | `CFD::Scalar` | Scalar CPU solvers | CFD::Core |
| `cfd_simd` | `CFD::SIMD` | AVX2/NEON SIMD solvers | CFD::Core, CFD::Scalar |
| `cfd_omp` | `CFD::OMP` | OpenMP parallelized solvers | CFD::Core, CFD::Scalar |
| `cfd_cuda` | `CFD::CUDA` | CUDA GPU solvers | CFD::Core |
| `cfd_library` | `CFD::Library` | Unified library (all backends) | All above |

**Implementation Tasks:**

- [x] Split `lib/CMakeLists.txt` into multiple library targets
- [x] Create separate source lists for each backend
- [x] Define public/private header boundaries per library
- [x] Add CMake aliases for clean target references
- [x] Add comprehensive test suite (`test_modular_libraries.c`)
- [x] Support shared library builds (both static and shared supported)
- [ ] Update examples/tests to link against specific backends (optional)
- [ ] Plugin loading system for dynamic backend selection

**Known Limitations:**

The modular libraries have circular dependencies that are handled via linker groups:

- `cfd_scalar`/`cfd_simd` call `poisson_solve()` (defined in `cfd_api`)
- `cfd_api` links against `cfd_scalar`/`cfd_simd`

**Current Solution:**
- On Linux: GNU linker groups (`-Wl,--start-group` ... `-Wl,--end-group`) resolve circular references
- On Windows/macOS: Linker automatically handles multiple passes
- Shared builds: Recompile sources into unified shared library
- Static builds: INTERFACE library linking modular libraries

**Future Improvements:**
To eliminate circular dependencies entirely, consider refactoring using:

- Weak symbols (platform-specific)
- Conditional registration at runtime
- Plugin architecture with dynamic loading

**Usage Examples:**

```cmake
# Currently, use the unified library for full functionality
target_link_libraries(my_app PRIVATE CFD::Library)

# Future: Link only what you need (after refactoring)
# target_link_libraries(my_app PRIVATE CFD::Core CFD::SIMD)
```

### 4.3 GPU Improvements (P2)

**Implemented:**

- [x] CUDA device detection and selection
- [x] GPU device information queries (compute capability, memory)
- [x] Projection method with Jacobi Poisson solver on GPU
- [x] GPU memory management (persistent memory, basic async transfers)
- [x] GPU boundary condition kernels
- [x] Configurable GPU settings (block size, convergence tolerance)
- [x] GPU solver statistics (kernel time, transfer time, iterations)

**Still needed:**

- [ ] Multi-GPU support
- [ ] Unified memory optimization
- [ ] Advanced async transfers (multi-stream overlap, double buffering)
- [ ] GPU-aware MPI
- [ ] Red-Black SOR GPU kernel (CPU SIMD version available in `poisson_redblack_simd.c`)

### 4.3 Performance Tools (P2)

- [ ] Built-in profiling
- [ ] Memory usage tracking
- [ ] Roofline analysis integration
- [ ] Scaling benchmarks

---

## Phase 5: I/O & Post-processing

**Goal:** Industry-standard data formats.

### 5.1 HDF5 Output (P1)

- [ ] Parallel HDF5 support
- [ ] Compression options
- [ ] Chunked storage
- [ ] XDMF metadata

### 5.2 Modern VTK (P1)

**Implemented:**

- [x] VTK legacy ASCII format output
- [x] Scalar field output (`write_vtk_output`)
- [x] Vector field output (`write_vtk_vector_output`)
- [x] Full flow field output (`write_vtk_flow_field`)
- [x] Timestamped run directories (`write_vtk_*_run` functions)

**Still needed:**

- [ ] VTK XML format (.vtu, .pvtu)
- [ ] Parallel VTK files
- [ ] Time series support
- [ ] Binary encoding

### 5.3 CSV Output (Implemented)

- [x] Timeseries data (step, time, dt, velocity stats, pressure stats)
- [x] Centerline profiles (horizontal/vertical)
- [x] Global statistics (min/max/avg of all fields)
- [x] Velocity magnitude columns
- [x] Automatic header creation and append mode

### 5.4 Restart Files (P1)

- [ ] Efficient binary format
- [ ] Incremental checkpoints
- [ ] Automatic recovery

### 5.5 In-situ Visualization (P3)

- [ ] Catalyst/ParaView integration
- [ ] ADIOS2 integration

---

## Phase 6: Validation & Documentation

**Goal:** Prove correctness, usability, and robustness.

### 6.0 API & Robustness Testing (P0 - Critical)

**Implemented:**

- [x] Core functionality tests (`tests/core/`)
- [x] Input validation tests (`test_input_validation.c`)
- [x] Error handling tests (`test_error_handling.c`)
- [x] Re-entrancy/thread-safety tests (`test_reentrancy.c`)
- [x] Solver tests organized by architecture (CPU, SIMD, OMP, GPU)
- [x] Simulation API tests (`tests/simulation/`)
- [x] I/O tests (VTK, CSV, output paths)
- [x] Physics validation tests (`test_physics_validation.c`)

**Still needed:**

- [ ] Negative testing suite (more edge cases)
- [ ] Memory leak checks (Valgrind/ASan integration in CI)

### 6.1 Mathematical Accuracy Validation (P0 - Critical)

**Goal:** Verify numerical correctness of all math-oriented computations.

#### 6.1.1 Finite Difference Stencil Tests

Unit tests for individual stencil operations:

- [ ] First derivative (central difference) - verify O(h²) accuracy
- [ ] Second derivative - verify O(h²) accuracy
- [ ] 2D Laplacian (5-point stencil) - verify O(h²) accuracy
- [ ] Divergence operator - verify O(h²) accuracy
- [ ] Gradient operator - verify O(h²) accuracy

**Test approach:** Use smooth analytical functions (e.g., `sin(kx)*sin(ky)`), compute numerical derivatives, compare to analytical derivatives, verify error scaling.

#### 6.1.2 Convergence Order Verification

- [ ] Spatial convergence tests (h-refinement: 16→32→64→128)
- [ ] Temporal convergence tests (dt-refinement)
- [ ] Automated order-of-accuracy computation
- [ ] Verify 2nd order spatial, 1st order temporal (Euler)

**Success criteria:** Measured convergence rate within 10% of expected order.

#### 6.1.3 Method of Manufactured Solutions (MMS)

- [ ] Define manufactured velocity/pressure fields with known derivatives
- [ ] Compute analytical source terms from Navier-Stokes substitution
- [ ] Run solver with manufactured source terms
- [ ] Compare numerical to manufactured solution
- [ ] Verify convergence order

**Example manufactured solution:**

```c
u(x,y,t) = sin(πx) * cos(πy) * exp(-2νπ²t)
v(x,y,t) = -cos(πx) * sin(πy) * exp(-2νπ²t)
```

#### 6.1.4 Poisson Solver Accuracy

- [ ] Zero RHS test (solution should remain constant)
- [ ] Uniform RHS test (quadratic solution)
- [ ] Sinusoidal RHS test with analytical solution comparison
- [ ] Convergence rate verification for all Poisson variants (SOR, Jacobi, Red-Black)
- [ ] Residual convergence tracking

#### 6.1.5 Laplacian Operator Validation

**Test the discrete Laplacian against manufactured solutions:**

- [ ] Manufactured solution: p = sin(πx)sin(πy) → ∇²p = -2π²p
- [ ] Verify 2nd-order accuracy O(dx²) with grid refinement
- [ ] Compare CPU, AVX2, and CG implementations

**Files:** `lib/src/solvers/linear/cpu/linear_solver_cg.c` - `apply_laplacian()`

#### 6.1.6 Divergence-Free Constraint Validation

**Verify projection method enforces incompressibility:**

- [ ] Measure max|∇·u| after projection step (should be < tolerance)
- [ ] Test with various initial velocity fields
- [ ] Verify all projection backends (CPU, AVX2, OMP, GPU)

#### 6.1.7 Linear Solver Convergence Validation

**Verify convergence rates match theory:**

- [ ] Jacobi: spectral radius ρ < 1
- [ ] SOR: optimal ω ≈ 2/(1 + sin(πh)) for Poisson
- [ ] Red-Black SOR: same convergence as SOR, parallelizable
- [ ] CG: convergence in ≤ n iterations for n×n system

**Files to create:**

- `tests/math/test_finite_differences.c`
- `tests/math/test_convergence_order.c`
- `tests/math/test_mms.c`
- `tests/math/test_linear_solver_convergence.c`
- `tests/math/manufactured_solutions.h`

### 6.2 Benchmark Validation (P0 - Critical)

#### 6.2.1 Lid-Driven Cavity Validation

**Status:** Tests implemented, solver NOT meeting scientific tolerance (RMS ~0.38, target < 0.10)

**Test files created:**
- `tests/validation/test_cavity_setup.c` - Basic setup and BC tests (7 tests)
- `tests/validation/test_cavity_flow.c` - Flow development and stability (8 tests)
- `tests/validation/test_cavity_validation.c` - Conservation and Ghia comparison (5 tests)
- `tests/validation/test_cavity_reference.c` - Reference-based regression tests (5 tests)
- `tests/validation/lid_driven_cavity_common.h` - Shared utilities
- `tests/validation/cavity_reference_data.h` - Ghia reference data
- `docs/validation/lid_driven_cavity.md` - Validation methodology documentation

**TODO - Honest Validation (MUST achieve full Ghia convergence):**

1. **Match Ghia et al. parameters EXACTLY:**
   - [ ] Grid: 129×129 (Ghia used 129×129 for all Re)
   - [ ] Boundary conditions: Regularized lid velocity at corners
   - [ ] Steady-state criterion: Residual < 1e-6 or 50000+ iterations
   - [ ] Reynolds numbers: Re=100, 400, 1000 (all must pass)

2. **Fix current solver convergence issues (RMS ~0.38 → target < 0.10):**
   - [ ] Increase pressure solver iterations (Jacobi may need 100+ per step)
   - [ ] Use smaller time step (dt < 0.0001 for stability)
   - [ ] Run to true steady state (20000+ time steps minimum)
   - [ ] Implement corner singularity regularization
   - [ ] Consider multigrid or CG for pressure solve

3. **Test ALL solver backends (run tests in parallel for speed):**
   - [ ] CPU scalar (explicit Euler, projection)
   - [ ] AVX2/SIMD (explicit Euler, projection)
   - [ ] OpenMP (explicit Euler, projection)
   - [ ] CUDA GPU (projection Jacobi)
   - [ ] Each backend must independently achieve RMS < 0.10

4. **Verification that tests are honest:**
   - [ ] Tests MUST fail if RMS > 0.10 (no loose tolerances)
   - [ ] Tests compare computed values at EXACT Ghia sample points
   - [ ] Tests report actual vs expected values transparently
   - [ ] No "current baseline" workarounds - fix solver, not tolerance

**Acceptance Criteria (non-negotiable):**

- RMS error vs Ghia < 0.10 for Re=100, 400, 1000
- All 7 solver backends produce identical results (within 0.1%)
- Grid convergence: error decreases monotonically with refinement
- Tests run in parallel to complete in < 60 seconds

#### 6.2.1.1 Grid Convergence Validation (P1)

**Issue:** Current grid convergence tests use relaxed tolerance (`prev_error + 0.08`) because RMS error does not strictly decrease with grid refinement when using the scalar Red-Black SOR Poisson solver.

**Observed behavior:**

- 17×17: RMS ~0.056
- 25×25: RMS ~0.037
- 33×33: RMS ~0.094 (worse than 25×25!)

**Root cause:** The scalar Poisson solver has accuracy limitations at larger grid sizes that prevent proper convergence.

**TODO - Strict Grid Convergence Validation:**

1. **Fix Poisson solver convergence at larger grids:**
   - [ ] Investigate why 33×33 produces worse results than 25×25
   - [ ] May need more Poisson iterations for larger grids
   - [ ] Consider using SIMD Red-Black SOR or multigrid for better accuracy

2. **Validate RMS monotonically decreases:**
   - [ ] Remove relaxed tolerance (`+ 0.08`) from grid convergence tests
   - [ ] Ensure RMS(33×33) < RMS(25×25) < RMS(17×17)
   - [ ] Test larger grids: 65×65, 129×129

3. **Add strict grid convergence test:**
   - [ ] Test must FAIL if RMS increases with refinement
   - [ ] No tolerance workarounds allowed

**Acceptance Criteria:**

- RMS error strictly decreases with each grid refinement level
- Convergence order approaches O(h²) asymptotically

#### 6.2.2 Taylor-Green Vortex Validation (P0)

**Analytical solution with known decay rate - ideal for full NS validation:**

```c
u(x,y,t) = cos(x) * sin(y) * exp(-2νt)
v(x,y,t) = -sin(x) * cos(y) * exp(-2νt)
p(x,y,t) = -0.25 * (cos(2x) + cos(2y)) * exp(-4νt)
```

**Tests to implement:**
- [ ] Verify velocity decay rate matches exp(-2νt)
- [ ] Verify pressure decay rate matches exp(-4νt)
- [ ] Test kinetic energy decay: KE(t) = KE₀ * exp(-4νt)
- [ ] Verify vorticity conservation
- [ ] Compare all solver backends (CPU, AVX2, OMP, GPU)

**Files to create:**
- `tests/validation/test_taylor_green_vortex.c`
- `tests/validation/taylor_green_reference.h`

#### 6.2.3 Poiseuille Flow Validation (P1)

**Analytical parabolic profile for channel flow:**

```c
u(y) = 4 * U_max * y * (H - y) / H²
```

**Tests to implement:**
- [ ] Steady-state velocity profile vs analytical
- [ ] Mass conservation verification
- [ ] Pressure gradient accuracy
- [ ] Inlet BC accuracy (parabolic profile)

**Files to create:**
- `tests/validation/test_poiseuille_flow.c`

#### 6.2.4 Other Benchmarks (P2)

- [ ] Backward-facing step - compare to Armaly et al. (1983)
- [ ] Flow over cylinder - compare to Williamson (1996)

#### 6.2.5 Release Validation Workflow (P1)

**Goal:** Full-length validation tests that run during releases (too slow for CI).

**Rationale:** CI tests use reduced grid sizes and iteration counts for fast feedback (~30 seconds). Release validation uses full parameters for scientific accuracy verification.

**CI vs Release Parameters:**

| Test | CI Mode | Release Mode |
|------|---------|--------------|
| Cavity Flow Development | 17×17, 500 steps | 33×33, 5000 steps |
| Cavity Re=100 Stability | 21×21, 500 steps | 33×33, 10000 steps |
| Cavity Re=400 Stability | 25×25, 500 steps | 65×65, 20000 steps |
| Reynolds Dependency | 17×17, 400 steps | 33×33, 5000 steps |
| Ghia Validation | 33×33, 5000 steps | 129×129, 50000 steps |
| Grid Convergence | 17→25→33 | 33→65→129 |
| Taylor-Green Vortex | 32×32, 1000 steps | 128×128, 10000 steps |

**Release Validation Tests to Implement:**

- [ ] Full Ghia validation at 129×129 grid for Re=100, 400, 1000
- [ ] Extended cavity flow convergence (run to true steady-state, residual < 1e-8)
- [ ] Multi-Reynolds grid convergence study (Richardson extrapolation)
- [ ] Taylor-Green vortex decay rate verification (extended time)
- [ ] Cross-architecture consistency check (all backends produce identical results)
- [ ] Memory usage and performance regression benchmarks

**Implementation:**

```bash
# CI mode (default) - fast, reduced parameters
cmake -DCAVITY_FULL_VALIDATION=0 ..
ctest -R validation

# Release mode - full scientific validation
cmake -DCAVITY_FULL_VALIDATION=1 ..
ctest -R validation --timeout 3600
```

**Files to create:**

- `tests/validation/test_ghia_full.c` - Full 129×129 Ghia validation
- `tests/validation/test_taylor_green_full.c` - Extended Taylor-Green decay
- `tests/validation/test_release_validation.c` - All backends consistency check
- `.github/workflows/release-validation.yml` - GitHub Actions workflow for releases

**Acceptance Criteria:**

- All tests pass with full parameters
- RMS error vs Ghia < 0.05 at 129×129 grid
- Taylor-Green decay rate within 1% of analytical
- All solver backends produce identical results (within floating-point tolerance)
- Total runtime < 30 minutes on release CI runner

### 6.3 Convergence Studies (P1)

- [ ] Grid independence studies for benchmark cases
- [ ] Time step independence studies
- [ ] Richardson extrapolation for error estimation
- [ ] Automated convergence reporting

### 6.4 Documentation (P1)

- [ ] Doxygen API documentation
- [ ] Theory/mathematics guide
- [ ] User tutorials
- [ ] Installation guide
- [ ] Troubleshooting guide
- [ ] Performance tuning guide
- [ ] Developer guide

### 6.5 Examples (P1)

**Implemented (12 examples):**

- [x] `minimal_example.c` - Simplest usage, basic setup
- [x] `basic_simulation.c` - Standard incompressible Navier-Stokes
- [x] `animated_flow_simulation.c` - Time-stepping with visualization
- [x] `simple_animated_flow.c` - Simpler animation variant
- [x] `velocity_visualization.c` - Output velocity field visualization
- [x] `performance_comparison.c` - Compare different solvers
- [x] `runtime_comparison.c` - Detailed timing comparisons
- [x] `solver_selection.c` - Demonstrate solver registry and selection
- [x] `custom_boundary_conditions.c` - Example BC usage
- [x] `custom_source_terms.c` - Source term implementation
- [x] `csv_data_export.c` - CSV output examples
- [x] `lid_driven_cavity.c` - Lid-driven cavity with Dirichlet BCs

**Still needed:**

- [ ] Heat transfer examples
- [ ] Turbulent flow examples
- [ ] Parallel computing examples (MPI)
- [ ] Python interface examples

---

## Phase 7: Python Integration

**Goal:** First-class Python support.

### 7.1 Python Bindings (P1)

- [ ] Complete C extension module
- [ ] NumPy array integration
- [ ] Pythonic API design
- [ ] Type hints and stubs
- [ ] Pre-built wheels (manylinux, macOS, Windows)

### 7.2 High-level Python API (P1)

- [ ] Problem definition classes
- [ ] Mesh generation helpers
- [ ] Post-processing utilities
- [ ] Jupyter notebook integration

---

## Version Milestones

### v0.1.0 - Foundation (Released)

- [x] Boundary condition abstraction layer with runtime backend selection
- [x] Neumann and Periodic boundary conditions (all backends)
- [x] SIMD Poisson solvers (Jacobi, Red-Black SOR)
- [x] GPU Jacobi Poisson solver
- [x] Comprehensive test suite
- [x] 11 example programs
- [x] VTK and CSV output

### v0.1.x - Boundary Condition Improvements

**Completed:**

- [x] Dirichlet (fixed value) boundary conditions
- [x] No-slip wall boundary conditions
- [x] Inlet velocity boundary conditions (uniform, parabolic, custom profiles)
- [x] Outlet boundary conditions (zero-gradient, convective)
- [x] Conjugate Gradient (CG) Krylov solver
- [x] Lid-driven cavity validation (Ghia et al. 1982 benchmark)
- [x] Cross-architecture solver validation (CPU, AVX2, OpenMP, CUDA)
- [x] Per-architecture Ghia validation tests
- [x] Basic documentation (OMP solvers, API constants)

### v0.1.5 - Current Release

- [x] OpenMP solver variants documented (explicit_euler_omp, projection_omp)
- [x] Updated solver type constants with NS_SOLVER_TYPE_ prefix
- [x] Doxyfile version synchronized with VERSION file
- [x] Ghia validation tests use appropriate step counts for Euler solvers

### v0.2.0 - 3D Support

- [ ] Full 3D solver capability
- [ ] 3D boundary conditions
- [ ] 3D validation cases

### v0.3.0 - Heat Transfer

- [ ] Energy equation
- [ ] Thermal boundary conditions
- [ ] Natural convection validation

### v0.4.0 - Turbulence

- [ ] At least one RANS model (k-epsilon or SA)
- [ ] Wall functions
- [ ] Turbulent channel flow validation

### v0.5.0 - Parallel Computing

- [ ] MPI parallelization
- [ ] Scalability benchmarks
- [ ] HDF5 parallel I/O

### v0.6.0 - Unstructured Meshes

- [ ] Unstructured mesh support
- [ ] Gmsh import
- [ ] Complex geometry examples

### v1.0.0 - Production Ready

- [ ] All Phase 1-6 features complete
- [ ] Comprehensive validation suite
- [ ] Complete documentation
- [ ] Stable API
- [ ] Python bindings complete
- [ ] Performance optimized

---

## Priority Legend

| Priority | Meaning |
|----------|---------|
| P0 | Critical - blocks v1.0 release |
| P1 | Important - required for v1.0 |
| P2 | Valuable - nice to have for v1.0 |
| P3 | Future - post v1.0 |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

When working on roadmap items:

1. Create an issue referencing the roadmap item
2. Create a feature branch
3. Implement with tests
4. Update documentation
5. Submit PR referencing the issue

---

## References

### CFD Validation Benchmarks

- Ghia et al. (1982) - Lid-driven cavity
- Kim & Moin (1985) - Turbulent channel flow
- Armaly et al. (1983) - Backward-facing step
- Williamson (1996) - Vortex shedding from cylinder

### Numerical Methods

- Ferziger & Peric - "Computational Methods for Fluid Dynamics"
- Versteeg & Malalasekera - "An Introduction to CFD"
- Moukalled et al. - "The Finite Volume Method in CFD"

### Turbulence Modeling

- Wilcox - "Turbulence Modeling for CFD"
- Pope - "Turbulent Flows"
