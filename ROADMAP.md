# CFD Library Roadmap to v1.0

This document outlines the development roadmap for achieving a commercial-grade, open-source CFD library.

## Current State (v0.1.7)

### What We Have

- [x] Pluggable solver architecture (function pointers, registry pattern)
- [x] Multiple solver backends (CPU, SIMD/AVX2, OpenMP, CUDA)
- [x] 2D incompressible Navier-Stokes solver
- [x] Explicit Euler and projection methods
- [x] Cross-platform builds (Windows, Linux, macOS)
- [x] CI/CD with GitHub Actions
- [x] Unity test framework integration
- [x] VTK and CSV output with timestamped directories
- [x] Visualization library (cfd-visualization)
- [x] Thread-safe library initialization
- [x] SIMD Poisson solvers (Jacobi and Red-Black SOR with AVX2)
- [x] Boundary condition abstraction layer with runtime backend selection
- [x] Neumann and Periodic boundary conditions (all backends)
- [x] GPU boundary condition kernels (CUDA)
- [x] Comprehensive test suite (core, solvers, simulation, I/O)
- [x] 12 example programs demonstrating various features

### Backend Coverage Summary

Each algorithm should have scalar (CPU) + SIMD + OMP variants. Track gaps here.

| Category            | Algorithm      | CPU  | AVX2     | NEON     | OMP      | GPU  |
| ------------------- | -------------- | ---- | -------- | -------- | -------- | ---- |
| **N-S Solvers**     | Explicit Euler | done | done     | —        | done     | —    |
|                     | Projection     | done | done     | —        | done     | done |
|                     | RK2 (Heun)    | done | **TODO** | —        | **TODO** | —    |
| **Linear Solvers**  | Jacobi         | done | done     | done     | —        | —    |
|                     | SOR            | done | **TODO** | **TODO** | —        | —    |
|                     | Red-Black SOR  | done | done     | done     | done     | —    |
|                     | CG / PCG       | done | done     | done     | —        | —    |
|                     | BiCGSTAB       | done | done¹    | done¹    | —        | —    |

¹ AVX2/NEON implementations include OpenMP parallelization internally. No separate OMP backend exists.

### Critical Gaps

- [ ] Only 2D (no 3D support)
- [x] ~~Limited boundary conditions (no symmetry planes)~~ Symmetry planes now supported
- [ ] Only structured grids
- [ ] No turbulence models
- [ ] Limited linear solvers (no multigrid)
- [ ] No restart/checkpoint capability

### Known Issues

#### OMP Red-Black SOR Poisson Solver Convergence (P1)

**Status:** Workaround implemented (switched OMP projection to CG)

**Issue:** The OMP Red-Black SOR Poisson solver fails to converge on certain problem configurations (e.g., 33×33 grids with dt=5e-4), hitting max iterations (1000) without reaching tolerance (1e-6).

**Impact:**

- OMP projection solver switched to CG as workaround (commit be356a3)
- Red-Black SOR remains available but unreliable for production use with OMP backend
- CG provides reliable convergence (O(√κ) vs SOR's O(n))

**Root Cause:** Unknown - requires investigation of:

- Omega parameter tuning for Neumann BCs (currently uses default 1.5)
- Parallel race conditions in red/black sweeps
- Boundary condition application in OMP implementation
- Comparison with working AVX2 Red-Black SOR implementation

**Action Items:**

- [ ] Profile OMP Red-Black SOR to identify convergence bottleneck
- [ ] Compare OMP vs AVX2 Red-Black implementations for differences
- [ ] Test omega parameter sweep (1.0 to 1.9) for optimal convergence
- [ ] Add convergence diagnostics (residual history logging)
- [ ] Consider switch to Chebyshev acceleration or SSOR

**Workaround:** Use CG or switch to AVX2/CPU backends for production

#### ~~Stretched Grid Formula Bug~~ (FIXED in v0.1.7)

**File:** `lib/src/core/grid.c` (`grid_initialize_stretched`)

**Resolution:** Fixed the hyperbolic stretching formula using tanh-based stretching:
```c
x[i] = xmin + (xmax - xmin) * (1.0 + tanh(beta * (2.0 * xi - 1.0)) / tanh(beta)) / 2.0
```

**Fixed behavior:**

- Grid spans full domain from `xmin` to `xmax`
- Points cluster near **boundaries** (useful for boundary layer resolution)
- Higher beta = more clustering near boundaries
- Edge case: beta=0 falls back to uniform grid

**Tests added:** `tests/core/test_grid.c` with 16 unit tests covering:

- Uniform grid spans full domain, equal spacing, non-unit domains
- Stretched grid spans full domain, clusters near boundaries, higher beta = more clustering
- beta=0 equals uniform, non-unit domains, monotonically increasing coordinates
- Error handling for invalid inputs

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

### 0.4 API & Robustness Testing

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

### 0.5 Error Handling & Robustness (P0) ✅

**Status:** COMPLETE (PR #139, Feb 2026)

**Problem:** Projection solvers used thread-unsafe static `warned` flag for Poisson convergence failures, continuing execution with inaccurate pressure fields instead of returning errors.

**Solution implemented:**

- [x] Removed static `warned` flag from all projection solvers (CPU, OMP, AVX2) ✅
- [x] Return `CFD_ERROR_MAX_ITER` immediately on Poisson convergence failure ✅
- [x] **Breaking API change:** `run_simulation_step()` and `run_simulation_solve()` now return `cfd_status_t` instead of `void` ✅
- [x] Updated solver registry wrappers to propagate errors properly ✅
- [x] Updated all ~15 test files to check return values ✅
- [x] Updated all examples to handle errors in loops ✅
- [x] Updated README with new function signatures ✅
- [x] Proper distinction between `CFD_ERROR_UNSUPPORTED` (backend unavailable) and other errors ✅

**Impact:**

- Thread-safe error handling (no data races in OpenMP builds)
- Users can now detect and respond to convergence failures
- Better diagnostics for validation debugging
- Compile-time safety (void → cfd_status_t breaks code that ignores errors)

**Files modified:**

- `lib/include/cfd/api/simulation_api.h` - Function signature changes
- `lib/src/api/simulation_api.c` - Error propagation implementation
- `lib/src/api/solver_registry.c` - Wrapper error handling
- `lib/src/solvers/navier_stokes/{cpu,omp,avx2}/solver_projection*.c` - Return errors
- `tests/**/*.c` - Test updates for error checking
- `examples/*.c` - Example error handling
- `README.md` - API documentation updates

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
- [x] Symmetry planes
- [x] Moving wall boundaries (via Dirichlet BCs, see lid-driven cavity example)
- [x] Time-varying boundary conditions

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
- `lib/src/boundary/boundary_conditions_time.h` - Time modulation helpers
- `lib/src/boundary/cpu/boundary_conditions_inlet_time_scalar.c` - Time-varying inlet scalar implementation
- `tests/core/test_boundary_conditions_symmetry.c` - Symmetry BC unit tests

#### 1.1.1 Boundary Conditions Code Refactoring (P2) ✅

**Refactoring Complete.** Reduced ~500 lines of duplicated code across BC backends using parameterized header templates.

1. **Consolidate Inlet BC Implementations (Priority 1)**
   - [x] Deleted 3 redundant inlet files (OMP/AVX2/NEON) — all delegate to scalar
   - **Savings:** 196 lines removed

2. **Extract Common Outlet SIMD Template (Priority 2)**
   - [x] Created `boundary_conditions_outlet_simd.h` parameterized by SIMD intrinsics
   - [x] AVX2/NEON outlet files reduced from ~135 lines each to ~33 lines
   - **Savings:** 99 lines removed

3. **Templatize OMP vs Scalar (Priority 3)**
   - [x] Created `boundary_conditions_core_impl.h` with token-pasting macros
   - [x] Scalar and OMP files include shared template with `BC_CORE_USE_OMP` flag
   - **Savings:** 29 lines removed

4. **Unify AVX2 and NEON (Priority 4)**
   - [x] Created `boundary_conditions_simd_impl.h` parameterized by STORE/LOAD/BROADCAST/VEC_TYPE/WIDTH/MASK
   - [x] AVX2/NEON main BC files reduced from ~218 lines each to ~55 lines
   - **Savings:** 181 lines removed

**Files created:**

- `lib/src/boundary/boundary_conditions_core_impl.h`
- `lib/src/boundary/boundary_conditions_outlet_simd.h`
- `lib/src/boundary/boundary_conditions_simd_impl.h`

**Files deleted:**

- `lib/src/boundary/omp/boundary_conditions_inlet_omp.c`
- `lib/src/boundary/avx2/boundary_conditions_inlet_avx2.c`
- `lib/src/boundary/neon/boundary_conditions_inlet_neon.c`

**Total savings:** ~505 lines removed across 4 priorities

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
  - **Note:** CG is now the default Poisson solver for all projection methods (PR #139)
  - CPU/OMP use CG_SCALAR, AVX2 uses CG_SIMD for reliable O(√κ) convergence
- [x] BiCGSTAB for non-symmetric systems ✅
  - [x] BiCGSTAB Scalar (CPU) ✅
  - [x] BiCGSTAB SIMD (AVX2/NEON with integrated OpenMP parallelization) ✅

  **Note:** AVX2 and NEON implementations include OpenMP parallelization via `#pragma omp parallel for` in inner loops. There is no separate `POISSON_BACKEND_OMP` backend for BiCGSTAB.

  **Files created (BiCGSTAB backends):**

  - `lib/src/solvers/linear/avx2/linear_solver_bicgstab_avx2.c` - AVX2 SIMD implementation with OpenMP
  - `lib/src/solvers/linear/neon/linear_solver_bicgstab_neon.c` - NEON SIMD implementation with OpenMP
  - `tests/math/test_bicgstab_avx2.c` - AVX2 vs scalar consistency test
  - `tests/math/test_bicgstab_neon.c` - NEON vs scalar consistency test

  Updated files:

  - `lib/src/solvers/linear/simd/linear_solver_simd_dispatch.c` - Added BiCGSTAB dispatcher
  - `lib/src/solvers/linear/linear_solver_internal.h` - Added BiCGSTAB SIMD factory declaration

- [ ] GMRES (Generalized Minimal Residual) for non-symmetric systems
  - [ ] GMRES scalar
  - [ ] GMRES AVX2
  - [ ] GMRES NEON
  - [ ] GMRES OMP
- [x] Jacobi (diagonal) preconditioner for CG (scalar, AVX2, NEON backends)
- [ ] SSOR (Symmetric SOR) preconditioner
- [ ] SOR SIMD variants (currently CPU-only)
  - [ ] SOR AVX2
  - [ ] SOR NEON
- [ ] ILU preconditioner
- [ ] Geometric multigrid
- [ ] Algebraic multigrid (AMG) solver
- [ ] AMG preconditioner (for use with CG/GMRES/BiCGSTAB)
- [ ] Performance benchmarking in Release mode

**Note:** Current SIMD Poisson solvers produce valid results but may not converge to strict tolerance (1e-6) on challenging problems like sinusoidal RHS within iteration limits. They converge properly on simpler problems (zero RHS, uniform RHS). See `docs/simd-optimization-analysis.md` for details.

#### 1.2.1 Poisson Solver Accuracy Tests ✅

- [x] Zero RHS test (solution should remain constant)
- [x] Uniform RHS test (quadratic solution)
- [x] Sinusoidal RHS test with analytical solution comparison
- [x] Convergence rate verification for all Poisson variants (SOR, Jacobi, Red-Black)
- [x] Residual convergence tracking

**Files created:**

- `tests/math/test_poisson_accuracy.c` - 15 tests covering all accuracy requirements

#### 1.2.2 Laplacian Operator Validation ✅

**Test the discrete Laplacian against manufactured solutions:**

- [x] Manufactured solution: p = sin(πx)sin(πy) → ∇²p = -2π²p
- [x] Verify 2nd-order accuracy O(dx²) with grid refinement
- [x] Compare CPU and CG implementations (SIMD compared when available)
- [x] Verify stencil symmetry (self-adjoint property)

**Files created:**

- `tests/math/test_laplacian_accuracy.c` - 4 tests covering all accuracy requirements

**Files tested:** `lib/src/solvers/linear/cpu/linear_solver_cg.c` - `apply_laplacian()`

#### 1.2.3 Linear Solver Convergence Validation ✅

**Verify convergence rates match theory:**

- [x] Jacobi: spectral radius ρ = cos(πh) verified to <1% accuracy (Dirichlet BCs)
- [x] SOR: over-relaxation (ω > 1) converges faster than Gauss-Seidel
- [x] Red-Black SOR: comparable convergence to standard SOR (ratio 0.5-2.0)
- [x] CG: convergence in O(√κ) iterations verified

**Files created:**

- `tests/math/test_linear_solver_convergence.c` - 6 tests covering convergence properties

**Note:** The Jacobi spectral radius test uses Dirichlet BCs (p=0 on boundary) because the ρ = cos(πh) formula applies only to the Dirichlet problem. With Neumann BCs, the discrete Laplacian has a constant null space giving eigenvalue 1. The SOR optimal ω = 2/(1 + sin(πh)) also applies to Dirichlet BCs; with Neumann BCs optimal ω is typically lower (1.5-1.7).

**Files still to create (future work):**

- `src/solvers/linear/multigrid.c`

#### 1.2.4 Preconditioned Conjugate Gradient (PCG) ✅

**Jacobi (diagonal) preconditioner for CG solver:**

- [x] Added `poisson_precond_type_t` enum (NONE, JACOBI) to `poisson_solver.h`
- [x] Added `preconditioner` field to `poisson_solver_params_t`
- [x] Implemented PCG algorithm in scalar CG solver
- [x] Implemented PCG in AVX2 backend with SIMD-vectorized preconditioner application
- [x] Implemented PCG in NEON backend with SIMD-vectorized preconditioner application
- [x] Backward compatible: POISSON_PRECOND_NONE (default) gives standard CG behavior

**PCG Algorithm:**
```
z = M⁻¹r           (apply preconditioner)
p = z              (search direction from preconditioned residual)
ρ = (r, z)         (instead of (r, r))
```

**Note:** For the uniform-grid Laplacian with constant coefficients, the Jacobi preconditioner M⁻¹ = 1/(2/dx² + 2/dy²) is a constant scalar, which doesn't improve the condition number. PCG provides benefit for problems with variable coefficients or non-uniform grids.

**Files modified:**

- `lib/include/cfd/solvers/poisson_solver.h` - Added preconditioner enum and params field
- `lib/src/solvers/linear/linear_solver.c` - Default params initialization
- `lib/src/solvers/linear/cpu/linear_solver_cg.c` - Scalar PCG implementation
- `lib/src/solvers/linear/avx2/linear_solver_cg_avx2.c` - AVX2 PCG implementation
- `lib/src/solvers/linear/neon/linear_solver_cg_neon.c` - NEON PCG implementation

**Files created:**

- `tests/math/test_pcg_convergence.c` - 4 tests verifying PCG correctness and consistency

### 1.3 Numerical Schemes (P1)

- [ ] Upwind differencing (1st order) for stability
- [ ] Central differencing with delayed correction
- [ ] High-resolution TVD schemes (Van Leer, Superbee)
- [ ] Gradient limiters (Barth-Jespersen, Venkatakrishnan)

#### 1.3.1 Finite Difference Stencil Tests ✅

Unit tests for individual stencil operations:

- [x] First derivative (central difference) - verify O(h²) accuracy
- [x] Second derivative - verify O(h²) accuracy
- [x] 2D Laplacian (5-point stencil) - verify O(h²) accuracy
- [x] Divergence operator - verify O(h²) accuracy
- [x] Gradient operator - verify O(h²) accuracy

**Test approach:** Use smooth analytical functions (e.g., `sin(kx)*sin(ky)`), compute numerical derivatives using shared stencil implementations, compare to analytical derivatives, verify error scaling.

**Files created:**

- `lib/include/cfd/math/stencils.h` - Shared stencil implementations (header-only)
- `tests/math/test_finite_differences.c` - 9 tests verifying O(h²) convergence

**Future work:**

- [ ] Migrate solver code to use `cfd/math/stencils.h` (currently inline implementations)
  - `solver_explicit_euler.c`, `solver_projection.c`, `linear_solver_jacobi.c`, etc.
  - This ensures tests exercise the exact production code paths

#### 1.3.2 Convergence Order Verification ✅

- [x] Spatial convergence tests (h-refinement: 16→32→64→128)
- [x] Temporal convergence tests (dt-refinement)
- [x] Automated order-of-accuracy computation
- [x] Verify super-linear spatial convergence (~1.5 order, BC-limited)

**Results:**

- Spatial: ~O(h^1.5) achieved (theoretical O(h²) limited by first-order BCs)
- Temporal: O(dt) difficult to isolate; spatial error dominates on practical grids

**Success criteria:** Spatial rate > 1.4, error decreases monotonically with refinement.

**Files created:**

- `tests/math/test_convergence_order.c`

#### 1.3.3 Method of Manufactured Solutions (MMS)

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

#### 1.3.4 Divergence-Free Constraint Validation ✅

**Verify projection method enforces incompressibility:**

- [x] Measure max|∇·u| after projection step (should be bounded)
- [x] Test with various initial velocity fields (sinusoidal, Taylor-Green, vortex pair)
- [x] Verify all projection backends (CPU, AVX2, OMP, GPU)

**Results:**

- Divergence computation verified against analytically div-free fields (<1e-10)
- All backends produce consistent results (within 10%)
- Divergence stays bounded (<10.0) - matches existing solver behavior
- GPU test skipped when CUDA not available

**Files created:**

- `tests/math/test_divergence_free.c`

### 1.4 Steady-State Solver (P1)

- [ ] SIMPLE algorithm for incompressible flow
- [ ] SIMPLEC / PISO variants
- [ ] Pseudo-transient continuation
- [ ] Convergence acceleration (relaxation)

**Files to create:**

- `include/cfd/solvers/steady_state.h`
- `src/solvers/steady/simple.c`
- `src/solvers/steady/simplec.c`
- `src/solvers/steady/piso.c`

### 1.5 Time Integration (P1)

Each time integrator requires a scalar (CPU) reference implementation first, then backend variants.
See `/add-ns-time-integrator` command for the cross-backend workflow.

**Algorithms:**

- [x] RK2 (Heun's method) — scalar
  - [ ] RK2 AVX2+OMP (`rk2_optimized`)
  - [ ] RK2 OpenMP (`rk2_omp`)
  - [ ] RK2 CUDA (`rk2_gpu`)
- [ ] RK4 (classical Runge-Kutta) — scalar
  - [ ] RK4 AVX2+OMP (`rk4_optimized`)
  - [ ] RK4 OpenMP (`rk4_omp`)
  - [ ] RK4 CUDA (`rk4_gpu`)
- [ ] Implicit Euler (backward Euler)
- [ ] Crank-Nicolson (2nd order implicit)
- [ ] BDF2 (backward differentiation)
- [ ] Adaptive time stepping with error control

**RK2 Implementation:**

- O(dt²) temporal accuracy verified via self-convergence test (ratio ≈ 4.0)
- Uses periodic stencil indexing in RHS evaluation to avoid ghost-cell order reduction
- BCs applied only after full RK2 step (not between stages)

**Files created:**

- `lib/src/solvers/navier_stokes/cpu/solver_rk2.c`
- `tests/solvers/navier_stokes/cpu/test_solver_rk2.c`

**Files modified:**

- `lib/include/cfd/solvers/navier_stokes_solver.h` — added `NS_SOLVER_TYPE_RK2`
- `lib/src/api/solver_registry.c` — registered RK2 factory
- `lib/CMakeLists.txt` — added source file
- `CMakeLists.txt` — added test entries

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

### 4.4 Performance Tools (P2)

- [ ] Built-in profiling
- [ ] Memory usage tracking
- [ ] Roofline analysis integration
- [ ] Scaling benchmarks

### 4.5 Structured Logging & Diagnostics (P2)

**Status:** Partial - `cfd_set_log_callback()` API exists but not used consistently

**Current Issues:**

- Raw `fprintf(stderr, ...)` and `snprintf()` scattered throughout codebase
- No log levels (can't filter INFO vs WARNING vs ERROR)
- No redirection (always stderr, can't send to file/syslog/GUI)
- No timestamps or structured metadata
- Not thread-safe (garbled output from multiple threads)
- Mixed purposes (diagnostics vs error messages)

**Proposed Structured Logging API:**

```c
// Log levels
typedef enum {
    CFD_LOG_DEBUG = 0,
    CFD_LOG_INFO = 1,
    CFD_LOG_WARNING = 2,
    CFD_LOG_ERROR = 3
} cfd_log_level_t;

// Logging function with structured metadata
void cfd_log(cfd_log_level_t level, const char* component,
             const char* format, ...);

// Example usage
cfd_log(CFD_LOG_WARNING, "poisson_solver",
        "Failed to converge (grid %zux%zu, dt=%.4e)",
        nx, ny, dt);
```

**Implementation Tasks:**

- [ ] Define `cfd_log()` API with log levels and component tags
- [ ] Implement default console handler (with timestamps, colored output)
- [ ] Add thread-safe logging (mutex-protected or per-thread buffers)
- [ ] Replace all `fprintf(stderr, ...)` calls with `cfd_log()`
- [ ] Replace diagnostic `snprintf()` with `cfd_log()` where appropriate
- [ ] Add log filtering by level (suppress DEBUG in production)
- [ ] Add log filtering by component (e.g., only show "boundary" logs)
- [ ] Support custom log handlers via callback
- [ ] Add structured data API for metrics (convergence stats, timings)

**Benefits:**

- Users can redirect logs to files, syslog, or application UI
- Fine-grained control (enable DEBUG for specific components)
- Better debugging (timestamps, thread IDs, component context)
- Thread-safe by design
- Statistics aggregation ("Poisson failed 15 times this run")
- Can mute logs entirely for embedded/production use

**Files to Modify:**

- `lib/src/solvers/linear/cpu/linear_solver_*.c` - Replace fprintf
- `lib/src/solvers/navier_stokes/cpu/solver_*.c` - Replace fprintf
- `lib/src/solvers/navier_stokes/omp/solver_*.c` - Replace fprintf
- `lib/src/solvers/navier_stokes/avx2/solver_*.c` - Replace fprintf
- `tests/validation/lid_driven_cavity_common.h` - Replace snprintf for diagnostics

### 4.6 Derived Fields Optimization (P2)

**Status:** OpenMP implemented, SIMD and CUDA pending

**Current Implementation:**

Located in `lib/src/core/derived_fields.c`:

- [x] OpenMP parallelization for velocity magnitude computation
- [x] OpenMP parallel reduction for field statistics (min/max/avg)
- [x] Threshold-based parallelization (OMP_THRESHOLD = 1000 cells)
- [x] Fallback to sequential for small grids

**Still needed:**

#### SIMD Optimization (AVX2/NEON)

Best for: All grid sizes, especially when combined with OpenMP

**Implementation approach:**

```c
// Velocity magnitude with AVX2 (4 doubles per iteration)
for (size_t i = 0; i < n; i += 4) {
    __m256d u = _mm256_loadu_pd(&field->u[i]);
    __m256d v = _mm256_loadu_pd(&field->v[i]);
    __m256d u2 = _mm256_mul_pd(u, u);
    __m256d v2 = _mm256_mul_pd(v, v);
    __m256d sum = _mm256_add_pd(u2, v2);
    __m256d mag = _mm256_sqrt_pd(sum);
    _mm256_storeu_pd(&vel_mag[i], mag);
}
// Handle remainder elements
```

**Tasks:**

- [ ] Add AVX2 implementation for velocity magnitude
- [ ] Add NEON implementation for velocity magnitude (ARM64)
- [ ] Combine with OpenMP (`#pragma omp simd` or manual intrinsics)
- [ ] Add SIMD horizontal reduction for statistics
- [ ] Add runtime CPU feature detection
- [ ] Benchmark against scalar+OpenMP version

**Benefits:**

- No thread overhead (works on small grids)
- Combines well with OpenMP for multi-core
- Reference: existing SIMD solvers in `lib/src/solvers/linear/avx2/`

**Challenges:**

- Architecture-specific code paths needed
- Remainder handling for non-aligned sizes
- Statistics reductions more complex with SIMD

#### CUDA GPU Acceleration

Best for: Very large grids (100k+ cells), GPU available

**Implementation approach:**

```cuda
__global__ void compute_velocity_magnitude_kernel(
    double* __restrict__ vel_mag,
    const double* __restrict__ u,
    const double* __restrict__ v,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vel_mag[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
    }
}
```

**Tasks:**

- [ ] Add CUDA kernel for velocity magnitude
- [ ] Add parallel reduction for statistics (use CUB library or custom)
- [ ] Share GPU memory with CUDA solvers (avoid CPU<->GPU transfers)
- [ ] Add threshold logic (only use GPU for large grids)
- [ ] Benchmark transfer overhead vs compute benefit

**Benefits:**

- Massive parallelism for large grids
- Can share GPU memory with CUDA solvers
- Reference: existing CUDA infrastructure in `lib/src/solvers/gpu/`

**Challenges:**

- GPU memory transfer overhead for small grids
- More complex reduction implementation
- Only beneficial if data already on GPU

**Performance Thresholds (Approximate):**

| Grid Size        | Recommended Approach                         |
|------------------|----------------------------------------------|
| < 1,000 cells    | Sequential (overhead not worth it)           |
| 1,000 - 10,000   | OpenMP (2-4 threads) ✅ Implemented          |
| 10,000 - 100,000 | OpenMP + SIMD                                |
| > 100,000        | CUDA (if GPU available) or OpenMP + SIMD     |

**Note:** These operations are memory-bound, not compute-bound:

- Velocity magnitude: Read 2 arrays, write 1 array
- Statistics: Read 1 array, compute 3 values

For memory-bound operations:

- SIMD helps with cache line utilization
- OpenMP helps with memory bandwidth aggregation across cores
- CUDA helps only if data is already on GPU

**Files to Modify:**

- `lib/src/core/derived_fields.c` - Add SIMD/CUDA variants
- `lib/src/core/derived_fields_simd.c` - New file for SIMD implementation
- `lib/src/core/derived_fields_gpu.cu` - New file for CUDA kernels
- `tests/io/test_csv_output.c` - Verify correctness after optimization

### 4.7 SIMD Projection Solver Optimization (P2)

**Status:** SIMD Poisson integration completed (December 2024), further optimizations pending

**Current State:**

- [x] SIMD Poisson solvers implemented (Jacobi and Red-Black SOR with AVX2)
- [x] SIMD Poisson integrated into projection solver (`solver_projection_simd.c`)
- [x] Full projection method with SIMD corrector step
- [x] All tests pass with results matching scalar implementation
- [x] `u_new` buffer reused as temp for Poisson solver (in-place Red-Black SOR)

**Reference:** See [docs/technical-notes/simd-optimization-analysis.md](docs/technical-notes/simd-optimization-analysis.md) for detailed technical analysis.

**Performance Analysis:**

Current SIMD implementation provides ~1.3-1.5x speedup with Poisson solver being 70-80% of total runtime. Speedup limited by Amdahl's law:

- Parallelizable fraction: ~80% (Poisson + Corrector)
- SIMD speedup: 1.5-2x (Red-Black has gather/scatter overhead)
- Expected overall: 1.3-1.5x (matches observations)

#### High Priority

##### 1. Improve Poisson Convergence for Non-Trivial Problems

SIMD Poisson solvers don't fully converge to 1e-6 tolerance on challenging problems (e.g., sinusoidal RHS) within current iteration limits.

**Tasks:**

- [ ] Increase `POISSON_MAX_ITER` from current 1000/2000 to higher values
- [ ] Implement adaptive tolerance based on problem scale
- [ ] Add optional multigrid preconditioner for faster convergence
- [ ] Investigate Red-Black omega parameter tuning for specific problem classes

##### 2. Performance Benchmarking in Release Mode

Current tests run in Debug mode where SIMD may be slower than scalar due to lack of compiler optimizations.

**Tasks:**

- [ ] Create Release mode benchmark suite
- [ ] Measure actual speedup vs scalar implementation
- [ ] Profile to identify remaining bottlenecks
- [ ] Compare against theoretical Amdahl's law predictions
- [ ] Document performance characteristics for different grid sizes

#### Medium Priority

##### 3. OpenMP + SIMD Hybrid

Combine thread parallelism with SIMD vectorization for maximum CPU utilization.

**Current State:** Separate OMP and SIMD backends exist, but no hybrid implementation.

**Tasks:**

- [ ] Implement hybrid projection solver using OpenMP across rows + SIMD within rows
- [ ] Use `#pragma omp parallel for` on outer loop with SIMD intrinsics in inner loop
- [ ] Benchmark scaling efficiency (measure speedup vs threads)
- [ ] Compare against pure OMP and pure SIMD implementations
- [ ] Handle load balancing for Red-Black coloring with OpenMP

**Benefits:**

- Jacobi allows full parallelization across rows
- Red-Black allows parallelization within each color
- Could achieve 4-8x speedup on modern multi-core CPUs

**Reference Files:**

- `lib/src/solvers/navier_stokes/omp/solver_projection_omp.c` - OMP implementation
- `lib/src/solvers/navier_stokes/avx2/solver_projection_avx2.c` - SIMD implementation
- `lib/src/solvers/linear/avx2/` - SIMD Poisson solvers

#### Low Priority

##### 4. Multigrid SIMD Implementation

Achieve O(N) complexity vs O(N²) for iterative methods.

**Benefits:**

- Multigrid offers O(N) complexity vs O(N²) for Jacobi/SOR
- Natural parallelism at each grid level
- Can use SIMD at each level for additional speedup
- Essential for large-scale 3D simulations

**Tasks:**

- [ ] Implement multigrid V-cycle framework
- [ ] Add restriction/prolongation operators
- [ ] Use SIMD Jacobi/Red-Black as smoothers at each level
- [ ] Benchmark convergence rate vs pure iterative methods
- [ ] Validate against analytical solutions

**Challenges:**

- Complex implementation requiring careful design
- Grid hierarchy management
- Operator construction at multiple levels
- Testing and validation complexity

**Priority Justification:** Low priority because:

- Current SIMD implementation adequate for 2D problems
- Critical for 3D but 3D support itself is Phase 2
- Better to implement multigrid after 3D infrastructure exists

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

## Phase 6: Benchmark Validation & Documentation

**Goal:** Validate against reference solutions and provide comprehensive documentation.

**Note:** API/robustness testing is in Phase 0.4. Mathematical accuracy validation (stencils, convergence, MMS) is in Phase 1.3. Linear solver validation is in Phase 1.2.

### 6.1 Benchmark Validation (P0 - Critical)

#### 6.1.1 Lid-Driven Cavity Validation ✅

**Status:** COMPLETE - Solver meets scientific tolerance (RMS ~0.04, target < 0.10)

**Test files created:**
- `tests/validation/test_cavity_setup.c` - Basic setup and BC tests (7 tests)
- `tests/validation/test_cavity_flow.c` - Flow development and stability (8 tests)
- `tests/validation/test_cavity_validation.c` - Conservation and Ghia comparison (5 tests)
- `tests/validation/test_cavity_reference.c` - Reference-based regression tests (5 tests)
- `tests/validation/lid_driven_cavity_common.h` - Shared utilities
- `tests/validation/cavity_reference_data.h` - Ghia reference data
- `docs/validation/lid_driven_cavity.md` - Validation methodology documentation

**What Was Fixed (PR #139, Feb 2026):**

1. **Switched Projection Solvers to CG:**
   - [x] CPU projection: Red-Black SOR → CG_SCALAR ✅
   - [x] OMP projection: Red-Black SOR → CG_SCALAR ✅
   - [x] AVX2 projection: Red-Black SOR → CG_SIMD ✅
   - **Rationale:** CG has O(√κ) convergence vs SOR's O(n) - typically 20-64 iterations vs 100s-1000s

2. **Increased Iteration Limits:**
   - [x] max_iterations: 1000 → 5000 (accommodates CG on fine grids) ✅

3. **Achieved Scientific Target:**
   - [x] RMS_u: 0.0382 (target < 0.10) ✅
   - [x] RMS_v: 0.0440 (target < 0.10) ✅
   - [x] Grid convergence now monotonic: 17×17 (0.046) → 25×25 (0.037) → 33×33 (0.032) ✅

**Backend Validation Complete (Feb 2026):**

3. **Test ALL solver backends (systematic validation):** ✅
   - [x] CPU scalar (explicit Euler, projection) ✅
   - [x] AVX2/SIMD (explicit Euler, projection) ✅
   - [x] OpenMP (explicit Euler, projection) ✅
   - [x] CUDA GPU (projection Jacobi) - test created, runs when GPU available ✅
   - [x] Each backend independently achieves accuracy target ✅

   **Test file:** `tests/validation/test_cavity_backends.c`

   **Results (33×33 grid, Re=100):**
   - Projection (CPU): RMS_u=0.0382, RMS_v=0.0440 < 0.10 ✅
   - Projection (AVX2): RMS_u=0.0382, RMS_v=0.0440 < 0.10 ✅
   - Projection (OMP): RMS_u=0.0382, RMS_v=0.0440 < 0.10 ✅
   - Explicit Euler (CPU): RMS_u=0.0957, RMS_v=0.1284 < 0.15 ✅
   - Explicit Euler (AVX2): RMS_u=0.0954, RMS_v=0.1312 < 0.15 ✅
   - Explicit Euler (OMP): RMS_u=0.0957, RMS_v=0.1284 < 0.15 ✅
   - Backend consistency: All 3 backends within 0.1% ✅

   **Documentation:**
   - `tests/validation/test_cavity_backends.c` - Comprehensive backend validation
   - `docs/validation/cavity-backends-validation.md` - Test methodology and results

4. **Verification that tests are honest:** ✅
   - [x] Tests MUST fail if RMS >= target (no loose tolerances) ✅
   - [x] Tests compare computed values at EXACT Ghia sample points ✅
   - [x] Tests report actual vs expected values transparently ✅
   - [x] No "current baseline" workarounds - fix solver, not tolerance ✅

   **Target enforcement:**
   - Projection method: RMS < 0.10 (production solver, strict)
   - Explicit Euler: RMS < 0.15 (simpler method, relaxed)
   - Tests FAIL immediately if RMS >= target

**Remaining Work (for 129×129 full validation):**

- [ ] Run full validation at 129×129 grid
- [ ] All backends at publication-quality grid resolution
- [ ] Expected runtime: ~30 minutes
- [ ] Use CAVITY_FULL_VALIDATION=1 build flag

**Acceptance Criteria (non-negotiable):**

- RMS error vs Ghia < 0.10 for Re=100, 400, 1000
- All 7 solver backends produce identical results (within 0.1%)
- Grid convergence: error decreases monotonically with refinement
- Tests run in parallel to complete in < 60 seconds

#### 6.1.1.1 Grid Convergence Validation (P1)

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

#### 6.1.2 Taylor-Green Vortex Validation (P0) ✅

**Analytical solution with known decay rate - ideal for full NS validation:**

```c
u(x,y,t) = cos(x) * sin(y) * exp(-2νt)
v(x,y,t) = -sin(x) * cos(y) * exp(-2νt)
p(x,y,t) = -0.25 * (cos(2x) + cos(2y)) * exp(-4νt)
```

**Tests implemented:**

- [x] Verify velocity decay rate matches exp(-2νt)
- [x] Verify kinetic energy decay: KE(t) = KE₀ * exp(-4νt)
- [x] Test L2 error remains bounded
- [x] Verify grid convergence (error decreases with refinement)
- [x] Verify divergence-free constraint (incompressibility)
- [x] Compare solver backends (CPU, OpenMP)
- [x] Long-time stability tests
- [x] Low viscosity stability tests

**Files created:**

- `tests/validation/test_taylor_green_vortex.c` - 9 validation tests
- `tests/validation/taylor_green_reference.h` - Analytical solutions and test utilities

#### 6.1.3 Poiseuille Flow Validation (P1) ✅

**Analytical parabolic profile for channel flow:**

```c
u(y) = 4 * U_max * y * (H - y) / H²
```

**Tests implemented:**

- [x] Velocity profile stability (analytical solution preserved by solver)
- [x] Mass conservation verification (flux in = flux mid = flux out)
- [x] Pressure gradient accuracy vs analytical dp/dx = -8μU_max/H²
- [x] Inlet BC accuracy (parabolic profile applied exactly)

**Results:**

- Profile RMS error: <1% (tolerance 1%)
- Mass flux conservation: <1% (tolerance 1%)
- Pressure gradient error: <5% (tolerance 5%)
- Inlet BC: machine precision

**Strategy:** Initialize with analytical solution, run projection steps, verify stability.

**Files created:**
- `tests/validation/test_poiseuille_flow.c`

#### 6.1.4 Other Benchmarks (P2)

- [ ] Backward-facing step - compare to Armaly et al. (1983)
- [ ] Flow over cylinder - compare to Williamson (1996)

#### 6.1.5 Release Validation Workflow (P1)

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
| Taylor-Green Vortex | 32×32, 200 steps | 128×128, 10000 steps |

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

### 6.2 Convergence Studies (P1)

- [ ] Grid independence studies for benchmark cases
- [ ] Time step independence studies
- [ ] Richardson extrapolation for error estimation
- [ ] Automated convergence reporting

### 6.3 Documentation (P1)

- [ ] Doxygen API documentation
- [ ] Theory/mathematics guide
- [ ] User tutorials
- [ ] Installation guide
- [ ] Troubleshooting guide
- [ ] Performance tuning guide
- [ ] Developer guide

### 6.4 Examples (P1)

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

---

## Phase 7: ML Inference Engine

**Goal:** Enable fast inference of pre-trained neural network surrogate models in pure C, without Python dependencies.

**Priority:** P3 - Future enhancement

### Rationale

Train models in Python (PyTorch/JAX), deploy inference in C for:

- ~1000× speedup over traditional CFD for inference
- No external dependencies at runtime
- SIMD-optimized matrix operations
- Embedded systems / HPC integration

### 7.1 Weight Format & Loader (P3)

- [ ] Define binary weight format (.cfdnn)
  ```c
  typedef struct {
      uint32_t magic;           // "CFDN"
      uint32_t version;
      uint32_t num_layers;
      uint32_t input_dim;
      uint32_t output_dim;
      // Layer descriptors follow
  } cfdnn_header_t;
  ```

- [ ] JSON metadata support for model info
- [ ] Weight loading API
  ```c
  cfdnn_model_t* cfdnn_load(const char* path);
  void cfdnn_free(cfdnn_model_t* model);
  ```

### 7.2 Layer Implementations (P3)

- [ ] Dense (fully connected) layer
  ```c
  void cfdnn_dense_forward(
      const double* input, size_t in_features,
      const double* weights, const double* bias,
      double* output, size_t out_features
  );
  ```

- [ ] Activation functions
  - [ ] ReLU, Leaky ReLU
  - [ ] Tanh, Sigmoid
  - [ ] GELU (for transformers)
  - [ ] Swish/SiLU

- [ ] Batch normalization
- [ ] Layer normalization
- [ ] Dropout (inference mode = identity)

### 7.3 SIMD-Optimized Kernels (P3)

- [ ] AVX2 matrix-vector multiply
- [ ] AVX2 matrix-matrix multiply (for batched inference)
- [ ] NEON equivalents for ARM
- [ ] Cache-optimized tiling for large layers

```c
// Example: SIMD dense layer
void cfdnn_dense_forward_avx2(
    const double* input, size_t in_features,
    const double* weights, const double* bias,
    double* output, size_t out_features
);
```

### 7.4 Model Architectures (P3)

| Architecture | Status | Notes |
|--------------|--------|-------|
| Fully Connected (MLP) | Planned | Priority - most common for PINNs |
| Convolutional (Conv2D) | Future | For grid-based models |
| Fourier Neural Operator | Future | State-of-the-art for PDEs |

### 7.5 Inference API (P3)

```c
// High-level inference API
typedef struct {
    size_t nx, ny;
    double Re;
    double* boundary_conditions;  // Flattened BC values
} cfdnn_input_t;

typedef struct {
    double* u;   // Velocity x-component (nx * ny)
    double* v;   // Velocity y-component (nx * ny)
    double* p;   // Pressure field (nx * ny)
} cfdnn_output_t;

cfd_status_t cfdnn_predict(
    const cfdnn_model_t* model,
    const cfdnn_input_t* input,
    cfdnn_output_t* output
);

// Batch inference for parameter sweeps
cfd_status_t cfdnn_predict_batch(
    const cfdnn_model_t* model,
    const cfdnn_input_t* inputs, size_t batch_size,
    cfdnn_output_t* outputs
);
```

### 7.6 Hybrid Solver Integration (P3)

- [ ] Use ML prediction as initial guess for iterative solver
  ```c
  // ML-accelerated projection method
  cfd_status_t ns_solve_projection_ml_init(
      const cfdnn_model_t* surrogate,
      ns_solver_t* solver,
      // ... other params
  );
  ```

- [ ] Adaptive switching: ML for steady-state estimate, CFD for accuracy
- [ ] Uncertainty quantification (ensemble models)

### 7.7 Benchmarking & Validation (P3)

- [ ] Inference time benchmarks vs Python
- [ ] Accuracy comparison with CFD solver
- [ ] Memory usage profiling
- [ ] Example: Cavity flow surrogate

### Success Criteria

- Load and run inference on exported PyTorch models
- <1ms inference time for 64×64 grid
- <5% L2 error compared to CFD solver (Re < 200)
- SIMD kernels show >2× speedup over scalar

### References

- [GGML](https://github.com/ggerganov/ggml) - Lightweight C tensor library
- [ONNX Runtime C API](https://onnxruntime.ai/) - Alternative: use ONNX format
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers) - Embedded ML reference

---

## Phase 8: Hybrid ML Integration (Optional)

**Goal:** Provide C-level building blocks for ML workflows while Python handles training and high-level orchestration.

**Priority:** P3 - Optional/Alternative to Phase 7

### Rationale

Alternative to full C inference (Phase 7). Best of both worlds:

- Python handles training, model management, and orchestration
- C provides optimized compute kernels callable from Python
- Lower implementation effort than full C inference engine
- Easier to support new architectures (just add kernels)

### 8.1 Optimized Compute Kernels (P3)

Expose SIMD-optimized operations for ML workloads via Python bindings:

```c
// Matrix operations optimized for neural network inference
cfd_status_t cfd_matmul(
    const double* A, size_t A_rows, size_t A_cols,
    const double* B, size_t B_cols,
    double* C  // Output: A_rows x B_cols
);

cfd_status_t cfd_matmul_add_bias(
    const double* A, size_t A_rows, size_t A_cols,
    const double* B, size_t B_cols,
    const double* bias,  // B_cols
    double* C
);

// Activation functions (in-place)
cfd_status_t cfd_relu(double* data, size_t n);
cfd_status_t cfd_tanh(double* data, size_t n);
cfd_status_t cfd_gelu(double* data, size_t n);
```

### 8.2 Python-Callable Kernels (P3)

Expose kernels to Python via bindings:

```python
from cfd_kernels import matmul, relu, gelu

# Use C kernels in custom PyTorch module
class CFDAcceleratedLayer(torch.nn.Module):
    def forward(self, x):
        # Offload to SIMD-optimized C kernel
        out = matmul(x.numpy(), self.weight.numpy())
        return torch.from_numpy(relu(out))
```

### 8.3 Physics Residual Kernels (P3)

Optimized physics loss computation for PINN training:

```c
// Compute Navier-Stokes residuals efficiently
cfd_status_t cfd_ns_residual(
    const double* u, const double* v, const double* p,
    size_t nx, size_t ny, double dx, double dy, double Re,
    double* continuity_residual,
    double* momentum_x_residual,
    double* momentum_y_residual
);

// Finite difference operators (2nd order central)
cfd_status_t cfd_gradient_x(const double* f, size_t nx, size_t ny, double dx, double* df_dx);
cfd_status_t cfd_gradient_y(const double* f, size_t nx, size_t ny, double dy, double* df_dy);
cfd_status_t cfd_laplacian(const double* f, size_t nx, size_t ny, double dx, double dy, double* lap_f);
```

Python usage:

```python
from cfd_kernels import ns_residual

# Fast physics loss for PINN training
def physics_loss(model, coords, Re):
    u, v, p = model(coords)
    cont, mom_x, mom_y = ns_residual(u, v, p, nx, ny, dx, dy, Re)
    return (cont**2 + mom_x**2 + mom_y**2).mean()
```

### 8.4 Data Generation Acceleration (P3)

Fast CFD simulation for training data generation:

```c
// Batch simulation for dataset generation
cfd_status_t cfd_generate_samples(
    const cfd_sample_params_t* params, size_t n_samples,
    cfd_sample_result_t* results,
    int n_threads  // OpenMP parallelization
);
```

```python
from cfd_ml import generate_training_data

# Generate 1000 cavity flow samples in parallel
samples = generate_training_data(
    problem="cavity",
    Re_range=(10, 200),
    n_samples=1000,
    n_threads=8  # Uses OpenMP in C
)
```

### 8.5 Memory-Mapped Data Sharing (P3)

Zero-copy data sharing between C and Python/NumPy:

```c
// Create memory-mapped buffer accessible from Python
cfd_buffer_t* cfd_create_shared_buffer(size_t size);
double* cfd_buffer_data(cfd_buffer_t* buf);
void cfd_buffer_free(cfd_buffer_t* buf);
```

```python
from cfd_memory import SharedBuffer

# Zero-copy data sharing
buf = SharedBuffer(shape=(128, 128))
arr = buf.as_numpy()  # No copy, direct memory access

# Pass to C functions without copying
cfd.run_simulation(output_buffer=buf)
```

### Comparison: Phase 7 vs Phase 8

| Aspect | Phase 7 (Full C Inference) | Phase 8 (Hybrid) |
|--------|---------------------------|------------------|
| Implementation effort | High | Medium |
| Python dependency | None at runtime | Required |
| New architecture support | Requires C changes | Just Python |
| Deployment | Embedded/HPC friendly | Python environment |
| Performance | Highest | High (kernel-level) |
| Flexibility | Fixed architectures | Any PyTorch model |

### Phase 8 Success Criteria

- Kernels provide >2x speedup over NumPy equivalents
- Zero-copy memory sharing works with NumPy arrays
- Physics residual kernel matches Python implementation to 1e-10
- Dataset generation scales linearly with thread count

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

### v0.1.5

- [x] OpenMP solver variants documented (explicit_euler_omp, projection_omp)
- [x] Updated solver type constants with NS_SOLVER_TYPE_ prefix
- [x] Doxyfile version synchronized with VERSION file
- [x] Ghia validation tests use appropriate step counts for Euler solvers

### v0.1.6

- [x] CI: Doxygen API documentation generation in release workflow
- [x] CI: macOS OpenMP support via Homebrew libomp
- [x] CI: Improved CUDA toolkit configuration for Windows builds
- [x] CI: Fixed library path detection for modular builds

### v0.1.7 - Current Release

- [x] Fix stretched grid formula to correctly span domain [xmin, xmax]
- [x] Add tanh-based stretching for boundary layer clustering
- [x] Add 16 unit tests for grid initialization functions

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
