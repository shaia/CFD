# CFD Library Roadmap to v1.0

This document outlines the development roadmap for achieving a commercial-grade, open-source CFD library.

## Current State (v0.2.0)

### What We Have

- [x] Pluggable solver architecture (function pointers, registry pattern)
- [x] Multiple solver backends (CPU, SIMD/AVX2, OpenMP, CUDA)
- [x] 2D/3D incompressible Navier-Stokes solver
- [x] Full 3D support with branch-free stride_z=0 pattern for 2D fallback
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
|                     | RK2 (Heun)     | done | done     | —        | done     | —    |
| **Linear Solvers**  | Jacobi         | done | done     | done     | —        | —    |
|                     | SOR            | done | done     | done     | —        | —    |
|                     | Red-Black SOR  | done | done     | done     | done     | —    |
|                     | CG / PCG       | done | done     | done     | done     | —    |
|                     | BiCGSTAB       | done | done     | done     | —        | —    |
| **Boundary Conds**  | All types      | done | done     | done     | done     | done |

### What's Missing

- [ ] No 3D validation benchmarks yet
- [ ] No turbulence models
- [ ] Limited linear solvers (no multigrid)
- [ ] No restart/checkpoint capability

### Known Issues

#### Active Bugs

**OMP Red-Black SOR Poisson Solver Convergence (P1)** ✅ RESOLVED

Root cause: Hard-coded omega=1.5 was suboptimal for larger grids. For a 33×33 grid, the optimal SOR omega is ~1.83. With omega=1.5, the spectral radius was too high, causing convergence to exceed max iterations.

Fix: All SOR and Red-Black SOR solvers now auto-compute optimal omega from grid dimensions using the Jacobi spectral radius formula: ω_opt = 2/(1+√(1-ρ_J²)). Users can override by setting `params.omega > 0`. Projection backends remain on CG (grid-size-independent).

Action items:

- [x] Profile OMP Red-Black SOR to identify convergence bottleneck — suboptimal omega
- [x] Compare OMP vs AVX2 Red-Black implementations for differences — identical algorithm
- [x] Test omega parameter sweep (1.0 to 1.9) for optimal convergence — auto-computed
- [x] Add convergence diagnostics (residual history logging) — test_optimal_omega.c
- [x] Consider switch to Chebyshev acceleration or SSOR — not needed, optimal omega suffices

**Grid Convergence Non-Monotonic Behavior (P1)** ✅ RESOLVED

Root cause: Red-Black SOR Poisson solver had insufficient convergence on larger grids. Switching all projection backends to CG (Conjugate Gradient) resolved the issue. Grid convergence is now strictly monotonic (17×17: 0.046, 25×25: 0.037, 33×33: 0.032).

Action items:

- [x] Investigate why 33×33 produces worse results than 25×25 — Red-Black SOR Poisson solver
- [x] May need more Poisson iterations for larger grids — solved by switch to CG
- [x] Consider using SIMD Red-Black SOR or multigrid for better accuracy — CG sufficient
- [x] Remove relaxed tolerance from grid convergence tests
- [x] Add strict grid convergence test that FAILs if RMS increases with refinement

#### Limitations

**SIMD Poisson Strict Tolerance (Section 1.2)**

Current SIMD Poisson solvers produce valid results but may not converge to strict tolerance (1e-6) on challenging problems like sinusoidal RHS within iteration limits. They converge properly on simpler problems (zero RHS, uniform RHS). See `docs/simd-optimization-analysis.md` for details.

**Convergence Order BC-Limited (Section 1.3.2)**

Spatial convergence achieves ~O(h^1.5) rather than theoretical O(h²), limited by first-order boundary conditions. Temporal convergence O(dt) is difficult to isolate; spatial error dominates on practical grids.

**Jacobi Preconditioner on Uniform Grids (Section 1.2.4)**

For the uniform-grid Laplacian with constant coefficients, the Jacobi preconditioner M⁻¹ = 1/(2/dx² + 2/dy²) is a constant scalar, which doesn't improve the condition number. PCG provides benefit for problems with variable coefficients or non-uniform grids.

**Modular Library Circular Dependencies (Section 4.2)**

The modular libraries have circular dependencies: `cfd_scalar`/`cfd_simd` call `poisson_solve()` (defined in `cfd_api`), while `cfd_api` links against `cfd_scalar`/`cfd_simd`. Resolved on Linux with linker groups; Windows/macOS handle automatically. Future: consider weak symbols or plugin architecture.

**OMP Loop Variable int Overflow for Large Grids (P3)**

Status: Deferred — low risk, no practical impact yet

OMP backends cast `size_t` loop variables to `int` for MSVC OpenMP 2.0 compatibility. Overflows for grids where `nx * ny > INT_MAX` (~46K × 46K).

- [ ] Audit all OMP backends for `size_t` → `int` casts
- [ ] Add `CFD_ASSERT(nx * ny <= INT_MAX)` guards if targeting large grids
- [ ] Consider requiring OpenMP 3.0+ when dropping MSVC OMP 2.0 support

**Debug Mode SIMD Benchmarking (Section 1.10)**

Current tests run in Debug mode where SIMD may be slower than scalar due to lack of compiler optimizations. Release mode benchmark suite needed for accurate performance measurements.

#### Technical Notes

**Spectral Radius Test BC Assumption (Section 1.2.3)**

The Jacobi spectral radius test uses Dirichlet BCs (p=0 on boundary) because the ρ = cos(πh) formula applies only to the Dirichlet problem. The SOR optimal ω = 2/(1 + sin(πh)) also applies to Dirichlet BCs; with Neumann BCs optimal ω is typically lower (1.5-1.7).

---

## Phase 0: Architecture & Robustness (P0 - Critical) ✅

**Status:** COMPLETE

- [x] 0.1 Safe Error Handling — `cfd_status_t`, error propagation, resource cleanup
- [x] 0.2 Thread Safety — removed static buffers, thread-safe registry, re-entrant solvers
- [x] 0.3 API Robustness — input validation, logging callback, version header, symbol visibility
- [x] 0.4 API & Robustness Testing — comprehensive test suites for all subsystems
- [x] 0.5 Error Handling & Robustness — removed static `warned` flag, `run_simulation_step()` returns `cfd_status_t`
- [x] 0.6 Structured Logging & Diagnostics — `cfd_log()` API with levels, component tags, thread-safe callbacks

**0.6 Deferred items:**

- [ ] Log filtering by component (e.g., only show "boundary" logs)
- [ ] Timestamps and colored output
- [ ] Structured data API for metrics (convergence stats, timings)

---

## Phase 1: Core Solver Improvements

**Goal:** Make the solver practically usable for real problems.

### 1.1 Boundary Conditions (P0 - Critical) ✅

All BC types implemented across all backends (Scalar, AVX2, NEON, OMP, GPU): Dirichlet, Neumann, Periodic, No-slip, Inlet (uniform/parabolic/custom), Outlet, Symmetry, Moving wall, Time-varying. Code refactored with shared templates (~505 lines removed).

### 1.2 Linear Solvers (P0 - Critical)

**Implemented:** Jacobi, SOR, Red-Black SOR, CG/PCG, BiCGSTAB (all with SIMD backends). CG is the default Poisson solver for all projection methods.

**Still needed:**

- [ ] GMRES (Generalized Minimal Residual) for non-symmetric systems
  - [ ] GMRES scalar
  - [ ] GMRES AVX2
  - [ ] GMRES NEON
  - [ ] GMRES OMP
  - [ ] GMRES GPU
- [ ] SSOR (Symmetric SOR) preconditioner
- [ ] GPU (CUDA) linear solver backends
  - [ ] Standalone Jacobi GPU (refactor from monolithic `solver_projection_jacobi_gpu.cu`)
  - [ ] Red-Black SOR GPU
  - [ ] CG GPU
  - [ ] BiCGSTAB GPU
- [ ] ILU preconditioner
- [ ] Geometric multigrid
- [ ] Algebraic multigrid (AMG) solver
- [ ] AMG preconditioner (for use with CG/GMRES/BiCGSTAB)
- [ ] Performance benchmarking in Release mode

**Completed sub-sections:** Poisson Accuracy Tests (1.2.1), Laplacian Validation (1.2.2), Convergence Validation (1.2.3), PCG (1.2.4).

### 1.3 Numerical Schemes (P1)

- [ ] Upwind differencing (1st order) for stability
- [ ] Central differencing with delayed correction
- [ ] High-resolution TVD schemes (Van Leer, Superbee)
- [ ] Gradient limiters (Barth-Jespersen, Venkatakrishnan)
- [ ] Migrate solver code to use `cfd/math/stencils.h` (currently inline implementations)

**Completed sub-sections:** Stencil Tests (1.3.1), Convergence Order (1.3.2), MMS (1.3.3), Divergence-Free (1.3.4).

### 1.4 Steady-State Solver (P1)

- [ ] SIMPLE algorithm for incompressible flow
- [ ] SIMPLEC / PISO variants
- [ ] Pseudo-transient continuation
- [ ] Convergence acceleration (relaxation)

### 1.5 Time Integration (P1)

Each time integrator requires a scalar (CPU) reference implementation first, then backend variants. See `/add-ns-time-integrator` for the cross-backend workflow.

**Implemented:** RK2 (Heun's method) — all CPU/AVX2/OMP backends, O(dt²) verified.

**Still needed:**

- [ ] RK2 CUDA (`rk2_gpu`)
- [ ] RK4 (classical Runge-Kutta) — scalar
  - [ ] RK4 AVX2+OMP (`rk4_optimized`)
  - [ ] RK4 OpenMP (`rk4_omp`)
  - [ ] RK4 CUDA (`rk4_gpu`)
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

### 1.8 Eigenvalue Solvers (P3)

Find eigenvalues/eigenvectors for stability analysis.

- [ ] Power iteration
- [ ] Inverse iteration
- [ ] Arnoldi iteration
- [ ] Stability analysis framework

### 1.9 Derived Fields (P2)

**Status:** OpenMP implemented, SIMD and CUDA pending

**Current:** OpenMP parallelization for velocity magnitude and field statistics with threshold-based parallelization (OMP_THRESHOLD = 1000 cells).

**Still needed:**

#### SIMD Optimization (AVX2/NEON)

- [ ] Add AVX2 implementation for velocity magnitude
- [ ] Add NEON implementation for velocity magnitude (ARM64)
- [ ] Combine with OpenMP (`#pragma omp simd` or manual intrinsics)
- [ ] Add SIMD horizontal reduction for statistics
- [ ] Add runtime CPU feature detection
- [ ] Benchmark against scalar+OpenMP version

#### CUDA GPU Acceleration

- [ ] Add CUDA kernel for velocity magnitude
- [ ] Add parallel reduction for statistics (use CUB library or custom)
- [ ] Share GPU memory with CUDA solvers (avoid CPU<->GPU transfers)
- [ ] Add threshold logic (only use GPU for large grids)
- [ ] Benchmark transfer overhead vs compute benefit

### 1.10 SIMD Projection Solver Optimization (P2)

**Status:** SIMD Poisson integration completed, further optimizations pending. Current ~1.3-1.5x speedup limited by Amdahl's law (parallelizable fraction ~80%).

#### High Priority

- [ ] Increase `POISSON_MAX_ITER` or implement adaptive tolerance
- [ ] Add optional multigrid preconditioner for faster convergence
- [ ] Investigate Red-Black omega parameter tuning
- [ ] Create Release mode benchmark suite
- [ ] Profile to identify remaining bottlenecks

#### Medium Priority — OpenMP + SIMD Hybrid

- [ ] Implement hybrid projection solver using OpenMP across rows + SIMD within rows
- [ ] Benchmark scaling efficiency
- [ ] Compare against pure OMP and pure SIMD implementations

#### Low Priority — Multigrid SIMD

- [ ] Implement multigrid V-cycle framework
- [ ] Add restriction/prolongation operators
- [ ] Use SIMD Jacobi/Red-Black as smoothers at each level
- [ ] Benchmark convergence rate vs pure iterative methods

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

### 3.1 3D Support (P0 - Critical) ✅

**Status:** Complete. "2D as subset of 3D" approach — `nz=1` produces bit-identical results to previous 2D code. Branch-free solver loops using precomputed constants (`stride_z=0`, `inv_dz2=0.0`). All 8 phases done: indexing macros, core data structures, 3D stencils, NS solvers, BCs, SIMD, OMP, CUDA, I/O, examples, docs.

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

**Status:** Implemented in v0.1.5. Split monolithic library into per-backend targets.

| Library | CMake Target | Contents | Dependencies |
| ------- | ------------ | -------- | ------------ |
| `cfd_core` | `CFD::Core` | Grid, memory, I/O, status, common utilities | None |
| `cfd_scalar` | `CFD::Scalar` | Scalar CPU solvers | CFD::Core |
| `cfd_simd` | `CFD::SIMD` | AVX2/NEON SIMD solvers | CFD::Core, CFD::Scalar |
| `cfd_omp` | `CFD::OMP` | OpenMP parallelized solvers | CFD::Core, CFD::Scalar |
| `cfd_cuda` | `CFD::CUDA` | CUDA GPU solvers | CFD::Core |
| `cfd_library` | `CFD::Library` | Unified library (all backends) | All above |

**Still needed:**

- [ ] Update examples/tests to link against specific backends (optional)
- [ ] Plugin loading system for dynamic backend selection

See [Known Issues](#known-issues) — modular library circular dependencies.

### 4.3 GPU Improvements (P2)

**Implemented:** CUDA device detection, GPU projection with Jacobi Poisson, GPU memory management, GPU BC kernels, configurable GPU settings, GPU solver statistics.

**Still needed:**

- [ ] Multi-GPU support
- [ ] Unified memory optimization
- [ ] Advanced async transfers (multi-stream overlap, double buffering)
- [ ] GPU-aware MPI
- GPU linear solver backends tracked in section 1.2

### 4.4 Performance Tools (P2)

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

**Implemented:** VTK legacy ASCII format, scalar/vector/flow field output, timestamped run directories.

**Still needed:**

- [ ] VTK XML format (.vtu, .pvtu)
- [ ] Parallel VTK files
- [ ] Time series support
- [ ] Binary encoding

### 5.3 CSV Output ✅

Implemented: timeseries data, centerline profiles, global statistics, velocity magnitude, automatic headers.

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

### 6.1 Benchmark Validation (P0 - Critical)

#### 6.1.1 Lid-Driven Cavity Validation ✅

**Status:** COMPLETE — All backends validated at 33×33, Re=100. Projection RMS < 0.10, Explicit Euler RMS < 0.15, all backends within 0.1% of each other. Grid convergence now monotonic.

**Remaining work (129×129 full validation):**

- [ ] Confirm all backends pass at 129×129 on EC2 and record RMS values
- [ ] Use CAVITY_FULL_VALIDATION=1 build flag

**Acceptance Criteria:**

- RMS error vs Ghia < 0.10 for Re=100, 400, 1000
- All 7 solver backends produce identical results (within 0.1%)
- Grid convergence: error decreases monotonically with refinement

##### 6.1.1.1 Grid Convergence Validation (P1) ✅

Resolved by switching projection backends from Red-Black SOR to CG. Grid convergence tests now enforce strict monotonicity with convergence rate reporting.

#### 6.1.2 Taylor-Green Vortex Validation (P0) ✅

Velocity/energy decay, L2 error, grid convergence, divergence-free, backend consistency, long-time and low-viscosity stability — all verified.

#### 6.1.3 Poiseuille Flow Validation (P1) ✅

Profile stability (<1% RMS), mass conservation (<1%), pressure gradient (<5% error), inlet BC (machine precision) — all verified.

#### 6.1.4 Other Benchmarks (P2)

- [ ] Backward-facing step - compare to Armaly et al. (1983)
- [ ] Flow over cylinder - compare to Williamson (1996)

#### 6.1.5 Release Validation Workflow (P1)

**Goal:** Full-length validation tests that run during releases (too slow for CI).

**CI vs Release Parameters:**

| Test | CI Mode | Release Mode |
|------|---------|--------------|
| Cavity Ghia Validation | 33×33, 5000 steps | 129×129, 50000 steps |
| Cavity Re=400 Stability | 25×25, 500 steps | 65×65, 20000 steps |
| Grid Convergence | 17→25→33 | 33→65→129 |
| Taylor-Green Vortex | 32×32, 200 steps | 128×128, 10000 steps |

**Release Validation Tests to Implement:**

- [ ] Full Ghia validation at 129×129 grid for Re=100, 400, 1000
- [ ] Extended cavity flow convergence (run to true steady-state, residual < 1e-8)
- [ ] Multi-Reynolds grid convergence study (Richardson extrapolation)
- [ ] Taylor-Green vortex decay rate verification (extended time)
- [ ] Cross-architecture consistency check (all backends produce identical results)
- [ ] Memory usage and performance regression benchmarks

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

**Implemented:** 12 examples (minimal, basic simulation, animated flow, velocity visualization, performance/runtime comparison, solver selection, custom BCs, custom source terms, CSV export, lid-driven cavity).

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
- [ ] JSON metadata support for model info
- [ ] Weight loading API

### 7.2 Layer Implementations (P3)

- [ ] Dense (fully connected) layer
- [ ] Activation functions (ReLU, Leaky ReLU, Tanh, Sigmoid, GELU, Swish/SiLU)
- [ ] Batch normalization
- [ ] Layer normalization
- [ ] Dropout (inference mode = identity)

### 7.3 SIMD-Optimized Kernels (P3)

- [ ] AVX2 matrix-vector multiply
- [ ] AVX2 matrix-matrix multiply (for batched inference)
- [ ] NEON equivalents for ARM
- [ ] Cache-optimized tiling for large layers

### 7.4 Model Architectures (P3)

| Architecture | Status | Notes |
|--------------|--------|-------|
| Fully Connected (MLP) | Planned | Priority - most common for PINNs |
| Convolutional (Conv2D) | Future | For grid-based models |
| Fourier Neural Operator | Future | State-of-the-art for PDEs |

### 7.5 Inference API (P3)

- [ ] High-level `cfdnn_predict()` and `cfdnn_predict_batch()` API
- [ ] Model loading/freeing lifecycle

### 7.6 Hybrid Solver Integration (P3)

- [ ] Use ML prediction as initial guess for iterative solver
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

Alternative to full C inference (Phase 7). Python handles training/orchestration, C provides optimized compute kernels. Lower effort, easier to support new architectures.

### 8.1 Optimized Compute Kernels (P3)

- [ ] SIMD-optimized matrix operations (`cfd_matmul`, `cfd_matmul_add_bias`)
- [ ] Activation functions (ReLU, Tanh, GELU — in-place)

### 8.2 Python-Callable Kernels (P3)

- [ ] Python bindings for C compute kernels

### 8.3 Physics Residual Kernels (P3)

- [ ] Optimized NS residual computation for PINN training
- [ ] Finite difference operators (gradient, laplacian)

### 8.4 Data Generation Acceleration (P3)

- [ ] Batch simulation API for dataset generation with OpenMP parallelization

### 8.5 Memory-Mapped Data Sharing (P3)

- [ ] Zero-copy shared buffers between C and Python/NumPy

### Comparison: Phase 7 vs Phase 8

| Aspect | Phase 7 (Full C Inference) | Phase 8 (Hybrid) |
|--------|---------------------------|------------------|
| Implementation effort | High | Medium |
| Python dependency | None at runtime | Required |
| New architecture support | Requires C changes | Just Python |
| Deployment | Embedded/HPC friendly | Python environment |
| Performance | Highest | High (kernel-level) |
| Flexibility | Fixed architectures | Any PyTorch model |

---

## Version Milestones

### v0.1.7 - Current Release

- [x] Stretched grid formula fix, tanh-based stretching, grid unit tests

### v0.2.0 - 3D Support

- [x] Phases 0–4: Indexing macros, core data structures, 3D stencils, NS solvers, BCs
- [ ] Phase 5-7: SIMD, OMP, CUDA backends for 3D
- [ ] Phase 8: 3D VTK output, examples, validation (Taylor-Green 3D, Poiseuille 3D)

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
| -------- | ------- |
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

## References & Bibliography

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
