# CFD Library Roadmap to v1.0

This document outlines the development roadmap for achieving a commercial-grade, open-source CFD library.

## Current State (v0.1.0)

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
- [x] SIMD Poisson solvers (Jacobi and Red-Black SOR with AVX2/SSE2)
- [x] Boundary condition abstraction layer with runtime backend selection
- [x] Neumann and Periodic boundary conditions (all backends)
- [x] GPU boundary condition kernels (CUDA)
- [x] Comprehensive test suite (core, solvers, simulation, I/O)
- [x] 11 example programs demonstrating various features

### Critical Gaps

- [ ] Only 2D (no 3D support)
- [ ] Limited boundary conditions (no no-slip walls, inlets, outlets)
- [ ] Only structured grids
- [ ] No turbulence models
- [ ] Limited linear solvers (SOR/Jacobi only, no CG/BiCGSTAB/multigrid)
- [ ] No restart/checkpoint capability

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
- [x] SIMD-optimized BC application (AVX2/SSE2)
- [x] OpenMP-parallelized BC application
- [x] CUDA GPU BC kernels
- [x] Dirichlet (fixed value) boundary conditions
- [ ] No-slip wall conditions
- [ ] Inlet velocity specification
- [ ] Outlet (zero-gradient/convective)
- [ ] Symmetry planes
- [ ] Moving wall boundaries
- [ ] Time-varying boundary conditions

**Implemented files:**

- `lib/include/cfd/boundary/boundary_conditions.h` - Public API with backend selection
- `lib/include/cfd/boundary/boundary_conditions_gpu.cuh` - GPU API declarations
- `lib/src/boundary/cpu/boundary_conditions.c` - Scalar + runtime dispatch
- `lib/src/boundary/simd/boundary_conditions_simd.c` - AVX2/SSE2 optimizations
- `lib/src/boundary/omp/boundary_conditions_omp.c` - OpenMP parallelization
- `lib/src/boundary/gpu/boundary_conditions_gpu.cu` - CUDA kernels
- `lib/src/boundary/boundary_conditions_internal.h` - Internal declarations

### 1.2 Linear Solvers (P0 - Critical)

**Implemented:**

- [x] SOR (Successive Over-Relaxation) - CPU baseline
- [x] Jacobi SIMD (`poisson_jacobi_simd.c`) - AVX2/SSE2 vectorized, fully parallelizable
- [x] Red-Black SOR SIMD (`poisson_redblack_simd.c`) - AVX2/SSE2 with SOR convergence rate
- [x] GPU Jacobi Poisson solver (`solver_projection_jacobi_gpu.cu`)
- [x] Integrate SIMD Poisson into projection solver (`solver_projection_simd.c`)

**Still needed:**

- [ ] Solver abstraction interface
- [ ] Conjugate Gradient (CG) for SPD systems
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

### 4.2 GPU Improvements (P2)

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

**Files to create:**

- `tests/math/test_finite_differences.c`
- `tests/math/test_convergence_order.c`
- `tests/math/test_mms.c`
- `tests/math/manufactured_solutions.h`

### 6.2 Benchmark Validation (P0 - Critical)

- [ ] Lid-driven cavity (Re 100, 400, 1000) - compare to Ghia et al. (1982)
- [ ] Channel flow (Poiseuille) - compare to analytical parabolic profile
- [ ] Backward-facing step - compare to Armaly et al. (1983)
- [ ] Flow over cylinder - compare to Williamson (1996)
- [ ] Taylor-Green vortex decay - compare to analytical decay rate

**Files to create:**

- `tests/validation/test_benchmark_cases.c`
- `tests/validation/reference_data/ghia_cavity.h`

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

### v0.1.0 - Foundation (Current)

**Completed:**

- [x] Boundary condition abstraction layer with runtime backend selection
- [x] Neumann and Periodic boundary conditions (all backends)
- [x] SIMD Poisson solvers (Jacobi, Red-Black SOR)
- [x] GPU Jacobi Poisson solver
- [x] Comprehensive test suite
- [x] 11 example programs
- [x] VTK and CSV output

**Remaining for v0.1.0:**

- [ ] Proper boundary conditions (walls, inlet/outlet)
- [ ] At least one Krylov solver (CG or BiCGSTAB)
- [ ] Lid-driven cavity validation
- [ ] Basic documentation

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
