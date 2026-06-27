# CFD Library Roadmap to v1.0

The development roadmap for achieving a commercial-grade, open-source CFD library.
This file tracks **what's next**; shipped history lives in [CHANGELOG.md](CHANGELOG.md).

## Conventions

**Status markers** (used throughout this file):

| Marker | Meaning |
| ------ | ------- |
| `[ ]`  | Pending — not started |
| `[~]`  | In progress |
| `[x]`  | Done (kept only where it adds context; full history is in the CHANGELOG) |

Completed phases are collapsed to a single `✅ Done in vX.Y — see CHANGELOG` line rather
than itemized here.

**Priority legend:**

| Priority | Meaning |
| -------- | ------- |
| P0 | Critical — blocks v1.0 release |
| P1 | Important — required for v1.0 |
| P2 | Valuable — nice to have for v1.0 |
| P3 | Future — post v1.0 |

---

## Current State (v0.3.0)

A pluggable, multi-backend (CPU / AVX2 / NEON / OpenMP / CUDA) 2D/3D incompressible
Navier-Stokes library with projection and explicit time-stepping methods, an energy
equation with Boussinesq buoyancy, a full linear-solver suite, VTK/CSV output, and a
visualization library. Phase 0 (architecture, error handling, thread safety, structured
logging) is complete. See [CHANGELOG.md](CHANGELOG.md) for the shipped feature history.

### Backend Coverage Matrix

The single source of truth for backend gaps. Each algorithm targets scalar (CPU) + SIMD
(AVX2/NEON) + OMP + GPU variants.

| Category            | Algorithm      | CPU  | AVX2     | NEON     | OMP      | GPU  |
| ------------------- | -------------- | ---- | -------- | -------- | -------- | ---- |
| **N-S Solvers**     | Explicit Euler | done | done     | —        | done     | done |
|                     | Projection     | done | done     | —        | done     | done |
|                     | RK2 (Heun)     | done | done     | —        | done     | done |
|                     | RK4 (classical)| done | done     | —        | done     | done |
| **Energy Eq.**      | Advec-diff + Boussinesq + thermal BCs | done | done | — | done | done |
| **Linear Solvers**  | Jacobi         | done | done     | done     | —        | done |
|                     | SOR            | done | done     | done     | —        | —    |
|                     | Red-Black SOR  | done | done     | done     | done     | done |
|                     | CG / PCG       | done | done     | done     | done     | done |
|                     | BiCGSTAB       | done | done     | done     | —        | done |
| **Boundary Conds**  | All types      | done | done     | done     | done     | done |

### Known Limitations

Genuine constraints to be aware of (not backlog items):

- **SIMD Poisson strict tolerance** — SIMD Poisson solvers produce valid results but may
  not reach strict tolerance (1e-6) on challenging problems (e.g. sinusoidal RHS) within
  iteration limits; they converge fine on simpler RHS. See
  `docs/technical-notes/simd-optimization-analysis.md`.
- **Convergence order BC-limited** — spatial convergence achieves ~O(h^1.5) rather than the
  theoretical O(h²), limited by first-order boundary conditions. Temporal O(dt) is hard to
  isolate as spatial error dominates on practical grids.
- **Jacobi preconditioner on uniform grids** — for the uniform-grid constant-coefficient
  Laplacian, M⁻¹ = 1/(2/dx² + 2/dy²) is a constant scalar and doesn't improve conditioning.
  PCG benefits only variable-coefficient or non-uniform-grid problems.
- **Modular library circular dependencies** — `cfd_scalar`/`cfd_simd` call `poisson_solve()`
  (in `cfd_api`) while `cfd_api` links against them. Resolved on Linux via linker groups;
  Windows/macOS handle automatically. Future option: weak symbols or a plugin architecture.

> The SOR optimal-ω and Jacobi spectral-radius formulas (ρ = cos(πh),
> ω = 2/(1 + sin(πh))) assume Dirichlet BCs; with Neumann BCs optimal ω is typically lower
> (1.5–1.7).

---

## Roadmap at a Glance

| Phase | Theme | Priority | Status — what remains |
| ----- | ----- | -------- | --------------------- |
| 1 | Core Solver Improvements | P0–P3 | GMRES, multigrid, implicit integrators, SIMPLE/PISO, nonlinear & eigenvalue solvers |
| 2 | Physics Extensions | P1–P3 | Turbulence (RANS), compressible, species, multiphase; energy-eq. extensions |
| 3 | Geometry & Mesh | P1–P2 | Unstructured meshes, mesh I/O, adaptive refinement (3D ✅) |
| 4 | Scalability & Performance | P1–P2 | MPI, GPU improvements, profiling tools (modular libs ✅) |
| 5 | I/O & Post-processing | P1–P3 | HDF5, modern VTK XML, in-situ viz (CSV ✅) |
| 6 | Validation & Documentation | P0–P1 | 129×129 release validation, convergence studies, docs |
| 7 | ML Integration | P3 | Future — full-C inference vs hybrid kernels |

---

## Phase 1: Core Solver Improvements

**Goal:** make the solver practically usable for real problems.

### 1.1 Boundary Conditions (P0)

✅ Done in v0.1.5 — all BC types across all backends (Dirichlet, Neumann, Periodic,
No-slip, Inlet, Outlet, Symmetry, Moving wall, Time-varying). See CHANGELOG.

### 1.2 Linear Solvers (P0)

Implemented: Jacobi, SOR, Red-Black SOR, CG/PCG, BiCGSTAB (all with SIMD backends; CG is the
default Poisson solver for projection methods). GPU standalone Jacobi, CG, Red-Black SOR, and
BiCGSTAB are done and validated vs CPU; `solve_projection_method_gpu` uses on-device CG.

**Still needed:**

- [ ] GMRES (Generalized Minimal Residual) for non-symmetric systems — scalar, AVX2, NEON,
      OMP, GPU
- [ ] SSOR (Symmetric SOR) preconditioner
- [ ] ILU preconditioner
- [ ] Geometric multigrid
- [ ] Algebraic multigrid (AMG) — solver and preconditioner (for CG/GMRES/BiCGSTAB)
- [ ] GPU plain SOR (the one remaining backend gap in the matrix above)

### 1.3 Numerical Schemes (P1)

Stencil tests, convergence-order, MMS, and divergence-free validation are done (see CHANGELOG).

**Still needed:**

- [ ] Upwind differencing (1st order) for stability
- [ ] Central differencing with delayed correction
- [ ] High-resolution TVD schemes (Van Leer, Superbee)
- [ ] Gradient limiters (Barth-Jespersen, Venkatakrishnan)
- [ ] Migrate solver code to use `cfd/math/stencils.h` (currently inline)

### 1.4 Steady-State Solver (P1)

- [ ] SIMPLE algorithm for incompressible flow
- [ ] SIMPLEC / PISO variants
- [ ] Pseudo-transient continuation
- [ ] Convergence acceleration (relaxation)

### 1.5 Time Integration (P1)

Implemented: RK2 (Heun) and RK4 (classical), all CPU/AVX2/OMP/GPU backends, O(dt²)/O(dt⁴)
verified. See `/add-ns-time-integrator` for the cross-backend workflow.

**Still needed:**

- [ ] Implicit Euler (backward Euler)
- [ ] Crank-Nicolson (2nd order implicit)
- [ ] BDF2 (backward differentiation)
- [ ] Adaptive time stepping with error control

### 1.6 Restart / Checkpoint (P1)

- [~] Binary checkpoint format (`.cfdchk`) — portable, versioned, CRC-protected
      (`lib/src/io/checkpoint.c`)
- [~] Save/restore complete simulation state (grid, field, scalar params, time, solver name)
- [~] Portable across platforms (little-endian fixed-width, endianness marker)
- [~] Version compatibility (format-version header, rejects unknown versions)

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

OpenMP parallelization for velocity magnitude and field statistics is done. SIMD and CUDA
pending.

- [ ] AVX2 + NEON velocity magnitude (with `#pragma omp simd` / intrinsics, runtime feature
      detection, SIMD horizontal reduction for statistics)
- [ ] CUDA velocity magnitude + parallel reduction (CUB or custom), GPU-memory sharing with
      CUDA solvers, large-grid threshold
- [ ] Benchmark SIMD/GPU vs scalar+OpenMP (transfer overhead vs compute benefit)

### 1.10 SIMD Projection Solver Optimization (P2)

SIMD Poisson integration is done; current ~1.3–1.5× speedup is Amdahl-limited
(parallelizable fraction ~80%). Remaining optimization work:

- [ ] Increase `POISSON_MAX_ITER` or implement adaptive tolerance
- [ ] Optional multigrid preconditioner for faster convergence (see §1.2 multigrid)
- [ ] Red-Black omega parameter tuning
- [ ] Profile to identify remaining bottlenecks
- [ ] OpenMP+SIMD hybrid projection (OMP across rows, SIMD within rows); benchmark vs pure OMP
      and pure SIMD

---

## Phase 2: Physics Extensions

**Goal:** support more physical phenomena.

### 2.1 Energy Equation (P1)

Temperature advection-diffusion, thermal BCs, Boussinesq buoyancy, and heat source terms are
done across scalar/OMP/AVX2/CUDA (GPU validated vs the de Vahl Davis benchmark; GPU heat
source via host callback). See CHANGELOG.

**Still needed:**

- [ ] Conjugate heat transfer
- [ ] Variable properties (viscosity/density as a function of T)
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

**Goal:** support complex geometries.

### 3.1 3D Support (P0)

✅ Done in v0.2.0 — "2D as subset of 3D" (`nz=1` is bit-identical to the old 2D path);
branch-free solver loops across all backends, plus 3D I/O, examples, and validation
(Taylor-Green 3D, Poiseuille 3D). See CHANGELOG.

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
- [ ] Mesh quality metrics and validation

### 3.4 Adaptive Mesh Refinement (P2)

- [ ] Cell-based refinement
- [ ] Refinement criteria (gradient, error)
- [ ] Coarsening
- [ ] Load balancing
- [ ] Hanging nodes treatment

---

## Phase 4: Scalability & Performance

**Goal:** scale to large problems.

### 4.1 MPI Parallelization (P1)

- [ ] Domain decomposition
- [ ] Ghost cell exchange
- [ ] Parallel I/O
- [ ] Load balancing
- [ ] Hybrid MPI+OpenMP

### 4.2 Modular Backend Libraries (P1)

✅ Done in v0.1.5 — split into per-backend targets (`CFD::Core`, `CFD::Scalar`, `CFD::SIMD`,
`CFD::OMP`, `CFD::CUDA`, `CFD::Library`). See the table in `CLAUDE.md` and the
modular-library circular-dependency note under [Known Limitations](#known-limitations).

**Still needed:**

- [ ] Update examples/tests to link against specific backends (optional)
- [ ] Plugin loading system for dynamic backend selection

### 4.3 GPU Improvements (P2)

Implemented: CUDA device detection, GPU projection, GPU memory management, GPU BC kernels,
configurable GPU settings, GPU solver statistics. (GPU linear-solver backends are tracked in
§1.2.)

- [ ] Multi-GPU support
- [ ] Unified memory optimization
- [ ] Advanced async transfers (multi-stream overlap, double buffering)
- [ ] GPU-aware MPI

### 4.4 Performance Tools (P2)

- [ ] Built-in profiling
- [ ] Memory usage tracking
- [ ] Roofline analysis integration
- [ ] Scaling benchmarks
- [ ] Release-mode benchmark suite — needed for accurate SIMD/scalar comparison (current
      tests run in Debug, where SIMD can be slower for lack of optimization)

### 4.5 Tech Debt (P3, deferred)

- [ ] **OMP loop-variable `int` overflow on large grids** — OMP backends cast `size_t` loop
      vars to `int` for MSVC OpenMP 2.0 compatibility, overflowing when `nx*ny > INT_MAX`
      (~46K×46K). Low risk, no practical impact yet. Audit casts, add
      `CFD_ASSERT(nx*ny <= INT_MAX)` guards if targeting large grids, or require OpenMP 3.0+
      when dropping MSVC OMP 2.0 support.
- [ ] Structured-logging follow-ups (Phase 0.6): log filtering by component; timestamps +
      colored output; structured metrics API (convergence stats, timings).

---

## Phase 5: I/O & Post-processing

**Goal:** industry-standard data formats.

### 5.1 HDF5 Output (P1)

- [ ] Parallel HDF5 support
- [ ] Compression options
- [ ] Chunked storage
- [ ] XDMF metadata

### 5.2 Modern VTK (P1)

VTK legacy ASCII (scalar/vector/flow-field, timestamped run dirs) is done.

- [ ] VTK XML format (.vtu, .pvtu)
- [ ] Parallel VTK files
- [ ] Time series support
- [ ] Binary encoding

### 5.3 CSV Output

✅ Done — timeseries, centerline profiles, global statistics, velocity magnitude, automatic
headers. See CHANGELOG.

### 5.4 In-situ Visualization (P3)

- [ ] Catalyst/ParaView integration
- [ ] ADIOS2 integration

> Restart/checkpoint file I/O is tracked under [§1.6](#16-restart--checkpoint-p1).

---

## Phase 6: Validation & Documentation

**Goal:** validate against reference solutions and provide comprehensive documentation.

### 6.1 Benchmark Validation (P0)

Lid-driven cavity (33×33, Re=100, all backends), Taylor-Green vortex, and Poiseuille flow are
all validated. See CHANGELOG and `docs/validation/`.

**Still needed — 129×129 release validation** (too slow for CI; uses `CAVITY_FULL_VALIDATION=1`):

- [ ] Full Ghia validation at 129×129 for Re=100, 400, 1000 (record RMS on EC2)
- [ ] Extended cavity convergence to true steady-state (residual < 1e-8)
- [ ] Multi-Reynolds grid-convergence study (Richardson extrapolation)
- [ ] Extended-time Taylor-Green decay-rate verification
- [ ] Cross-architecture consistency (all backends identical within 0.1%)
- [ ] Memory + performance regression benchmarks

**Other benchmarks (P2):**

- [ ] Backward-facing step — compare to Armaly et al. (1983)
- [ ] Flow over cylinder — compare to Williamson (1996)

**CI vs Release parameters:**

| Test | CI Mode | Release Mode |
|------|---------|--------------|
| Cavity Ghia Validation | 33×33, 5000 steps | 129×129, 50000 steps |
| Cavity Re=400 Stability | 25×25, 500 steps | 65×65, 20000 steps |
| Grid Convergence | 17→25→33 | 33→65→129 |
| Taylor-Green Vortex | 32×32, 200 steps | 128×128, 10000 steps |

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

12 examples implemented (minimal, basic simulation, animated flow, visualization,
performance comparison, solver selection, custom BCs/source terms, CSV export, cavity).

- [ ] Heat transfer examples
- [ ] Turbulent flow examples
- [ ] Parallel computing examples (MPI)

---

## Phase 7: ML Integration (P3, future)

**Goal:** integrate pre-trained neural-network surrogate models for ~1000× faster inference.
Train in Python (PyTorch/JAX); use this library for compute. Two approaches are under
consideration — pick one before implementing.

### Approach A — Full C Inference

Pure-C inference with no runtime Python dependency (embedded/HPC friendly).

- [ ] Binary weight format (`.cfdnn`) + JSON metadata + loader API
- [ ] Layers: Dense, activations (ReLU/LeakyReLU/Tanh/Sigmoid/GELU/Swish), batch/layer norm,
      dropout (identity at inference)
- [ ] SIMD kernels: AVX2/NEON matrix-vector and matrix-matrix multiply, cache-tiled
- [ ] Architectures: MLP (priority), Conv2D (future), Fourier Neural Operator (future)
- [ ] Inference API (`cfdnn_predict` / `cfdnn_predict_batch`) + model lifecycle
- [ ] Hybrid solver use: ML initial guess, adaptive ML↔CFD switching, ensemble UQ
- [ ] Benchmarking + validation (inference time vs Python, accuracy vs CFD, cavity surrogate)

### Approach B — Hybrid Python + C Kernels

Python handles training/orchestration; C provides optimized compute kernels. Lower effort,
easier to support new architectures.

- [ ] SIMD matrix ops (`cfd_matmul`, `cfd_matmul_add_bias`) + in-place activations
- [ ] Python bindings for the C compute kernels
- [ ] Physics residual kernels for PINN training (NS residual, FD gradient/laplacian)
- [ ] Batch simulation API for dataset generation (OpenMP)
- [ ] Memory-mapped zero-copy buffers between C and Python/NumPy

### A vs B

| Aspect | A (Full C Inference) | B (Hybrid) |
|--------|----------------------|------------|
| Implementation effort | High | Medium |
| Python dependency | None at runtime | Required |
| New architecture support | Requires C changes | Just Python |
| Deployment | Embedded/HPC friendly | Python environment |
| Performance | Highest | High (kernel-level) |
| Flexibility | Fixed architectures | Any PyTorch model |

**Success criteria (Approach A):** load/run exported PyTorch models; <1 ms inference for a
64×64 grid; <5% L2 error vs CFD (Re < 200); SIMD kernels >2× over scalar.

**References:** [GGML](https://github.com/ggerganov/ggml) ·
[ONNX Runtime C API](https://onnxruntime.ai/) ·
[TF Lite Micro](https://www.tensorflow.org/lite/microcontrollers)

---

## Version Milestones

**Shipped** (see [CHANGELOG.md](CHANGELOG.md) for details):

- ✅ **v0.1.7** — Stretched-grid fix, tanh stretching, grid unit tests
- ✅ **v0.2.0** — 3D support (indexing, stencils, NS solvers, BCs, SIMD/OMP/CUDA, VTK,
  validation)
- ✅ **v0.3.0** — Heat transfer (energy equation, thermal BCs, natural-convection validation,
  GPU backends) *(current release)*

**Planned:**

| Milestone | Target |
| --------- | ------ |
| **v0.4.0 — Turbulence** | At least one RANS model (k-ε or SA), wall functions, turbulent channel-flow validation |
| **v0.5.0 — Parallel Computing** | MPI parallelization, scalability benchmarks, HDF5 parallel I/O |
| **v0.6.0 — Unstructured Meshes** | Unstructured mesh support, Gmsh import, complex-geometry examples |
| **v1.0.0 — Production Ready** | All Phase 1–6 features, comprehensive validation, complete docs, stable API, performance optimized |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). When working on roadmap items:

1. Create an issue referencing the roadmap item
2. Create a feature branch
3. Implement with tests
4. Update documentation
5. Submit a PR referencing the issue

---

## References & Bibliography

**CFD Validation Benchmarks:** Ghia et al. (1982) — lid-driven cavity · Kim & Moin (1985) —
turbulent channel flow · Armaly et al. (1983) — backward-facing step · Williamson (1996) —
vortex shedding from cylinder.

**Numerical Methods:** Ferziger & Peric, *Computational Methods for Fluid Dynamics* ·
Versteeg & Malalasekera, *An Introduction to CFD* · Moukalled et al., *The Finite Volume
Method in CFD*.

**Turbulence Modeling:** Wilcox, *Turbulence Modeling for CFD* · Pope, *Turbulent Flows*.
