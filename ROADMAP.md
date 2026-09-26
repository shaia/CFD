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
|                     | Upwind convection (1st order) | done | done | — | done | —² |
|                     | Implicit viscous (BE / CN)³   | done | —    | — | done | —  |
| **Energy Eq.**      | Advec-diff + Boussinesq + thermal BCs | done | done | — | done | done |
| **Turbulence**      | k-ε / SA + wall functions             | done | done | — | done | —    |
| **Linear Solvers**  | Jacobi         | done | done     | done     | done     | done |
|                     | SOR            | done | done     | done     | —¹       | done |
|                     | Red-Black SOR  | done | done     | done     | done     | done |
|                     | CG / PCG       | done | done     | done     | done     | done |
|                     | BiCGSTAB       | done | done     | done     | done     | done |
|                     | GMRES(m)       | done | done     | done     | done     | —    |
|                     | Multigrid (GMG)| done | —        | —        | done     | —    |
| **Boundary Conds**  | All types      | done | done     | done     | done     | done |

¹ Plain (lexicographic) SOR is inherently sequential — each update reads
already-updated neighbors. Its parallel form is **Red-Black SOR**, which has an
OMP backend; a "plain SOR OMP" would either change the numerics silently or need
low-value wavefront machinery, so it is intentionally omitted.

² GPU solvers reject `NS_CONVECTION_SCHEME_UPWIND` with `CFD_ERROR_UNSUPPORTED`.

³ `params.viscous_scheme` on the projection solvers; laminar only (a turbulence model is
refused, since ν + ν_t needs a variable-coefficient implicit operator). Every other solver
refuses an implicit scheme through `NS_SOLVER_CAP_IMPLICIT_VISCOUS`.

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
- **Multigrid grid-dimension constraint** — the geometric multigrid solver requires
  2^k+1 points per active dimension (e.g. 33, 65, 129) for exact coarsening; other sizes
  return `CFD_ERROR_INVALID`. Use CG for arbitrary grid sizes. On conforming grids,
  MG-preconditioned CG (`POISSON_PRECOND_MULTIGRID`) gives grid-size-independent
  iteration counts on uniform grids where Jacobi PCG gives no benefit.

> SOR's automatic ω is chosen for the walls in use. With the default zero-gradient walls,
> points beside a wall relax by ω·F/(F − W), which makes the iteration SOR on the Neumann matrix,
> whose optimum is higher than the Dirichlet ω = 2/(1 + sin(πh)), not lower; the automatic ω
> estimates it from just below. With a custom `apply_bc` the exact Dirichlet formula applies. See
> the SOR section of `docs/reference/solvers.md`.

---

## Roadmap at a Glance

| Phase | Theme | Priority | Status — what remains |
| ----- | ----- | -------- | --------------------- |
| 1 | Core Solver Improvements | P0–P3 | GMRES ✅, geometric multigrid ✅; AMG, implicit integrators, SIMPLE/PISO, nonlinear & eigenvalue solvers |
| 2 | Physics Extensions | P1–P3 | RANS k-ε/SA ✅; realizable k-ε, k-ω SST remain; compressible, species, multiphase; energy-eq. extensions |
| 3 | Geometry & Mesh | P1–P2 | Unstructured meshes, mesh I/O, adaptive refinement (3D ✅) |
| 4 | Scalability & Performance | P1–P2 | MPI, GPU improvements, profiling tools (modular libs ✅) |
| 5 | I/O & Post-processing | P1–P3 | HDF5, modern VTK XML, in-situ viz (CSV ✅) |
| 6 | Validation & Documentation | P0–P1 | 129×129 release validation, convergence studies, docs |
| 7 | ML Integration | P3 | Approach A chosen; algebraic ν_t correction shipped, inference engine built here |

---

## Phase 1: Core Solver Improvements

**Goal:** make the solver practically usable for real problems.

### 1.1 Boundary Conditions (P0)

✅ Done in v0.1.5 — all BC types across all backends (Dirichlet, Neumann, Periodic,
No-slip, Inlet, Outlet, Symmetry, Moving wall, Time-varying). See CHANGELOG.

### 1.2 Linear Solvers (P0)

Implemented: Jacobi, SOR, Red-Black SOR, CG/PCG, BiCGSTAB, GMRES(m) (all with SIMD backends;
CG is the default Poisson solver for projection methods), and geometric multigrid (scalar, OMP).
GPU standalone Jacobi, CG, Red-Black SOR, plain SOR, and BiCGSTAB are done and validated vs
CPU; `solve_projection_method_gpu` uses on-device CG.

GMRES(m) with restart is implemented across scalar, AVX2, NEON, and OMP backends (right-
preconditioned Jacobi seam; validated vs CG on the SPD Poisson problem, with restart-no-stall
and cross-backend consistency tests). Since the current linear systems are all the symmetric
pressure-Poisson operator, GMRES is a forward-looking addition (it becomes load-bearing once
non-symmetric operators arrive, e.g. implicit advection-diffusion in §1.5).

**Still needed:**

- [x] GMRES (Generalized Minimal Residual) for non-symmetric systems — scalar, AVX2, NEON, OMP
      (done). GPU variant deferred.
- [ ] GMRES GPU backend (deferred from the initial GMRES landing)
- [x] Jacobi and BiCGSTAB OpenMP backends — completes the OMP linear tier
      (Jacobi, Red-Black SOR, CG/PCG, BiCGSTAB, GMRES). Plain lexicographic SOR
      stays scalar/SIMD-only; its parallel form is Red-Black SOR OMP.
- [ ] SSOR (Symmetric SOR) preconditioner
- [ ] ILU preconditioner
- [x] Geometric multigrid — scalar backend; V/W/F(FMG) cycles, Red-Black GS or weighted
      Jacobi smoothers, full-weighting restriction (Neumann-folded at boundaries) +
      bilinear/trilinear prolongation, Neumann (default) and Dirichlet BC modes, 2D/3D.
      Grid dims must be 2^k+1. Wired into the projection method
      (`ns_solver_params_t.pressure_solver`, scalar backend) and available as a CG
      preconditioner (`POISSON_PRECOND_MULTIGRID`, scalar CG). SIMD/GPU variants
      deferred (see `.claude/specs/multigrid-projection-integration.md`).
- [x] Geometric multigrid OpenMP backend — `POISSON_BACKEND_OMP` (`multigrid_omp`,
      `POISSON_PRESET_MULTIGRID` with `backend = POISSON_BACKEND_OMP`). One algorithm
      template shared with scalar; the
      OMP backend supplies row-parallel smoother/residual/transfer/BC primitives with no
      parallel reductions, so results are bit-identical to scalar at 1/2/4 threads.
      AUTO still resolves multigrid to scalar. Deferred: SIMD/GPU multigrid.
      Files created: `lib/src/solvers/linear/multigrid_template/linear_solver_multigrid_template.h`,
      `lib/src/solvers/linear/omp/linear_solver_multigrid_omp.c`.
      Files modified: `lib/src/solvers/linear/cpu/linear_solver_multigrid.c`,
      `lib/src/solvers/linear/cpu/multigrid_transfer.c`,
      `lib/src/solvers/linear/multigrid_internal.h`,
      `lib/src/solvers/linear/linear_solver_internal.h`,
      `lib/src/solvers/linear/linear_solver.c`, `lib/include/cfd/solvers/poisson_solver.h`,
      `lib/CMakeLists.txt`, `tests/math/test_omp_consistency.c`,
      `tests/math/test_multigrid_convergence.c`, `tests/solvers/test_linear_solver.c`.
- [x] Multigrid pressure solve in `projection_omp` and MG-preconditioned OpenMP CG —
      `NS_PRESSURE_SOLVER_MULTIGRID` → `POISSON_PRESET_MULTIGRID` and
      `NS_PRESSURE_SOLVER_PCG_MG` → `POISSON_PRESET_MULTIGRID_PCG`, both on
      `POISSON_BACKEND_OMP` (OMP CG with an OMP
      multigrid V-cycle preconditioner), with the scalar projection's init gating and no
      scalar sub-solver on the OMP path. Both CG backends share one preconditioner
      setup (`poisson_solver_create_mg_precond`).
      Files modified: `lib/src/solvers/navier_stokes/omp/solver_projection_omp.c`,
      `lib/src/solvers/linear/omp/linear_solver_cg_omp.c`,
      `lib/src/solvers/linear/cpu/linear_solver_cg.c`,
      `lib/src/solvers/linear/omp/linear_solver_multigrid_omp.c`,
      `lib/src/solvers/linear/multigrid_internal.h`,
      `lib/src/solvers/linear/linear_solver_internal.h`,
      `lib/src/solvers/linear/linear_solver.c`, `lib/src/api/solver_registry.c`,
      `lib/include/cfd/solvers/poisson_solver.h`,
      `lib/include/cfd/solvers/navier_stokes_solver.h`,
      `tests/solvers/navier_stokes/cpu/test_projection_pressure_solver.c`,
      `tests/math/test_mg_pcg_convergence.c`, `tests/math/test_omp_consistency.c`.
- [ ] Algebraic multigrid (AMG) — solver and preconditioner (for CG/GMRES/BiCGSTAB)
- [x] GPU plain SOR (Block SOR: per-thread tile sweep, red-black tile coloring, in-place; closes the matrix)

### 1.3 Numerical Schemes (P1)

Stencil tests, convergence-order, MMS, and divergence-free validation are done (see CHANGELOG).

**Still needed:**

- [x] Upwind differencing (1st order) for stability — `ns_solver_params_t.convection_scheme`
      (`NS_CONVECTION_SCHEME_UPWIND`) switches the convective first derivatives of momentum
      and temperature to first-order upwind on the scalar, OpenMP and AVX2 backends of all
      four solvers; pressure gradients, viscous terms and divergence stay central. GPU
      solvers reject it at init and at step. Validated: O(h) stencil and solver-level
      convergence (rate ~0.95 vs ~2.0 central), bounded step advection where central
      overshoots, and OMP/AVX2 agreement with scalar within 1e-10 in 2D and 3D.
      Files created: `lib/src/solvers/navier_stokes/ns_convection_internal.h`,
      `lib/src/solvers/navier_stokes/avx2/upwind_avx2.h`,
      `tests/math/test_upwind_stencils.c`, `tests/math/test_upwind_convergence.c`,
      `tests/solvers/navier_stokes/test_convection_scheme.c`.
      Files modified: `lib/include/cfd/math/stencils.h`,
      `lib/include/cfd/solvers/navier_stokes_solver.h`, `lib/src/api/solver_registry.c`,
      the explicit Euler / projection / RK kernels under `lib/src/solvers/navier_stokes/`
      (`cpu/`, `omp/`, `avx2/`, `momentum_rhs/`), the energy solvers under
      `lib/src/solvers/energy/`, the GPU drivers `solver_projection_gpu.cu` and
      `solver_rk_gpu.cu`, `tests/solvers/energy/test_energy_solver.c`,
      `tests/solvers/navier_stokes/cpu/test_ns_solver_3d.c`.
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

Implicit viscous term (`params.viscous_scheme`): backward Euler (θ = 1, L-stable) and
Crank–Nicolson (θ = ½) on the scalar and OpenMP projection solvers. Each is solved as a
Helmholtz-shifted CG problem per velocity component, which removes the diffusion limit on
dt. Viscous-part order is verified at 1.0 / 2.0 on a discrete eigenmode. The full step
stays O(dt), from explicit convection and non-incremental Chorin splitting. See
`docs/reference/solvers.md#viscous-time-discretization`.

**Still needed:**

- [x] Implicit Euler (backward Euler) — viscous term, scalar + OMP projection
- [x] Crank-Nicolson (2nd order implicit) — viscous term, scalar + OMP projection
- [ ] Implicit viscous on AVX2 / CUDA projection (needs the Helmholtz shift in SIMD and GPU CG)
- [ ] Variable-coefficient implicit viscous for ν + ν_t (turbulence)
- [ ] Second order overall: AB2/CN with incremental pressure correction
- [ ] BDF2 (backward differentiation)
- [ ] Adaptive time stepping with error control

### 1.6 Restart / Checkpoint (P1)

✅ Done — portable, versioned, CRC-protected binary checkpoint format (`.cfdchk`) saving and
restoring complete simulation state (grid, field, scalar params, time, solver name);
little-endian fixed-width encoding with endianness marker and a format-version header that
rejects unknown versions (`lib/src/io/checkpoint.c`). See CHANGELOG.

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
- [x] Optional multigrid preconditioner for faster convergence (see §1.2 multigrid) —
      done for scalar and OpenMP CG (`POISSON_PRECOND_MULTIGRID`, symmetric V(2,2) Jacobi
      cycle) and selectable in the scalar and OpenMP projections via
      `pressure_solver = NS_PRESSURE_SOLVER_PCG_MG`; SIMD PCG-MG is pending
- [x] Red-Black omega parameter tuning — the automatic ω is at or just below the optimum for the
      walls in use, with wall-adjacent points relaxed so that SOR theory holds for the Neumann matrix
      (`poisson_solver_resolve_omega`)
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

- [x] Spalart-Allmaras (1-equation)
- [x] k-epsilon standard
- [ ] k-epsilon realizable
- [ ] k-omega SST
- [x] Wall functions
- [ ] Low-Reynolds number treatment

**Done (2D, uniform grids):** standard k-ε and Spalart-Allmaras with log-law wall functions
on scalar/OMP/AVX2 backends; validated against turbulent channel flow at Re_τ = 395
(k-ε: u_τ error 2.9%; SA: 3.1%). GPU turbulence not yet implemented.

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

**129×129 release validation** (`-DCAVITY_FULL_VALIDATION=ON`; runs in the EC2 GPU workflow on
every push to master, about 50 minutes):

- [x] Full Ghia validation at 129×129 for Re=100, 400, 1000 — AVX2/OMP/GPU projection RMS
  ≤ 0.033, recorded in `docs/validation/cavity-backends-validation.md`
- [ ] Extended cavity convergence to true steady-state (residual < 1e-8)
- [x] Explicit Euler cavity cases stopped at 11,300–11,800 steps (t ≈ 1.1–1.2) through the
  harness's kinetic-energy exit. The exit compared the change in kinetic energy **per step**
  against a fixed threshold, which scales with dt and so measured the step size as much as
  the flow; it is now a rate, `|d(ln KE)/dt| < 1e-6`, taken from the step the solver
  actually used. The 129×129 Euler cases are dropped (the "or" of this item): they cost
  ~1 h of EC2 to hold a non-production solver to a relaxed target, and were never evidence
  of 129×129 accuracy. Euler stays validated at 33×33. See
  `docs/validation/cavity-backends-validation.md`
- [ ] Multi-Reynolds grid-convergence study (Richardson extrapolation)
- [x] Extended-time Taylor-Green decay-rate verification — `test_taylor_green_decay.c` fits the
  kinetic-energy decay rate over t = 10 (86% of the energy gone; t = 20 in full validation)
  on every projection backend: within 0.04% of −4ν at 65×65, no early/late drift, second-order
  convergence. It runs on one wall-bounded vortex cell. On the periodic vortex, the projection
  (no periodic pressure solve) reaches 0.92 of the rate and the pseudo-compressible Euler/RK
  solvers 0.42 on every grid. See `docs/validation/taylor-green-decay.md`
- [x] Cross-architecture consistency (all backends identical within 0.1%) — `test_solver_architecture.c`
  compares the whole field of AVX2, OpenMP and CUDA against scalar for projection and Euler
  (cavity) and Euler, RK2, RK4 (periodic Taylor-Green). Closing it took a fix to the CUDA
  Euler/RK boundary handling, which differed from the CPU solvers by 75% (Euler, cavity) and
  0.18% (RK)
- [ ] Memory + performance regression benchmarks

**Other benchmarks (P2):**

- [ ] Backward-facing step — compare to Armaly et al. (1983)
- [ ] Flow over cylinder — compare to Williamson (1996)

**CI vs Release parameters:**

| Test | CI Mode | Release Mode |
|------|---------|--------------|
| Cavity Ghia Validation | 33×33, 5000 steps | 129×129; 50000 steps (Re=100), 60000 (Re=400), 100000 (Re=1000) |
| Cavity Re=400 Stability | 25×25, 500 steps | 65×65, 20000 steps |
| Grid Convergence (monotone RMS vs Ghia) | 17→25→33 | 17→25→33 |
| Grid Convergence (Richardson) | Re=100: 17/33/65 | Re=100: 33/65/129; Re=400: 65/129/257; Re=1000: 129/193/257 |
| Taylor-Green Vortex | 32×32, 200 steps | 128×128, 10000 steps |
| Taylor-Green decay rate | 65×65, t=10 | 129×129, t=20 |

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

**Decision made:** **Approach A — full C inference**, with a **learned eddy-viscosity
correction** as the first consumer. Rationale, rejected alternatives, format and integration
design in [ml-integration-design.md](docs/technical-notes/ml-integration-design.md).

**Governing rule established by this decision:** *ML earns its place only where no analytic
or algorithmic answer exists.* This library has twice replaced a tuned constant with an
analytic one (SOR omega via Young's formula; multigrid in place of iteration tuning). A
learned model that competes with a provably optimal algorithm loses on cost, on maintenance,
and on having unbounded failure modes where the algorithm has proven ones.

### Approach A — Full C Inference (chosen)

Pure-C inference with no runtime Python dependency (embedded/HPC friendly).

- [x] Binary weight format (`.cfdnn`) + loader API, modelled line-for-line on `.cfdchk`
- [x] Layers: Dense + activations (Identity/ReLU/LeakyReLU/Tanh/Sigmoid/Softplus)
- [ ] SIMD kernels: AVX2/NEON, vectorized across cells so OMP stays bit-identical to scalar
      (the table exports a NULL kernel today; scalar and OMP are done, OMP bit-identical)
- [ ] Architectures: MLP (priority), Conv2D (future), Fourier Neural Operator (future)
- [x] Inference API (`cfd_nn_predict_batch`) + model lifecycle
- [x] Correction seam at `turb_update_nu_t()`, reaching all three CPU backends, with the
      algebraic `NS_NUT_CORRECTION_S_STAR` and the learned `params.turb_closure` as
      mutually exclusive alternatives on it
- [x] **Algebraic competitor measured first, per the governing rule.** A two-constant
      strain-rate power law cuts channel TKE error 14.96% → 6.23% where it was fitted and
      12.80% → 9.27% at a held-out Re_tau, with `u_tau` unmoved (design note §2.7)
- [ ] Trained model + Python exporter (`tools/cfdnn/`) — **blocked**: a network must now
      beat the algebraic correction above, and §2.7 showed the only non-circular target
      left on this case is `k+` at 9 and 14 nodes
- [ ] Separated or adverse-pressure-gradient validation case — **prerequisite** for the
      learned closure: in channel flow the momentum balance pins the shear stress, so the
      closure has little authority over the quantities the gate measures
- [ ] Validation: must beat tuned-Cs Smagorinsky, and must vanish in laminar regions
      (the latter holds structurally and is asserted in `test_nut_correction.c`)

**Three corrections to this section** (reasoning in the design note):

1. **No JSON metadata.** There is no JSON parser in the repo, and adding one would be the
   first third-party runtime dependency in project history. A sidecar also creates a
   two-file consistency problem the CRC cannot cover. The format is self-describing binary:
   one file, one CRC.
2. **No batch-norm or dropout.** Dropout is identity at inference. Batch-norm folds exactly
   into the preceding layer's weights at export time. The format *rejects* a batch-norm
   layer kind so a non-folding exporter fails loudly rather than silently producing a wrong
   model.
3. **MLP-first was correct** and is retained — the pointwise closure needs Dense only.

**Success criteria (rewritten).** The original "~1000x faster inference / <5% L2 error vs
CFD (Re < 200)" targeted a flow surrogate. It is not a defensible bar in a library that
validates with MMS convergence *order*, Ghia RMS to four decimals across backends, and
bit-identical 3D degeneracy — and it was never achievable against a warm-started,
multigrid-preconditioned projection method. Superseded by the closure ship gate in the
design note.

### Approach B — Hybrid Python + C Kernels (not chosen)

Rejected because it makes a Python environment a *runtime* requirement of a zero-dependency
C library, across a five-configuration CI matrix including Windows x86 and macOS ARM64.
Recorded for the future: Python bindings for the C compute kernels, physics residual kernels
for PINN training, a batch simulation API for dataset generation, and memory-mapped
zero-copy NumPy buffers.

### Rejected: ML for the pressure Poisson initial guess

Investigated and rejected on evidence — **do not re-propose without reading the design
note.** Three independent findings: under the library's relative stopping rule a better
initial guess shrinks the convergence target by the same factor it shrinks the residual, so
it cannot buy iterations; a local network removes the high-frequency error that was already
cheap and leaves the low-frequency error that sets the iteration count; and MG-preconditioned
CG already reaches 5 grid-independent iterations at 33^2-129^2 while the projection method
already warm-starts.

### Redirected work (non-ML, surfaced by the Phase 7 investigation)

- [ ] Remove the duplicated residual dot product in the unpreconditioned CG branch —
      `rho_new = (r,r)` and `res_norm = sqrt((r,r))` are the same quantity computed by two
      full O(N) passes, in all three CPU backends (`cpu/linear_solver_cg.c`,
      `avx2/linear_solver_cg_avx2.c`, `omp/linear_solver_cg_omp.c` — grep
      `rho_new = dot_product`). The preconditioned branch must keep its own `(r,r)`,
      since `rho_new = (r,z)` there.
- [ ] Blend the wall-function branches in `turbulence_wall_u_tau()` (Spalding's law). With
      `WALL_KAPPA = 0.41` / `WALL_B = 5.2` the linear and log laws do not intersect exactly
      at `WALL_YPLUS_LAMINAR`, so `u_tau` jumps at the switch. Measure the jump first.
      Note `test_turbulent_channel` uses this function as its measurement instrument.

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
| ✅ **v0.4.0 — Turbulence** | *(complete: k-ε + SA, log-law wall functions, turbulent channel-flow validation at Re_τ = 395)* |
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
