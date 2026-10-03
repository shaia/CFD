# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The turbulence release. RANS k-ε and Spalart-Allmaras models with wall functions, validated
against channel DNS at Re_τ = 392 and 587, together with what they needed: geometric multigrid
and GMRES(m) Poisson solvers, implicit viscous time integration, first-order upwind convection,
checkpoint/restart, and pure-C neural-network inference for learned closures. New validations
cover the laminar backward-facing step, the 129×129 Ghia cavity on every optimized backend,
multi-Reynolds Richardson extrapolation and extended-time Taylor-Green decay. The Poisson
configuration API changed incompatibly; see *Upgrading from 0.3.0*.

### Upgrading from 0.3.0

Each item is detailed in the entry it names.

- **Poisson parameters are grouped by method** — `params.omega` → `params.sor.omega`,
  `params.preconditioner` / `params.restart` → `params.krylov.*`, `params.mg_*` →
  `params.multigrid.*`. A group the resolved method does not read is refused (Changed).
- **`poisson_solver_type` and `DEFAULT_POISSON_SOLVER` are gone**: build a
  `poisson_solver_config_t` from `poisson_solver_config_preset()`. `poisson_solve()` takes that
  config and returns `cfd_status_t`; `poisson_solve_3d()` and `poisson_solve_3d_params()` are
  removed (Changed).
- **The default Krylov walls are zero-gradient, so the operator is singular**: an RHS with
  non-zero interior mean is refused with `POISSON_INCOMPATIBLE_RHS`. Call
  `poisson_make_rhs_compatible()` or prescribe a face through `params.walls` (Fixed).
- **A caller's `apply_bc` is refused** on multigrid and every GPU solver, and alongside
  `params.walls` (Changed).
- **`run_simulation_step()` / `run_simulation_solve()` step with `sim->params.dt`** instead of a
  hard-coded 0.005, so an unset dt now steps at `init_simulation()`'s 0.001; a zero, negative or
  non-finite dt is refused (Fixed).
- **`compute_time_step()` applies the laminar viscous limit**, which binds on fine grids (Fixed).
- **`init_simulation()` / `init_simulation_with_solver()` return NULL** when the solver fails to
  initialize, and `explicit_euler` / `explicit_euler_omp` return their step errors (Fixed).
- **The `*_optimized` solvers need AVX2 compiled in** (`CFD_ENABLE_AVX2`, default OFF) as well as
  an AVX2+FMA3 CPU, and refuse init otherwise (Fixed).
- **`ns_solver_params_t` and `flow_field` grew**, so code must be rebuilt against the new
  headers. Each new parameter is off when zero.

### Added

- **RANS turbulence: standard k-ε and Spalart-Allmaras with wall functions**
  (`params.turb_model`, `params.turb_bc`, `cfd/solvers/turbulence_solver.h`). Built like the
  energy-equation module: per-backend kernels, a workspace-aware step, in-module BCs. Upwind
  advection, conservative face-averaged diffusion, semi-implicit Patankar sinks for positivity,
  a production limiter and nu_t realizability clipping. Eddy viscosity couples into every
  projection/Euler/RK momentum path through a conservative `nu_eff = nu + nu_t`, leaving the
  laminar path bitwise identical. `turb_k`, `turb_eps`, `turb_nu_tilde` and `nu_t` live on
  `flow_field` and are written to VTK/CSV. Scalar, OpenMP and AVX2 backends; **2D uniform grids only**. The GPU
  paths return `CFD_ERROR_UNSUPPORTED` when turbulence is enabled, as does an implicit
  viscous scheme combined with a turbulence model.
  - **Wall functions** on `BC_TYPE_NOSLIP` faces derive the friction velocity from the law of
    the wall, set equilibrium k/epsilon (or nu_tilde) at the first interior node, and set the
    wall-face viscosity so the discrete wall shear equals u_tau^2 exactly. The law is
    selectable (`ns_turbulence_bc_config_t.wall_law`, `ns_wall_law_t`), and
    `turbulence_wall_u_tau()` takes it as its first argument, so callers compute exactly what
    the wall function imposes. `NS_WALL_LAW_LOG` (0, default) is the linear/log law
    (κ = 0.41, B = 5.2), switching where the two cross at y+ = 11.06 so u_τ is continuous.
    `NS_WALL_LAW_SPALDING` is Spalding's single smooth y+(u+). Which is closer to reality
    depends on where the first node sits: against channel DNS, Spalding is closer below
    y+ ≈ 17 and the log law from y+ ≈ 20 on (u+ 0.7% under DNS at y+ = 40, against 4.4%).
    Hence the log law as the default for wall-function grids and Spalding as the option for
    first nodes in the buffer layer. An unknown `wall_law` is `CFD_ERROR_INVALID` from
    `turbulence_apply_bcs()`.
  - **Turbulence BCs on part of a face** (`ns_turbulence_bc_segment_t`,
    `turbulence_bc_add_segment()`, `ns_turbulence_bc_config_t.segments`/`n_segments`), so a
    backward-facing step with its inlet at the step plane can have wall functions on the step
    face and a turbulent inflow above it. A segment overrides its face's type over a
    normalized range of the edge, with the same node-index convention and tolerance as
    `bc_inlet_set_range()` (one shared helper, `lib/src/boundary/bc_edge_range.h`). Segment
    types are NEUMANN, DIRICHLET (fixed values, or a `ns_turbulence_profile_fn` profile
    evaluated per node at the position a velocity inlet over the same range sees) and NOSLIP;
    PERIODIC is refused. Up to `NS_TURB_BC_MAX_SEGMENTS` (4); the last covering segment wins.
    With no segments every node gets the operations it got before, which is checked by a
    whole-face segment reproducing the face type bit for bit. Spalart-Allmaras wall distance
    measures to the wall parts of a face only. `turbulence_bc_add_segment()` refuses a
    malformed segment and leaves the config unchanged; hand-filled configs are re-checked by
    solver init, by the turbulence step and by `turbulence_apply_bcs()`, each before touching
    any field, which also refuse a k-ε DIRICHLET segment with no profile and `eps <= 0`. A
    profile that returns a negative or non-finite value is refused by
    `turbulence_apply_bcs()`, which resolves every boundary node first and writes nothing
    unless all of them pass.
  - **Validation.** Turbulent channel flow at Re_τ = 395 (`test_turbulent_channel`,
    `examples/turbulent_channel.c`) steps at dt = 1e-3 and stops when no velocity changes
    faster than 1e-3 per unit time (t ≈ 64). A step of 2e-3 is past the limit forward Euler
    with central convection needs, dt below about 2ν_eff/|u|², and grows a sinuous mode
    from roundoff. At steady state u_τ is 0.9742 for both models with the log law (2.6% from
    the log-law target) and 0.9427 with Spalding (5.7%). Because the wall function imposes
    log-law behaviour at the first node, the log-law check cannot measure closure error, so
    the test also scores u+, k+ and -uv+ against Moser-Kim-Mansour DNS at Re_τ 392 and 587
    (`docs/validation/turbulent-channel-dns.md`). Unit tests cover decay, wall functions, SA
    closures, laminar regression and cross-backend consistency. The example takes an
    optional second argument, `spalding`, and measures u_τ with its own log-law inversion,
    as the test does, so both laws are read with one yardstick
  (`lib/src/solvers/turbulence/`, `lib/include/cfd/solvers/turbulence_solver.h`,
  `tests/solvers/turbulence/`, `tests/validation/test_turbulent_channel.c`,
  `examples/turbulent_channel.c`).

- **Strain-rate correction to the k-epsilon eddy viscosity** (`NS_NUT_CORRECTION_S_STAR`,
  `ns_solver_params_t.turb_nut_correction`). A two-constant power law in `S* = |S| k / eps`
  -- a `C_mu` that varies with the local strain -- fitted against MKM channel DNS. It cuts the
  channel TKE error from 14.96% to 6.23% at Re_tau 392 where it was fitted, and from 12.80% to
  9.27% at a held-out Re_tau 587, with `u_tau` unmoved. Off by default and bit-identical when
  off; k-epsilon only, refused with `CFD_ERROR_UNSUPPORTED` for any other model, and refused
  together with `turb_closure`. Applied in the scalar, OpenMP and AVX2 turbulence steps. Why
  this, and why not an ML pressure guess, is in `docs/technical-notes/ml-integration-design.md`
  (`tests/solvers/turbulence/test_nut_correction.c`).

- **Pure-C neural-network inference for learned closures** (`cfd/nn/cfdnn.h`, `.cfdnn` format
  version 1). Evaluates a pre-trained multilayer perceptron with no runtime dependencies: Dense
  layers with a fused identity, ReLU, leaky ReLU, tanh, sigmoid or softplus activation,
  binary32 weights and arithmetic behind a binary64 API. A loaded model is immutable and may
  be shared across threads; a context owns the scratch and belongs to one thread. The reader
  refuses any header field it does not understand (version, endianness, dtype, flag bits,
  tensor layout, reserved words) before allocating. Scalar, OpenMP (over the batch) and SIMD
  backends; `CFD_NN_BACKEND_AUTO` resolves SIMD > OMP > scalar.
  - **As a k-ε closure**, through `ns_solver_params_t.turb_closure`: a context whose model
    maps 3 inputs to 1 output. The prediction, clamped to [0.1, 10], multiplies nu_t under the
    existing realizability bound, so it can lower nu_t as well as raise it. Solver init and
    the turbulence step refuse it with any model but k-ε, with a model of the wrong shape, or
    together with `turb_nut_correction`, and a non-finite prediction fails the step with
    `CFD_ERROR_DIVERGED` rather than continuing uncorrected. The correction walks the grid in
    tiles, so any context capacity works (`cfd_nn_context_capacity()`).
  - **AVX2 and NEON kernels** (`CFD_NN_BACKEND_SIMD`, chosen at runtime). Lanes carry samples,
    not features, so nothing is reduced across lanes: AVX2 stays within 3.4e-6 of scalar
    (tolerance 1e-5). `tanh`, `sigmoid` and `softplus` run in the vector unit too: Cephes
    `exp`/`log` and an odd rational `tanh`, all within 4.3 ulp of float64 truth and
    NaN-preserving, so a corrupt model still fails with `CFD_ERROR_DIVERGED`. On a
    closure-sized network (3 → 16 tanh → 16 tanh → 1 softplus) AVX2 is 6.9x faster than
    scalar, against 1.6x with scalar transcendentals. NEON shares the AVX2 code through one
    template and has not yet run on ARM hardware. Avoid an OMP context for `turb_closure`: at
    the closure's 256-cell tiles it is slower than scalar.
  - **Python exporter** (`tools/cfdnn/`, numpy only). Writes, reads and evaluates the format,
    folds input standardization and BatchNorm exactly into Dense layers, and converts a
    PyTorch `Sequential` (Linear, ReLU, LeakyReLU, Tanh, Sigmoid, Softplus, BatchNorm1d,
    Dropout); anything else is refused, not skipped. It is developer tooling, outside the
    build and CI. `test_cfdnn_python_export` embeds a network distilled from the algebraic
    `NS_NUT_CORRECTION_S_STAR` law and asserts the C reader loads it, the C writer emits the
    same bytes, every kernel backend matches the Python reference (1.3e-7), and the network
    run as `params.turb_closure` reproduces the algebraic correction's `nu_t` (2.7e-4). The
    fixture is a pipeline check, not a closure model
  (`lib/include/cfd/nn/cfdnn.h`, `lib/src/nn/`, `docs/technical-notes/ml-integration-design.md`).

- **Geometric multigrid Poisson solver** (`POISSON_METHOD_MULTIGRID`, `POISSON_PRESET_MULTIGRID`)
  — V/W/F(FMG) cycles with Red-Black Gauss-Seidel or weighted-Jacobi smoothers, full-weighting
  restriction and bilinear/trilinear prolongation, 2D/3D, configured through
  `params.multigrid`. Two BC modes: Neumann zero-gradient (default, matching the other Poisson
  solvers; nullspace handled via Neumann-folded restriction weights and coarse-level mean
  projection) and Dirichlet (inhomogeneous boundary data supported). Grid dims must be 2^k+1
  per active dimension. Grid-size-independent convergence (< 0.15 residual reduction per
  V(2,2) cycle, 9²–129² validated). Grids whose `nx` or `ny` exceeds `INT_MAX` are
  `CFD_ERROR_LIMIT_EXCEEDED` at init.
  - **OpenMP backend** (`multigrid_omp`; `POISSON_BACKEND_AUTO` still resolves multigrid to
    scalar, so request OMP explicitly). The algorithm lives in one template shared with the
    scalar solver; the OpenMP backend supplies row-parallel smoother, residual,
    restriction/prolongation and boundary primitives with no parallel reductions, so its
    solutions are bit-identical to `multigrid_scalar` at any thread count (verified over a
    V/W/F x smoother x BC-mode matrix at 1, 2 and 4 threads). A kernel uses the thread team
    only when its loop covers at least 32,768 points of a plane; smaller loops run on the
    calling thread, where a team would cost more than the work.
  - **As a CG preconditioner** (`POISSON_PRECOND_MULTIGRID`, `POISSON_PRESET_MULTIGRID_PCG`):
    one symmetric V(2,2) weighted-Jacobi multigrid cycle in Dirichlet mode per apply, on
    scalar and OpenMP CG (SIMD/GPU CG and GMRES reject it with `CFD_ERROR_UNSUPPORTED`). Grid-
    size-independent convergence: 5 CG iterations at 33²–129² (tol 1e-8) vs 50–170
    unpreconditioned. The inner cycle takes `pre_smooth`, `post_smooth`, `coarse_max_iter`
    and `max_levels` from `params.multigrid`; `cycle`, `smoother`, `bc` and an unequal
    pre/post sweep count are refused with `CFD_ERROR_INVALID`, since a V-cycle with
    weighted-Jacobi smoothing, Dirichlet coarse corrections and equal sweeps is what makes one
    apply a symmetric operator, and CG depends on that. OpenMP PCG-MG matches scalar within
    1e-9 at 1, 2 and 4 threads.
  - **As the projection's pressure solve** (`ns_solver_params_t.pressure_solver`;
    `ns_pressure_solver_t`, 0 = CG as before): `NS_PRESSURE_SOLVER_MULTIGRID` (RHS interior
    mean subtracted for Neumann compatibility) or `NS_PRESSURE_SOLVER_PCG_MG`, on the scalar
    `projection` and on `projection_omp`, each on its own backend, so no scalar sub-solver runs
    on the OpenMP path. Non-2^k+1 grids return `CFD_ERROR_UNSUPPORTED` and degenerate grids
    `CFD_ERROR_INVALID` at init; `projection_optimized` and `projection_gpu` reject the
    multigrid modes. The OpenMP multigrid projection matches the scalar one within 1e-10 on
    257x257, and on a 257x257 Re=1000 cavity takes 16 ms/step against 140 ms for the AVX2 CG
    solve
  (`lib/src/solvers/linear/multigrid_template/linear_solver_multigrid_template.h`,
  `lib/src/solvers/linear/cpu/linear_solver_multigrid.c`,
  `lib/src/solvers/linear/cpu/multigrid_transfer.c`,
  `lib/src/solvers/linear/omp/linear_solver_multigrid_omp.c`,
  `lib/src/solvers/linear/cpu/linear_solver_cg.c`,
  `lib/src/solvers/linear/omp/linear_solver_cg_omp.c`,
  `lib/src/solvers/navier_stokes/cpu/solver_projection.c`,
  `lib/src/solvers/navier_stokes/omp/solver_projection_omp.c`,
  `tests/math/test_multigrid_operators.c`, `tests/math/test_multigrid_convergence.c`,
  `tests/math/test_mg_pcg_convergence.c`, `tests/math/test_omp_consistency.c`,
  `tests/solvers/navier_stokes/cpu/test_projection_pressure_solver.c`).

- **Restarted GMRES(m) Poisson solver** (`POISSON_METHOD_GMRES`) — Arnoldi with modified
  Gram-Schmidt and incremental Givens rotations for non-symmetric systems, 2D/3D, restart
  length via `params.krylov.restart` (0 = default 30) and optional right Jacobi
  preconditioning. Scalar, AVX2, NEON and OpenMP backends (`gmres_scalar`, `gmres_simd`,
  `gmres_omp`) share one algorithm template and differ only in their O(n) vector
  primitives. `poisson_solver_init` rejects restart lengths whose Hessenberg matrix cannot
  be indexed with `int` (m ≥ 46341), and grids too large to index (`nx` or `ny` above
  `INT_MAX`, or an `(m+1)`-vector Krylov basis overflowing `size_t` bytes), with
  `CFD_ERROR_LIMIT_EXCEEDED`
  (`lib/src/solvers/linear/gmres_template/linear_solver_gmres_template.h`,
  `tests/math/test_gmres.c`, `tests/math/test_omp_consistency.c`).

- **Jacobi and BiCGSTAB OpenMP backends** — completes the OpenMP linear-solver
  tier (previously CG and Red-Black SOR). Both are selectable via
  `poisson_solver_create(method, POISSON_BACKEND_OMP)`; per-element updates are
  identical to the scalar reference, so results match within reduction rounding
  (verified by scalar-vs-OMP consistency tests). Both reject grids whose `nx` or
  `ny` exceeds `INT_MAX`, or whose `nx*ny*nz` overflows `size_t`, with
  `CFD_ERROR_LIMIT_EXCEEDED` at init. Plain lexicographic SOR remains
  scalar/SIMD-only — its parallel form is the existing Red-Black SOR OMP solver
  (`lib/src/solvers/linear/omp/linear_solver_jacobi_omp.c`,
  `lib/src/solvers/linear/omp/linear_solver_bicgstab_omp.c`,
  `tests/math/test_omp_consistency.c`, `tests/solvers/test_linear_solver.c`).

- **Per-face walls on the Poisson solver** — new `poisson_solver_params_t.walls`
  (`poisson_walls_t`; zero-init = all zero-gradient, the operator these solvers have always
  claimed). Each face is independently `POISSON_WALL_ZERO_GRADIENT` or
  `POISSON_WALL_DIRICHLET` with a prescribed value, which states explicitly what was
  previously inferred from whether an `apply_bc` hook happened to be NULL.
  Krylov solvers apply only the **homogeneous** part of the walls to their search directions
  — a zero halo on a prescribed face — and the full condition to the iterate. Applying a
  prescribed value to a direction would make the operator affine rather than linear, which
  the Krylov recurrences do not describe; `test_dirichlet_lift_is_linear` asserts the split
  holds by checking that raising one face by V shifts the solution by exactly the ramp
  `V*x/Lx` (measured 1.1e-14).
  Honoured by CG, BiCGSTAB and GMRES on the scalar, OpenMP and SIMD backends. The stationary
  and multigrid solvers apply whole-domain walls inside their sweeps and the GPU backend
  applies them on the device, so all of those return `CFD_ERROR_UNSUPPORTED` at init rather
  than silently solving a different problem; walls together with an `apply_bc` hook return
  `CFD_ERROR_INVALID`, since both prescribe wall values. A z-face on a 2D grid is refused
  only when it is the caller's only prescribed face, the case where they meant to pin the
  pressure level and nothing did. A linear field `a*x + b` is reproduced to 2e-15 across the
  supported methods and backends.
  At the NS layer, `ns_solver_params_t.pressure_bc` carries the configuration to the scalar,
  OpenMP and AVX2 projection solvers, following `ns_thermal_bc_config_t`. The projection's
  Neumann compatibility projection is conditional: with a face prescribed the operator is
  nonsingular and mean-subtracting would change the answer rather than make it exist
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/include/cfd/solvers/navier_stokes_solver.h`,
  `lib/src/solvers/navier_stokes/{cpu,omp,avx2}/solver_projection*.c`,
  `lib/src/api/solver_registry.c`, `tests/math/test_poisson_walls.c`,
  `tests/validation/test_poiseuille_flow.c`).
- **Pressure-driven channels are driven.** `PoiseuilleFlowTest` and `Poiseuille3DTest` now
  prescribe the pressure on the streamwise faces, which is what a pressure-driven channel
  needs: with zero-gradient everywhere the only streamwise forcing is the divergence of the
  boundary velocities, a dipole at the first and last interior column worth half the momentum
  balance, so the profile decays. Measured `dp/dx` goes from -0.891 to **-1.604** against an
  analytical -1.600 (0.23% error), profile RMS from 0.121 to **0.00079**, and the mass-flux
  imbalance from 22.6% to **0.02%**. Prescribing the gradient makes
  `test_pressure_gradient` partly a consistency check; the independent content is in the
  profile-RMS and mass-conservation assertions, which only pass if the momentum balance
  sustains the parabola under that pressure difference.

- **Helmholtz shift in the Poisson solvers** — new `params.helmholtz_shift` (sigma; 0 = the
  existing pure-Poisson path), in the common block beside `walls`, since it describes the
  operator rather than the algorithm. The solved equation becomes
  `nabla^2 x - sigma*x = rhs`, which is what implicit diffusion needs:
  `(I - nu*dt*nabla^2)u = b` rearranges to sigma = 1/(nu*dt) with rhs = -b/(nu*dt).
  Because `-nabla^2` is already SPD, a positive shift makes the operator strictly
  diagonally dominant, so it is better conditioned than the pressure solve beside it:
  `lambda_min = sigma` exactly and `cond = 1 + 8d` with `d = nu*dt/h^2`. Measured on a
  65x65 mixed-mode problem, CG takes 160 iterations at sigma = 0 and 5 at sigma = 1e6; over
  a 33 -> 129 refinement the unshifted count grows 78 -> 325 while the shifted one moves
  4 -> 6. A shift also removes the Neumann nullspace, so a shifted solve is nonsingular and
  the zero-interior-mean rule does not apply to it.
  The shift is applied through the existing `axpy` primitive at the call site rather than
  fused into the kernels, so at sigma = 0 the branch is not taken and the unshifted path
  executes the instructions it always did — bit-identity by construction rather than by an
  IEEE argument, and immune to `0.0 * inf`. Verified by memcmp against an untouched
  `params_default()` solve, including a `-0.0` shift.
  Support is default-deny from one central table: a backend that ignored the shift would
  silently solve the unshifted equation, which is a wrong answer rather than a slow one.
  Scalar and OpenMP CG implement it (with `POISSON_PRECOND_JACOBI`; OpenMP matches scalar to
  4e-15); every other method, backend and the multigrid preconditioner return
  `CFD_ERROR_UNSUPPORTED` at init, and negative or non-finite shifts return
  `CFD_ERROR_INVALID` — a negative shift is the indefinite Helmholtz operator, which is
  not SPD. Verified exact against a discrete manufactured solution to 3e-16 in 2D and
  5e-15 in 3D across sigma from 0 to 1e8, with a sweep asserting every (method, backend)
  pair behaves as the table says
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/linear/linear_solver_internal.h`,
  `lib/src/solvers/linear/cpu/linear_solver_cg.c`, `tests/math/test_helmholtz_shift.c`,
  `tests/solvers/test_linear_solver.c`).

- **Implicit viscous time integration** (`ns_solver_params_t.viscous_scheme`):
  `NS_VISCOUS_SCHEME_BACKWARD_EULER` (θ = 1, L-stable) and `NS_VISCOUS_SCHEME_CRANK_NICOLSON`
  (θ = ½) on the scalar `projection` and OpenMP `projection_omp` solvers. The predictor's
  explicit increment is replaced by the solution of `(I − θν·dt·∇²)δ = b`, one shifted CG
  solve per velocity component, which removes the diffusion limit on dt; `compute_time_step()`
  leaves that limit out for an implicit scheme. The gain is stability, not overall order:
  convection stays explicit and the projection is non-incremental, so a full step remains
  O(dt). A viscous-bound cavity at 20× the explicit limit stays bounded where the explicit
  scheme saturates. `explicit` (0) is the default and runs exactly as before.
  - Refusals go through a new capability flag, `NS_SOLVER_CAP_IMPLICIT_VISCOUS`, checked in
    `solver_init`, `solver_step` and `solver_solve`. The exported GPU entry points
    (`solve_*_gpu`, `gpu_solver_step`) skip those wrappers, so they refuse it themselves.
    Every other solver returns `CFD_ERROR_UNSUPPORTED`, including when the scheme is set on
    params after init, as does an implicit scheme combined with a turbulence model. Unknown
    values are `CFD_ERROR_INVALID`.
  - The projection context owns a second Poisson solver for the viscous solve. It is rebuilt
    when dt (and so the shift) changes.

- **First-order upwind convection** — new `ns_solver_params_t.convection_scheme` field
  (`ns_convection_scheme_t`; 0 = existing central differencing, unchanged).
  `NS_CONVECTION_SCHEME_UPWIND` takes each convective first derivative from the side the
  local velocity comes from, for momentum (`u·∇u`) and temperature advection (`u·∇T`), so
  convection-dominated flows stay free of the wiggles central differencing produces. The
  scheme is O(h) with numerical diffusion |u|h/2; pressure gradients, viscous terms and
  the divergence stay central. Implemented on the scalar, OpenMP and AVX2 backends of the
  explicit Euler, projection, RK2 and RK4 solvers (the AVX2 kernels blend the one-sided
  differences with a mask). GPU solvers reject upwind with `CFD_ERROR_UNSUPPORTED` at init
  and at step, and unknown values return `CFD_ERROR_INVALID` at init. The derivative is
  also available as `stencil_upwind_deriv_x/y/z()` in `cfd/math/stencils.h`.
  Verified: stencil and solver-level refinement give first order (RK2 advection rates
  0.90 and 0.95, central 2.02 and 2.01); a step advected at CFL 0.5 stays within its
  initial range on all four scalar solvers and in the energy equation while central differencing
  overshoots; OpenMP and AVX2 upwind match scalar to round-off (relative L2 below 2e-16)
  in 2D and 3D
  (`lib/src/solvers/navier_stokes/ns_convection_internal.h`,
  `lib/src/solvers/navier_stokes/avx2/upwind_avx2.h`, `lib/src/api/solver_registry.c`,
  `tests/math/test_upwind_stencils.c`, `tests/math/test_upwind_convergence.c`,
  `tests/solvers/navier_stokes/test_convection_scheme.c`,
  `tests/solvers/energy/test_energy_solver.c`,
  `tests/solvers/navier_stokes/cpu/test_ns_solver_3d.c`).

- **Restart / checkpoint support** (`cfd/io/checkpoint.h`, `CFD_CHECKPOINT_FORMAT_VERSION` 8) — a
  portable, versioned, CRC-protected binary format (`.cfdchk`) that saves and restores the
  complete simulation state: grid; flow field, including the turbulence arrays `turb_k`,
  `turb_eps`, `turb_nu_tilde` and `nu_t`; time; solver name; and the solver parameters, among
  them the turbulence model and BCs (wall law and segments included), the eddy-viscosity
  correction, the pressure solver and pressure BCs, the convection and viscous schemes and the
  thermal BCs. Little-endian fixed-width encoding with an endianness marker, and a
  format-version header that rejects any other version. Settings read back are validated, so
  an unknown wall law or viscous scheme, or a segment count above the array, is refused. N
  steps, a checkpoint and M more match N+M continuous steps bit for bit under k-epsilon and SA
  on the scalar RK2 solver and under k-epsilon on the AVX2 one.
  - Caller-owned pointers are not stored: the `source_func` / `heat_source_func` callbacks,
    the learned closure `turb_closure`, and a turbulence segment's profile callback. After
    `load_simulation_from_checkpoint()` the caller re-supplies them;
    `restore_simulation_checkpoint()`, which restores in place, carries the callbacks and
    `turb_closure` across.
  - A profiled DIRICHLET segment's values are stored as NaN. Loading succeeds, since solver
    init accepts that marker, but `solver_step()` and `solver_solve()` refuse the segment, for
    every model and before any field moves, until the profile is re-attached, rather than run
    on placeholders. An in-place restore carries the live profile across by slot, but only onto
    a segment stored that way, never onto one saved with constant values
  (`lib/src/io/checkpoint.c`, `lib/src/api/simulation_api.c`, `tests/io/test_checkpoint.c`).

- **`ns_solver_stats_t.dt_used`** reports the time step a solve actually advanced by.
  The explicit Euler solvers clamp their own step to the new public `NS_EULER_DT_LIMIT`
  regardless of `params.dt`, so a caller measuring simulated time could not get it right
  from the parameters alone. `solver_step()` and `solver_solve()` default the field to
  `params.dt` and the Euler wrappers, CUDA included, override it, so every solver reports a
  usable value; the clamp itself now has one definition instead of four
  (`lib/include/cfd/solvers/navier_stokes_solver.h`, `lib/src/api/solver_registry.c`,
  the explicit Euler kernels under `lib/src/solvers/navier_stokes/`).

- **Inlet on part of an edge** — `bc_inlet_set_range(&cfg, start, end)` restricts an inlet
  to the nodes whose normalized edge position lies in `[start, end]`, leaves every other node
  of the edge untouched, and lays the profile over the range, so a parabola is zero at both
  ends. A new `range` member of `bc_inlet_config_t`; zero-initialized (every factory) means
  the whole edge, exactly as before. Honoured by the shared CPU inlet (scalar, OpenMP, AVX2,
  NEON), the time-varying inlet and the GPU inlet kernels. A range that is not
  `0 <= start < end <= 1`, or one on a z-face, is `CFD_ERROR_INVALID`
  (`tests/core/test_boundary_conditions_inlet.c`).

- **Laminar backward-facing step validation** (`tests/validation/test_backward_facing_step.c`,
  `validation` label). Gartling's geometry, expansion ratio 2, inflow over the upper half of
  the left edge through the range above, outlet at `p = 0`. Lower-wall reattachment length at
  Re = 100 (scalar projection, 3.12 at 33 nodes across H) and Re = 400 (OpenMP) against
  Armaly et al. and 2D computations. With the inlet at the step the solver converges about
  6% above the channel-inlet references at Re = 100, as Barton (1997) predicts for this
  geometry. Time step and domain length are shown not to matter. See
  `docs/validation/backward-facing-step.md`.
- **129×129 lid-driven cavity validation recorded** — the AVX2, OpenMP and CUDA projection
  backends match Ghia et al. (1982) at 129×129 with RMS_u / RMS_v of 0.0017 / 0.0024
  (Re=100), 0.0096 / 0.0328 (Re=400) and 0.0299 / 0.0300 (Re=1000), identical across
  backends to four decimals and across three EC2 workflow runs. Each backend line of
  `test_cavity_backends` now also prints the steps actually run, whether the run reached
  steady state, and its residual
  (`docs/validation/cavity-backends-validation.md`, `tests/validation/test_cavity_backends.c`).
- **Multi-Reynolds grid-convergence study with Richardson extrapolation**
  (`test_cavity_richardson.c`, ROADMAP 6.1). The steady cavity is solved on three grids at
  Re = 100 (33/65/129), 400 (65/129/257) and 1000 (129/257/513), and the observed order,
  extrapolated value and GCI are computed per Celik et al. (2008) for u at the centre and the
  centreline extrema. CI runs Re = 100 on 17/33/65. It runs on the OpenMP projection with the
  multigrid pressure solve, selected through `cavity_run_with_pressure_solver_ctx()`. At
  Re=1000 the extrapolated extrema are within 0.11% of Botella & Peyret (1998). The observed
  order is about 1.3, not 2: see `docs/validation/cavity-grid-convergence.md`.
- **Cross-architecture consistency across every solver family** (ROADMAP 6.1).
  `test_solver_architecture.c` compares the whole field of every backend (AVX2, OpenMP,
  CUDA) with the scalar reference for projection and Explicit Euler on the 33x33 cavity,
  and for Explicit Euler, RK2 and RK4 on a periodic Taylor-Green vortex. All must agree
  within 0.1% (velocity against max |u|; pressure with its mean removed against its range).
  The test it replaces compared one center value, with absolute tolerances of 0.002 and
  0.01, and never covered RK2/RK4 or CUDA Euler.
- **Extended-time Taylor-Green decay rate** (`test_taylor_green_decay.c`, ROADMAP 6.1).
  The kinetic-energy decay rate is fitted over t = 10 (t = 20 in full validation), during
  which the energy falls by 86% (98%). The fit must match −4ν, with no drift between the
  early and late thirds of the run, and converge at second order. It runs on one wall-bounded
  vortex cell, because the projection's pressure solve has no periodic option.
  `docs/validation/taylor-green-decay.md` records why the periodic vortex cannot serve: the
  projection reaches 0.92 of the rate there, and the pseudo-compressible Euler/RK solvers 0.42.

- **Examples for features that had none.** `natural_convection.c` (energy equation,
  Boussinesq buoyancy, thermal BCs; within 1.1% of de Vahl Davis at Ra = 1000),
  `steady_flow_multigrid.c` (the multigrid pressure solve, timed against CG, plus
  checkpoint/restart that reproduces an uninterrupted run: bit-identically on one thread,
  to round-off with several, since threaded reductions do not sum in a fixed order), and
  `pressure_driven_channel.c` (Dirichlet pressure faces and implicit viscous stepping,
  against the exact Poiseuille profile). `taylor_green_convergence.c` adds RK4 to its
  solver comparison; `lid_driven_cavity_direct.c` takes an `upwind` argument for
  `params.convection_scheme`.

### Changed

- **The Poisson API no longer accepts configuration it cannot honour.** An audit found
  eleven places where a parameter could be set and then silently ignored. `poisson_solver_init`
  now refuses each of them, with `cfd_get_last_error()` carrying the sentence that names the
  fix. This is a breaking change throughout; there are no external users yet, so it is made
  outright rather than behind deprecations.
  - **Parameters are grouped by the method that reads them**: `params.omega` → `params.sor.omega`,
    `params.preconditioner` / `params.restart` → `params.krylov.*`, and the seven `params.mg_*`
    fields → `params.multigrid.{cycle,smoother,bc,pre_smooth,post_smooth,coarse_max_iter,max_levels}`.
    `walls`, `helmholtz_shift`, `tolerance`, `absolute_tolerance`, `max_iterations`,
    `check_interval` and `verbose` stay common. The grouping is what makes the refusal
    expressible: with a flat struct, "the caller set an SOR knob on a CG solve" and "the
    caller left it alone" are the same bytes. A non-zero group the resolved method does not
    read is `CFD_ERROR_INVALID`.
  - Refused within an owned group: a preconditioner on BiCGSTAB, which implements none on any
    backend, and on every GPU solver, none of which has an `M^-1` apply
    (`CFD_ERROR_UNSUPPORTED`); `krylov.restart` outside GMRES; a multigrid preconditioner
    outside scalar and OpenMP CG; a `multigrid` group that breaks the inner cycle's symmetry
    while it serves as a preconditioner; `check_interval` below 1, which reached `iter % 0`.
    Gauss-Seidel overriding `sor.omega` to 1 is documented on the enum member and pinned by
    `test_gauss_seidel_is_sor_at_omega_one`.
  - **A caller's `apply_bc` is refused where it would be ignored**: on multigrid, and on every
    GPU solver. Multigrid used to install its own routine into that same public slot, so a
    caller following the documented create/assign/init workflow overwrote multigrid's hook and
    their own boundary condition then had no effect on the solve. Solvers now install their
    walls in a separate `internal_apply_bc`, which makes `apply_bc != NULL` mean "the caller
    prescribed walls" everywhere it is asked.
  - **`poisson_solver_type` is replaced by `poisson_solver_config_t` + `poisson_preset_t`.**
    The old enum was 13 hand-maintained (method × backend) pairs duplicating two existing enums,
    omitting GMRES, BiCGSTAB and the GPU entirely, and unable to express `params` at all.
    `poisson_solver_config_preset()` now returns an editable config — six intents × any
    backend × any parameters. Its `DEFAULT` is CG, which *is* subject to the
    incompatible-RHS refusal; the old `DEFAULT_POISSON_SOLVER` was Red-Black SOR, which is
    exempt, so switching preset used to change silently whether an RHS was legal.
    `POISSON_PRESET_SMOOTHER` is the exempt one and says so. Both multigrid presets name
    `POISSON_BACKEND_SCALAR`, since AUTO resolves to SIMD where no CG implements the
    multigrid preconditioner.
  - **`poisson_solve()` returns `cfd_status_t`** and takes a config. Every failure — unknown
    preset, create failure, init rejection, max-iter, divergence, unsolvable RHS — used to
    collapse to `-1`. `poisson_solve_3d`, `poisson_solve_3d_params` and three never-called
    wrappers are deleted; the arity and first-parameter type both change, so an old call site
    is a compile error rather than a silent misinterpretation.
  - **The 13-slot solver cache is deleted** (~210 lines, with its `atexit` and a `memcmp`
    cache key over struct padding). It served one caller: each projection solver now owns its
    pressure solver for its lifetime (`ns_pressure_internal.h`), rebuilding only when the
    configuration or grid actually changes, and refuses a field whose dimensions differ from
    the grid that solver was built for.
  - `ns_check_pressure_solver()` joins `ns_check_pressure_bc()` in all nine time-integrator
    inits, so `ns_solver_params_t.pressure_solver` and `pressure_bc` are validated by every
    integrator, the AVX2 ones included.
  - A refused solve resets the whole stats struct, so a reused one cannot show the previous
    solve's residual beside `POISSON_INCOMPATIBLE_RHS`.
  - `poisson_solver_status_string()` is new: `POISSON_INCOMPATIBLE_RHS` had no name and printed
    as a generic "error" in both examples, each of which hand-rolled its own ternary chain.
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/navier_stokes/ns_pressure_internal.h`,
  `lib/src/solvers/navier_stokes/ns_convection_internal.h`, `lib/src/api/solver_registry.c`,
  `tests/math/test_poisson_config.c`)

- **Boundary-condition OpenMP regions open only for large edges.** Entering an OpenMP region
  costs ~12-15 us at 4 threads on MSVC, while a 2D edge copy is well under 1 us, and every
  OpenMP and SIMD BC kernel opened one per edge (per z-plane in 3D): a threaded 2D Neumann
  call took ~50 us against 0.06-6 us serial, and 129^3 took ~5 ms against 63 us. A region
  now uses the team only when it writes at least `BC_OMP_MIN_POINTS` (32,768) points, the
  crossover measured for 2D Neumann and the value multigrid uses; it replaces the 256-point
  `BC_SIMD_THRESHOLD` (`lib/src/boundary/boundary_conditions_internal.h`,
  `docs/technical-notes/openmp-vs-scalar.md`).
- **The Krylov halo refresh no longer opens an OpenMP region per iteration.** A Neumann
  face is a copy from the adjacent interior line -- O(nx + ny), about 130 writes on a
  33x33 plane -- and it runs once per Krylov iteration (twice for BiCGSTAB). An MSVC
  parallel region costs ~12 us to enter at 4 threads, more than the Laplacian sweep it
  accompanied. It is serial on every backend now; all three produced identical values.
  Measured A/B, `GhiaProjectionAvx2Test` run alone: 53.7 s before, 47.6-50.1 s after.
- **Unpreconditioned CG computes the residual norm once per iteration.** `res_norm` reuses
  `rho_new = (r,r)` instead of a second O(N) pass, in the scalar, AVX2, NEON and OMP backends.
- **The GPU projection's compatibility projection stays on the device.** It computed
  the RHS interior mean on the GPU, copied it back, and `cudaStreamSynchronize`d before
  dividing -- one full pipeline drain per inner iteration. The subtract kernel now reads the
  device-side sum and the interior count and divides itself, and the discarded
  `cudaMemsetAsync` status is checked. No measurable speedup (`GhiaProjectionGpuTest` on an
  RTX 4090: 135.7 s before, 129.0-136.1 s after), because `cg_gpu_solve_device`
  synchronizes for its own dot products anyway; kept as a correctness and clarity fix.

### Fixed

- **The Krylov pressure solve uses the zero-gradient walls it always claimed, and forms
  its residual against them.** CG, BiCGSTAB and GMRES update interior points only, so the
  operator they invert is whatever their vectors' halos say it is. Their search directions
  carried a permanent zero halo, making the operator Dirichlet, while the initial residual
  was formed from `x` after `apply_bc` had filled its halo with zero-gradient copies of the
  interior. Two defects, one on top of the other.
  The residual and the operator disagreed for any non-zero initial guess, so the solve
  converged to a field satisfying neither system — it returned `x` with
  `A_dirichlet*x = b + (A_dirichlet - A_neumann)*x0`, an error of O(x0/h²) along every wall.
  Measured on a 33×33 Poisson problem, re-solving from an already-converged field moved it
  by 9.4% of its own magnitude on all six CPU Krylov backends. The projection solver
  warm-starts each pressure solve from the previous pressure, so every inner iteration
  after the first took that path.
  And the operator itself was wrong: holding the pressure at zero on every wall is not the
  boundary condition a projection method wants, and not the one the solver documented.
  Each solver now applies the **homogeneous** part of the boundary condition to its search
  directions before every operator apply, and the **full** condition to the iterate before
  any residual — including GMRES at each restart, where the iterate has moved and the
  extension is stale. For the default walls the homogeneous part is the zero-gradient
  extension itself; for an `apply_bc` hook, whose prescribed values belong to the iterate,
  it is a zero halo, since an inhomogeneous condition on a direction would make the
  operator affine rather than linear.
  Re-solving from a converged field now returns it in 0–3 iterations on every backend
  (`tests/math/test_krylov_warm_start.c`, which states this as idempotence at the solution
  and needs no reference to halos).
  **Behaviour change:** with the default walls the operator is singular, the constants
  being its nullspace, so the RHS must now have zero interior mean — the same requirement
  standalone multigrid has in `MG_BC_NEUMANN` mode. The projection solvers subtract the
  interior mean of `div(u*)` on every preset; callers who relied on a uniform RHS converging
  should either call `poisson_make_rhs_compatible()` (now exported) or prescribe a face
  through `params.walls`, which makes the operator nonsingular and admits any RHS.
  An incompatible RHS is refused up front with `CFD_ERROR_INVALID` and
  `stats.status = POISSON_INCOMPATIBLE_RHS`, instead of being iterated on: such a system
  has no solution, and iterating either stalls on the component in the nullspace or drives
  the rest of the field away chasing it. `test_laplacian_accuracy` had been reaching a
  residual of 1e21 that way. The 3D projection goldens were re-pinned: `L2(p)` now stays
  ~1.0, since a constant is in the nullspace and the caller's pressure level is carried
  rather than driven to zero walls
  (`lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/linear/linear_solver_internal.h`, the six CPU Krylov backends and the
  two GPU ones, `lib/src/solvers/navier_stokes/{cpu,omp,avx2}/solver_projection*.c`,
  `tests/math/test_krylov_warm_start.c`, `tests/math/test_bicgstab.c`,
  `tests/math/test_gmres.c`, `tests/solvers/test_linear_solver.c`,
  `tests/solvers/navier_stokes/cpu/test_ns_solver_3d.c`).
- **`run_simulation_step()` and `run_simulation_solve()` now step with the caller's
  `params.dt`.** Both overwrote it with a hard-coded 0.005 on every call ("for animation
  stability"), so the documented `sim->params.dt = ...` had no effect. On a fine grid that
  fixed step breaks the diffusive stability limit: a 129x129 Re=100 cavity, whose explicit
  limit is about 1.2e-3, diverged to the velocity clamp. `init_simulation()` still defaults
  dt to 0.001, so a simulation that never sets it now steps at 0.001, not 0.005.
  `tests/simulation/test_simulation_api.c` checks that both entry points advance by exactly
  the dt set, including a change between steps; against the old code both checks fail
  (`Expected 0.0003 Was 0.005`). The Taylor-Green output quoted in
  `docs/guides/examples.md` is refreshed.
- **A step of zero, a negative step or a non-finite step is refused.** `solver_step()`,
  `solver_solve()` (and through them `run_simulation_step()` / `run_simulation_solve()`)
  and the exported GPU entry points (`solve_*_gpu`, `gpu_solver_step`) return
  `CFD_ERROR_INVALID` unless `params.dt` is finite and positive. Before, `dt = 0` succeeded
  with `current_time` frozen, a negative `dt` integrated backwards, and NaN filled every field
  without an error. `run_simulation_step()` and `run_simulation_solve()` also reset
  `last_stats` before each call, so a solve refused before it ran does not advance
  `current_time` by the previous call's step.
- **The 3D time step is sized from every plane, and `current_time` advances by the step
  actually taken.** `ns_dt_convective()` scanned only the k = 0 plane of a 3D grid, so a
  field whose fastest flow sat above it got a step sized from stagnant fluid: 0.01 against a
  correct 4.34e-4 in the regression test, 23x past the CFL limit. Separately,
  `run_simulation_step()` and `run_simulation_solve()` accumulated `current_time` from
  `params.dt` while the explicit Euler kernels clamp their step to `NS_EULER_DT_LIMIT`, so
  `current_time`, which checkpoints and output record, could run 50x ahead of the time
  simulated. Both now advance by `stats.dt_used` (see Added), and the CUDA wrappers report
  the iteration count they ran (`lib/src/solvers/navier_stokes/ns_dt_internal.h`,
  `lib/src/api/simulation_api.c`).
- **The laminar viscous stability limit is now applied.** `compute_time_step()` gated the
  viscous constraint `dt < h^2 / (2*nu_eff*ndim)` behind `turb_model != TURB_MODEL_NONE`,
  so for laminar flow it never ran and `compute_dt` could return an unstable step. The
  limit scales as `h^2` against the convective limit's `h`, so it binds as grids refine:
  on a 129x129 unit cavity at Re=100 (nu=1e-2, cfl=0.5) it is 7.6e-4 against a CFL limit
  of 3.3e-3. It now applies to the molecular viscosity unconditionally, still adding any
  eddy viscosity when a turbulence model is active. `compute_time_step()` is also
  decomposed into `ns_dt_convective` / `ns_dt_viscous` / `ns_dt_thermal`, each returning
  INFINITY where its process imposes no limit, so a solver that treats a term implicitly
  can leave that constraint out. `tests/core/test_cfl.c` had no viscous case at all and
  gains five; three existing sound-speed tests now set `mu = 0` to keep isolating the
  acoustic branch
  (`lib/src/solvers/navier_stokes/cpu/solver_explicit_euler.c`,
  `lib/src/solvers/navier_stokes/ns_dt_internal.h`, `tests/core/test_cfl.c`).
- **A Poisson solve started at its own solution no longer runs to `max_iterations`.** The
  residual of a converged field cannot be measured below round-off, about
  ε·|x|·(2/dx² + 2/dy²), and that floor grows as |x|/h² while `absolute_tolerance` is fixed
  at 1e-10. Near steady state, where each pressure solve warm-starts from the last one, the
  stopping target fell below anything the solver could reach. Stationary solvers and
  multigrid then spun to the cap and returned `CFD_ERROR_MAX_ITER`, which is how a 513×513
  Re=1000 cavity on the multigrid pressure solve failed at step 60,895. Krylov solvers
  chased noise for up to 125 iterations, and in the worst cases moved the field: SIMD and
  GPU CG by 2.5% and 3.6%, and GPU BiCGSTAB diverged. Every stopping rule, on every backend,
  now stops at max(tolerance·r₀, absolute_tolerance, floor), where the floor is ten times
  that round-off estimate (`poisson_solver_residual_floor`,
  `lin_gpu_residual_floor_l2`). A solve started from a cold guess is unaffected, since its
  floor is zero. `tests/math/test_poisson_roundoff_floor.c` starts all 27 method/backend
  pairs from a converged 129×129 field whose floor is ten times `absolute_tolerance`.
  Before this fix none returned at once; now all return after 0 iterations.
- SOR and Red-Black SOR ran far from their best omega with the default zero-gradient walls.
  The walls are copied from the interior after each sweep, so a point beside a wall read its own
  previous value through the copy, and the automatic omega came from the Dirichlet formula, well
  below the best one: Red-Black SOR on a 33x33 seeded-noise problem took 361 sweeps at the
  automatic omega and 193 at the best. Wall-adjacent points now relax by
  omega * factor / (factor - wall weight), which makes the iteration SOR on the Neumann matrix
  itself, and the automatic omega estimates that matrix's optimum from just below, through a
  Rayleigh quotient of its slowest mode. The same problem now takes 117 sweeps, and 65x65 to
  257x257 grids about a third of their
  former sweeps (709 to 235 at 65x65). The converged solution is unchanged. A custom `apply_bc`
  keeps the Dirichlet formula and no wall scaling; wall values other than the default copy now
  have to be set through `apply_bc` rather than written between iterations. The CUDA solvers apply
  the walls on the device and never call `apply_bc`, so their init rejects one with
  `CFD_ERROR_UNSUPPORTED`. Applies to every SOR and Red-Black SOR backend: scalar, OpenMP, AVX2,
  NEON and CUDA (`lib/src/solvers/linear/linear_solver_internal.h`, `lib/src/solvers/linear/cpu/`,
  `lib/src/solvers/linear/omp/`, `lib/src/solvers/linear/avx2/`, `lib/src/solvers/linear/neon/`,
  `lib/src/solvers/linear/gpu/`, `tests/math/test_optimal_omega.c`,
  `tests/math/test_poisson_accuracy.c`, `tests/math/test_poisson_sor_gpu.c`).
- The SIMD SOR solvers (AVX2 and NEON) ran a Block SOR that read the left neighbour inside each
  SIMD block from the previous sweep. That is not the SOR iteration, and on the AVX2 build it
  diverged for omega between 1.40 and 1.50 on every grid measured, below the automatic omega: a
  33x33 solve at the automatic omega reported convergence after 1,764 sweeps with a residual of
  exactly zero. Each row is now swept in two passes, the stencil terms the sweep does not write
  with SIMD and then the relaxation in order, which is scalar SOR sweep for sweep
  (`lib/src/solvers/linear/avx2/linear_solver_sor_avx2.c`,
  `lib/src/solvers/linear/neon/linear_solver_sor_neon.c`, `tests/solvers/test_linear_solver.c`,
  `docs/technical-notes/block-sor-simd.md`).
- `POISSON_METHOD_GAUSS_SEIDEL` created the SOR solvers and ran at SOR's automatic omega, not at
  1: a 33x33 zero-gradient solve took 380 sweeps where omega = 1 takes 2,376. It now always
  relaxes with omega = 1, whatever `params.sor.omega` says
  (`lib/src/solvers/linear/linear_solver.c`, `lib/src/solvers/linear/linear_solver_internal.h`,
  `tests/solvers/test_linear_solver.c`).
- A solve on the shared loop that blew up could report convergence.
  `poisson_solver_compute_residual()` kept the largest residual with `>`, which is false for
  NaN, so a field that had overflowed read as a zero residual and passed the tolerance test.
  The residual of such a field is now NaN, and the solve stops with `POISSON_DIVERGED` and
  `CFD_ERROR_DIVERGED` at the first convergence check that finds it no longer finite. The shared
  loop also checks the last iteration, whatever `check_interval` is, so a solve never ends on a
  residual older than the field it returns, and a solve whose starting residual is not finite
  stops before its first sweep, which could otherwise overwrite the bad value and report
  convergence. The CUDA Jacobi, SOR and Red-Black SOR solvers, which turned their checks off and
  ran out `max_iterations` on a residual that was not finite, report divergence the same way
  (`lib/src/solvers/linear/linear_solver.c`, `lib/include/cfd/solvers/poisson_solver.h`,
  `lib/src/solvers/linear/gpu/`, `tests/solvers/test_linear_solver.c`,
  `tests/math/test_poisson_sor_gpu.c`, `tests/math/test_poisson_jacobi_gpu.c`).
- Solvers on the shared solve loop (Jacobi, SOR and Red-Black SOR on every CPU backend)
  reported one more iteration than they ran when they exhausted `max_iterations`;
  `stats.iterations` now counts the iterations performed, as CG and BiCGSTAB already did
  (`lib/src/solvers/linear/linear_solver.c`, `tests/solvers/test_linear_solver.c`,
  `tests/math/test_omp_consistency.c`).
- `cg_omp` and `bicgstab_simd` now reject grids whose `nx` or `ny` exceeds `INT_MAX`, or
  whose `nx*ny*nz` overflows `size_t`, with `CFD_ERROR_LIMIT_EXCEEDED` at init. Their
  primitives loop over `int` bounds, which such grids silently truncated (`cg_omp`) or
  emptied (`bicgstab_simd`, where a solve then reported convergence from a zero residual).
- **`explicit_euler_optimized` updates every interior column.** The AVX2 row loop processed
  4-wide groups with no scalar remainder, so when `(nx-2) % 4 != 0` the last 1-3 interior
  columns of each row kept their old values (3 per row at 33×33 and 129×129). The remainder
  now runs through the solver's scalar row path, and a 19×19 AVX2-vs-scalar run agrees to
  1e-17 (`lib/src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c`,
  `tests/solvers/navier_stokes/avx2/test_solver_explicit_euler_avx2.c`).
- **`explicit_euler_optimized` applies momentum source terms in its AVX2 lanes.** The
  vectorized columns skipped `source_func` and the default sinusoidal sources, while the
  scalar remainder columns applied them, so a run with sources on disagreed from column to
  column. Both paths now call `compute_source_terms()` as the scalar `explicit_euler` step
  does, which also passes the z coordinate to `source_func` in 3D and applies negative
  amplitudes and a v-amplitude set without a u-amplitude. With the default sources on,
  AVX2 matches scalar to 4e-18 on 19×19 and 3e-17 on 32×32 (previously 2.5e-3 on 19×19).
  Runs with both amplitudes zero and no `source_func`, such as the cavity validation,
  are unchanged
  (`lib/src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c`,
  `tests/solvers/navier_stokes/avx2/test_solver_explicit_euler_avx2.c`).
- **`explicit_euler` and `explicit_euler_omp` return their errors from `step` and `solve`.**
  The registry wrappers discarded the implementation's status and always returned
  `CFD_SUCCESS`, so a diverged (NaN/Inf) field, an allocation failure, non-uniform z
  spacing or a failed energy or thermal-BC step went unreported and the
  simulation kept stepping. The wrappers now return the status before filling in stats,
  as the projection wrappers do. A new test seeds a NaN pressure value and checks that
  both calls return `CFD_ERROR_DIVERGED` on each backend
  (`lib/src/api/solver_registry.c`, `tests/solvers/navier_stokes/test_solver_helpers.h`,
  `lib/src/solvers/navier_stokes/omp/solver_explicit_euler_omp.c`,
  `tests/solvers/navier_stokes/cpu/test_solver_explicit_euler.c`,
  `tests/solvers/navier_stokes/omp/test_solver_explicit_euler_omp.c`).
- **`init_simulation` and `init_simulation_with_solver` return NULL when the solver fails
  to initialize.** They ignored `solver_init()`'s status and returned a simulation whose
  solver was never set up, so the failure only surfaced on the first step, if at all. They
  now free the partial simulation and return NULL, with `cfd_get_last_status()` reporting
  the solver's status (for example `CFD_ERROR_INVALID` from `explicit_euler_optimized` on a
  grid narrower than 3 points)
  (`lib/src/api/simulation_api.c`, `lib/include/cfd/api/simulation_api.h`,
  `tests/simulation/test_simulation_api.c`).
- **CUDA Explicit Euler, RK2 and RK4 now treat boundaries as their CPU counterparts do.**
  The shared GPU driver (`solver_rk_gpu.cu`) restored the caller's boundary values after
  every step, and ran Explicit Euler through the RK kernel's wrap-around stencil. The CPU
  RK2/RK4 solvers instead make every field periodic after the step, and CPU Explicit Euler
  reads the ghost cells, keeps the caller's velocity boundaries and makes p and T periodic.
  As a result, CUDA Explicit Euler never saw a cavity lid (75% field difference from the
  scalar solver), and CUDA RK2/RK4 lagged the scalar ghost values by one step (0.18%). Each
  order now follows its CPU reference, and all three agree with the scalar solver to
  round-off.
- **The exported GPU Runge-Kutta entry points refused nothing on a small grid.**
  `solve_explicit_euler_method_gpu`, `solve_rk2_method_gpu` and `solve_rk4_method_gpu` checked
  whether the grid was large enough for the GPU before validating their parameters, so an
  unsupported configuration (a turbulence model, upwind convection, a prescribed pressure
  face, a host callback) on a grid below the GPU threshold came back as a bare `CFD_ERROR`
  instead of its refusal. The size check now runs after validation, as it already did in
  `solve_projection_method_gpu`.
- **SIMD backend availability now requires the compiled-in kernels, not just the CPU.**
  `cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)` returned `cfd_has_simd()`, a pure
  runtime CPUID query, while `CFD_ENABLE_AVX2` defaults to `OFF`. The default build on any
  AVX2-capable machine therefore advertised a SIMD backend it did not contain, and
  `cfd_solver_create_checked()` handed back `*_optimized` solvers that could not run. The
  check is now compile-time AND runtime, through one shared predicate
  (`ns_simd_backend_available()`) that the registry and all four SIMD solvers share, so
  they cannot disagree. There are no NEON Navier-Stokes kernels, so NEON CPUs correctly
  report the SIMD backend unavailable
  (`lib/src/solvers/navier_stokes/simd/ns_simd_backend.c`,
  `lib/src/solvers/navier_stokes/ns_simd_backend_internal.h`, `lib/src/api/solver_registry.c`,
  `tests/solvers/test_solver_backend_api.c`, `tests/core/test_modular_libraries.c`,
  `tests/core/test_modular_core_simd.c`).
- **One failure mode for a build without SIMD.** The same configuration produced three
  different outcomes: `explicit_euler_optimized` silently ran scalar loops and returned
  `CFD_SUCCESS` -- the cross-backend fallback the error-handling rules forbid --
  `rk2_optimized`/`rk4_optimized` returned `CFD_ERROR_UNSUPPORTED`, and
  `projection_optimized` returned it indirectly via a NULL sub-solver probe whose message
  named the wrong cause. All four now fail at init with `CFD_ERROR_UNSUPPORTED` and a
  message that says whether the build or the CPU is missing AVX2; the dead scalar
  fallback kernel in the AVX2 Euler solver is deleted. `rk2`/`rk4` additionally gained the
  runtime half of the check, which they lacked -- an AVX2 build on a pre-AVX2 CPU
  previously initialized successfully and then executed unsupported instructions
  (`lib/src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c`,
  `solver_projection_avx2.c`, `solver_rk2_avx2.c`, `solver_rk4_avx2.c`,
  `tests/solvers/navier_stokes/avx2/test_solver_explicit_euler_avx2.c`,
  `tests/solvers/navier_stokes/cpu/test_solver_explicit_euler.c`,
  `tests/simulation/test_simulation_api.c`, `docs/reference/solvers.md`).
- **A Poisson SIMD factory that returns NULL now says why.** `log_no_simd_available` only
  logged at DEBUG and set no error state, so `poisson_solver_create(..., SIMD)` on a
  build without AVX2 handed back NULL with `cfd_get_last_error()` reading `(null)`. It now
  sets an error naming the fix. Its doc comment also told callers to "fall back to scalar if
  needed", which is the cross-backend fallback the library forbids.
- **AVX2 is only reported when FMA3 is present too.** `cfd_detect_simd_arch()` checked the
  AVX2 bit alone, but FMA3 is a separate CPUID capability: the AVX2 CG, BiCGSTAB, GMRES and
  `.cfdnn` kernels call `_mm256_fmadd_*`, and GCC builds the AVX2 library with `-mfma`. A CPU
  or VM exposing AVX2 without FMA would have selected those kernels and faulted with an
  illegal instruction. It now reports `CFD_SIMD_NONE`, so SIMD requests return
  `CFD_ERROR_UNSUPPORTED` and AUTO falls back (`lib/src/core/cpu_features.c`).
- **Boundary-condition availability messages say what is actually true.**
  `BC_BACKEND_CUDA` reported "CUDA not yet implemented" while the GPU boundary kernels
  existed and were already driving the GPU solvers. They are device-side -- device
  pointers plus a CUDA stream -- and so cannot be reached through `bc_backend_impl_t`,
  a host-pointer table; routing them through it would force a host round-trip per call
  or reinterpret host pointers as device pointers. The host API still reports the
  backend unavailable, now saying why and pointing at
  `cfd/boundary/boundary_conditions_gpu.cuh`. Likewise `BC_TYPE_INLET`/`BC_TYPE_OUTLET`
  warned "not implemented" from `apply_scalar_field_bc()` though both are implemented on
  every backend; what they cannot do is travel through the config-free type-enum path,
  exactly like `BC_TYPE_DIRICHLET` and `BC_TYPE_NOSLIP`. They now name
  `bc_apply_inlet()`/`bc_apply_outlet()` and return `CFD_ERROR_INVALID` rather than
  `CFD_ERROR_UNSUPPORTED`, which tests key on to skip a genuinely missing backend
  (`lib/src/boundary/boundary_conditions.c`, `lib/include/cfd/boundary/boundary_conditions.h`).
- **Run-directory paths that do not fit are refused instead of truncated.** With a long base
  path the composed run directory was silently truncated, so a wrong directory could be
  created and used. Every run-directory variant now checks the composed length and, on
  overflow, leaves the buffer empty, creates nothing and sets `CFD_ERROR_LIMIT_EXCEEDED`; the
  output registry returns NULL, the VTK run writers skip the write, and
  `cfd_get_run_directory()` never returns a truncated path.
- **`ensure_directory_exists()` creates missing parents, like `mkdir -p`.** It made only the
  last level, so run directories under a base such as `../../artifacts` failed whenever that
  base did not exist yet, which is how the examples ran from a fresh checkout.
- **Cavity validation runs to a real steady state.** The harness stopped once the relative
  kinetic-energy change per step fell below 1e-8. That quantity scales with dt, so it
  registered a slow transient as convergence: the Explicit Euler cases ended at t ≈ 1.13 on
  a flow that needs t ≈ 10-20 to develop, and passed their RMS target without a developed
  solution. A kinetic-energy rate test would still stop early, because the cavity's KE
  overshoots before settling and the rate passes through zero at the turning point: a
  129x129 Re=1000 run stopped at t = 45.2, with u at the centre still 7.4e-4 from its
  settled value, as large as the differences between grids. The exit is now a field
  residual, `max |u^{n+1} - u^n| / (dt * U_lid) < 1e-6`, which every cavity test shares; the
  same run continues to t = 106.6. The 129x129 Explicit Euler cases are dropped: they cost
  about an hour of EC2 per run to hold a non-production solver to a relaxed target that
  every projection case clears with over 3x margin
  (`tests/validation/lid_driven_cavity_common.h`,
  `docs/validation/cavity-backends-validation.md`).
- **Natural-convection validation stops at a steady state.** It stopped when the relative
  kinetic-energy change per step fell below 1e-6, which happened at t* = 0.11 with the
  hot-wall Nusselt number still rising (1.089 against 1.117). It now waits until no velocity
  or temperature changes faster than 1e-4 in diffusive units (shared helper
  `tests/validation/steady_state.h`): errors against de Vahl Davis drop from 0.9 / 1.5 / 2.5%
  (u, v, Nu) to 1.0 / 0.0 / 0.4%.
- `examples/poisson_solver_tuning.c` benchmarked SOR and Red-Black SOR at a hard-coded omega of
  1.5 rather than the automatic value, and capped the iterations it printed for an off-by-one that
  is fixed. The `poisson_solver.h` usage example called `poisson_solver_init()` without `nz` and
  `dz`, and `max_iterations` was documented as defaulting to 1000 where the default is 5000.
- `scripts/ec2-validate.sh` runs every `CavityBackend_*` ctest entry. It ran the test binary
  without a filter, which skips the Re=400 and Re=1000 cases, and under `set -e` a failing run
  ended the script before its FAILED summary.

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
