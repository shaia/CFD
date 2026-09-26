# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Wall functions solve Spalding's law of the wall.** `turbulence_wall_u_tau()` switched from
  the linear law to the log law at y+ = 11.63, but with κ = 0.41, B = 5.2 the two cross at
  y+ = 11.06, so u_τ jumped 3.3% at the switch (wall shear 6.6%, first-node ε 10%).
  Spalding's single smooth y+(u+) removes the jump. It sits below the log law until
  y+ ≈ 100, so the Re_τ = 395 channel's u_τ error, measured against the log law, moves from
  2.9% to 5.9% (k-ε) and 3.1% to 6.0% (SA). `WALL_YPLUS_LAMINAR` is removed
  (`lib/src/solvers/turbulence/cpu/turbulence_solver.c`,
  `tests/solvers/turbulence/test_turbulence_wall_functions.c`,
  `tests/validation/test_turbulent_channel.c`).

- **Unpreconditioned CG computes the residual norm once per iteration.** `res_norm` reuses
  `rho_new = (r,r)` instead of a second O(N) pass, in the scalar, AVX2, NEON and OMP backends.

- **Checkpoint format version 5.** `.cfdchk` files now carry `viscous_scheme`, validated on
  read. Version-4 files are rejected as unsupported, as the format has always done on a
  layout change.

- **Checkpoints carry the eddy-viscosity correction setting**, so
  `CFD_CHECKPOINT_FORMAT_VERSION` goes 3 -> 4 and files written by earlier versions are
  rejected. `ns_solver_params_t.turb_nut_correction` is now serialized, and range-checked
  when read back, so resuming a run that used it no longer comes back with the correction
  silently off. The learned-closure context `turb_closure` is a caller-owned pointer and
  still cannot be stored, but `restore_simulation_checkpoint()` now carries it across an
  in-place restore alongside the source callbacks, and the exclusion is documented on both
  load paths (`lib/include/cfd/io/checkpoint.h`, `lib/src/io/checkpoint.c`,
  `lib/src/api/simulation_api.c`, `tests/io/test_checkpoint.c`).

- **The Poisson API no longer accepts configuration it cannot honour.** An audit found
  eleven places where a parameter could be set and then silently ignored. `poisson_solver_init`
  now refuses each of them, with `cfd_get_last_error()` carrying the sentence that names the
  fix. This is a breaking change throughout; there are no external users yet, so it is made
  outright rather than behind deprecations.
  - **Parameters are grouped by the method that reads them**: `params.omega` → `params.sor.omega`,
    `params.preconditioner` / `params.restart` → `params.krylov.*`, and the seven `params.mg_*`
    fields → `params.multigrid.{cycle,smoother,bc,pre_smooth,post_smooth,coarse_max_iter,max_levels}`.
    `walls`, `tolerance`, `absolute_tolerance`, `max_iterations`, `check_interval` and `verbose`
    stay common. The grouping is what makes the refusal expressible: with a flat struct, "the
    caller set an SOR knob on a CG solve" and "the caller left it alone" are the same bytes.
    A non-zero group the resolved method does not read is `CFD_ERROR_INVALID`.
  - Refused within an owned group: a preconditioner on BiCGSTAB, which implements none on any
    backend (`CFD_ERROR_UNSUPPORTED`); `krylov.restart` outside GMRES; a multigrid preconditioner
    outside scalar and OpenMP CG; a `multigrid` group that breaks the inner cycle's symmetry
    while it serves as a preconditioner; `check_interval` below 1, which reached `iter % 0`.
    Gauss-Seidel overriding `sor.omega` to 1 stays as it was — documented on the enum member and
    pinned by `test_gauss_seidel_is_sor_at_omega_one`.
  - **A caller's `apply_bc` is refused where it would be ignored**: on multigrid, and on every
    GPU solver. Multigrid used to install its own routine into that same public slot, so a
    caller following the documented create/assign/init workflow overwrote multigrid's hook and
    their own boundary condition then had no effect on the solve. Solvers now install their
    walls in a separate `internal_apply_bc`, which makes `apply_bc != NULL` mean "the caller
    prescribed walls" everywhere it is asked.
  - **`poisson_solver_type` is replaced by `poisson_solver_config_t` + `poisson_preset_t`.**
    The old enum was 13 hand-maintained (method × backend) pairs duplicating two existing enums,
    omitting GMRES, BiCGSTAB and the GPU entirely, and unable to express `params` at all. A
    preset now returns an editable config — six intents × any backend × any parameters. Its
    `DEFAULT` is CG, which *is* subject to the incompatible-RHS refusal; the old
    `DEFAULT_POISSON_SOLVER` was Red-Black SOR, which is exempt, so switching preset used to
    change silently whether an RHS was legal. `POISSON_PRESET_SMOOTHER` is the exempt one and
    says so.
  - **`poisson_solve()` returns `cfd_status_t`** and takes a config. Every failure — unknown
    preset, create failure, init rejection, max-iter, divergence, unsolvable RHS — used to
    collapse to `-1`. `poisson_solve_3d`, `poisson_solve_3d_params` and three never-called
    wrappers are deleted; the arity and first-parameter type both change, so an old call site
    is a compile error rather than a silent misinterpretation.
  - **The 13-slot solver cache is deleted** (~210 lines, with its `atexit` and a `memcmp`
    cache key over struct padding). It served one caller: each projection solver now owns its
    pressure solver for its lifetime (`ns_pressure_internal.h`), rebuilding only when the
    configuration or grid actually changes — which also removes three throwaway solvers that
    existed purely to probe whether a grid was 2^k+1.
  - `ns_check_pressure_solver()` joins `ns_check_pressure_bc()` in all nine time-integrator
    inits. `ns_solver_params_t.pressure_solver` was previously validated by no integrator at
    all, and the three AVX2 integrators were skipping the pressure-BC check, so
    `explicit_euler` rejected a pressure BC while `explicit_euler_optimized` accepted and
    ignored it.
  - **The multigrid preconditioner now reads the caller's `multigrid` group.** It used
    to build its inner cycle from hardcoded values and drop the rest on the floor, so
    `pre_smooth`, `post_smooth`, `coarse_max_iter` and `max_levels` were inert for
    PCG-MG — the same silent-ignore in miniature, and one the new ownership table would
    otherwise have blessed by declaring CG the owner of that group. Those four are now
    forwarded and tunable for the first time; `cycle`, `smoother`, `bc` and an unequal
    pre/post sweep count are refused with `CFD_ERROR_INVALID` rather than overridden,
    since a V-cycle with weighted-Jacobi smoothing, Dirichlet coarse corrections and
    equal sweeps is what makes one apply a symmetric operator, and CG minimising over a
    Krylov space depends on that. `test_pcg_mg_reads_the_multigrid_group` pins the
    forwarding by measurement: a V(1,1) inner cycle needs strictly more CG iterations
    than V(4,4) (33 vs 25 at 33², 65 vs 49 at 129²).
  - **A SIMD factory that returns NULL now says why.** `log_no_simd_available` only
    logged at DEBUG and set no error state, so `poisson_solver_create(..., SIMD)` on a
    build without AVX2 handed back NULL with `cfd_get_last_error()` reading `(null)` --
    visible as a bare `CFD_ERROR_UNSUPPORTED` from the AVX2 projection's init on a
    `CFD_ENABLE_AVX2=OFF` build. It now sets an error naming the fix. Its doc comment
    also told callers to "fall back to scalar if needed", which is the cross-backend
    fallback the library forbids.
  - **The GPU projection's compatibility projection stays on the device.** It computed
    the RHS interior mean on the GPU, copied it back, and `cudaStreamSynchronize`d before
    dividing -- one full pipeline drain per inner iteration, in the loop whose pressure
    solve is device-resident specifically to avoid host round-trips. The subtract kernel
    now reads the device-side sum and the interior count and divides itself, and the
    discarded `cudaMemsetAsync` status is checked. Measured A/B on an RTX 4090,
    `GhiaProjectionGpuTest`: 135.7 s before, 129.0-136.1 s after -- no measurable change,
    because `cg_gpu_solve_device` synchronizes for its own dot products anyway and one
    more drain per projection iteration is lost beside them. Kept as a correctness and
    clarity fix, not a speedup.
  - **The Krylov halo refresh no longer opens an OpenMP region per iteration.** A Neumann
    face is a copy from the adjacent interior line -- O(nx + ny), about 130 writes on a
    33x33 plane -- and it runs once per Krylov iteration (twice for BiCGSTAB). An MSVC
    parallel region costs ~12 us to enter at 4 threads, more than the Laplacian sweep it
    accompanied. It is serial on every backend now; all three produced identical values.
    Measured A/B, `GhiaProjectionAvx2Test` run alone: 53.7 s before, 47.6-50.1 s after.
  - **Fixes from review of the above.** `POISSON_PRESET_MULTIGRID_PCG` left `backend` at
    AUTO, which resolves to SIMD wherever AVX2 or NEON is present, and no SIMD CG
    implements the multigrid preconditioner -- so the preset was refused at init on the
    machines it was for. Both multigrid presets now name `POISSON_BACKEND_SCALAR`, and
    `test_every_preset_runs_at_its_own_backend` solves with every preset as shipped.
    A prescribed `walls.front`/`.back` on a 2D grid was accepted and then skipped, a
    silent ignore reached through the validator's own per-face branch; it is now
    `CFD_ERROR_INVALID`. The scalar and OpenMP projections check that the field they were
    handed matches the grid their pressure solver was built for -- the buffers are sized
    from the field and the Poisson kernels sweep to the solver's dimensions, which could
    not diverge while the solve took its size from the field. The AVX2 projection
    re-derives its pressure configuration each step as the other two do, instead of
    reading `params->pressure_bc` fresh for the mean subtraction while solving with a
    solver frozen at init, and releases that solver on the two init failure paths that
    ran after it was built. `ns_pressure_ensure` reports what the factory reported rather
    than calling every NULL an unsupported backend. A refused solve resets the whole
    stats struct, so a reused one cannot show the previous solve's residual beside
    `POISSON_INCOMPATIBLE_RHS`.
  - **Second review pass.** No GPU solver implements a preconditioner -- there is no
    `M^-1` apply anywhere under `linear/gpu/` -- so `POISSON_PRECOND_JACOBI` on GPU CG
    was accepted and then never used; only the multigrid preconditioner had been
    refused there. The AVX2 projection now checks that a pressure solver rebuilt
    mid-run still matches the buffers allocated at init, the guard the scalar and
    OpenMP paths carry: without it, stepping with a larger grid rebuilt the solver for
    it and swept past the end of `p_new`, `rhs` and `u_new`. `ns_pressure_ensure` tests
    the 2^k+1 grid shape itself instead of assuming every `CFD_ERROR_INVALID` from init
    was that rejection -- with the validator in place, INVALID now means a dozen other
    things, and all of them were being reported as "resize your grid".
    `poisson_make_rhs_compatible()` screens degenerate dimensions, which used to wrap
    `nx - 1` to `SIZE_MAX`. The group-emptiness test is field-by-field rather than a
    raw byte scan, which depended on three structs having no padding and on a caller's
    `= {0}` zeroing padding that C11 leaves unspecified. `test_poiseuille_flow` checks
    the init and step statuses it was discarding, so a refused configuration says so
    instead of failing later on profile accuracy. `hold_walls_at_zero` and the halo
    contract it documents live in one header rather than three test files.
  - **Merged with the Helmholtz shift (#216).** `helmholtz_shift` joins `walls` in the
    common block of the regrouped params rather than a method group: it describes the
    operator, not the algorithm. Its support table becomes one more rule in
    `poisson_solver_check_config`, which is where that branch's own "default-deny, one
    central table" reasoning was already heading. Three interactions the merge itself
    could not show: a shift removes the Neumann nullspace, so
    `poisson_solver_krylov_is_singular()` reports a shifted solve as nonsingular and the
    zero-interior-mean rule no longer applies to it; the shift's manufactured-solution
    tests vanish on the boundary and so now request Dirichlet walls explicitly, having
    previously relied on the Krylov halo sitting at zero by accident; and the 2D z-face
    refusal was narrowed, because `poisson_walls_uniform()` sets all six faces and the
    library was rejecting a configuration its own constructor produced. A z-face on a 2D
    grid is refused only when it is the caller's only prescribed face — the case where
    they meant to pin the pressure level and nothing did.
  - `poisson_solver_status_string()` is new: `POISSON_INCOMPATIBLE_RHS` had no name and printed
    as a generic "error" in both examples, each of which hand-rolled its own ternary chain.
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/navier_stokes/ns_pressure_internal.h`,
  `lib/src/solvers/navier_stokes/ns_convection_internal.h`, `lib/src/api/solver_registry.c`,
  `tests/math/test_poisson_config.c`)

### Added

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
- **Multi-Reynolds grid-convergence study with Richardson extrapolation**
  (`test_cavity_richardson.c`, ROADMAP 6.1). The steady cavity is solved on three grids at
  Re = 100 (33/65/129), 400 (65/129/257) and 1000 (129/257/513), and the observed order,
  extrapolated value and GCI are computed per Celik et al. (2008) for u at the centre and the
  centreline extrema. CI runs Re = 100 on 17/33/65. It runs on the OpenMP projection with the
  multigrid pressure solve (`NS_PRESSURE_SOLVER_MULTIGRID`): on a 257x257 Re=1000 cavity that
  is 16 ms/step against 140 ms for the AVX2 CG solve, with the same velocity field to 5e-14.
  The test harness gains `cavity_run_with_pressure_solver_ctx()` to select it. The observed order is about 1.3, not 2:
  see `docs/validation/cavity-grid-convergence.md`.

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
- **Helmholtz shift on the OpenMP CG solver.** `params.helmholtz_shift` is now honoured by
  OpenMP CG as well as scalar CG, through the same guarded `axpy` at the same two call sites,
  so σ = 0 runs exactly the old instructions. It matches scalar to 4e-15.

- **Strain-rate correction to the k-epsilon eddy viscosity** (`NS_NUT_CORRECTION_S_STAR`,
  `ns_solver_params_t.turb_nut_correction`). A two-constant power law in `S* = |S| k / eps`
  -- a `C_mu` that varies with the local strain -- fitted against MKM channel DNS. It cuts the
  channel TKE error from 14.96% to 6.23% at Re_tau 392 where it was fitted, and from 12.80% to
  9.27% at a held-out Re_tau 587, with `u_tau` unmoved. Off by default and bit-identical when
  off; k-epsilon only, refused with `CFD_ERROR_UNSUPPORTED` for any other model. Applied in
  the scalar, OpenMP and AVX2 turbulence steps; GPU turbulence is already refused. Why this,
  and why not an ML pressure guess, is in `docs/technical-notes/ml-integration-design.md`
  (`tests/solvers/turbulence/test_nut_correction.c`).

- **Helmholtz shift in the Poisson solvers** — new
  `poisson_solver_params_t.helmholtz_shift` (sigma; 0 = the existing pure-Poisson path).
  The solved equation becomes `nabla^2 x - sigma*x = rhs`, which is what implicit diffusion
  needs: `(I - nu*dt*nabla^2)u = b` rearranges to sigma = 1/(nu*dt) with rhs = -b/(nu*dt).
  Because `-nabla^2` is already SPD, a positive shift makes the operator strictly
  diagonally dominant, so it is better conditioned than the pressure solve beside it:
  `lambda_min = sigma` exactly and `cond = 1 + 8d` with `d = nu*dt/h^2`. Measured on a
  65x65 mixed-mode problem, CG takes 160 iterations at sigma = 0 and 5 at sigma = 1e6; over
  a 33 -> 129 refinement the unshifted count grows 78 -> 325 while the shifted one moves
  4 -> 6.
  The shift is applied through the existing `axpy` primitive at the call site rather than
  fused into the kernels, so at sigma = 0 the branch is not taken and the unshifted path
  executes the instructions it always did — bit-identity by construction rather than by an
  IEEE argument, and immune to `0.0 * inf`. Verified by memcmp against an untouched
  `params_default()` solve, including a `-0.0` shift.
  Support is default-deny from one central table rather than a check per `*_init`: a
  backend that ignored the shift would silently solve the unshifted equation, which is a
  wrong answer rather than a slow one. Scalar CG implements it (with
  `POISSON_PRECOND_JACOBI`), as does OpenMP CG (see above); every other method,
  backend and the multigrid preconditioner
  return `CFD_ERROR_UNSUPPORTED` at init, and negative or non-finite shifts return
  `CFD_ERROR_INVALID` — a negative shift is the indefinite Helmholtz operator, which is
  not SPD. Verified exact against a discrete manufactured solution to 3e-16 in 2D and
  5e-15 in 3D across sigma from 0 to 1e8, with a sweep asserting all 20 (method, backend)
  pairs behave as the table says
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/linear/linear_solver_internal.h`,
  `lib/src/solvers/linear/cpu/linear_solver_cg.c`, `tests/math/test_helmholtz_shift.c`,
  `tests/solvers/test_linear_solver.c`).
- **`ns_solver_stats_t.dt_used`** reports the time step a solve actually advanced by.
  The explicit Euler solvers clamp their own step to the new public `NS_EULER_DT_LIMIT`
  regardless of `params.dt`, so a caller measuring simulated time could not get it right
  from the parameters alone. `solver_step()` and `solver_solve()` default the field to
  `params.dt` and the Euler wrappers override it, so every solver reports a usable value;
  the clamp itself now has one definition instead of four
  (`lib/include/cfd/solvers/navier_stokes_solver.h`, `lib/src/api/solver_registry.c`,
  the explicit Euler kernels under `lib/src/solvers/navier_stokes/`).
- **Per-face walls on the Poisson solver** — new `poisson_solver_params_t.walls`
  (`poisson_walls_t`; zero-init = all zero-gradient, the operator these solvers have always
  used, so zero-initialization is fully backward compatible). Each face is independently
  `POISSON_WALL_ZERO_GRADIENT` or `POISSON_WALL_DIRICHLET` with a prescribed value, which
  states explicitly what was previously inferred from whether an `apply_bc` hook happened to
  be NULL.
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
  `CFD_ERROR_INVALID`, since both prescribe wall values. A linear field `a*x + b` is
  reproduced to 2e-15 across the supported methods and backends.
  At the NS layer, `ns_solver_params_t.pressure_bc` carries the configuration to the scalar,
  OpenMP and AVX2 projection solvers, following `ns_thermal_bc_config_t`. The projection's
  Neumann compatibility projection is now conditional: with a face prescribed the operator is
  nonsingular and mean-subtracting would change the answer rather than make it exist. The
  field is serialized, so `CFD_CHECKPOINT_FORMAT_VERSION` goes 2 -> 3
  (`lib/include/cfd/solvers/poisson_solver.h`, `lib/src/solvers/linear/linear_solver.c`,
  `lib/include/cfd/solvers/navier_stokes_solver.h`,
  `lib/src/solvers/navier_stokes/{cpu,omp,avx2}/solver_projection*.c`,
  `lib/src/api/solver_registry.c`, `lib/src/io/checkpoint.c`,
  `tests/math/test_poisson_walls.c`, `tests/validation/test_poiseuille_flow.c`).
- **Pressure-driven channels are driven.** `PoiseuilleFlowTest` and `Poiseuille3DTest` now
  prescribe the pressure on the streamwise faces, which is what a pressure-driven channel
  needs: with zero-gradient everywhere the only streamwise forcing is the divergence of the
  boundary velocities, a dipole at the first and last interior column worth half the momentum
  balance, so the profile decays. Measured `dp/dx` goes from -0.891 to **-1.604** against an
  analytical -1.600 (0.23% error), profile RMS from 0.121 to **0.00079**, and the mass-flux
  imbalance from 22.6% to **0.02%**. Note that prescribing the gradient makes
  `test_pressure_gradient` partly a consistency check; the independent content is in the
  profile-RMS and mass-conservation assertions, which only pass if the momentum balance
  sustains the parabola under that pressure difference.

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
- **129×129 lid-driven cavity validation recorded** — the AVX2, OpenMP and CUDA projection
  backends match Ghia et al. (1982) at 129×129 with RMS_u / RMS_v of 0.0017 / 0.0024
  (Re=100), 0.0096 / 0.0328 (Re=400) and 0.0299 / 0.0300 (Re=1000), identical across
  backends to four decimals and across three EC2 workflow runs. Each backend line of
  `test_cavity_backends` now also prints the steps actually run, whether the kinetic-energy
  residual stopped the run early, and that residual
  (`docs/validation/cavity-backends-validation.md`, `tests/validation/test_cavity_backends.c`).
- **Multigrid pressure solve on the OpenMP projection solver** — `projection_omp`
  now accepts `NS_PRESSURE_SOLVER_MULTIGRID` (OpenMP multigrid V-cycles, RHS
  interior mean subtracted for Neumann compatibility) and
  `NS_PRESSURE_SOLVER_PCG_MG` (OpenMP CG preconditioned by an OpenMP multigrid
  V-cycle), gated at init like the scalar `projection`: non-2^k+1 grids return
  `CFD_ERROR_UNSUPPORTED` and degenerate grids `CFD_ERROR_INVALID`. No scalar
  sub-solver runs on either path. OpenMP CG therefore accepts
  `POISSON_PRECOND_MULTIGRID`, also exposed as the new `POISSON_SOLVER_PCG_MG_OMP`
  convenience preset; both CG backends build the preconditioner through one
  shared helper, so they apply the same V(2,2) weighted-Jacobi Dirichlet cycle.
  OpenMP PCG-MG solves match scalar within 1e-9 at 1, 2 and 4 threads (257x257,
  where the multigrid kernels run threaded, and 17^3), and the OpenMP multigrid
  projection matches the scalar one within 1e-10 on 257x257.
  `projection_optimized` and `projection_gpu` still reject the multigrid modes
  (`lib/src/solvers/navier_stokes/omp/solver_projection_omp.c`,
  `lib/src/solvers/linear/omp/linear_solver_cg_omp.c`,
  `lib/src/solvers/linear/linear_solver_internal.h`,
  `lib/src/api/solver_registry.c`,
  `tests/solvers/navier_stokes/cpu/test_projection_pressure_solver.c`,
  `tests/math/test_mg_pcg_convergence.c`, `tests/math/test_omp_consistency.c`).
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
  exceeds `INT_MAX` with `CFD_ERROR_LIMIT_EXCEEDED` at init
  (`lib/src/solvers/linear/multigrid_template/linear_solver_multigrid_template.h`,
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
  `NS_PRESSURE_SOLVER_PCG_MG` (MG-preconditioned CG). Non-2^k+1 grids and the
  `projection_optimized` and `projection_gpu` backends reject the selection with
  `CFD_ERROR_UNSUPPORTED` at init
  (`lib/src/solvers/navier_stokes/cpu/solver_projection.c`,
  `lib/src/api/solver_registry.c`,
  `tests/solvers/navier_stokes/cpu/test_projection_pressure_solver.c`).
- **Multigrid-preconditioned CG** — `POISSON_PRECOND_MULTIGRID` runs one
  symmetric V(2,2) weighted-Jacobi multigrid cycle in Dirichlet mode per
  preconditioner apply (scalar and OpenMP CG; SIMD/GPU CG and GMRES reject it
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
  solver parameters, time, solver name). The parameters include the turbulence model,
  pressure solver, convection scheme and the thermal and turbulence boundary conditions,
  so a restored run keeps them; only the `source_func` / `heat_source_func` callbacks must
  be re-supplied. Little-endian fixed-width encoding with an endianness marker and a
  format-version header that rejects unknown versions
  (`lib/src/io/checkpoint.c`, `lib/include/cfd/io/checkpoint.h`, `tests/io/test_checkpoint.c`).

- **RANS turbulence models: standard k-ε and Spalart-Allmaras with log-law wall
  functions.** The v0.4.0 turbulence milestone (PR #199), mirroring the energy-equation
  module architecture: per-backend kernels, workspace-aware step, in-module BCs. Upwind
  advection, conservative face-averaged diffusion, semi-implicit Patankar sinks for
  positivity, a production limiter and nu_t realizability clipping. Wall functions on
  `BC_TYPE_NOSLIP` faces derive the friction velocity from the log law, set equilibrium
  k/epsilon (or nu_tilde) at the first interior node, and set the wall-face viscosity so
  the discrete wall shear equals u_tau^2 exactly. Eddy viscosity couples into every
  projection/Euler/RK momentum path through a conservative `nu_eff = nu + nu_t`, leaving
  the laminar path bitwise identical. New public API in `turbulence_solver.h`,
  `turb_model`/`turb_bc` in params, and `k`/`eps`/`nu_tilde`/`nu_t` on `flow_field` and in
  VTK/CSV output. Validated against turbulent channel flow at Re_tau = 395 (k-ε u_tau
  error 2.9%, SA 3.1%; u+ within a few percent of the log law), with unit tests for decay,
  wall functions, SA closures, laminar regression and cross-backend consistency, plus a
  `turbulent_channel` example.
  Scalar, OpenMP and AVX2 backends; **2D uniform grids only**, and the GPU paths return
  `CFD_ERROR_UNSUPPORTED` when turbulence is enabled
  (`lib/src/solvers/turbulence/`, `lib/include/cfd/solvers/turbulence_solver.h`,
  `tests/solvers/turbulence/`, `examples/turbulent_channel.c`).

### Fixed

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
  whether the grid was large enough for the GPU before validating their parameters, so a
  turbulence model, upwind convection, a prescribed pressure face or a host callback on a grid
  below the GPU threshold came back as a bare `CFD_ERROR` instead of its refusal. The size
  check now runs after validation, as it already did in `solve_projection_method_gpu`.

- **Cavity validation now measures steady state as a rate, not a per-step change.** The
  harness stopped a run once the relative change in kinetic energy **per step** fell below
  1e-8. That quantity scales with dt, so it registered a slow transient as convergence:
  the Explicit Euler cases ended at ~11,300 of 25,000 steps (t ≈ 1.13) on a flow that needs
  t ≈ 10-20 to develop, and passed their RMS target without a developed solution. The test
  is now `|d(ln KE)/dt| < 1e-6`, in units of 1/time, evaluated after t > 1.0, and computed
  from the step the solver actually took rather than `params.dt`. Running to the real
  budget improves the 33x33 Euler result from RMS_u 0.0957 / RMS_v 0.1284 to 0.0777 /
  0.0334, scalar and OpenMP agreeing to four decimals; the projection results are
  unchanged at 0.0382 / 0.0440. The 129x129 Explicit Euler cases are dropped: they cost
  about an hour of EC2 per run to hold a non-production solver to a relaxed target that
  every projection case clears with over 3x margin, and were never evidence of 129x129
  accuracy (ROADMAP §6.1)
  (`tests/validation/lid_driven_cavity_common.h`, `CMakeLists.txt`,
  `docs/validation/cavity-backends-validation.md`, `ROADMAP.md`).
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
  standalone multigrid already has in `MG_BC_NEUMANN` mode. The projection solvers
  subtract the interior mean of `div(u*)` on every preset; callers who relied on a uniform
  RHS converging should either call `poisson_make_rhs_compatible()` or prescribe a face
  through `params.walls`, which makes the operator nonsingular and admits any RHS. (Both
  are better than the `apply_bc` hook an earlier draft of this entry suggested: walls and
  a hook together are now refused, since both prescribe wall values.)
  The 3D projection goldens were re-pinned: `L2(p)` now stays ~1.0, since a constant is in
  the nullspace and the iteration keeps every direction mean-free, so the caller's pressure
  level is carried rather than driven to zero walls; `L2(u)`/`L2(v)` move by ~3e-7 relative
  (`lib/src/solvers/linear/linear_solver.c`,
  `lib/src/solvers/linear/linear_solver_internal.h`, the six CPU Krylov backends and the
  two GPU ones, `lib/src/solvers/navier_stokes/{cpu,omp,avx2}/solver_projection*.c`,
  `tests/math/test_krylov_warm_start.c`, `tests/math/test_bicgstab.c`,
  `tests/math/test_gmres.c`, `tests/solvers/test_linear_solver.c`,
  `tests/solvers/navier_stokes/cpu/test_ns_solver_3d.c`).
  An incompatible RHS is now refused up front with `CFD_ERROR_INVALID` and
  `stats.status = POISSON_INCOMPATIBLE_RHS`, instead of being iterated on: such a system
  has no solution, and iterating either stalls on the component in the nullspace or drives
  the rest of the field away chasing it. `test_laplacian_accuracy` had been reaching a
  residual of 1e21 that way. `poisson_make_rhs_compatible()` is the supported way to
  comply, and is exported — the helper the docs previously pointed at was not.
- **`explicit_euler_optimized` updates every interior column.** The AVX2 row loop processed
  4-wide groups with no scalar remainder, so when `(nx-2) % 4 != 0` the last 1-3 interior
  columns of each row kept their old values (3 per row at 33×33 and 129×129). The remainder
  now runs through the solver's scalar row path, and a 19×19 AVX2-vs-scalar run agrees to
  1e-17. The 129×129 Re=100 AVX2 Euler case now matches OpenMP (11,775 steps, RMS_v 0.1277
  instead of 0.1293)
  (`lib/src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c`,
  `tests/solvers/navier_stokes/avx2/test_solver_explicit_euler_avx2.c`,
  `docs/validation/cavity-backends-validation.md`).
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
  spacing or a failed energy, thermal-BC or turbulence step went unreported and the
  simulation kept stepping. The wrappers now return the status before filling in stats,
  as the projection wrappers do. A new test seeds a NaN pressure value and checks that
  both calls return `CFD_ERROR_DIVERGED` on each backend. The OpenMP implementation also
  frees its turbulence workspace when the energy step fails, which it previously leaked
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
- `scripts/ec2-validate.sh` runs every `CavityBackend_*` ctest entry. It ran the test binary
  without a filter, which skips the Re=400 and Re=1000 cases, and under `set -e` a failing run
  ended the script before its FAILED summary.
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
- The SIMD SOR solvers (AVX2 and NEON) ran a Block SOR that read the left neighbour inside each
  SIMD block from the previous sweep. That is not the SOR iteration, and on the AVX2 build it
  diverged for omega between 1.40 and 1.50 on every grid measured, below the automatic omega: a
  33x33 solve at the automatic omega reported convergence after 1,764 sweeps with a residual of
  exactly zero. Each row is now swept in two passes, the stencil terms the sweep does not write
  with SIMD and then the relaxation in order, which is scalar SOR sweep for sweep
  (`lib/src/solvers/linear/avx2/linear_solver_sor_avx2.c`,
  `lib/src/solvers/linear/neon/linear_solver_sor_neon.c`, `tests/solvers/test_linear_solver.c`,
  `docs/technical-notes/block-sor-simd.md`).
- `examples/poisson_solver_tuning.c` benchmarked SOR and Red-Black SOR at a hard-coded omega of
  1.5 rather than the automatic value, and capped the iterations it printed for an off-by-one that
  is fixed. The `poisson_solver.h` usage example called `poisson_solver_init()` without `nz` and
  `dz`, and `max_iterations` was documented as defaulting to 1000 where the default is 5000.
- `POISSON_METHOD_GAUSS_SEIDEL` created the SOR solvers and ran at SOR's automatic omega, not at
  1: at 9aa06d1 a 33x33 zero-gradient solve took 380 sweeps where omega = 1 takes 2,376. It now
  always relaxes with omega = 1, whatever `params.omega` says
  (`lib/src/solvers/linear/linear_solver.c`, `lib/src/solvers/linear/linear_solver_internal.h`,
  `tests/solvers/test_linear_solver.c`).

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
