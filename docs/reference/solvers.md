# Solvers & Numerical Methods

Comprehensive guide to solvers and numerical methods in the CFD Framework.

## Overview

The CFD Framework implements multiple solvers for the incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
∇·u = 0
```

All solvers support both 2D (`nz=1`) and 3D (`nz>1`) grids using a branch-free approach: `stride_z=0` when `nz==1` causes z-terms to vanish, producing bit-identical 2D results with zero overhead.

## Solver Families

### 1. Explicit Euler Solvers

Simple forward Euler time integration using finite differences.

**Equation:**
```
u^(n+1) = u^n + dt * [-(u·∇)u + ν∇²u - ∇p/ρ]
```

**Characteristics:**
- First-order accurate in time
- Simple and fast
- Does not enforce incompressibility strictly
- Good for learning and debugging

**Stability:** CFL condition must be satisfied:
```
dt ≤ min(dx²/(4ν), dx/u_max)
```

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `explicit_euler` | Scalar | Basic implementation |
| `explicit_euler_optimized` | SIMD | AVX2 (requires `-DCFD_ENABLE_AVX2=ON`; no NEON kernels) |
| `explicit_euler_omp` | OpenMP | Multi-threaded |
| `explicit_euler_gpu` | CUDA | GPU-accelerated |

### 2. Projection Method Solvers

Chorin's projection method - properly enforces incompressibility constraint.

**Algorithm (Fractional Step Method):**

1. **Predictor Step** - Compute intermediate velocity (ignore pressure):
   ```
   u* = u^n + dt * [-(u·∇)u + ν∇²u]
   ```

2. **Pressure Poisson Solve** - Enforce incompressibility:
   ```
   ∇²p^(n+1) = (ρ/dt) * ∇·u*
   ```

3. **Corrector Step** - Project velocity to divergence-free space:
   ```
   u^(n+1) = u* - (dt/ρ) * ∇p^(n+1)
   ```

**Characteristics:**
- Second-order accurate in space (central differences; first-order with
  [upwind convection](#convection-scheme))
- First-order accurate in time
- Viscous term explicit by default, or implicit (backward Euler / Crank–Nicolson)
  on `projection` and `projection_omp` — see
  [Viscous Time Discretization](#viscous-time-discretization)
- Properly enforces ∇·u = 0
- More expensive due to Poisson solve

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `projection` | Scalar | Basic implementation |
| `projection_optimized` | SIMD | AVX2 + OpenMP (requires `-DCFD_ENABLE_AVX2=ON`; no NEON kernels) |
| `projection_omp` | OpenMP | Multi-threaded |
| `projection_gpu` | GPU | CUDA-accelerated (CG pressure solve) |

**Pressure solver selection** (`ns_solver_params_t.pressure_solver`):

Each backend pairs with a CG Poisson preset by default. On the scalar
`projection` and OpenMP `projection_omp` solvers the pressure solve can be
switched to geometric multigrid, on the solver's own backend:

Each projection solver owns one `poisson_solver_t` for its lifetime, built at
init from the preset below on the projection's own backend — never on another,
which would be a silent cross-backend fallback.

| `pressure_solver` value | preset | `projection` | `projection_omp` |
|-------------------------|--------|--------------|------------------|
| `NS_PRESSURE_SOLVER_DEFAULT` (0) | `POISSON_PRESET_DEFAULT` | CG, scalar | CG, OpenMP |
| `NS_PRESSURE_SOLVER_MULTIGRID` | `POISSON_PRESET_MULTIGRID` | MG, scalar | MG, OpenMP |
| `NS_PRESSURE_SOLVER_PCG_MG` | `POISSON_PRESET_MULTIGRID_PCG` | PCG-MG, scalar | PCG-MG, OpenMP |

`NS_PRESSURE_SOLVER_MULTIGRID` runs multigrid V-cycles (the RHS interior mean is
subtracted first, for Neumann compatibility); `NS_PRESSURE_SOLVER_PCG_MG` runs CG
preconditioned by one multigrid V-cycle. The MG modes require 2^k+1 grid points
per active dimension (33, 65, 129, ...); `solver_init` returns
`CFD_ERROR_UNSUPPORTED` otherwise. `projection_optimized` and `projection_gpu`
reject any non-default value with `CFD_ERROR_UNSUPPORTED` at init — they do not
wire a multigrid pressure solve, and the library never falls back across
backends silently.

```c
ns_solver_params_t params = ns_solver_params_default();
params.pressure_solver = NS_PRESSURE_SOLVER_MULTIGRID;  // 2^k+1 grids only
ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
cfd_status_t status = solver_init(slv, grid, &params);   // UNSUPPORTED on 128x128
```

## Convection Scheme

Every Navier-Stokes solver discretizes the momentum convection `(u·∇)u` and the
temperature advection `u·∇T` with O(h²) central differences by default. When
convection dominates (cell Péclet number `Pe = |u|h/ν > 2`), central differencing
produces wiggles: the solution over- and undershoots around sharp gradients, and
without enough viscosity the oscillations grow.

`ns_solver_params_t.convection_scheme` selects first-order upwind differencing
instead:

| `convection_scheme` value | Convective derivative ∂f/∂x | Accuracy |
|---------------------------|-----------------------------|----------|
| `NS_CONVECTION_SCHEME_CENTRAL` (0) | `(f[i+1] - f[i-1]) / 2h` | O(h²) |
| `NS_CONVECTION_SCHEME_UPWIND` | `(f[i] - f[i-1]) / h` when the local velocity is ≥ 0, else `(f[i+1] - f[i]) / h` | O(h) |

Upwind adds numerical diffusion `|u|h/2`. At CFL ≤ 1 an advected profile stays
within its initial bounds, but gradients smear and grid refinement only converges
at first order. Use it for coarse grids, high-Re startup transients and
convection-dominated flows where central differencing oscillates. Only the
convective first derivatives change: pressure gradients, viscous terms and the
divergence stay central. Turbulence transport (k, ε, ν̃) is always upwind,
independent of this setting.

Explicit time steps must still satisfy both the convective CFL limit and the
diffusion limit; the numerical diffusion of upwind relaxes neither.

| Backend | Upwind convection |
|---------|-------------------|
| Scalar (`explicit_euler`, `projection`, `rk2`, `rk4`) | Yes |
| OpenMP (`*_omp`) | Yes |
| AVX2 (`*_optimized`) | Yes (blend-mask vectorized) |
| CUDA (`*_gpu`) | No: `CFD_ERROR_UNSUPPORTED` at init and at step |

The `*_optimized` solvers require a build configured with `-DCFD_ENABLE_AVX2=ON`
(off by default) **and** a CPU that supports AVX2. Where either is missing they
return `CFD_ERROR_UNSUPPORTED` at init rather than running scalar kernels -- a
silent fallback would turn a configuration mistake into a performance mystery.
Use the scalar names (`explicit_euler`, `projection`, `rk2`, `rk4`) for a
guaranteed scalar path, and `cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)`
to test for the SIMD backend before selecting it.

`projection_optimized` is stricter still: its SIMD Poisson sub-solver is
threaded, so it also needs OpenMP and reports `CFD_ERROR_UNSUPPORTED` without it
even when the other three `*_optimized` solvers are available.

Convection-scheme values other than the two above are rejected with
`CFD_ERROR_INVALID` at init.

```c
ns_solver_params_t params = ns_solver_params_default();
params.convection_scheme = NS_CONVECTION_SCHEME_UPWIND;
ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION_OMP);
cfd_status_t status = solver_init(slv, grid, &params);  // UNSUPPORTED on projection_gpu
```

The upwind derivative itself is `stencil_upwind_deriv_x/y/z()` in
`cfd/math/stencils.h`.

## Viscous Time Discretization

Every solver advances the viscous term explicitly by default, so the time step is
bounded by the diffusion limit `dt < h² / (2ν·ndim)`. That limit scales as h²,
against the convective limit's h, so it binds at low Reynolds number and on fine
grids. `ns_solver_params_t.viscous_scheme` lets the projection solver advance the
viscous term implicitly instead:

| `viscous_scheme` value | θ | Viscous term | Stability |
|------------------------|---|--------------|-----------|
| `NS_VISCOUS_SCHEME_EXPLICIT` (0) | 0 | O(dt) | `dt < h²/(2ν·ndim)` |
| `NS_VISCOUS_SCHEME_BACKWARD_EULER` | 1 | O(dt) | unconditional, L-stable |
| `NS_VISCOUS_SCHEME_CRANK_NICOLSON` | ½ | O(dt²) | unconditional, A-stable only |

The predictor's explicit increment `b = dt·(−(u·∇)u + f + ν∇²uⁿ)` becomes the
right-hand side of

```
(I − θ·ν·dt·∇²) δ = b,    u* = uⁿ + δ,    δ = 0 on every boundary node
```

solved once per velocity component per step as the shifted Poisson problem
`∇²δ − σδ = −σb` with `σ = 1/(θ·ν·dt)` (the Poisson solvers'
`helmholtz_shift`), by CG at relative tolerance 1e-10. The shift makes the
operator better conditioned than the pressure Poisson operator, so this costs
a handful of CG iterations and barely grows with the grid. δ = 0 on the
boundary is not a new assumption: the predictor already holds boundary values at
their step-start values for the whole step, and the caller's boundary conditions
apply after it.

**What it buys is stability, not a higher overall order.** Convection stays
forward Euler and the projection is Chorin's non-incremental splitting, so a
full step is O(dt) under either scheme. Crank–Nicolson makes the viscous part
second order, which is visible where diffusion dominates, but as `ν·dt·|λ|`
grows its amplification tends to −1: stiff, high-wavenumber modes flip sign each
step instead of decaying. Backward Euler damps them, which makes it the right
choice for marching to a steady state with a large dt; prefer Crank–Nicolson for
transients at a dt near the explicit limit.

`compute_time_step()` leaves the diffusion limit out when the scheme is
implicit. The convective CFL limit still applies, and so does the thermal
diffusion limit when the energy equation is active, because temperature is
still advanced explicitly.

| Solver | Implicit viscous |
|--------|------------------|
| `projection` | Yes (scalar CG) |
| `projection_omp` | Yes (OpenMP CG) |
| All others | No: `CFD_ERROR_UNSUPPORTED` at init and at step |

The two supporting solvers advertise `NS_SOLVER_CAP_IMPLICIT_VISCOUS`, which
`solver_init`, `solver_step` and `solver_solve` check. An implicit scheme
combined with a turbulence model is rejected with `CFD_ERROR_UNSUPPORTED`,
because `ν + ν_t` varies in space and the implicit operator has constant
coefficients. Values outside the enum are `CFD_ERROR_INVALID` on every solver.

```c
ns_solver_params_t params = ns_solver_params_default();
params.viscous_scheme = NS_VISCOUS_SCHEME_BACKWARD_EULER;
params.dt = 20.0 * h * h / (4.0 * params.mu);  // 20x the 2D explicit limit
ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
cfd_status_t status = solver_init(slv, grid, &params);  // UNSUPPORTED on rk4
```

Verified in `tests/solvers/navier_stokes/test_viscous_scheme.c`. On a discrete
Laplacian eigenmode, one step reproduces the θ-method amplification
`(1 + (1−θ)ν·dt·λ)/(1 − θ·ν·dt·λ)` to 1e-15. Temporal order measures 2.00 for
Crank–Nicolson and 0.98 for backward Euler, at time steps all above the explicit
limit. A viscous-bound cavity at 20× the explicit limit stays bounded by the lid
speed, while the explicit scheme saturates at the velocity clamp.

## Linear Solvers (Poisson Equation)

The projection method requires solving the pressure Poisson equation:
```
∇²p = f
```

### Available Methods

#### 1. Jacobi Iteration

**Algorithm:**
```
p_ij^(k+1) = (p_i-1,j + p_i+1,j + p_i,j-1 + p_i,j+1 - h²f_ij) / 4
```

**Characteristics:**
- Simple, parallel-friendly
- Slow convergence: O(n²) iterations for n×n grid
- Unconditionally stable
- Good for GPU/SIMD

**Convergence Rate:** ρ ≈ 1 - π²/(2n²)

**Usage:**
```c
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_JACOBI,
                                                 POISSON_BACKEND_SIMD);
```

#### 2. Successive Over-Relaxation (SOR)

**Algorithm:**
```
p_ij^(k+1) = (1-ω)p_ij^k + (ω/4)(p_i-1,j + p_i+1,j + p_i,j-1 + p_i,j+1 - h²f_ij)
```

**Relaxation factor:** `params.sor.omega = 0` (the default) chooses ω for the grid and the walls:
the optimum itself for walls set by `apply_bc`, and an estimate at or just below it for the default
zero-gradient walls. Any `omega > 0` is used as given. `POISSON_METHOD_GAUSS_SEIDEL` creates the
same solvers and always relaxes with ω = 1, whatever `omega` says — the one documented case where a
field a method owns is deliberately overridden rather than refused.

- **Default zero-gradient walls.** The walls are copied from the interior after each sweep, so
  during a sweep a point next to a wall reads its own previous value through the copy. Every SOR
  and Red-Black SOR backend therefore relaxes such a point by ω·F/(F − W), where
  F = 2(1/dx² + 1/dy² + 1/dz²) and W is the summed weight of its wall neighbours (1/dx² for a wall
  in x). On a square 2D grid that is 4/3·ω beside an edge and 2·ω in a corner. It makes the
  iteration SOR on the Neumann matrix itself, so
  Young's ω_opt = 2/(1 + √(1 − ρ_J²)) applies, and the automatic ω takes ρ_J from a Rayleigh
  quotient of the slowest mode against that matrix (`poisson_solver_compute_neumann_omega` in
  `lib/src/solvers/linear/linear_solver_internal.h`). It never exceeds the optimum. A grid with
  only two interior points has no slow mode, just the alternating one that ω = 1 removes in a
  sweep, so there the automatic ω is 1. The converged solution is the same as without the scaling.
- **Walls set by the caller's `apply_bc`**, installed before `poisson_solver_init()`, which
  chooses ω. There is no wall scaling, and the automatic ω is the Dirichlet optimum:

  ```
  ρ_J = [cos(π/(nx−1))/dx² + cos(π/(ny−1))/dy² + cos(π/(nz−1))/dz²] / [1/dx² + 1/dy² + 1/dz²]
  ω_opt = 2 / (1 + √(1 − ρ_J²))          which is 2 / (1 + sin(πh)) on a square grid
  ```

  with the z terms only in 3D. The CUDA SOR and Red-Black SOR solvers apply the walls on the device,
  so their init rejects a custom `apply_bc` with `CFD_ERROR_UNSUPPORTED`.

**Characteristics:**
- Faster than Jacobi (ω > 1)
- Sequential row updates (row j depends on j-1)
- Optimal ω depends on the grid and the walls (see above)
- SIMD variant sweeps each row in two passes: the stencil terms that do not depend on the sweep, SIMD_WIDTH cells at a time, then the relaxation in order against the updated left neighbor. It is the scalar iteration, sweep for sweep (see [SIMD SOR technical note](../technical-notes/block-sor-simd.md))
- GPU variant uses Block SOR: each thread sweeps an 8×8 tile sequentially (Gauss-Seidel inside the tile), with red-black *tile* coloring (red pass then black pass, two launches per iteration) so a tile's halo is never written by another tile in the same pass — the update is in-place, race-free, and provably convergent for 0<ω<2

**Convergence Rate:** ρ ≈ 1 - 2πh (with optimal ω)

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `sor_scalar` | Scalar | Sequential Gauss-Seidel + SOR relaxation |
| `sor_simd` | SIMD | SOR with SIMD stencil terms (auto-detects AVX2/NEON) |
| `sor_gpu` | GPU | Block SOR (CUDA; per-thread tile sweep, red-black tile coloring, in-place) |

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.sor.omega = 0.0;  // 0 = automatic, at or just below the optimum; > 0 overrides

// Scalar (fully sequential, best convergence per iteration)
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_SOR,
                                                 POISSON_BACKEND_SCALAR);

// SIMD (the scalar iteration, with the stencil terms vectorized)
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_SOR,
                                                 POISSON_BACKEND_SIMD);
```

#### 3. Red-Black SOR

**Algorithm:**
- Split grid into red and black points (checkerboard)
- Update all red points (can parallelize)
- Update all black points (can parallelize)

**Characteristics:**
- Same convergence as SOR
- Parallelizable (red/black decoupled)
- SIMD and GPU friendly
- Preferred over standard SOR for performance

**Usage:**
```c
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_REDBLACK_SOR,
                                                 POISSON_BACKEND_SIMD);
```

#### 4. Conjugate Gradient (CG)

**Algorithm:** Krylov subspace method for symmetric positive definite systems.

**Characteristics:**
- Theoretically converges in n iterations
- Practical convergence: O(√κ) where κ = condition number
- Memory efficient (no matrix storage)
- Best for large grids

**Convergence:** For Poisson equation, κ ≈ 4(n-1)²/π², giving O(n) iterations.

**Usage:**
```c
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                 POISSON_BACKEND_SIMD);
```

**What the Krylov solvers invert.** CG, BiCGSTAB and GMRES update interior points
only, so the operator they invert is whatever their vectors' halos say it is. Each
applies the **homogeneous** part of the boundary condition to its search directions
before every operator apply, and the **full** condition to the iterate before the
initial residual — an inhomogeneous condition on a direction would make the
operator affine rather than linear, which the Krylov recurrences do not describe.

Residual and operator therefore agree, so **warm-starting is safe**: a solve begun
from the previous step's field converges to the field a cold start reaches, and gets
there in far fewer iterations. (Before v0.3.0 the residual was built from the
zero-gradient extension while the directions carried a zero halo, so any non-zero
initial guess converged to a field solving neither system.)

**Per-face walls.** `params.walls` states each face independently:

```c
poisson_solver_params_t params = poisson_solver_params_default();
params.walls.left  = POISSON_WALL_DIRICHLET;   /* inlet  */
params.walls.right = POISSON_WALL_DIRICHLET;   /* outlet */
params.walls.values.left  = 0.0;
params.walls.values.right = -6.4;              /* the rest stay zero-gradient */
```

Zero-initialisation is all-zero-gradient, the operator these solvers have always
used. `poisson_walls_uniform(POISSON_WALL_DIRICHLET, 0.0)` gives the whole-domain
homogeneous Dirichlet operator in one call.

Per-face walls are honoured by **CG, BiCGSTAB and GMRES** on the scalar, OpenMP and
SIMD backends. The stationary (Jacobi, SOR, Red-Black SOR) and multigrid solvers
apply whole-domain walls inside their sweeps, and the GPU backend applies them on
the device, so all of those reject a non-default `params.walls` at init with
`CFD_ERROR_UNSUPPORTED` rather than silently solving a different problem.

**The all-zero-gradient operator is singular** — the constants are its nullspace —
so, exactly as for standalone multigrid in `MG_BC_NEUMANN` mode, the RHS must have
zero interior mean:

```c
poisson_make_rhs_compatible(rhs, nx, ny, nz);   /* what the projection solvers do */
```

An incompatible RHS describes a system with **no solution**. The solve refuses it
up front, returning `CFD_ERROR_INVALID` with `stats.status = POISSON_INCOMPATIBLE_RHS`,
rather than iterating on it — the solvers do not quietly project the RHS onto the
compatible subspace, because that answers a different question than the one asked.
The solution also carries a free additive constant; the iteration keeps every
direction mean-free, so the level is whatever the initial guess had.

**Prescribing any face removes all of that**: the operator becomes nonsingular, any
RHS is admissible, and the level is pinned. Do **not** mean-subtract then — it would
change the answer rather than make it exist.

An `apply_bc` hook remains the way to prescribe **spatially varying** wall values.
It supplies the lift while the homogeneous part stays a zero halo, which is the
standard way to handle inhomogeneous Dirichlet walls. A hook and a non-default
`params.walls` both prescribe wall values, so installing both is rejected with
`CFD_ERROR_INVALID`. The stationary and multigrid solvers re-apply a hook on every
sweep and honour either kind.

#### 5. Preconditioned CG (PCG)

**Algorithm:** CG with preconditioner M to improve conditioning.

**Available Preconditioners:**
- **Jacobi (Diagonal)**: M = diag(A)
  - Simple, cheap per iteration
  - Variable coefficients: can reduce iterations
  - Constant coefficients (uniform grid): no benefit
- **Multigrid** (`POISSON_PRECOND_MULTIGRID`): one geometric-multigrid V-cycle
  per apply
  - Grid-size-independent outer iteration count (measured 5 CG iterations at
    33²–129², tol 1e-8, vs 50–170 for plain CG)
  - Scalar and OpenMP CG backends (OpenMP CG runs the cycle on the OpenMP
    multigrid backend); SIMD and GPU CG and GMRES reject it with
    `CFD_ERROR_UNSUPPORTED`
  - Requires 2^k+1 grid points per active dimension (inherited from the
    multigrid hierarchy); init fails with `CFD_ERROR_INVALID` otherwise
  - The inner cycle is a symmetric V(2,2) with weighted-Jacobi smoothing in
    Dirichlet mode, so the preconditioner is SPD as CG requires
- **SSOR**: M = (D + L)D⁻¹(D + U) (future)
- **ILU**: Incomplete LU factorization (future)

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.krylov.preconditioner = POISSON_PRECOND_JACOBI;  // Enable preconditioning

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                 POISSON_BACKEND_SIMD);
poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
```

The `krylov` group belongs to CG, BiCGSTAB and GMRES. Setting it on a
stationary method is refused at init with `CFD_ERROR_INVALID` rather than
ignored — and BiCGSTAB, which does read the group, refuses a preconditioner
specifically with `CFD_ERROR_UNSUPPORTED`, because it implements none on any
backend.

**Multigrid-preconditioned CG (scalar or OpenMP backend, 2^k+1 dims):**

```c
poisson_solver_params_t params = poisson_solver_params_default();
params.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
// With a multigrid preconditioner selected, CG also reads params.multigrid.
// pre_smooth/post_smooth (equal), coarse_max_iter and max_levels shape the
// inner cycle and are yours. The cycle type, the smoother and the boundary
// mode are what make one apply a symmetric operator, so init refuses those
// with CFD_ERROR_INVALID rather than overriding them.
params.multigrid.pre_smooth = 3;   // V(3,3) inner cycle: a stronger
params.multigrid.post_smooth = 3;  // preconditioner, fewer CG iterations

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                 POISSON_BACKEND_SCALAR);  // or POISSON_BACKEND_OMP
poisson_solver_init(solver, 65, 65, 1, dx, dy, 0.0, &params);
```
The convenience API exposes the same configuration as the
`POISSON_PRESET_MULTIGRID_PCG` preset, on whichever backend `cfg.backend`
names.

#### 6. BiCGSTAB

**Algorithm:** Bi-Conjugate Gradient Stabilized for non-symmetric systems.

**Characteristics:**
- Handles non-symmetric matrices
- More robust than CG for difficult problems
- Higher cost per iteration than CG

**Backends:** scalar, SIMD (AVX2/NEON), and CUDA GPU (`POISSON_BACKEND_GPU`).

**Usage:**
```c
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_BICGSTAB,
                                                 POISSON_BACKEND_SCALAR);
```

#### 7. GMRES(m)

**Algorithm:** Restarted Generalized Minimal Residual for non-symmetric systems.
Builds an orthonormal Krylov basis via Arnoldi + modified Gram-Schmidt and
minimizes the residual over that basis using incremental Givens rotations,
restarting every `m` inner iterations to bound memory.

**Characteristics:**
- Handles non-symmetric matrices; minimizes the residual norm at every step
- Restart length `m` set via `params.krylov.restart` (0 = auto, default 30); bounds the
  Krylov basis to `m+1` grid-sized vectors
- Right-preconditioned Jacobi seam (`POISSON_PRECOND_JACOBI`); like PCG, Jacobi
  preconditioning gives no benefit on a uniform grid (constant diagonal)
- On the symmetric Poisson operator it converges to the same solution as CG (a
  useful independent cross-check); its real value is future non-symmetric systems

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `gmres_scalar` | Scalar | Reference implementation |
| `gmres_simd` | SIMD | AVX2/NEON vector primitives + OpenMP (auto-detects AVX2/NEON) |
| `gmres_omp` | OpenMP | OpenMP-parallelized vector primitives |

All three backends include one implementation of the algorithm
(`lib/src/solvers/linear/gmres_template/linear_solver_gmres_template.h`) and differ
only in their O(n) vector primitives; the dense Hessenberg/Givens work is serial
everywhere. Parallel dot products split their sums into per-thread partial sums,
and OpenMP leaves the order in which those are combined unspecified, so
`gmres_simd` and `gmres_omp` agree with `gmres_scalar` to rounding rather than
bit-for-bit, and their last bits can differ with the thread count and between runs
(a single-threaded 2D `gmres_omp` solve matches `gmres_scalar` exactly). A CUDA GPU
variant is not yet implemented.

`poisson_solver_init` rejects restart lengths whose `(m+1)·m` Hessenberg matrix
cannot be indexed with `int` (m ≥ 46341), and grids too large to index (`nx` or
`ny` above `INT_MAX`, or an `(m+1)`-vector Krylov basis overflowing `size_t`
bytes), with `CFD_ERROR_LIMIT_EXCEEDED`.

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.krylov.restart = 30;  // GMRES(30); 0 selects the default

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_GMRES,
                                                 POISSON_BACKEND_SCALAR);
poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &params);
```

`restart` is the GMRES basis size and no other method reads it, so setting it
on CG or BiCGSTAB is refused with `CFD_ERROR_INVALID`.

#### 8. Geometric Multigrid

**Algorithm:** V-cycle (default), W-cycle, or F-cycle (full multigrid) on a
hierarchy of coarsened grids. Each cycle pre-smooths, restricts the residual
with full weighting, recursively solves the coarse correction equation,
prolongates with bilinear (2D) / trilinear (3D) interpolation, and post-smooths.
Smoothers: Red-Black Gauss-Seidel (default) or weighted Jacobi (ω=2/3).

**Characteristics:**
- **Grid-size-independent convergence**: ~0.05–0.1 residual reduction per
  V(2,2) cycle regardless of resolution — O(N) total work, the optimal
  complexity for structured-grid Poisson problems
- **Grid constraint**: every active dimension must have 2^k+1 points
  (5, 9, 17, 33, 65, 129, ...); other sizes return `CFD_ERROR_INVALID`
- Two BC modes via `params.multigrid.bc`:
  - `MG_BC_NEUMANN` (default) — zero-gradient BCs matching the other Poisson
    solvers. The system is singular (solution defined up to a constant; RHS
    should have zero interior mean). Restriction uses Neumann-folded boundary
    weights and coarse RHS/corrections are mean-projected internally.
  - `MG_BC_DIRICHLET` — caller-supplied boundary values of `x` are held fixed
    (supports inhomogeneous data); coarse corrections use zero boundaries.
- `MG_CYCLE_F` runs one full-multigrid pass (coarsest-first nested iteration)
  on the first cycle — reaching discretization accuracy immediately — then
  continues with V-cycles
- Parameters, all under `params.multigrid`: `cycle`, `smoother`, `bc`,
  `pre_smooth`/`post_smooth` (default 2/2), `coarse_max_iter` (default 50),
  `max_levels` (0 = auto)

**Backends:** scalar (`multigrid_scalar`) and OpenMP (`POISSON_BACKEND_OMP`,
`multigrid_omp`). `POISSON_PRESET_MULTIGRID` names `POISSON_BACKEND_SCALAR`
rather than leaving AUTO, which would resolve to SIMD — a backend multigrid
lacks — so set `cfg.backend = POISSON_BACKEND_OMP` for the threaded one.
Both share one algorithm template
(`lib/src/solvers/linear/multigrid_template/linear_solver_multigrid_template.h`);
the OpenMP backend parallelizes the smoother, residual, grid-transfer and
boundary kernels over rows with no parallel reductions (the Neumann interior mean
and the convergence residual stay serial), so its results are bit-identical to
the scalar backend at any thread count. Each kernel uses the thread team only when
its loop covers at least 32,768 points of a plane: the interior points of the plane
it writes, or the 2(nx+ny) copies of the Neumann boundary pass. Smaller loops (the
coarser levels, small grids, and the boundary pass unless nx+ny reaches 16,384) run
on the calling thread, where starting a thread team would cost more than the work.
`POISSON_BACKEND_AUTO` resolves to scalar; request `POISSON_BACKEND_OMP` explicitly.
Grids whose `nx` or `ny` exceeds `INT_MAX` return `CFD_ERROR_LIMIT_EXCEEDED` at init.
Also available as a CG
preconditioner (`POISSON_PRECOND_MULTIGRID`, scalar and OpenMP CG — see §5) and as
the pressure solver of the scalar and OpenMP projection methods
(`ns_solver_params_t.pressure_solver`). SIMD/GPU variants are planned follow-ups.

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.multigrid.cycle = MG_CYCLE_V;  // or MG_CYCLE_W / MG_CYCLE_F
params.multigrid.bc = MG_BC_NEUMANN;  // default; matches other solvers

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                                 POISSON_BACKEND_SCALAR);  // or POISSON_BACKEND_OMP
poisson_solver_init(solver, 65, 65, 1, dx, dy, 0.0, &params);  // dims 2^k+1
```

### Linear Solver Performance Comparison

**Problem:** 65×65 grid, tolerance = 1e-6

| Method | Iterations | Time (ms) | Notes |
|--------|-----------|-----------|-------|
| Jacobi | ~8000 | 45 | Simple, slow |
| SOR (automatic ω) | 268 | — | Good serial performance; seeded-noise RHS, default walls |
| Red-Black SOR (automatic ω) | 235 | — | Parallelizable; seeded-noise RHS, default walls |
| CG | ~80 | 5 | Best for large grids |
| PCG (Jacobi) | ~80 | 5.5 | No benefit on uniform grid |
| PCG (Multigrid) | ~5 | — | Grid-size-independent; needs 2^k+1 dims |
| BiCGSTAB | ~40 | 4 | Fastest convergence |
| Multigrid V(2,2) | ~8 cycles | — | O(N), grid-size-independent; needs 2^k+1 dims |

**Note:** Jacobi preconditioning provides no benefit on uniform grids with constant coefficients (diagonal is constant 4/h²).

## Backend Performance

### CPU Scalar

Basic C implementation:
- No SIMD, no parallelization
- Portable to all platforms
- Good baseline for correctness

**Use when:**
- Debugging
- Small grids (<50×50)
- Platform doesn't support SIMD

### SIMD (AVX2/NEON)

Vectorized implementation using intrinsics:

**AVX2 (x86-64):**
- 4 double operations per instruction (256-bit)
- Fused multiply-add (FMA) for performance
- 2-3x speedup over scalar

**NEON (ARM64):**
- 2 double operations per instruction (128-bit)
- ARM64 architecture optimization
- 1.5-2x speedup over scalar

**Features:**
- Aligned memory allocation
- Vectorized inner loops
- Scalar remainder handling

**Use when:**
- Medium to large grids (>100×100)
- Modern CPU (Intel Haswell+ or ARM Cortex-A57+)
- Single-threaded workload

**Example:**
```c
// AVX2 inner loop (simplified)
for (size_t i = 0; i < n; i += 4) {
    __m256d u_vec = _mm256_load_pd(&u[i]);
    __m256d v_vec = _mm256_load_pd(&v[i]);
    __m256d result = _mm256_fmadd_pd(u_vec, v_vec, sum_vec);
    _mm256_store_pd(&out[i], result);
}
```

### OpenMP

Multi-threaded parallelization:

**Features:**
- Thread pool for hot loops
- Static scheduling for load balance
- Scales with CPU cores

**Speedup:** ~3-6x on 8-core CPU (depends on grid size and cache)

**Use when:**
- Large grids (>200×200)
- Multi-core CPU available
- Long simulations

**Example:**
```c
#pragma omp parallel for schedule(static)
for (int j = 1; j < ny-1; j++) {
    for (int i = 1; i < nx-1; i++) {
        // Stencil operation...
    }
}
```

### CUDA GPU

GPU-accelerated using CUDA:

**Features:**
- Thousands of parallel threads
- Coalesced memory access
- Shared memory tiling

**Speedup:** 10-50x for very large grids (>500×500)

**Use when:**
- Very large grids (>200×200)
- Many time steps
- NVIDIA GPU available

**Crossover Point:**
- Small grids: CPU faster (data transfer overhead)
- Grid >200×200: GPU becomes beneficial
- Grid >500×500: GPU significantly faster

**Example:**
```c
// Check if GPU should be used
gpu_config_t config = gpu_config_default();
if (gpu_should_use(&config, nx, ny, num_steps)) {
    solver = cfd_solver_create(registry, "projection_gpu");
} else {
    solver = cfd_solver_create(registry, "projection_optimized");
}
```

## Performance Benchmarks

### Solver Comparison (100×50 grid, 50 steps, Release mode)

| Solver | Time (ms) | Speedup | Accuracy |
|--------|-----------|---------|----------|
| explicit_euler | 2.6 | 1.0x | Low |
| explicit_euler_optimized | 0.9 | 2.9x | Low |
| explicit_euler_omp (8 cores) | 0.8 | 3.3x | Low |
| projection | 19.0 | 1.0x | High |
| projection_optimized | 5.3 | 3.6x | High |
| projection_omp (8 cores) | 4.2 | 4.5x | High |
| projection_gpu | 8.4 | 0.45x† | High |

† GPU slower on small grids due to data transfer overhead

### Grid Size Scaling (projection_optimized, 100 steps)

| Grid | Time (s) | Memory (MB) | Iterations/step |
|------|----------|-------------|-----------------|
| 50×50 | 0.5 | 0.02 | ~150 |
| 100×100 | 2.1 | 0.08 | ~300 |
| 200×200 | 9.8 | 0.31 | ~600 |
| 500×500 | 82.4 | 1.91 | ~1500 |

### GPU vs SIMD (500×500 grid, 1000 steps)

| Solver | Time (s) | Speedup |
|--------|----------|---------|
| projection_optimized | 824 | 1.0x |
| projection_gpu | 68 | 12.1x |

## Choosing a Solver

### Decision Tree

```
Need strict incompressibility enforcement?
├─ No  → Use Explicit Euler family (faster)
│        ├─ Small grid (<100×100) → explicit_euler
│        ├─ Medium grid (100-500) → explicit_euler_optimized or explicit_euler_omp
│        └─ Large grid (>500)     → explicit_euler_omp or explicit_euler_gpu
│
└─ Yes → Use Projection Method family
         ├─ Small grid (<100×100) → projection
         ├─ Medium grid (100-500) → projection_optimized or projection_omp
         └─ Large grid (>500)     → projection_gpu

GPU available and grid >200×200?
└─ Use CUDA variant for 10-50x speedup
```

### Recommendations by Use Case

**Learning/Debugging:**
- `explicit_euler` or `projection`
- Simple, predictable behavior
- Easy to inspect intermediate results

**Production Simulations:**
- `projection_optimized` or `projection_omp` (medium grids)
- `projection_gpu` (large grids)
- Best accuracy and performance

**Benchmarking:**
- Always use `CMAKE_BUILD_TYPE=Release`
- Disable I/O during timing measurements
- Run multiple iterations for statistical significance

## Boundary Conditions

### Supported Types

1. **Dirichlet** - Fixed values: u(boundary) = u_bc
2. **Neumann** - Fixed gradient: ∂u/∂n(boundary) = g_bc
3. **Periodic** - u(0) = u(L)
4. **No-slip** - u = 0, v = 0 (walls)
5. **Inlet** - Specified velocity profile
6. **Outlet** - Zero-gradient: ∂u/∂n = 0

### Implementation

Boundary conditions are applied after each solver step:

```c
void apply_boundary_conditions(flow_field* field, grid_t* grid) {
    // No-slip walls (u=0, v=0)
    for (size_t i = 0; i < grid->nx; i++) {
        field->u[i + 0 * grid->nx] = 0.0;           // Bottom
        field->u[i + (grid->ny-1) * grid->nx] = 0.0; // Top
    }

    // Lid-driven cavity (top wall moves)
    for (size_t i = 0; i < grid->nx; i++) {
        field->u[i + (grid->ny-1) * grid->nx] = 1.0;
    }
}
```

## Validation

### Ghia Lid-Driven Cavity

Classic benchmark problem validated against Ghia et al. (1982):

**Setup:**
- Square cavity [0,1] × [0,1]
- Top wall moves with velocity u=1
- All other walls no-slip
- Reynolds number Re = ρUL/μ

**Results (129×129 grid, AVX2/OpenMP/CUDA projection):**
- Centerline velocity profiles match published data at Re = 100, 400 and 1000
- RMS error vs Ghia (u / v): 0.0017 / 0.0024 at Re=100, 0.0096 / 0.0328 at Re=400,
  0.0299 / 0.0300 at Re=1000

See [lid-driven-cavity.md](../validation/lid-driven-cavity.md) and
[cavity-backends-validation.md](../validation/cavity-backends-validation.md) for details.

### Taylor-Green Vortex

Analytical solution for decaying vortex:

**Exact Solution:**
```
u(x,y,t) = -cos(x)sin(y)exp(-2νt)
v(x,y,t) =  sin(x)cos(y)exp(-2νt)
p(x,y,t) = -0.25(cos(2x) + cos(2y))exp(-4νt)
```

**Results:**
- Velocity error < 1% at t=1.0
- Energy decay matches analytical solution

**3D Extension:**

```text
u(x,y,z,t) =  cos(x)sin(y)cos(z)exp(-3νt)
v(x,y,z,t) = -sin(x)cos(y)cos(z)exp(-3νt)
w(x,y,z,t) = 0
```

- Velocity decays as exp(-3νt), kinetic energy as exp(-6νt)
- Validated on 16×16×16 grid with ν=0.01

## Turbulence Models (RANS)

Two Reynolds-Averaged Navier-Stokes (RANS) turbulence models are implemented for 2D simulations
on uniform grids.  Set `params.turb_model` before calling `solver_init`; the default
(`TURB_MODEL_NONE = 0`) reproduces the laminar path bit-exactly with zero overhead.

### Turbulence Model Overview

| Model | Type | Transported variables | Extra cost |
|-------|----|----|----|
| `TURB_MODEL_K_EPSILON` | 2-equation | k, ε | ~2× laminar step |
| `TURB_MODEL_SPALART_ALLMARAS` | 1-equation | ν̃ | ~1.5× laminar step |

The turbulent viscosity ν_t is computed each step and stored in `flow_field->nu_t`.  All NS
solvers (scalar, OMP, AVX2) apply the effective viscosity

```
div((ν + ν_t) grad u)
```

in the momentum equation; ν_t = 0 when `TURB_MODEL_NONE`.

**Not modelled (standard practice):** Boussinesq-stress transpose term and the -(2/3)k δ_ij
isotropic stress (absorbed into modified pressure).  No turbulent Prandtl heat-flux model; the
energy equation is independent of turbulence in this release.

### Standard k-ε Model

**Transport equations:**

```
∂k/∂t + (u·∇)k = div((ν + ν_t/σ_k) grad k) + P_k - ε
∂ε/∂t + (u·∇)ε = div((ν + ν_t/σ_ε) grad ε) + C1 (ε/k) P_k - C2 ε²/k
```

**Turbulent viscosity:**

```
ν_t = C_μ k² / ε        (clipped at 1e5 ν)
```

**Constants (Launder & Spalding):**

| C_μ  | C1   | C2   | σ_k | σ_ε |
|------|------|------|-----|-----|
| 0.09 | 1.44 | 1.92 | 1.0 | 1.3 |

### Spalart-Allmaras Model

**Transport equation:**

```
∂ν̃/∂t + (u·∇)ν̃ = cb1 S̃ ν̃ + (1/σ) div((ν + ν̃) grad ν̃) + cb2/σ |grad ν̃|² - cw1 fw (ν̃/d)²
```

**Turbulent viscosity:**

```
ν_t = ν̃ fv1,    fv1 = χ³ / (χ³ + cv1³),    χ = ν̃/ν
```

**Constants (no-ft2 fully turbulent variant):**

| cb1    | cb2   | σ   | κ    | cw2 | cw3 | cv1 |
|--------|-------|-----|------|-----|-----|-----|
| 0.1355 | 0.622 | 2/3 | 0.41 | 0.3 | 2   | 7.1 |

Wall distance d is computed on the fly from faces marked `BC_TYPE_NOSLIP` in `turb_bc`.

### Discretization

- **Advection:** first-order upwind (guarantees positivity of k, ε, ν̃)
- **Diffusion:** conservative face-averaged effective viscosity
- **Source terms:** semi-implicit Patankar treatment — the destruction term is linearized so k,
  ε, and ν̃ can never be driven negative
- **Production limiter:** P_k ≤ 10 ε
- **Positivity floors:** k, ε, ν̃ ≥ 1e-10 enforced after each step
- **Time step:** the adaptive `compute_time_step` includes the viscous limit
  `dt < dx² / (2 ν_eff n_dim)` using max(ν_t) when turbulence is active

### Wall Functions

Wall-function treatment is applied at every face marked `BC_TYPE_NOSLIP` in `turb_bc`.

**Law of the wall** (`params.turb_bc.wall_law`, κ = 0.41, B = 5.2 for both):

| `ns_wall_law_t` | Law | Use when |
|---|---|---|
| `NS_WALL_LAW_LOG` (0, default) | u+ = y+ below y+_c, u+ = (1/κ) ln(y+) + B above | First node at y+ ≳ 20, including the usual wall-function range 30–100 |
| `NS_WALL_LAW_SPALDING` | y+ = u+ + e^(-κB) [e^(κu+) - 1 - κu+ - (κu+)²/2 - (κu+)³/6] | The first node may sit below y+ ≈ 17 (sublayer or lower buffer layer) |

The log law switches branch at y+_c = 11.06, where the linear and log laws meet, so u_τ is
continuous and only its slope changes. The switch is taken on the wall Reynolds number
u_p y_p / ν ≤ y+_c², which is known before u_τ. It used to sit at y+ = 11.63, where the laws
do not meet, and u_τ jumped 3.3% there (wall shear 6.6%, ε 10%).

Spalding's law is one smooth curve through the sublayer, buffer and log layers, solved by
safeguarded Newton iteration on u+. Below y+ ≈ 100 it sits under the log law: u_τ for a
given u_p is 3.1% above the log-law value at y+ = 40, 0.9% at y+ = 100.

Which is closer to reality depends on where the first node sits. Against Moser-Kim-Mansour
channel DNS at Re_τ ≈ 395:

| y+ | 5 | 11 | 15 | 20 | 30 | 39.5 | 100 |
|---|---|---|---|---|---|---|---|
| Log law vs DNS | +3.7% | +22.5% | +10.8% | +4.2% | 0.0% | −0.7% | −0.6% |
| Spalding vs DNS | +1.2% | −1.7% | −3.9% | −5.2% | −5.2% | −4.4% | −1.7% |

Spalding's law is closer up to y+ ≈ 17, the log law from about y+ = 20 on.
`turbulence_wall_u_tau(law, u_p, y_p, ν)` returns the u_τ each law imposes.

**Equilibrium values at the first interior node (distance y_p from the wall):**

```
k  = u_τ² / sqrt(C_μ)
ε  = u_τ³ / (κ y_p)          [k-ε]
ν̃  = κ u_τ y_p               [SA]
```

The wall-face effective viscosity is set to `max(2*(u_τ² y_p / u_p - ν), 0)` so that the
discrete wall shear equals u_τ² exactly and reduces to pure laminar shear in the viscous
sublayer.

**y+ guideline:** place the first interior node at 30 ≤ y+ ≤ 100.

### Backend Coverage

| Backend | k-ε | SA | Notes |
|---------|-----|----|-------|
| Scalar  | done | done | reference implementation |
| AVX2    | done | done | k-ε: i-vectorized 4-wide with blendv upwind + scalar tail; SA: OMP-parallel scalar rows (pow-heavy closures do not vectorize profitably; numerics identical) |
| NEON    | — | — | not yet implemented |
| OMP     | done | done | j-loop parallel, identical numerics to scalar |
| GPU     | — | — | returns `CFD_ERROR_UNSUPPORTED` when a turbulence model is enabled |

Cross-backend consistency is verified by a unit test asserting kernel-level L∞ agreement
between all available backends.

### Limitations

- **2D only** — 3D grids return `CFD_ERROR_UNSUPPORTED`
- **Uniform grid only** — non-uniform spacing returns `CFD_ERROR_UNSUPPORTED`
- **No GPU turbulence** — GPU NS solvers return `CFD_ERROR_UNSUPPORTED` when
  `turb_model != TURB_MODEL_NONE`
- **No turbulent Prandtl model** — the energy equation is not coupled to turbulence

### Turbulent Channel-Flow Validation

**Setup:** Re_τ = 395, δ = 1, 16×21 uniform grid, body-force-driven
(f_x = u_τ²/δ = 1 so exact steady u_τ = 1), first-node y+ ≈ 40.

u_τ is recovered from the first-node velocity by inverting the pure log law, independently
of the model's own wall law, so the yardstick does not move with the model it grades.

**Results** (u_τ error vs the exact 1; u+ error vs the log law):

| Model | Wall law | u_τ (recovered) | u_τ error | u+ at y+ = 39.5 | u+ at y+ = 79 |
|-------|----------|-----------------|-----------|-----------------|---------------|
| k-ε | log (default) | 0.971 | 2.9% | 0.5% | 2.4% |
| SA  | log (default) | 0.967 | 3.3% | 0.6% | 3.5% |
| k-ε | Spalding | 0.941 | 5.9% | 1.0% | 2.4% |
| SA  | Spalding | 0.940 | 6.0% | 1.1% | 3.4% |

Spalding's law sits below the log law at the first node (y+ = 39.5), imposing a 3.1% higher
u_τ for a given u_p, so the steady first-node velocity drops 3.6% and the log-law yardstick
reads it as a larger u_τ deficit. The test runs the default. The SA log-law figure was
recorded as 3.1% before, from a Debug build; this Release build reads 3.3%.

Source: `tests/validation/test_turbulent_channel.c` (ctest label `validation`).

## References

### Numerical Methods

- **Chorin, A.J.** (1968). "Numerical Solution of the Navier-Stokes Equations". Mathematics of Computation.
- **Ferziger & Peric** - "Computational Methods for Fluid Dynamics"
- **Versteeg & Malalasekera** - "An Introduction to CFD"

### Validation

- **Ghia, U., Ghia, K.N., Shin, C.T.** (1982). "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method". Journal of Computational Physics.
- **Kim, J., Moin, P., Moser, R.** (1987). "Turbulence statistics in fully developed channel flow at low Reynolds number". Journal of Fluid Mechanics.

### Linear Solvers

- **Saad, Y.** - "Iterative Methods for Sparse Linear Systems"
- **Barrett et al.** - "Templates for the Solution of Linear Systems"

### Turbulence Modeling

- **Launder, B.E. & Spalding, D.B.** (1974). "The numerical computation of turbulent flows". Computer Methods in Applied Mechanics and Engineering.
- **Spalart, P.R. & Allmaras, S.R.** (1992). "A one-equation turbulence model for aerodynamic flows". AIAA Paper 92-0439.
- **Wilcox, D.C.** - "Turbulence Modeling for CFD"

## Next Steps

- [Examples](../guides/examples.md) - See solvers in action
- [API Reference](api-reference.md) - API documentation
- [Building](../guides/building.md) - Build with specific solvers
