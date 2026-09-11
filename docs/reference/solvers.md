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
| `explicit_euler_optimized` | SIMD | SIMD-optimized (auto-detects AVX2/NEON) |
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
- Second-order accurate in space (central differences)
- First-order accurate in time
- Properly enforces ∇·u = 0
- More expensive due to Poisson solve

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `projection` | Scalar | Basic implementation |
| `projection_optimized` | SIMD | SIMD-optimized (runtime detection: AVX2/NEON) |
| `projection_omp` | OpenMP | Multi-threaded |
| `projection_gpu` | GPU | CUDA-accelerated (CG pressure solve) |

**Pressure solver selection** (`ns_solver_params_t.pressure_solver`):

Each backend pairs with a CG Poisson preset by default. On the scalar
`projection` solver the pressure solve can be switched to geometric multigrid:

| `pressure_solver` value | Pressure Poisson solve |
|-------------------------|------------------------|
| `NS_PRESSURE_SOLVER_DEFAULT` (0) | Backend's CG preset (existing behavior) |
| `NS_PRESSURE_SOLVER_MULTIGRID` | Multigrid V-cycles (`POISSON_SOLVER_MG_SCALAR`) |
| `NS_PRESSURE_SOLVER_PCG_MG` | CG + MG V-cycle preconditioner (`POISSON_SOLVER_PCG_MG_SCALAR`) |

The MG modes require 2^k+1 grid points per active dimension (33, 65, 129, ...);
`solver_init` returns `CFD_ERROR_UNSUPPORTED` otherwise. `projection_optimized`,
`projection_omp`, and `projection_gpu` reject any non-default value with
`CFD_ERROR_UNSUPPORTED` at init — multigrid has no SIMD/OMP/GPU backend yet and
the library never falls back across backends silently.

```c
ns_solver_params_t params = ns_solver_params_default();
params.pressure_solver = NS_PRESSURE_SOLVER_MULTIGRID;  // 2^k+1 grids only
ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
cfd_status_t status = solver_init(slv, grid, &params);   // UNSUPPORTED on 128x128
```

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

**Optimal Relaxation (Dirichlet BCs):**
```
ω_opt = 2 / (1 + sin(πh))
```

**Characteristics:**
- Faster than Jacobi (ω > 1)
- Sequential row updates (row j depends on j-1)
- Optimal ω depends on problem
- SIMD variant uses Block SOR: processes SIMD_WIDTH consecutive cells per block, with intra-block left-neighbor approximation (see [Block SOR technical note](../technical-notes/block-sor-simd.md))
- GPU variant also uses Block SOR: each thread sweeps an 8×8 tile sequentially (Gauss-Seidel inside the tile), with red-black *tile* coloring (red pass then black pass, two launches per iteration) so a tile's halo is never written by another tile in the same pass — the update is in-place, race-free, and provably convergent for 0<ω<2

**Convergence Rate:** ρ ≈ 1 - 2πh (with optimal ω)

**Available Backends:**
| Solver | Backend | Description |
|--------|---------|-------------|
| `sor_scalar` | Scalar | Sequential Gauss-Seidel + SOR relaxation |
| `sor_simd` | SIMD | Block SOR (auto-detects AVX2/NEON) |
| `sor_gpu` | GPU | Block SOR (CUDA; per-thread tile sweep, red-black tile coloring, in-place) |

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.omega = 1.5;  // Relaxation parameter

// Scalar (fully sequential, best convergence per iteration)
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_SOR,
                                                 POISSON_BACKEND_SCALAR);

// SIMD (Block SOR, higher throughput, slightly more iterations)
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
  - Scalar CG backend only; other backends reject it with
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
params.preconditioner = POISSON_PRECOND_JACOBI;  // Enable preconditioning

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                 POISSON_BACKEND_SIMD);
poisson_solver_init(solver, nx, ny, dx, dy, &params);  // Pass params with preconditioner
```

**Multigrid-preconditioned CG (scalar backend, 2^k+1 dims):**

```c
poisson_solver_params_t params = poisson_solver_params_default();
params.preconditioner = POISSON_PRECOND_MULTIGRID;

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_CG,
                                                 POISSON_BACKEND_SCALAR);
poisson_solver_init(solver, 65, 65, 1, dx, dy, 0.0, &params);
```
The convenience API exposes the same configuration as the
`POISSON_SOLVER_PCG_MG_SCALAR` preset for `poisson_solve()`/`poisson_solve_3d()`.

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
- Restart length `m` set via `params.restart` (0 = auto, default 30); bounds the
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
cannot be indexed with `int` (m ≥ 46341) with `CFD_ERROR_LIMIT_EXCEEDED`.

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.restart = 30;  // GMRES(30); 0 selects the default

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_GMRES,
                                                 POISSON_BACKEND_SCALAR);
poisson_solver_init(solver, nx, ny, nz, dx, dy, dz, &params);
```

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
- Two BC modes via `params.mg_bc`:
  - `MG_BC_NEUMANN` (default) — zero-gradient BCs matching the other Poisson
    solvers. The system is singular (solution defined up to a constant; RHS
    should have zero interior mean). Restriction uses Neumann-folded boundary
    weights and coarse RHS/corrections are mean-projected internally.
  - `MG_BC_DIRICHLET` — caller-supplied boundary values of `x` are held fixed
    (supports inhomogeneous data); coarse corrections use zero boundaries.
- `MG_CYCLE_F` runs one full-multigrid pass (coarsest-first nested iteration)
  on the first cycle — reaching discretization accuracy immediately — then
  continues with V-cycles
- Parameters: `mg_cycle`, `mg_smoother`, `mg_bc`, `mg_pre_smooth`/`mg_post_smooth`
  (default 2/2), `mg_coarse_max_iter` (default 50), `mg_max_levels` (0 = auto)

**Backends:** scalar only. Also available as a CG preconditioner
(`POISSON_PRECOND_MULTIGRID`, scalar CG only — see §5) and as the projection
method's pressure solver (`ns_solver_params_t.pressure_solver`). SIMD/OMP/GPU
variants are planned follow-ups.

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.mg_cycle = MG_CYCLE_V;        // or MG_CYCLE_W / MG_CYCLE_F
params.mg_bc = MG_BC_NEUMANN;        // default; matches other solvers

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                                 POISSON_BACKEND_SCALAR);
poisson_solver_init(solver, 65, 65, 1, dx, dy, 0.0, &params);  // dims 2^k+1
```

### Linear Solver Performance Comparison

**Problem:** 65×65 grid, tolerance = 1e-6

| Method | Iterations | Time (ms) | Notes |
|--------|-----------|-----------|-------|
| Jacobi | ~8000 | 45 | Simple, slow |
| SOR (ω=1.5) | ~2000 | 15 | Good serial performance |
| Red-Black SOR | ~2000 | 8 | Parallelizable |
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

**Results (Re=100, 129×129 grid):**
- Centerline velocity profiles match published data
- RMS error < 0.01

See [validation/lid-driven-cavity.md](validation/lid-driven-cavity.md) for details.

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

Log-law wall treatment is applied at every face marked `BC_TYPE_NOSLIP` in `turb_bc`.

**Log law:**

```
u+ = (1/κ) ln(y+) + B,    κ = 0.41,  B = 5.2
```

Below y+ = 11.63 the linear viscous-sublayer law is used.  The friction velocity u_τ is
recovered by Newton iteration on the log-law residual (`turbulence_wall_u_tau()`).

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

**k-ε results:**

| Quantity | Value | Error |
|----------|-------|-------|
| u_τ (recovered) | 0.971 | 2.9% |
| u+ at y+ = 39.5 | — | 0.5% vs log law |
| u+ at y+ = 79   | — | 2.4% vs log law |

**SA results:**

| Quantity | Value | Error |
|----------|-------|-------|
| u_τ (recovered) | 0.969 | 3.1% |
| u+ at y+ ≈ 39.5 | — | 0.5% vs log law |
| u+ at y+ ≈ 79   | — | 3.7% vs log law |

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
