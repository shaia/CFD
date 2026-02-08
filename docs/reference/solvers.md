# Solvers & Numerical Methods

Comprehensive guide to solvers and numerical methods in the CFD Framework.

## Overview

The CFD Framework implements multiple solvers for the incompressible Navier-Stokes equations:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
∇·u = 0
```

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
| `euler_cpu` | Scalar | Basic implementation |
| `euler_avx2` | SIMD | AVX2-optimized (x86-64) |
| `euler_neon` | SIMD | NEON-optimized (ARM64) |
| `euler_omp` | OpenMP | Multi-threaded |

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
| `projection_jacobi_gpu` | GPU | CUDA-accelerated (Jacobi iteration) |

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
- Sequential updates (harder to parallelize)
- Optimal ω depends on problem

**Convergence Rate:** ρ ≈ 1 - 2πh (with optimal ω)

**Usage:**
```c
poisson_solver_params_t params = poisson_solver_params_default();
params.omega = 1.5;  // Relaxation parameter

poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_SOR,
                                                 POISSON_BACKEND_SCALAR);
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

#### 6. BiCGSTAB

**Algorithm:** Bi-Conjugate Gradient Stabilized for non-symmetric systems.

**Characteristics:**
- Handles non-symmetric matrices
- More robust than CG for difficult problems
- Higher cost per iteration than CG

**Usage:**
```c
poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_BICGSTAB,
                                                 POISSON_BACKEND_SCALAR);
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
| BiCGSTAB | ~40 | 4 | Fastest convergence |

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
    solver = cfd_solver_create(registry, "projection_jacobi_gpu");
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
| projection_jacobi_gpu | 8.4 | 0.45x† | High |

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
| projection_jacobi_gpu | 68 | 12.1x |

## Choosing a Solver

### Decision Tree

```
Need strict incompressibility enforcement?
├─ No  → Use Explicit Euler family (faster)
│        ├─ Small grid (<100×100) → euler_cpu
│        ├─ Medium grid (100-500) → euler_avx2 or euler_omp
│        └─ Large grid (>500)     → euler_omp or euler_cuda
│
└─ Yes → Use Projection Method family
         ├─ Small grid (<100×100) → projection
         ├─ Medium grid (100-500) → projection_optimized or projection_omp
         └─ Large grid (>500)     → projection_jacobi_gpu

GPU available and grid >200×200?
└─ Use CUDA variant for 10-50x speedup
```

### Recommendations by Use Case

**Learning/Debugging:**
- `euler_cpu` or `projection_cpu`
- Simple, predictable behavior
- Easy to inspect intermediate results

**Production Simulations:**
- `projection_optimized` or `projection_omp` (medium grids)
- `projection_jacobi_gpu` (large grids)
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

## References

### Numerical Methods

- **Chorin, A.J.** (1968). "Numerical Solution of the Navier-Stokes Equations". Mathematics of Computation.
- **Ferziger & Peric** - "Computational Methods for Fluid Dynamics"
- **Versteeg & Malalasekera** - "An Introduction to CFD"

### Validation

- **Ghia, U., Ghia, K.N., Shin, C.T.** (1982). "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method". Journal of Computational Physics.

### Linear Solvers

- **Saad, Y.** - "Iterative Methods for Sparse Linear Systems"
- **Barrett et al.** - "Templates for the Solution of Linear Systems"

## Next Steps

- [Examples](../guides/examples.md) - See solvers in action
- [API Reference](api-reference.md) - API documentation
- [Building](../guides/building.md) - Build with specific solvers
