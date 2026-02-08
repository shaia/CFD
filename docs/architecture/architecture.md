# Architecture & Design Principles

This document describes the architectural principles and design patterns used in the CFD Framework.

## Overview

The CFD Framework follows a modular, layered architecture that separates:
- Core infrastructure (grid, memory, I/O)
- Numerical solvers (backend implementations)
- Public API (dispatcher layer)

## Design Principles

### 1. Separation of Concerns

**Solvers focus solely on solving** - They don't decide when or where to write output files.

Benefits:
- Makes solvers more testable and reusable
- Gives callers full control over output timing and location
- Prevents file pollution from automatic solver output
- Enables flexible output strategies (batch processing, streaming, etc.)

Example:
```c
// Solver only solves - no I/O side effects
cfd_status_t solver_step(ns_solver_t* solver, flow_field* field,
                         grid_t* grid, double dt);

// Caller controls when to write output
for (int step = 0; step < max_steps; step++) {
    solver_step(solver, field, grid, dt);

    if (step % output_interval == 0) {
        write_to_vtk(field, grid, filename);  // Caller's decision
    }
}
```

### 2. Zero-Branch Dispatch

The modern `ns_solver_t` interface uses function pointers for dispatch instead of switch statements:

```c
// Old pattern (branching):
void solver_step(solver_type type, ...) {
    switch (type) {
        case EULER: euler_step(...); break;
        case PROJECTION: projection_step(...); break;
        // ...
    }
}

// New pattern (function pointers):
struct ns_solver {
    cfd_status_t (*step)(ns_solver_t*, flow_field*, grid_t*, double);
    // ...
};

// Zero-branch call:
solver->step(solver, field, grid, dt);
```

Benefits:
- No branch misprediction overhead
- Optimal CPU pipeline utilization
- Compiler can inline hot paths
- Extensible without modifying core code

### 3. Pluggable Architecture

All components use registry patterns for extensibility:

#### Solver Registry

```c
// Register solvers at runtime
ns_solver_registry_t* registry = cfd_registry_create();
cfd_registry_register_defaults(registry);

// Add custom solver without touching existing code
cfd_registry_register(registry, "my_custom_solver",
                      my_solver_create_fn,
                      NS_SOLVER_METHOD_CUSTOM,
                      NS_SOLVER_BACKEND_SCALAR);

// Factory pattern for clean instantiation
ns_solver_t* solver = cfd_solver_create(registry, "projection_avx2");
```

#### Output Registry

```c
// Multiple output formats supported
output_registry_t* output_reg = output_registry_create();
output_registry_register(output_reg, "vtk", vtk_writer_create);
output_registry_register(output_reg, "csv", csv_writer_create);

// Runtime format selection
output_writer_t* writer = output_registry_get(output_reg, "vtk");
```

### 4. Backend Abstraction

Solvers are organized by computational backend:

```
lib/src/solvers/
├── cpu/              # Scalar implementations (always available)
├── simd/             # AVX2/NEON optimized
│   ├── avx2/         # x86-64 SIMD
│   └── neon/         # ARM64 SIMD
├── omp/              # OpenMP parallelized
└── gpu/              # CUDA kernels
```

**Runtime Detection:**
```c
// Check backend availability at runtime
if (cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
    solver = cfd_solver_create(registry, "projection_avx2");
} else {
    solver = cfd_solver_create(registry, "projection_cpu");
}
```

**Compile-Time Guards:**
- Public API dispatchers are technology-agnostic (no `#ifdef` for SIMD/OpenMP/CUDA)
- Backend files contain `#ifdef` guards
- Unavailable backends provide NULL stubs

### 5. Memory Efficiency

#### Contiguous Array Storage
```c
// Good: Contiguous, cache-friendly
typedef struct {
    double* u;      // nx * ny doubles
    double* v;      // nx * ny doubles
    double* p;      // nx * ny doubles
} flow_field;

// Bad: Array of structures (poor cache locality)
typedef struct {
    struct { double u, v, p; } cells[ny][nx];
} flow_field_aos;
```

#### SIMD-Friendly Layouts
```c
// Aligned allocation for SIMD
double* field = cfd_aligned_calloc(nx * ny, sizeof(double), 32);

// Structure-of-arrays pattern
for (size_t i = 0; i < nx * ny; i += 4) {
    __m256d u_vec = _mm256_load_pd(&field[i]);  // Aligned load
    // SIMD operations...
}
```

#### Explicit Memory Management
- Clear ownership: creator is responsible for cleanup
- No hidden allocations in hot paths
- Predictable memory patterns for profiling

### 6. Error Handling

#### Status Code Pattern
```c
typedef enum {
    CFD_SUCCESS = 0,
    CFD_ERROR = -1,
    CFD_ERROR_NOMEM = -2,
    CFD_ERROR_INVALID = -3,
    CFD_ERROR_UNSUPPORTED = -5,  // Backend unavailable
    CFD_ERROR_MAX_ITER = -7,     // Solver convergence failure
} cfd_status_t;
```

#### Thread-Safe Error Reporting
```c
// Set error (thread-local storage)
cfd_set_error(CFD_ERROR_INVALID, "Grid dimensions must be positive");

// Retrieve error
const char* msg = cfd_get_last_error();
cfd_status_t code = cfd_get_last_status();

// Clear error state
cfd_clear_error();
```

#### No Silent Fallbacks
```c
// Good: Return error if backend unavailable
if (!poisson_solver_backend_available(POISSON_BACKEND_SIMD)) {
    return CFD_ERROR_UNSUPPORTED;
}

// Bad: Silent fallback to different backend
if (!simd_available) {
    use_scalar_backend();  // Wrong - caller loses control
}
```

## Modular Library Architecture

### Library Components

| Library | Target | Description | Dependencies |
|---------|--------|-------------|--------------|
| `cfd_core` | `CFD::Core` | Grid, memory, I/O | None |
| `cfd_scalar` | `CFD::Scalar` | Scalar CPU solvers | CFD::Core |
| `cfd_simd` | `CFD::SIMD` | AVX2/NEON solvers | CFD::Core, CFD::Scalar |
| `cfd_omp` | `CFD::OMP` | OpenMP solvers | CFD::Core, CFD::Scalar |
| `cfd_cuda` | `CFD::CUDA` | CUDA GPU solvers | CFD::Core |
| `cfd_api` | `CFD::API` | Dispatcher layer | All backends |
| `cfd_library` | `CFD::Library` | Unified library | All above |

### Dependency Graph

```
cfd_library (unified)
    ├── cfd_api (dispatchers)
    │   ├── cfd_scalar
    │   ├── cfd_simd
    │   ├── cfd_omp
    │   └── cfd_cuda
    └── cfd_core (foundation)
```

### CMake Usage

```cmake
find_package(CFD REQUIRED)

# Link only what you need
target_link_libraries(my_simd_app PRIVATE CFD::SIMD)      # SIMD-only
target_link_libraries(my_omp_app PRIVATE CFD::OMP)        # OpenMP-only
target_link_libraries(my_gpu_app PRIVATE CFD::CUDA)       # GPU-only
target_link_libraries(my_full_app PRIVATE CFD::Library)   # Everything
```

### Benefits

- **Reduced binary size** - Link only the backends you use
- **Fewer dependencies** - No OpenMP/CUDA requirements for scalar-only builds
- **Faster compilation** - Compile only needed components
- **Clear dependencies** - Each library declares what it needs
- **Runtime backend detection** - Backends provide NULL stubs when unavailable

## Output Organization

All simulation output is organized in timestamped subdirectories:

- Pattern: `artifacts/output/<name>_<grid>_<timestamp>/`
- Example: `artifacts/output/cavity_100x50_20251128_115220/`

Benefits:
- Prevents file conflicts between runs
- Easy to manage and archive results
- Configurable base path via `cfd_set_output_base_dir()`

## Algorithm-Primitive Separation

High-level algorithms (CG, PCG, BiCGSTAB) are expressed in terms of primitive operations:

```c
// Algorithm (same across all backends)
double cg_solve(double* x, double* rhs, ...) {
    double* r = allocate(...);
    double* p = allocate(...);

    // High-level CG loop
    for (int iter = 0; iter < max_iter; iter++) {
        double alpha = dot_product(r, r) / dot_product(p, Ap);  // Primitive
        axpy(x, alpha, p);                                       // Primitive
        apply_laplacian(Ap, p);                                  // Primitive
        // ...
    }
}

// Backend-specific primitives
// CPU version:
double dot_product_cpu(double* a, double* b, size_t n);

// AVX2 version:
double dot_product_avx2(double* a, double* b, size_t n) {
    __m256d sum_vec = _mm256_setzero_pd();
    for (size_t i = 0; i < n; i += 4) {
        __m256d a_vec = _mm256_load_pd(&a[i]);
        __m256d b_vec = _mm256_load_pd(&b[i]);
        sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
    }
    // Horizontal sum...
}
```

Benefits:
- Algorithm logic is not duplicated across backends
- Easy to add new backends (just implement primitives)
- Performance critical code is isolated

## Boundary Conditions Architecture

The boundary condition system exemplifies the framework's design principles: backend abstraction, zero-branch dispatch, and code reuse through templating.

### Architecture Overview

```
User Code
    ↓
lib/include/cfd/boundary/boundary_conditions.h (Public API)
    ↓
lib/src/boundary/boundary_conditions.c (Dispatcher)
    ↓
    ├── cpu/boundary_conditions_scalar.c       (Scalar backend)
    ├── omp/boundary_conditions_omp.c          (OpenMP backend)
    ├── simd/boundary_conditions_simd_dispatch.c
    │   ├── avx2/boundary_conditions_avx2.c    (AVX2 SIMD)
    │   └── neon/boundary_conditions_neon.c    (NEON SIMD)
    └── gpu/boundary_conditions_gpu.cu         (CUDA kernels)
```

### Supported Boundary Condition Types

| BC Type                     | Description                             | Use Cases                                 |
|-----------------------------|-----------------------------------------|-------------------------------------------|
| **Neumann (Zero Gradient)** | ∂φ/∂n = 0                               | Pressure, intermediate velocity           |
| **Periodic**                | φ(x_min) = φ(x_max)                     | Fully periodic domains                    |
| **Dirichlet**               | φ = φ_bc                                | Fixed values (temperature, velocity)      |
| **No-slip**                 | u = v = 0                               | Stationary walls                          |
| **Inlet**                   | u = f(y,t), v = 0                       | Inlet velocity (uniform/parabolic/custom) |
| **Outlet**                  | ∂φ/∂n = 0 or convective                 | Outflow boundary                          |
| **Symmetry**                | ∂u_normal/∂n = 0, u_tangent = 0         | Symmetry planes                           |
| **Moving Wall**             | u = u_wall(x,t)                         | Lid-driven cavity, rotating walls         |

### Runtime Backend Selection

The dispatcher automatically selects the best available backend:

```c
// Public API - backend selected at runtime
void bc_apply_neumann_2d(double* field, size_t nx, size_t ny, bc_backend_t backend);

// Dispatcher logic
void bc_apply_neumann_2d(double* field, size_t nx, size_t ny, bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_AUTO:
            // Select best available: GPU > SIMD > OMP > Scalar
            if (bc_gpu_available()) {
                bc_apply_neumann_2d_gpu(field, nx, ny);
            } else if (bc_simd_available()) {
                bc_apply_neumann_2d_simd(field, nx, ny);
            } else if (bc_omp_available()) {
                bc_apply_neumann_2d_omp(field, nx, ny);
            } else {
                bc_apply_neumann_2d_scalar(field, nx, ny);
            }
            break;
        case BC_BACKEND_SIMD:
            bc_apply_neumann_2d_simd(field, nx, ny);
            break;
        // ...
    }
}
```

### Code Reuse Through Template Headers

To eliminate duplication across backends, the BC system uses parameterized header templates:

#### 1. Core Implementation Template

**File:** `boundary_conditions_core_impl.h`

Unifies scalar and OpenMP implementations using token-pasting macros:

```c
// Scalar version uses BC_CORE_FN(name) → name
// OMP version uses BC_CORE_FN(name) → name##_omp

void BC_CORE_FN(bc_apply_neumann_2d)(double* field, size_t nx, size_t ny) {
#ifdef BC_CORE_USE_OMP
    #pragma omp parallel for
#endif
    for (size_t j = 0; j < ny; j++) {
        field[0 + j*nx] = field[1 + j*nx];           // Left boundary
        field[(nx-1) + j*nx] = field[(nx-2) + j*nx]; // Right boundary
    }
    // ... top/bottom boundaries
}
```

**Usage:**

- `boundary_conditions_scalar.c` includes with no `BC_CORE_USE_OMP`
- `boundary_conditions_omp.c` includes with `#define BC_CORE_USE_OMP`

**Savings:** 29 lines of duplicated code removed

#### 2. SIMD Implementation Template

**File:** `boundary_conditions_simd_impl.h`

Parameterized by SIMD intrinsics (AVX2 vs NEON):

```c
// Parameters: SIMD_LOAD, SIMD_STORE, SIMD_VEC_TYPE, SIMD_WIDTH

void BC_SIMD_FN(bc_apply_dirichlet_2d)(
    double* field, size_t nx, size_t ny, double value)
{
    SIMD_VEC_TYPE val_vec = SIMD_BROADCAST(&value);

    // Vectorized loop
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i += SIMD_WIDTH) {
            size_t idx = i + j * nx;
            SIMD_STORE(&field[idx], val_vec);
        }
        // Scalar remainder
        for (size_t i = (nx / SIMD_WIDTH) * SIMD_WIDTH; i < nx; i++) {
            field[i + j*nx] = value;
        }
    }
}
```

**AVX2 Parameters:**

```c
#define SIMD_VEC_TYPE __m256d
#define SIMD_WIDTH 4
#define SIMD_LOAD(p) _mm256_loadu_pd(p)
#define SIMD_STORE(p, v) _mm256_storeu_pd(p, v)
#define SIMD_BROADCAST(p) _mm256_broadcast_sd(p)
```

**NEON Parameters:**

```c
#define SIMD_VEC_TYPE float64x2_t
#define SIMD_WIDTH 2
#define SIMD_LOAD(p) vld1q_f64(p)
#define SIMD_STORE(p, v) vst1q_f64(p, v)
#define SIMD_BROADCAST(p) vld1q_dup_f64(p)
```

**Savings:** 181 lines removed (AVX2 and NEON implementations unified)

#### 3. Outlet BC SIMD Template

**File:** `boundary_conditions_outlet_simd.h`

Specialized template for outlet boundary conditions:

```c
void BC_OUTLET_SIMD_FN(bc_apply_outlet_convective)(
    double* field, double* field_prev,
    size_t nx, size_t ny, double dx, double dt)
{
    // Convective BC: ∂φ/∂t + u∂φ/∂x = 0
    // Uses SIMD operations for vectorization
    // ...
}
```

**Savings:** 99 lines removed

### Benefits of Template Approach

1. **Single Source of Truth**: Algorithm logic defined once
2. **Type Safety**: Compile-time checking across all backends
3. **Performance**: Zero runtime overhead, same as hand-coded versions
4. **Maintainability**: Bug fixes propagate to all backends automatically
5. **Consistency**: Identical behavior across backends (verified by tests)

**Total Code Reduction:** ~505 lines eliminated through templating

### Time-Varying Boundary Conditions

Support for time-dependent BCs with modulation functions:

```c
// Define time-varying inlet velocity
bc_inlet_spec_t inlet = {
    .type = BC_INLET_PARABOLIC,
    .u_max = 1.0,
    .time_modulation = BC_TIME_SINUSOIDAL,
    .frequency = 2.0,  // 2 Hz oscillation
    .phase = 0.0
};

// Apply at each timestep
bc_apply_inlet_2d(field->u, field->v, nx, ny, &inlet, current_time);
```

**Modulation Types:**

- `BC_TIME_CONSTANT` - Steady boundary condition
- `BC_TIME_SINUSOIDAL` - Sinusoidal oscillation
- `BC_TIME_RAMP` - Linear ramp up/down
- `BC_TIME_CUSTOM` - User-defined function pointer

### GPU Boundary Conditions

CUDA kernels for boundary conditions on device memory:

```cuda
__global__ void bc_apply_neumann_2d_kernel(
    double* __restrict__ field,
    size_t nx, size_t ny)
{
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < ny) {
        // Left and right boundaries
        field[0 + j*nx] = field[1 + j*nx];
        field[(nx-1) + j*nx] = field[(nx-2) + j*nx];
    }
}
```

**Features:**

- Coalesced memory access patterns
- Shared memory optimization for complex BCs
- Minimal CPU-GPU transfers (works on device memory)

### Integration with Solvers

Solvers apply BCs at appropriate stages:

```c
// Projection method example
void projection_solve_step(flow_field* field, grid_t* grid, double dt) {
    // 1. Predictor: compute intermediate velocity
    compute_intermediate_velocity(field, grid, dt);

    // 2. Apply Neumann BC on intermediate velocity
    bc_apply_neumann_2d(field->u, grid->nx, grid->ny, BC_BACKEND_AUTO);
    bc_apply_neumann_2d(field->v, grid->nx, grid->ny, BC_BACKEND_AUTO);

    // 3. Solve pressure Poisson equation
    poisson_solve(field->p, rhs, grid);

    // 4. Apply Neumann BC on pressure
    bc_apply_neumann_2d(field->p, grid->nx, grid->ny, BC_BACKEND_AUTO);

    // 5. Corrector: project velocity to divergence-free
    correct_velocity(field, grid, dt);

    // 6. Apply final velocity BC (e.g., periodic or no-slip)
    bc_apply_periodic_2d(field->u, grid->nx, grid->ny, BC_BACKEND_AUTO);
    bc_apply_periodic_2d(field->v, grid->nx, grid->ny, BC_BACKEND_AUTO);
}
```

### Performance Characteristics

**Scalar vs SIMD (Neumann BC, 256×256 grid):**

- Scalar: 45 μs
- AVX2: 18 μs (2.5x speedup)
- NEON: 28 μs (1.6x speedup)

**OpenMP Scaling (1024×1024 grid, Neumann BC):**

- 1 thread: 720 μs
- 4 threads: 205 μs (3.5x)
- 8 threads: 118 μs (6.1x)

**GPU Benefits:**

- Small grids (<100×100): CPU faster (kernel launch overhead)
- Large grids (>500×500): GPU 5-10x faster
- Best when data already on GPU (no transfer overhead)

### Testing Strategy

**Unit Tests:**

- Each BC type tested independently
- Verify correct values at boundary points
- Check interior points unchanged

**Cross-Backend Validation:**

- All backends produce identical results
- L2 difference < 1e-14 (machine precision)
- Reference: `tests/core/test_boundary_conditions_symmetry.c`

**Integration Tests:**

- Validate BCs in full solver context
- Lid-driven cavity (Dirichlet + no-slip)
- Periodic channel flow
- Inlet/outlet pipe flow

### Future Extensions

Planned enhancements (see ROADMAP.md):

- [ ] Robin boundary conditions (α∂φ/∂n + βφ = γ)
- [ ] Pressure outlet BC (specify pressure at outlet)
- [ ] Free surface tracking
- [ ] Immersed boundary method
- [ ] Adaptive mesh refinement at boundaries

## Thread Safety

### Initialization

```c
// Thread-safe initialization (uses pthread_once / InitOnceExecuteOnce)
cfd_status_t cfd_init(void);

// Safe for concurrent applications:
// Thread 1:
cfd_init();
simulation* sim1 = simulation_create(...);

// Thread 2:
cfd_init();  // Safe - only initializes once
simulation* sim2 = simulation_create(...);
```

### Error Reporting

Thread-local storage for error messages:
```c
// Each thread has its own error state
// Thread 1:
cfd_set_error(CFD_ERROR_INVALID, "Thread 1 error");
const char* msg = cfd_get_last_error();  // Returns "Thread 1 error"

// Thread 2:
cfd_set_error(CFD_ERROR_NOMEM, "Thread 2 error");
const char* msg = cfd_get_last_error();  // Returns "Thread 2 error"
```

### OpenMP Safety

```c
// Safe OpenMP usage in solvers
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
    // No shared state modifications
    result[i] = compute(input[i]);
}
```

## File Organization Best Practices

```
lib/src/<module>/
├── <module>.c           # Dispatcher (technology-agnostic, no #ifdef)
├── cpu/                 # Scalar implementations
├── simd/                # SIMD dispatch
│   ├── avx2/
│   └── neon/
├── omp/                 # OpenMP parallelized
└── gpu/                 # CUDA kernels
```

Rules:
- Dispatcher/public API goes at module root
- Technology implementations go in subfolders
- Don't put dispatcher under `cpu/` - it's technology-agnostic

## Cross-Backend Implementation Checklist

When adding a new solver feature across backends:

1. **Header**: Add enum/params to public header
2. **Default params**: Update default parameter functions
3. **Scalar**: Implement in `cpu/` (reference implementation)
4. **AVX2**: Port to `simd/avx2/` (same algorithm, SIMD primitives)
5. **NEON**: Port to `simd/neon/` (same algorithm, NEON primitives)
6. **Tests**: Verify all backends produce identical results (L2 diff < 1e-10)
7. **CMakeLists.txt**: Add test entries (executable, link, register)
8. **Documentation**: Update ROADMAP.md, README.md

## References

- Ferziger & Peric - "Computational Methods for Fluid Dynamics"
- Versteeg & Malalasekera - "An Introduction to CFD"
- Moukalled et al. - "The Finite Volume Method in CFD"
