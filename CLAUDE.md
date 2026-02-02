# CFD Library Architecture Principles

## Project Overview

**CFD Framework** is a production-grade computational fluid dynamics library in pure C for solving 2D incompressible Navier-Stokes equations.

### Core Capabilities

- **Numerical Methods**: Projection method (Chorin's algorithm) with pressure-Poisson correction; Explicit Euler for simple cases
- **Linear Solvers**: Jacobi, SOR, Red-Black SOR, Conjugate Gradient (CG)
- **Boundary Conditions**: Dirichlet, Neumann, Periodic, No-slip walls, Inlet (uniform/parabolic/custom), Outlet (zero-gradient/convective)
- **Grid**: 2D structured grids with uniform or stretched (tanh) spacing

### Performance Backends

- **Scalar**: Portable C baseline (always available)
- **SIMD**: AVX2 (x86-64) and NEON (ARM64) with runtime CPU detection
- **OpenMP**: Multi-threaded parallelization
- **CUDA**: GPU acceleration (in progress)

### Output & Visualization

- VTK format for ParaView/VisIt
- CSV timeseries, centerline profiles, statistics

### Current Status

Version 0.1.x — Core architecture complete, approaching v1.0. Validated against Ghia lid-driven cavity and Taylor-Green vortex benchmarks.

## Backend Abstraction

- Public API dispatchers must be technology-agnostic (no `#ifdef` for SIMD/OpenMP/CUDA)
- Use function pointer tables (`backend_impl_t`) for backend abstraction
- Each backend provides its own impl table with NULL for unavailable functions
- Compile-time guards (`#ifdef`) belong in backend files only, not in dispatchers

## Folder Organization

- Dispatcher/public API goes at module root (e.g., `boundary/boundary_conditions.c`)
- Technology implementations go in subfolders (`cpu/`, `simd/`, `omp/`, `gpu/`)
- Don't put dispatcher under `cpu/` - it's technology-agnostic, not CPU-specific

## Error Handling

- Return `cfd_status_t`, not silent fallbacks
- Let caller decide how to handle `CFD_ERROR_UNSUPPORTED`
- No redundant `#else` error branches in internal functions - if developer requests SIMD/GPU, they know their target supports it

## Fixing Error

- when fixing an error make sure you update or create new workflow that prevent such family of bugs to ever return

## Code Style

- Avoid backwards-compatibility hacks (unused `_vars`, re-exports, `// removed` comments)
- If something is unused, delete it completely
- Before creating new helper functions, check for existing implementations in shared headers (e.g., `test_solver_helpers.h`)
- Before defining new macros (e.g., `M_PI`, constants), check if they are already defined elsewhere in the codebase or standard headers
- Refactor to use existing code instead of duplicating functionality

## Git Workflow

- Create a new commit for each change; do not use `--amend`
- Do not add `Co-Authored-By` lines to commit messages

## Testing Conventions

### Test Framework

- Use Unity test framework (included via FetchContent)
- All tests require `setUp()` and `tearDown()` functions (even if empty)
- Test functions must start with `test_`

### Test File Structure

```c
#include "unity.h"
#include "cfd/module/api.h"
#include <math.h>

/* Parameters as macros at top */
#define DOMAIN_XMIN 0.0
#define TOLERANCE 1e-6

/* Helper functions (static) */
static double* create_field(size_t nx, size_t ny) { ... }

void setUp(void) {}
void tearDown(void) {}

void test_feature(void) {
    // Arrange, Act, Assert pattern
    TEST_ASSERT_*(...);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_feature);
    return UNITY_END();
}
```

### Manufactured Solutions

For accuracy testing, use analytical solutions where the Laplacian is known:

- `p = sin(πx)sin(πy)` → `∇²p = -2π²p` (Dirichlet-compatible, zero on boundaries)
- `p = cos(2πx)cos(2πy)` → integrates to zero (Neumann-compatible)

### Convergence Verification

**Formula:** `rate = log(e_coarse/e_fine) / log(h_coarse/h_fine)`

**Rules for convergence studies:**

1. **Use actual grid spacing in ratios** - Grid spacing h = domain/(n-1), not domain/n. Ensure all ratios use the same quantity (e.g., h, not mixing n and h).

2. **Ensure consistent comparison conditions** - When comparing errors across refinement levels, all other parameters (final time, physical conditions) must be identical. Use `round()` for step counts to avoid truncation drift.

3. **Verify monotonic error decrease** - Always assert that error decreases with refinement, independent of rate calculation. If error doesn't decrease, the test should fail regardless of computed rate.

4. **Account for limiting factors** - Theoretical rates may not be achieved due to:
   - Lower-order boundary conditions limiting overall order
   - Coupling between spatial and temporal errors
   - One error type dominating another (e.g., spatial masking temporal)

5. **Set realistic thresholds** - For second-order schemes limited by first-order BCs, expect ~1.5 instead of 2.0. Document actual vs theoretical rates.

## Solver Architecture

### Boundary Conditions

- Poisson solvers use **Neumann BCs** internally (∂p/∂n = 0)
- For Dirichlet BC tests, manually apply BCs after each solver iteration
- Neumann-compatible RHS must integrate to zero over domain

### Theoretical Convergence Properties

- Jacobi spectral radius: ρ = cos(πh)
- SOR optimal ω (Dirichlet): ω = 2/(1 + sin(πh))
- SOR optimal ω (Neumann): typically 1.5-1.7
- CG iteration bound: O(√κ) where κ ≈ 4(n-1)²/π²

## ROADMAP Workflow

When completing roadmap items:

1. Create feature branch: `feat/<section-name>`
2. Implement and test
3. Mark items with `[x]` and add ✅ to section header
4. Add "Files created:" section with file paths
5. Commit with message: `feat: <description>`
6. Create PR referencing roadmap section

### Priority System

| Priority | Meaning |
|----------|---------|
| P0 | Critical - blocks v1.0 release |
| P1 | Important - required for v1.0 |
| P2 | Valuable - nice to have for v1.0 |
| P3 | Future - post v1.0 |

## File Organization

### Library Structure

```
lib/
├── include/cfd/<module>/    # Public headers
│   └── <module>.h
└── src/<module>/            # Implementation
    ├── <module>.c           # Dispatcher (technology-agnostic)
    ├── cpu/                 # Scalar implementations
    ├── simd/                # AVX2/NEON dispatch
    │   ├── avx2/
    │   └── neon/
    ├── omp/                 # OpenMP parallelized
    └── gpu/                 # CUDA kernels
```

### Test Organization

```
tests/
├── core/       # Grid, memory, error handling, BCs
├── math/       # Stencils, Poisson, Laplacian, convergence
├── validation/ # Physics benchmarks (Ghia, Taylor-Green)
├── solvers/    # Backend-specific solver tests
├── simulation/ # Full simulation API tests
└── io/         # VTK, CSV output tests
```

## Validation Benchmarks

### Reference Solutions

- **Lid-driven cavity:** Ghia et al. (1982) - target RMS < 0.10
- **Taylor-Green vortex:** Analytical decay exp(-2νt) - target 1% error
- **Poiseuille flow:** Analytical parabolic profile

### CI vs Release Parameters

CI uses reduced grids (17×17, 500 steps) for fast feedback.
Release uses full grids (129×129, 50000 steps) for accuracy.
