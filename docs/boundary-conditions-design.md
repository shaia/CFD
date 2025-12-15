# Boundary Condition Abstraction Layer Design

## Overview

This document describes the design for a unified boundary condition abstraction layer that eliminates code duplication across CPU, SIMD, OMP, and GPU solvers while supporting multiple BC types.

## Current State

### Existing BC Implementations (Duplicated)

| Location | Function | BC Type | Operates On |
|----------|----------|---------|-------------|
| `solver_explicit_euler.c:189-223` | `apply_boundary_conditions()` | Periodic | `flow_field*` |
| `solver_projection_simd.c:129-144` | `apply_velocity_bc()` | Neumann | Raw `double*` arrays |
| `solver_projection.c:215-228` | Inline code | Neumann | Raw `double*` arrays |
| `solver_projection_omp.c:89-99` | Inline code | Neumann | Raw `double*` arrays |
| `poisson_common_simd.c:17-28` | `poisson_apply_bc()` | Neumann | Raw `double*` array |
| `solver_projection_jacobi_gpu.cu:75-101` | CUDA kernels | Neumann | Device `double*` |

### BC Types Currently Used

1. **Periodic** - Final velocity after projection (wraps domain)
2. **Neumann (Zero Gradient)** - Intermediate velocity, pressure (du/dn = 0)

## Proposed Architecture

### New Files

```text
lib/include/cfd/core/boundary_conditions.h  (public header)
lib/src/core/boundary_conditions.c          (CPU implementation)
```

### BC Type Enum

```c
typedef enum {
    BC_TYPE_PERIODIC,       // Wrap-around: left=right-1, right=left+1
    BC_TYPE_NEUMANN,        // Zero gradient: boundary = interior neighbor
    BC_TYPE_DIRICHLET,      // Fixed value (future)
    BC_TYPE_NOSLIP,         // No-slip wall: u=v=0 (future)
    BC_TYPE_INLET,          // Inlet velocity (future)
    BC_TYPE_OUTLET          // Outlet/convective (future)
} bc_type_t;
```

### Unified API

```c
// Apply BC to a single scalar field (raw array)
void bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type);

// Apply BC to velocity pair (u, v arrays)
void bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type);

// Apply BC to entire flow_field struct
void bc_apply_flow_field(flow_field* field, const grid* grid, bc_type_t type);

// Convenience wrappers for common cases
#define bc_apply_neumann(f, nx, ny)  bc_apply_scalar(f, nx, ny, BC_TYPE_NEUMANN)
#define bc_apply_periodic(f, nx, ny) bc_apply_scalar(f, nx, ny, BC_TYPE_PERIODIC)
```

## Implementation Steps

### Step 1: Create Header File

**File:** `lib/include/cfd/core/boundary_conditions.h`

- Define `bc_type_t` enum
- Declare core BC functions with `CFD_LIBRARY_EXPORT`
- Include convenience macros

### Step 2: Create Implementation File

**File:** `lib/src/core/boundary_conditions.c`

- Implement `bc_apply_scalar()` with switch on BC type
- Implement `bc_apply_velocity()` calling scalar version twice
- Implement `bc_apply_flow_field()` for all flow variables
- Keep existing `apply_boundary_conditions()` as backward-compatible wrapper

### Step 3: Update CMakeLists.txt

**File:** `lib/CMakeLists.txt`

- Add `src/core/boundary_conditions.c` to CFD_CORE_SOURCES

### Step 4: Refactor Solvers (One at a Time)

#### 4a. Update `solver_projection_simd.c`

- Replace `apply_velocity_bc()` with `bc_apply_velocity(u, v, nx, ny, BC_TYPE_NEUMANN)`
- Remove the static function definition

#### 4b. Update `solver_projection.c`

- Replace inline BC loops with `bc_apply_velocity()` and `bc_apply_neumann()`

#### 4c. Update `solver_projection_omp.c`

- Replace parallelized BC loops with unified calls
- Note: OMP version may need OMP-aware BC functions (future enhancement)

#### 4d. Update `poisson_common_simd.c`

- Replace `poisson_apply_bc()` with `bc_apply_neumann()`
- Or keep as thin wrapper for backward compatibility

### Step 5: Update `apply_boundary_conditions()`

**File:** `solver_explicit_euler.c`

- Refactor to call `bc_apply_flow_field(field, grid, BC_TYPE_PERIODIC)`
- Maintains backward compatibility

### Step 6: Run Tests

- All existing tests should pass unchanged
- Verify no performance regression

## Key Design Decisions

1. **Raw Arrays First**: Core functions operate on `double*` arrays, not `flow_field*`
   - Allows use in Poisson solvers and intermediate velocity
   - `flow_field` wrapper calls raw array version

2. **BC Type as Parameter**: Runtime selection via enum
   - Enables future support for mixed BCs per boundary
   - Small overhead acceptable for flexibility

3. **Backward Compatibility**: Keep `apply_boundary_conditions()` working
   - Existing code doesn't break
   - Can migrate incrementally

4. **GPU Deferred**: CUDA kernels remain separate for now
   - Different memory model (device vs host)
   - Can add `bc_apply_*_gpu()` variants later

## Files to Modify

| File | Change |
|------|--------|
| `lib/include/cfd/core/boundary_conditions.h` | **NEW** - BC type enum and function declarations |
| `lib/src/core/boundary_conditions.c` | **NEW** - BC implementations |
| `lib/CMakeLists.txt` | Add new source file |
| `lib/src/solvers/simd/solver_projection_simd.c` | Remove `apply_velocity_bc()`, use new API |
| `lib/src/solvers/cpu/solver_projection.c` | Replace inline BC code |
| `lib/src/solvers/omp/solver_projection_omp.c` | Replace inline BC code |
| `lib/src/solvers/simd/poisson_common_simd.c` | Replace `poisson_apply_bc()` |
| `lib/src/solvers/cpu/solver_explicit_euler.c` | Refactor `apply_boundary_conditions()` |

## Testing Strategy

1. Run all existing solver tests after each step
2. No new tests needed initially (existing tests cover BC correctness)
3. Future: Add dedicated BC unit tests for each type

## Estimated Scope

- **New code**: ~150 lines (header + implementation)
- **Refactored code**: ~100 lines removed, ~50 lines updated
- **Risk**: Low - all changes are mechanical refactors with existing test coverage
