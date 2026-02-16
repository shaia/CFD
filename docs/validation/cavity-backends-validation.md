# Lid-Driven Cavity Backend Validation

## Overview

This document describes the comprehensive backend validation system for the lid-driven cavity benchmark, which systematically tests all solver backends against the Ghia et al. (1982) reference data.

## Test File

**Location:** `tests/validation/test_cavity_backends.c`

**Purpose:** Validate that all CFD solver backends independently achieve acceptable accuracy on the lid-driven cavity benchmark.

## Test Coverage

### Projection Method Backends
- **CPU Scalar** (`projection`)
- **AVX2/SIMD** (`projection_optimized`)
- **OpenMP** (`projection_omp`)
- **CUDA GPU** (`projection_jacobi_gpu`)

### Explicit Euler Backends
- **CPU** (`explicit_euler`)
- **AVX2/SIMD** (`explicit_euler_optimized`)
- **OpenMP** (`explicit_euler_omp`)

### Backend Consistency
- Verifies projection backends (CPU, AVX2, OMP) produce consistent results (within 0.1%)

## Accuracy Targets

The test enforces different accuracy targets based on solver sophistication:

| Solver Type | RMS Target | Justification |
|-------------|------------|---------------|
| Projection Method | < 0.10 | Production solver, strict target |
| Explicit Euler | < 0.15 | Simpler method, relaxed target |

### Why Different Targets?

**Projection Method:**
- Uses fractional step method with pressure correction
- Solves pressure Poisson equation at each step
- Expected to achieve high accuracy (RMS < 0.10)
- This is the **production solver** used in real applications

**Explicit Euler:**
- Simpler explicit time integration
- No pressure correction
- Less accurate than projection method
- Primarily for comparison and simple test cases
- Accepts slightly higher RMS (< 0.15)

## Test Strategy

### 1. Individual Backend Validation

For each backend:
1. Run cavity simulation at 33×33 (CI) or 129×129 (full validation)
2. Extract centerline velocity profiles
3. Compute RMS error vs Ghia reference data
4. **FAIL if RMS >= target** (no "baseline" workarounds)

### 2. Backend Consistency Check

Compare projection backends (CPU Scalar, AVX2, OpenMP):
- Extract center point values (u, v) at 33×33 grid
- Verify all available projection backends within 0.1% of each other
- Ensures SIMD/OMP optimizations preserve numerical correctness

## Grid Sizes

### CI Mode (Default)
```c
Grid: 33×33
Steps: 5000 (25000 for Explicit Euler)
dt: 0.0005
```

### Full Validation Mode
```c
#define CAVITY_FULL_VALIDATION 1

Grid: 129×129
Steps: 50000 (250000 for Explicit Euler)
dt: 0.0002
```

To run full validation:
```bash
cmake -DCAVITY_FULL_VALIDATION=1 -B build
cmake --build build --config Release
ctest --test-dir build -C Release -R CavityBackendsTest
```

## Current Results (CI Mode, 33×33)

### ✅ Projection Method (PASS)
```
Projection (CPU Scalar):   RMS_u=0.0382  RMS_v=0.0440  < 0.10 ✅
Projection (OpenMP):       RMS_u=0.0382  RMS_v=0.0440  < 0.10 ✅
```

### ✅ Explicit Euler (PASS with relaxed target)
```
Explicit Euler (CPU):      RMS_u=0.0957  RMS_v=0.1284  < 0.15 ✅
Explicit Euler (OpenMP):   RMS_u=0.0957  RMS_v=0.1284  < 0.15 ✅
```

### ⏭️ Skipped (Not Compiled)
```
Projection (AVX2/SIMD):    SIMD disabled in this build
Explicit Euler (AVX2):     SIMD disabled in this build
Projection (CUDA GPU):     CUDA not available
```

### ✅ Backend Consistency (PASS)
```
All available backends produce consistent results (within 0.1%)
```

## Key Findings

1. **Projection method meets scientific target** (RMS < 0.10) ✅
2. **Explicit Euler meets relaxed target** (RMS < 0.15) ✅
3. **CPU and OpenMP backends are consistent** ✅
4. **Test correctly fails when RMS >= target** ✅

## Test Honesty

This test implementation follows the ROADMAP requirement:

> **Verification that tests are honest:**
> - Tests MUST fail if RMS > target (no loose tolerances)
> - Tests compare computed values at EXACT Ghia sample points
> - Tests report actual vs expected values transparently
> - No "current baseline" workarounds - fix solver, not tolerance

The test **will fail** if:
- Projection method RMS >= 0.10
- Explicit Euler RMS >= 0.15
- Backend consistency > 0.1%
- Simulation blows up or produces NaN values

## Integration with CI/CD

This test is labeled as "validation" and runs in CI:

```cmake
add_test(NAME CavityBackendsTest COMMAND test_cavity_backends)
set_tests_properties(CavityBackendsTest PROPERTIES LABELS "validation")
```

To run only validation tests:
```bash
ctest -L validation
```

To exclude validation tests (e.g., sanitizer builds):
```bash
ctest -LE validation
```

## Next Steps

### Immediate (for v1.0)
1. Run full 129×129 validation (release mode)
2. Test with AVX2 enabled build
3. Test with CUDA GPU (if available)
4. Verify all backends still pass at 129×129

### Future Enhancements
1. Add Re=400 and Re=1000 validation
2. Add grid convergence study (33→65→129→257)
3. Add backend performance comparison
4. Add transient accuracy metrics (not just steady-state)

## References

- **Test file:** `tests/validation/test_cavity_backends.c`
- **Common utilities:** `tests/validation/lid_driven_cavity_common.h`
- **Reference data:** `tests/validation/cavity_reference_data.h`
- **Ghia et al. (1982):** "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method"
