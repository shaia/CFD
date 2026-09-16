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
- **CUDA GPU** (`projection_gpu`)

### Explicit Euler Backends
- **CPU** (`explicit_euler`)
- **AVX2/SIMD** (`explicit_euler_optimized`)
- **OpenMP** (`explicit_euler_omp`)

### Backend Consistency
- Verifies projection backends (CPU, AVX2, OMP) produce consistent results (within 0.1%)

### Reynolds Number Coverage
- **Re=100** — projection + Explicit Euler backends. CI 33×33 runs every backend including CPU-scalar; full 129×129 runs the optimized backends only (AVX2/OMP/GPU projection, AVX2/OMP Euler) — CPU-scalar is excluded per the long-validation scalar policy.
- **Re=400** — projection backends only (AVX2, OMP, GPU), full 129×129 validation
- **Re=1000** — projection backends only (AVX2, OMP, GPU), full 129×129 validation

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

Enabled with the CMake option `-DCAVITY_FULL_VALIDATION=ON`:

```text
Re=100:   Grid 129×129, 50000 steps  (250000 for Explicit Euler), dt 0.0002
Re=400:   Grid 129×129, 60000 steps  (projection only),           dt 0.0005
Re=1000:  Grid 129×129, 100000 steps (projection only),           dt 0.0005
```

Step counts are budgets: the harness stops a run early once the relative change in kinetic
energy per step drops below 1e-8 (after step 100), and each backend prints a `Steps run:` line
with the steps it actually ran.

Higher-Reynolds cases run the **projection backends only** (the production solver)
and require the finer 129×129 grid to resolve the stronger primary vortex and the
secondary corner vortices. Every projection case passes with at least a 3× margin on the
RMS target (see Full Validation Results below), so the step budgets are unchanged.

To run full validation:
```bash
cmake -DCAVITY_FULL_VALIDATION=ON -B build
cmake --build build --config Release
# Re=100 (all backends) — exclude the Re=400/1000 tests so only the base cases run
ctest --test-dir build -C Release -R "CavityBackend_" -E "_Re(400|1000)" --output-on-failure -j "$(nproc)"
# Re=400 / Re=1000 only
ctest --test-dir build -C Release -R "CavityBackend_.*_Re(400|1000)" --output-on-failure -j "$(nproc)"
```

The full suite is split into one ctest entry per (backend, Reynolds) so `ctest -j`
runs them concurrently; each carries a 4-hour timeout.

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

1. **Projection method meets scientific target** (RMS < 0.10) ✅ — at 33×33, and at 129×129 for Re=100, 400 and 1000
2. **Explicit Euler meets relaxed target** (RMS < 0.15) ✅ — but its 129×129 runs stop early (see Known Issue below)
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

Full-validation builds (`-DCAVITY_FULL_VALIDATION=ON`) replace `CavityBackendsTest` with one
ctest entry per backend and Reynolds number (`CavityBackend_*`, registered in `CMakeLists.txt`).
The Cross-Architecture Validation (EC2) workflow, `.github/workflows/gpu-validation.yml`, builds
that configuration on every push to master and on pull requests labelled `gpu-test`. It runs the
`validation` label on a g4dn.4xlarge with an NVIDIA T4 and writes a per-(Re, backend) RMS table
to the job summary. `scripts/ec2-validate.sh` runs the same `CavityBackend_*` entries on a
manually started instance.

## Full Validation Results (129×129)

All nine projection cases pass the RMS < 0.10 target with at least a 3× margin, and the AVX2,
OpenMP and CUDA backends give the same RMS to four decimals at every Reynolds number.

| Re   | Backend   | RMS_u  | RMS_v  | Pass? | Step budget × dt (t) | Wall time |
|------|-----------|--------|--------|-------|----------------------|-----------|
| 100  | AVX2/SIMD | 0.0017 | 0.0024 | Yes   | 50000 × 0.0002 (10)  | 293 s     |
| 100  | OpenMP    | 0.0017 | 0.0024 | Yes   | 50000 × 0.0002 (10)  | 428 s     |
| 100  | CUDA GPU  | 0.0017 | 0.0024 | Yes   | 50000 × 0.0002 (10)  | 516 s     |
| 400  | AVX2/SIMD | 0.0096 | 0.0328 | Yes   | 60000 × 0.0005 (30)  | 336 s     |
| 400  | OpenMP    | 0.0096 | 0.0328 | Yes   | 60000 × 0.0005 (30)  | 487 s     |
| 400  | CUDA GPU  | 0.0096 | 0.0328 | Yes   | 60000 × 0.0005 (30)  | 481 s     |
| 1000 | AVX2/SIMD | 0.0299 | 0.0300 | Yes   | 100000 × 0.0005 (50) | 516 s     |
| 1000 | OpenMP    | 0.0299 | 0.0300 | Yes   | 100000 × 0.0005 (50) | 801 s     |
| 1000 | CUDA GPU  | 0.0299 | 0.0300 | Yes   | 100000 × 0.0005 (50) | 805 s     |

**Source:** GitHub Actions run 34853615452 of the Cross-Architecture Validation (EC2) workflow,
commit `6e1b104` (2026-09-14). Runs 34815324268 (`23b196a`) and 30116961115 (2026-07-24, #201)
produced the same RMS values. Configuration: g4dn.4xlarge (16 vCPUs, NVIDIA T4), gcc Release
with AVX2 and CUDA enabled, `ctest -L validation -j 4` with `OMP_NUM_THREADS=4`. Wall times come
from that shared run, with several tests running at once, so they are not performance
benchmarks. These runs predate the `Steps run:` line, so the table lists step budgets; the next
workflow run reports the steps each case ran.

> If a later change pushes a case above RMS 0.10, increase its step budget (more physical
> time) before relaxing the tolerance — the target is a scientific standard, not a tunable.

### Known Issue: Explicit Euler at 129×129

The Explicit Euler cases are not evidence of 129×129 accuracy. Both solvers cap dt at 1e-4, and
the harness's kinetic-energy exit (relative change per step below 1e-8) ends the runs long before
their 250,000-step budget:

| Backend   | Steps run | Simulated time | KE residual | RMS_u  | RMS_v  |
|-----------|-----------|----------------|-------------|--------|--------|
| AVX2/SIMD | 11,323    | ≈ 1.1          | 8.3e-9      | 0.0957 | 0.1293 |
| OpenMP    | 11,775    | ≈ 1.2          | 9.8e-9      | 0.0957 | 0.1277 |

Measured on a local Windows (MSVC Release) build with 4 OpenMP threads; the RMS values match the
EC2 runs. The 33×33 CI case stops the same way: the scalar Euler solver ends at 11,300 of 25,000
steps with RMS_u 0.0957 and RMS_v 0.1284. The flow needs about 10–20 time units to reach steady
state at Re=100 (see [lid-driven-cavity.md](lid-driven-cavity.md)), so these runs pass the 0.15
target without a developed solution on either grid. The test logic is unchanged; the follow-up is
tracked in ROADMAP §6.1.

## Next Steps

### Immediate (for v1.0)
1. ROADMAP §6.1 cross-architecture consistency stays open: `CavityBackend_Consistency` compares
   CPU, AVX2 and OpenMP center values at 33×33 without the GPU, and matching RMS values at
   129×129 are not a field-level 0.1% comparison.
2. Make the Explicit Euler 129×129 cases run to a developed flow, or drop them from full
   validation (see Known Issue above).

### Future Enhancements
1. Add grid convergence study (33→65→129→257)
2. Add backend performance comparison
3. Add transient accuracy metrics (not just steady-state)

## References

- **Test file:** `tests/validation/test_cavity_backends.c`
- **Common utilities:** `tests/validation/lid_driven_cavity_common.h`
- **Reference data:** `tests/validation/cavity_reference_data.h`
- **Ghia et al. (1982):** "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method"
