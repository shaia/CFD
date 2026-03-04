# Math Subsystem Test Plan

## Overview

The math subsystem covers linear solvers, Poisson solvers, finite difference stencils, and manufactured solution (MMS) verification. This document tracks current coverage, identified gaps, and proposed new tests.

## Current Coverage

| File | Tests | Covers |
|------|-------|--------|
| `test_bicgstab.c` | 9 | BiCGSTAB scalar: create, zero/sinusoidal/Dirichlet RHS, CG comparison, error handling |
| `test_bicgstab_avx2.c` | 1 | AVX2 vs scalar consistency (L2 < 5e-9) |
| `test_bicgstab_neon.c` | 1 | NEON vs scalar consistency |
| `test_convergence_order.c` | 4 | O(h²) spatial, O(dt) temporal for Euler and Projection |
| `test_divergence_free.c` | 11 | Projection ∇·u constraint across CPU/AVX2/OMP/GPU |
| `test_finite_differences.c` | 9 | 2D: df/dx, df/dy, d²f/dx², d²f/dy², Laplacian, divergence, gradient |
| `test_laplacian_accuracy.c` | 4 | Direct stencil, CG backend comparison, residual convergence, symmetry |
| `test_linear_solver_convergence.c` | 6 | Jacobi spectral radius, SOR omega, SOR vs Jacobi speedup, CG bound |
| `test_mms.c` | 5 | MMS source callback, Euler/RK2 spatial + temporal convergence |
| `test_pcg_convergence.c` | 4 | PCG vs CG, iteration count, disabled preconditioner, SIMD |
| `test_poisson_3d.c` | 17 | 3D accuracy (all solvers), 2D backward compat, grid convergence |
| `test_poisson_accuracy.c` | 15 | Zero/uniform/sinusoidal RHS, all solver types, convergence rates |

**Total: 12 files, ~86 test functions**

## Gap Analysis

### P0 — Critical

**Gap 1: No 3D finite difference stencil tests.** `test_finite_differences.c` is 2D only. Functions `stencil_first_deriv_z`, `stencil_second_deriv_z`, `stencil_laplacian_3d`, `stencil_divergence_3d` in `lib/include/cfd/math/stencils.h` have no accuracy tests.

**Gap 2: No solver breakdown/stagnation detection tests.** CG and BiCGSTAB have breakdown thresholds (`1e-30`) in `lib/src/solvers/linear/linear_solver_internal.h`. No test exercises these paths: near-singular systems, max-iter reached, trivial (zero) systems.

**Gap 3: No OMP vs scalar consistency tests for linear solvers.** AVX2 and NEON have consistency tests (`test_bicgstab_avx2.c`, `test_bicgstab_neon.c`). OMP CG and OMP Red-Black SOR have no 2D cross-backend comparison.

### P1 — Important

**Gap 4: No non-uniform grid testing.** All math tests use uniform grids (dx = dy). Stencil accuracy and solver convergence on rectangular domains (dx ≠ dy) are untested.

**Gap 5: No solver robustness/edge-case tests.** Missing: minimal grids (5×5), large tolerances, SOR omega near boundaries, sequential solve consistency, create/destroy resource cycles.

**Gap 6: No residual computation tests.** `poisson_solver_compute_residual()` is used internally but never directly tested with known-answer verification.

### P2 — Nice-to-have

**Gap 7: No CG iteration scaling verification.** No test tracks iteration count trend across grid sizes to verify O(√κ) scaling.

## Proposed New Tests

### `test_finite_differences_3d.c` — P0

| Test | Method | Expected |
|------|--------|----------|
| `test_first_deriv_z_accuracy` | Grid refinement 17³→33³, f=sin(kx)sin(ky)sin(kz) | Rate > 1.7 |
| `test_second_deriv_z_accuracy` | Same approach, compare d²f/dz² | Rate > 1.7 |
| `test_laplacian_3d_accuracy` | Analytical = -3k²f | Rate > 1.7 |
| `test_divergence_3d_accuracy` | Known vector field, compare to analytical div | Rate > 1.7 |
| `test_divergence_3d_free_field` | Curl of vector potential, verify max|div| < h² | max < h² |

### `test_solver_breakdown.c` — P0

| Test | Method | Expected |
|------|--------|----------|
| `test_cg_incompatible_neumann` | Non-zero interior-sum RHS, tight tol | `POISSON_MAX_ITER` or `POISSON_STAGNATED` |
| `test_bicgstab_trivial_system` | x=0, rhs=0 | Converges in 0–1 iterations |
| `test_cg_trivial_system` | x=0, rhs=0 | Converges in 0–1 iterations |
| `test_bicgstab_max_iter` | tol=1e-15, max_iter=5 | `CFD_ERROR_MAX_ITER` |
| `test_cg_max_iter` | tol=1e-15, max_iter=5 | `CFD_ERROR_MAX_ITER` |

### `test_omp_consistency.c` — P0

| Test | Method | Expected |
|------|--------|----------|
| `test_cg_omp_vs_scalar` | Sinusoidal-RHS 33×33, both backends | L2 diff < 1e-9 |
| `test_redblack_omp_vs_scalar` | Same problem | L2 diff < 1e-6 |

### `test_solver_robustness.c` — P1

| Test | Method | Expected |
|------|--------|----------|
| `test_minimal_grid_5x5` | 5×5 grid, Jacobi + CG | L2 error < 0.1 |
| `test_early_termination_large_tol` | tol=0.1, 33×33 | Iterations < 20 |
| `test_sor_omega_boundary` | omega=1.99 vs 1.0 | Both converge |
| `test_sequential_solves_consistent` | Solve twice, same instance | L2 diff < 1e-10 |
| `test_solver_create_destroy_cycle` | 100× lifecycle loop | No crash |

### `test_residual_computation.c` — P1

| Test | Method | Expected |
|------|--------|----------|
| `test_residual_exact_solution` | Exact manufactured solution | Residual < 1e-10 |
| `test_residual_wrong_solution` | x=0, rhs≠0 | Residual > 0 |
| `test_residual_convergence_rate` | Grid refinement 17→33→65→129 | Rate > 1.7 |

### `test_nonuniform_grid.c` — P1

| Test | Method | Expected |
|------|--------|----------|
| `test_stencil_rectangular_domain` | 65×33 on [0,2π]×[0,π] | L2 error < 0.05 |
| `test_poisson_rectangular_domain` | CG on same grid | L2 error < 1e-2 |

### `test_cg_scaling.c` — P2

| Test | Method | Expected |
|------|--------|----------|
| `test_cg_sqrt_kappa_scaling` | Solve at 9/17/33/65, record iters | iter/√κ < 3.0 |
| `test_pcg_vs_cg_across_sizes` | Compare at 17/33/65/129 | PCG ≤ CG × 1.05 |

## Coverage Matrix (after implementation)

| Component | Correctness | Convergence | Backend Consistency | Breakdown | Edge Cases |
|-----------|-------------|-------------|---------------------|-----------|------------|
| Jacobi | poisson_accuracy, poisson_3d | linear_solver_conv | poisson_3d | **new** | **new** |
| SOR | poisson_accuracy | linear_solver_conv | — | — | **new** |
| Red-Black SOR | poisson_accuracy, poisson_3d | linear_solver_conv | poisson_3d, **new (OMP)** | — | — |
| CG | poisson_accuracy, pcg_conv | linear_solver_conv | laplacian_accuracy, **new (OMP)** | **new** | **new** |
| BiCGSTAB | bicgstab | — | bicgstab_avx2, bicgstab_neon | **new** | **new** |
| PCG | pcg_convergence | pcg_convergence, **new (scaling)** | pcg_convergence | — | — |
| FD stencils 2D | finite_differences, laplacian_accuracy | finite_differences | — | — | **new (rect)** |
| FD stencils 3D | **new** | **new** | — | — | — |
| Residual API | laplacian_accuracy (indirect) | — | — | — | **new** |
| MMS | mms | mms | — | — | — |

**Legend:** Named = existing test file, **new** = proposed in this plan, — = not applicable
