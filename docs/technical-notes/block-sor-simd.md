# SIMD SOR: Vectorizing Successive Over-Relaxation

## Problem Statement

The SOR (Successive Over-Relaxation) method uses Gauss-Seidel iteration with in-place updates. When computing `x[i,j]`, the left neighbor `x[i-1,j]` has already been updated in the current sweep:

```c
for (j = 1; j < ny - 1; j++) {
    for (i = 1; i < nx - 1; i++) {
        double p_new = -(rhs[idx]
            - (x[idx + 1] + x[idx - 1]) / dx2    // x[idx-1] already updated
            - (x[idx + nx] + x[idx - nx]) / dy2   // x[idx-nx] already updated
            ) * inv_factor;
        x[idx] = x[idx] + omega * (p_new - x[idx]);
    }
}
```

This creates a read-after-write dependency chain along the i-axis:

```
x[1] → x[2] → x[3] → x[4] → ...
```

AVX2 processes 4 doubles simultaneously, but the value at index 2 needs the result from index 1, the value at index 3 needs the result from index 2, and so on. This **read-after-write hazard** prevents naive SIMD vectorization of the whole update.

## Two Passes Per Row

Only one term of the stencil depends on the sweep in progress: the left neighbor. Every other term is fixed for the whole row:

| Term | Written during this row's sweep? |
|------|----------------------------------|
| `x[idx + 1]`, the right neighbor | No: it is updated after `x[idx]` |
| `x[idx - nx]`, the row below | No: that row was finished before this one started |
| `x[idx + nx]`, the row above | No: it is swept after this row |
| `x[idx ± stride_z]`, the planes either side (3D) | No |
| `rhs[idx]` | No |
| `x[idx - 1]`, the left neighbor | **Yes** |

So each row is swept in two passes:

```c
/* Pass 1, SIMD_WIDTH cells at a time (4 for AVX2, 2 for NEON) */
partial[i] = x[idx + 1] * inv_dx2
           + (x[idx + nx] + x[idx - nx]) * inv_dy2
           + (x[idx + stride_z] + x[idx - stride_z]) * inv_dz2
           - rhs[idx];

/* Pass 2, in order */
for (i = 1; i < nx - 1; i++) {
    double p_new = (partial[i] + x[idx - 1] * inv_dx2) * inv_factor;
    x[idx] = x[idx] + w * (p_new - x[idx]);   /* w: omega, or the wall factor at a row's ends */
}
```

Pass 1 reads exactly what the sequential sweep would read, so the result is the scalar SOR iteration. The only difference is rounding: the scalar kernel divides `(x[idx + 1] + x[idx - 1])` by `dx2`, while pass 2 adds `x[idx - 1] * inv_dx2` to a sum that already holds `x[idx + 1] * inv_dx2`. `test_sor_simd_matches_scalar_sweep_for_sweep` holds 40 sweeps of the two within 1e-10 of each other, and `test_sor_simd_converges_like_scalar` holds their sweep counts within one.

Because the iteration is SOR, the automatic omega applies to it, including the wall factor beside zero-gradient walls (see the SOR section of `docs/reference/solvers.md`). A per-row scratch buffer of `nx` doubles holds pass 1's output; it is allocated at init.

## Why Not Block SOR

Until September 2026 these solvers used a Block SOR: the whole update, left neighbor included, was computed SIMD_WIDTH cells at a time. Within a block, the left neighbor of every cell after the first came from the previous sweep:

```
Block SOR (SIMD_WIDTH=4):
  ┌─ SIMD block 0 ──────────────────────────┐
  │ x[1]ⁿ⁺¹ ← uses x[0]ⁿ⁺¹ (BC, fresh)     │
  │ x[2]ⁿ⁺¹ ← uses x[1]ⁿ   (stale)         │
  │ x[3]ⁿ⁺¹ ← uses x[2]ⁿ   (stale)         │
  │ x[4]ⁿ⁺¹ ← uses x[3]ⁿ   (stale)         │
  └──────────────────────────────────────────┘
  ┌─ SIMD block 1 ──────────────────────────┐
  │ x[5]ⁿ⁺¹ ← uses x[4]ⁿ⁺¹ (fresh)         │
  │ x[6]ⁿ⁺¹ ← uses x[5]ⁿ   (stale)         │
  │ ...                                      │
  └──────────────────────────────────────────┘
```

That is a different iteration from SOR, not an approximation of it, and it does not converge for every omega between 0 and 2. Measured on the AVX2 build at b9f31e4 (MSVC Debug, `examples/sor_omega_sweep.c`, seeded-noise right-hand side with zero mean, tolerance 1e-6 relative, omega from 1.00 in steps of 0.05):

| Grid | Walls | Largest omega that converged | Sweeps there | Diverged from | Scalar SOR at its automatic omega |
|------|-------|------------------------------|--------------|---------------|-----------------------------------|
| 17×17 | zero-gradient | 1.45 | 1,154 | 1.50 | 150 |
| 17×17 | Dirichlet | 1.40 | 233 | 1.45 | 45 |
| 33×33 | zero-gradient | 1.40 | 1,783 | 1.45 | 380 |
| 33×33 | Dirichlet | 1.40 | 869 | 1.45 | 90 |
| 65×65 | zero-gradient | 1.40 | 6,570 | 1.45 | 751 |
| 65×65 | Dirichlet | 1.40 | 2,404 | 1.45 | 186 |

Every automatic omega on those grids lies above the point where Block SOR diverged, so at its default setting the solver always diverged. It reported convergence anyway: once the field overflowed, `poisson_solver_compute_residual()` skipped the NaN residuals and returned zero. The shared solve loop now stops with `POISSON_DIVERGED` instead.

## Why Not OpenMP on Rows

SOR rows remain **sequential**. Row `j` depends on row `j-1` being fully swept. Applying `#pragma omp parallel for` across rows would break the y-direction Gauss-Seidel dependency and turn the iteration into something else again.

For thread-level parallelism with SOR convergence, use Red-Black SOR (`POISSON_METHOD_REDBLACK_SOR`), which decomposes the grid into independent color sweeps.

## Performance Characteristics

Pass 1 vectorizes seven loads, the stencil arithmetic and a store for every block; pass 2 is a scalar recurrence of one load, three multiply-adds and a store per cell. Per-sweep throughput against the scalar solver has not been benchmarked since the change, so no speedup is claimed here. The number of sweeps is the scalar solver's.

## Implementation Notes

The SIMD SOR implementations live in:
- `lib/src/solvers/linear/avx2/linear_solver_sor_avx2.c` — AVX2 (4-wide)
- `lib/src/solvers/linear/neon/linear_solver_sor_neon.c` — NEON (2-wide)

Key implementation details:
- The context holds the SIMD constants (`dx2_inv_vec`, `dy2_inv_vec`, `dz2_inv_vec`) and the `partial` row buffer
- Pass 1 has a scalar remainder for rows whose interior width is not a multiple of SIMD_WIDTH
- In 2D, `stride_z` is 0 and `inv_dz2` is 0, so the z terms vanish
- Boundary conditions are applied after each full sweep via `poisson_solver_apply_bc()`

## References

- Y. Saad, "Iterative Methods for Sparse Linear Systems", 2nd edition, SIAM, 2003 — Chapter 4 covers relaxation methods and their convergence properties
- D. M. Young, "Iterative Solution of Large Linear Systems", Academic Press, 1971 — consistent orderings and the optimal SOR omega
