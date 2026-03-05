# Block SOR: SIMD Vectorization of Successive Over-Relaxation

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

AVX2 processes 4 doubles simultaneously, but value at index 2 needs the result from index 1, value at index 3 needs the result from index 2, etc. This **read-after-write hazard** prevents naive SIMD vectorization.

## Block SOR Technique

Instead of processing cells one at a time, process SIMD_WIDTH consecutive cells as a single block:

- **AVX2**: blocks of 4 cells (`__m256d`, 256-bit)
- **NEON**: blocks of 2 cells (`float64x2_t`, 128-bit)

### Dependency Analysis

For AVX2 with block size 4, processing cells `[4k+1, 4k+2, 4k+3, 4k+4]`:

**Between blocks (satisfied):**
- Cell `4k+1` reads left neighbor `x[4k]` from the previous block
- That value IS already updated and stored back to memory
- The inter-block Gauss-Seidel dependency is fully preserved

**Within a block (approximated):**
- Cell `4k+2` reads `x[4k+1]` — NOT yet updated (same SIMD operation)
- Cell `4k+3` reads `x[4k+2]` — NOT yet updated
- Cell `4k+4` reads `x[4k+3]` — NOT yet updated
- These intra-block left-neighbor reads use stale (previous-iteration) values

**Y-direction (satisfied):**
- Row `j` is processed only after row `j-1` is completely swept
- The bottom neighbor `x[i,j-1]` is always the correctly updated value
- No approximation in the y-direction

### Visualization

```
Row j, iteration n:

Scalar SOR (fully sequential):
  x[1]ⁿ⁺¹ ← uses x[0]ⁿ⁺¹ (BC)
  x[2]ⁿ⁺¹ ← uses x[1]ⁿ⁺¹ (fresh!)
  x[3]ⁿ⁺¹ ← uses x[2]ⁿ⁺¹ (fresh!)
  x[4]ⁿ⁺¹ ← uses x[3]ⁿ⁺¹ (fresh!)
  x[5]ⁿ⁺¹ ← uses x[4]ⁿ⁺¹ (fresh!)
  ...

Block SOR (SIMD_WIDTH=4):
  ┌─ SIMD block 0 ──────────────────────────┐
  │ x[1]ⁿ⁺¹ ← uses x[0]ⁿ⁺¹ (BC, fresh!)   │
  │ x[2]ⁿ⁺¹ ← uses x[1]ⁿ   (stale)        │
  │ x[3]ⁿ⁺¹ ← uses x[2]ⁿ   (stale)        │
  │ x[4]ⁿ⁺¹ ← uses x[3]ⁿ   (stale)        │
  └──────────────────────────────────────────┘
  ┌─ SIMD block 1 ──────────────────────────┐
  │ x[5]ⁿ⁺¹ ← uses x[4]ⁿ⁺¹ (fresh!)       │
  │ x[6]ⁿ⁺¹ ← uses x[5]ⁿ   (stale)        │
  │ x[7]ⁿ⁺¹ ← uses x[6]ⁿ   (stale)        │
  │ x[8]ⁿ⁺¹ ← uses x[7]ⁿ   (stale)        │
  └──────────────────────────────────────────┘
```

## Convergence Impact

Block SOR behaves as a hybrid between Jacobi (all old values) and Gauss-Seidel (all fresh values):

| Property | Jacobi | Block SOR (AVX2) | Block SOR (NEON) | Scalar SOR |
|----------|--------|------------------|------------------|------------|
| Left-neighbor freshness | 0% fresh | 25% fresh (1/4) | 50% fresh (1/2) | 100% fresh |
| Bottom-neighbor freshness | 0% fresh | 100% fresh | 100% fresh | 100% fresh |
| Convergence per iteration | Slowest | Moderate | Moderate-Fast | Fastest |
| Parallelizability | Full | SIMD only | SIMD only | None |

**In practice:**
- Convergence rate is slightly slower than scalar SOR but significantly faster than Jacobi
- The effect diminishes on larger grids (more inter-block steps relative to intra-block)
- For NEON (width=2), the approximation is milder — only 1 of 2 left-neighbor reads is stale
- May require ~10-20% more iterations than scalar SOR to reach the same residual tolerance

## Why Not OpenMP on Rows

Unlike Jacobi or Red-Black SOR, Block SOR rows remain **sequential**. Row `j` depends on row `j-1` being fully swept. Applying `#pragma omp parallel for` across rows would break the y-direction Gauss-Seidel dependency and degrade convergence to Jacobi-like behavior (or worse, introduce race conditions).

The speedup comes purely from SIMD throughput:
- AVX2: ~4x arithmetic operations per cycle (256-bit / 64-bit per double)
- NEON: ~2x arithmetic operations per cycle (128-bit / 64-bit per double)

For thread-level parallelism with SOR convergence, use Red-Black SOR (`POISSON_METHOD_REDBLACK_SOR`) which decomposes the grid into independent color sweeps.

## Performance Characteristics

| Metric | Expected Value |
|--------|---------------|
| AVX2 speedup per iteration | ~2-3x over scalar |
| NEON speedup per iteration | ~1.5x over scalar |
| Additional iterations needed | ~10-20% more than scalar |
| Net wall-clock improvement | Positive for grids > ~32x32 |
| Memory overhead | None (in-place, no temporary buffer) |

## Implementation Notes

The Block SOR SIMD implementation lives in:
- `lib/src/solvers/linear/avx2/linear_solver_sor_avx2.c` — AVX2 (4-wide)
- `lib/src/solvers/linear/neon/linear_solver_sor_neon.c` — NEON (2-wide)

Key implementation details:
- Context struct stores precomputed SIMD vectors (`omega_vec`, `dx2_inv_vec`, etc.)
- SIMD loop with scalar remainder tail for non-aligned grid widths
- SOR relaxation computed in SIMD: `result = x_center + omega * (p_new - x_center)`
- Boundary conditions applied after each full sweep via `poisson_solver_apply_bc()`

## References

- Y. Saad, "Iterative Methods for Sparse Linear Systems", 2nd edition, SIAM, 2003 — Chapter 4 covers block relaxation methods and their convergence properties
- The block/chunked Gauss-Seidel technique is standard in vectorized PDE solvers where sequential dependencies prevent element-wise SIMD parallelism
