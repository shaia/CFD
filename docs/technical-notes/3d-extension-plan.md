# 3D Extension Plan: "2D as Subset of 3D"

## Context

The CFD library is currently 2D-only (v0.1.7). This plan extends it to full 3D using the "2D as subset of 3D" approach: `nz=1` means 2D. No separate 2D/3D codepaths — all code becomes 3D-aware, and existing 2D behavior is preserved when `nz=1`.

**Why now:** 3D support is the critical P0 gap blocking v1.0. The architecture is clean enough for systematic extension (~90 files, ~200 indexing expressions) rather than a rewrite.

**Key constraint:** `nz=1` must produce bit-identical results to the current code.

---

## Key Design Decisions

### Branch-Free Solver Loops (Critical Requirement)

**Goal: Zero conditional branches in inner solver loops.** All 2D/3D differences resolved via precomputed constants, not runtime checks.

**Mechanism:** The `grid` struct stores precomputed loop bounds and coefficients set once at init time:

```c
typedef struct {
    // ... existing fields (x, y, nx, ny, etc.) ...
    size_t nz;
    double* z;     // [nz]
    double* dz;    // [nz-1]
    double zmin, zmax;

    // Precomputed for branch-free solver loops (set once at grid init)
    size_t stride_z;    // nx*ny when nz>1, 0 when nz==1
    double inv_dz2;     // 1/dz^2 when nz>1, 0.0 when nz==1
    size_t k_start;     // 1 when nz>1, 0 when nz==1
    size_t k_end;       // nz-1 when nz>1, 1 when nz==1
} grid;
```

**How it works — zero z-contribution when nz=1:**

```c
// Same code runs for 2D and 3D. No if/else anywhere.
for (size_t k = grid->k_start; k < grid->k_end; k++) {    // 1 iteration when nz=1
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);  // stride_z=0 -> same as 2D
            // 7-point stencil: z-terms vanish naturally
            double z_term = (p[idx + stride_z] + p[idx - stride_z] - 2.0 * p[idx]) * inv_dz2;
            // When stride_z=0: p[idx+0]+p[idx-0]-2*p[idx] = 0, times inv_dz2=0.0 -> 0.0
        }
    }
}
```

- `stride_z=0` -> z-neighbor loads read the same cell -> difference is zero
- `inv_dz2=0.0` -> even if the difference were nonzero, the coefficient kills it
- `k_start=0, k_end=1` -> k-loop runs exactly once with k=0
- **Result: identical computation to current 2D code, with zero branches**

**W-momentum for nz=1:** w is allocated but initialized to zero. Since all w values are 0 and all z-derivatives evaluate to 0 (stride_z=0), w-momentum computation produces w_new=0 automatically. No branch needed. The only cost is redundant loads of the zero-filled w array.

**The ONLY branches in the entire solver path (per-call, not per-point):**

1. Z-face boundary conditions: skip when nz==1 (one check per BC call, not per grid point). This is necessary because z-face Neumann BC `field[i,j,0]=field[i,j,1]` would read out-of-bounds when nz=1.

### Cache Optimization Strategy

**Memory layout:** Row-major `idx = k*nx*ny + j*nx + i` (z-slowest, x-fastest).

**Access pattern analysis for 7-point stencil:**

| Neighbor | Stride | Distance (128x128 grid) | Cache level |
| -------- | ------ | ----------------------- | ----------- |
| x+/-1 | 1 double | 8 bytes | L1 (same cache line) |
| y+/-1 | nx doubles | 1 KB | L1/L2 |
| z+/-1 | nx*ny doubles | 128 KB | L2/L3 |

**Cache-aware design rules:**

1. **Loop order k->j->i (outermost to innermost):** Walks memory sequentially. Each i-step is stride-1. This is already the natural order and must not be changed.

2. **Three-plane working set:** Processing plane k requires planes k-1, k, k+1. For 128x128: 3 x 128KB = 384KB. Fits in L2 cache (typically 256KB-1MB per core). The k->j->i loop order naturally reuses plane k as k+1's neighbor in the next iteration.

3. **OpenMP: contiguous k-slabs per thread.** Using `collapse(2)` on k,j with `schedule(static)` assigns each thread a contiguous block of (k,j) pairs. Threads working on adjacent k-planes share z-neighbor data in shared L3. Avoid `schedule(dynamic)` which would scatter k-planes across threads.

4. **Precomputed `stride_z` in grid struct.** Avoids recomputing `nx*ny` in every inner loop iteration.

5. **SIMD vectorizes on i (x-direction):** This is contiguous memory — full cache line utilization. AVX2 loads 4 consecutive doubles (32 bytes, half a cache line). Never vectorize on j or k.

6. **W-array for nz=1:** The w array (nx*ny zeros) is allocated and may be touched, causing some cache pollution for 2D problems. This is acceptable because: (a) w is the same size as u or v, adding ~20% to the working set, (b) the zeros will be read from cache after the first access, (c) the alternative (branching) is worse.

### Indexing

- Introduce `IDX_2D(i,j,nx)` and `IDX_3D(i,j,k,nx,ny)` in new `lib/include/cfd/core/indexing.h`
- In practice, solvers use `k * stride_z + IDX_2D(i, j, nx)` rather than `IDX_3D` directly, because stride_z is precomputed (avoiding a multiply)
- Phase 0 replaces all inline `j*nx+i` with `IDX_2D` first (mechanical, zero-risk)

### API Backward Compatibility

- `grid_create(nx, ny, ...)` -> wrapper for `grid_create_3d(nx, ny, 1, ...)`
- `flow_field_create(nx, ny)` -> wrapper for `flow_field_create_3d(nx, ny, 1)`
- `init_simulation(nx, ny, ...)` -> wrapper for `init_simulation_3d(nx, ny, 1, ...)`
- BC functions: old `(nx, ny)` signatures become wrappers passing `nz=1`

### Red-Black Coloring in 3D

`(i+j) % 2` -> `(i+j+k) % 2`. When `k=0`: identical to current.

### SIMD Strategy

AVX2/NEON continue vectorizing over i-direction (contiguous memory). Add z-neighbor loads (`p[idx +/- stride_z]`). 7-point stencil: 2 extra loads per point vs 5-point, but same vectorization pattern.

### OMP Strategy

`#pragma omp parallel for collapse(2) schedule(static)` on k,j loops. Static scheduling keeps k-planes contiguous per thread for L2 cache reuse.

### CUDA Strategy

3D: `dim3(16,8,4)` = 512 threads per block. Thread indexing maps directly to i,j,k. When nz=1, z-dimension of grid is 1 block with 1 thread in z.

---

## Phases

### Phase 0: Indexing Macro (3-5 days)

Replace all inline `j * nx + i` with `IDX_2D(i, j, nx)`. Purely mechanical.

**New file:**

- `lib/include/cfd/core/indexing.h` — `IDX_2D`, `IDX_3D`, `STRIDE_Z`

**Files to modify (~20 in lib/src/):**

- `lib/src/solvers/navier_stokes/cpu/solver_projection.c`
- `lib/src/solvers/navier_stokes/cpu/solver_explicit_euler.c`
- `lib/src/solvers/navier_stokes/cpu/solver_rk2.c`
- `lib/src/solvers/navier_stokes/avx2/solver_projection_avx2.c`
- `lib/src/solvers/navier_stokes/avx2/solver_explicit_euler_avx2.c`
- `lib/src/solvers/navier_stokes/avx2/solver_rk2_avx2.c`
- `lib/src/solvers/navier_stokes/omp/solver_projection_omp.c`
- `lib/src/solvers/navier_stokes/omp/solver_explicit_euler_omp.c`
- `lib/src/solvers/navier_stokes/omp/solver_rk2_omp.c`
- `lib/src/solvers/linear/cpu/linear_solver_jacobi.c`
- `lib/src/solvers/linear/cpu/linear_solver_redblack.c`
- `lib/src/solvers/linear/cpu/linear_solver_cg.c`
- `lib/src/solvers/linear/cpu/linear_solver_bicgstab.c`
- `lib/src/solvers/linear/cpu/linear_solver_sor.c`
- `lib/src/solvers/linear/avx2/linear_solver_*.c` (4 files)
- `lib/src/solvers/linear/neon/linear_solver_*.c` (4 files)
- `lib/src/solvers/linear/omp/linear_solver_*.c` (2 files)
- `lib/src/io/vtk_output.c`
- `lib/src/boundary/boundary_conditions_core_impl.h`
- Shared test headers (`test_solver_helpers.h`, `taylor_green_reference.h`, etc.)

**Done:** All tests pass. `grep -r "j \* nx + i\|j \* field->nx" lib/src/` returns zero matches.

---

### Phase 1: Core Data Structures (1-2 weeks)

Extend grid, flow_field, BC structs. No solver logic changes yet.

**Critical files to modify:**

| File | Change |
| ---- | ------ |
| `lib/include/cfd/core/grid.h` | Add `nz, z[], dz[], zmin, zmax, stride_z, inv_dz2, k_start, k_end` to `grid` struct. Add `grid_create_3d()` |
| `lib/src/core/grid.c` | Implement 3D creation/init. Precompute `stride_z`, `inv_dz2`, `k_start`, `k_end`. Old `grid_create()` becomes wrapper with `stride_z=0, inv_dz2=0.0, k_start=0, k_end=1` |
| `lib/include/cfd/solvers/navier_stokes_solver.h` | Add `w` and `nz` to `flow_field`. Add `ns_source_func_3d_t`. Add `flow_field_create_3d()` |
| `lib/src/solvers/navier_stokes/cpu/solver_explicit_euler.c` | Update `flow_field_create/destroy`, `initialize_flow_field` (contains these utilities) |
| `lib/include/cfd/boundary/boundary_conditions.h` | Add `front/back` to `bc_dirichlet_values_t`. Add `BC_EDGE_FRONT=0x10`, `BC_EDGE_BACK=0x20` to `bc_edge_t` |
| `lib/include/cfd/solvers/poisson_solver.h` | Add `nz, dz` to `poisson_solver` struct |
| `lib/include/cfd/core/derived_fields.h` | Add `w_stats`, `nz` |
| `lib/src/core/derived_fields.c` | Add `w` to velocity magnitude: `sqrt(u^2+v^2+w^2)` |
| `lib/include/cfd/api/simulation_api.h` | Add `init_simulation_3d()` |
| `lib/src/api/simulation_api.c` | Implement. Old `init_simulation()` becomes wrapper |

**New tests:** `tests/core/test_grid_3d.c`, `tests/core/test_flow_field_3d.c`

**Done:** All existing tests pass unchanged (wrappers preserve old behavior). New tests verify 3D grid/field creation.

---

### Phase 2: Stencils + Scalar CPU Linear Solvers (1-2 weeks)

Get the math right in scalar code first. All optimized backends mirror these patterns.

**Files to modify:**

| File | Change |
| ---- | ------ |
| `lib/include/cfd/math/stencils.h` | Add `stencil_laplacian_3d`, `stencil_divergence_3d`, `stencil_first_deriv_z`, `stencil_second_deriv_z` |
| `lib/src/solvers/linear/cpu/linear_solver_jacobi.c` | Triple loop, 7-point stencil, `inv_factor` includes dz |
| `lib/src/solvers/linear/cpu/linear_solver_redblack.c` | Triple loop, `(i+j+k)%2` coloring |
| `lib/src/solvers/linear/cpu/linear_solver_sor.c` | Triple loop, 7-point stencil |
| `lib/src/solvers/linear/cpu/linear_solver_cg.c` | 3D `dot_product`, `axpy`, `apply_laplacian` primitives |
| `lib/src/solvers/linear/cpu/linear_solver_bicgstab.c` | Same as CG: 3D primitives |
| `lib/src/solvers/linear/linear_solver.c` | Update dispatcher, `poisson_solver_init` with nz |
| `lib/src/solvers/linear/linear_solver_internal.h` | Update signatures |

**New/updated tests:** `tests/math/test_stencils_3d.c`, update `test_poisson_accuracy.c`, `test_laplacian_accuracy.c`, `test_linear_solver_convergence.c`

**Done:** All 6 scalar linear solvers work with both nz=1 (identical results) and nz>1. 3D Poisson converges on 16x16x16 grid.

---

### Phase 3: Scalar CPU NS Solvers (1-2 weeks)

Add w-momentum equation to all 3 CPU solvers.

**Files to modify:**

| File | Change |
| ---- | ------ |
| `lib/src/solvers/navier_stokes/cpu/solver_projection.c` | 3D predictor (w-momentum), 3D divergence (add dw/dz), 3D corrector (dp/dz -> w), allocate w_star |
| `lib/src/solvers/navier_stokes/cpu/solver_explicit_euler.c` | Triple loop, w-velocity update, dz in CFL |
| `lib/src/solvers/navier_stokes/cpu/solver_rk2.c` | 3D RHS with z-terms, w_pred/rhs_w arrays |
| `lib/src/solvers/navier_stokes/boundary_copy_utils.h` | 6-face boundary copy |
| `lib/src/api/solver_registry.c` | Solvers read `grid->nz` for dimensionality |

**New tests:** `tests/solvers/navier_stokes/cpu/test_solver_projection_3d.c`. Update existing CPU solver tests.

**Done:** All 3 CPU solvers run with nz=1 (identical) and nz>1 (8x8x8). Projection achieves divergence-free in 3D.

---

### Phase 4: Boundary Conditions — All Backends (2 weeks)

Largest phase. The BC public API has ~50 exported functions, each taking `(nx, ny)`. Internally, add nz; externally, old signatures become wrappers.

**Approach:** Update the internal `bc_backend_impl_t` function pointer table to accept nz. All 15 backend implementation files cascade from this change.

**Files to modify:**

| Category | Files | Change |
| -------- | ----- | ------ |
| Internal API | `lib/src/boundary/boundary_conditions_internal.h` | Add nz to all function pointer typedefs |
| Dispatcher | `lib/src/boundary/boundary_conditions.c` | Route 3D calls, old API wraps with nz=1 |
| Scalar CPU | `cpu/boundary_conditions_scalar.c`, `cpu/*_outlet_scalar.c`, `cpu/*_inlet_scalar.c` | Add k-loops for front/back z-faces |
| OMP | `omp/boundary_conditions_omp.c`, `omp/*_outlet_omp.c` | Same + OMP over face planes |
| AVX2 | `avx2/boundary_conditions_avx2.c`, `avx2/*_outlet_avx2.c` | SIMD on z-face planes |
| NEON | `neon/boundary_conditions_neon.c`, `neon/*_outlet_neon.c` | Same |
| Templates | `boundary_conditions_core_impl.h`, `*_simd_impl.h`, `*_outlet_simd.h` | Add z-face template code |
| SIMD dispatch | `simd/boundary_conditions_simd_dispatch.c` | Updated signatures |
| Public header | `lib/include/cfd/boundary/boundary_conditions.h` | Add `_3d` variants for key functions |

**Z-face BC semantics:**

- Neumann: `field[i,j,0] = field[i,j,1]` and `field[i,j,nz-1] = field[i,j,nz-2]`
- Periodic: `field[...,0] = field[...,nz-2]`, `field[...,nz-1] = field[...,1]`
- Dirichlet: `field[...,0] = values->front`, `field[...,nz-1] = values->back`
- Symmetry: Z-plane -> w=0, du/dz=0, dv/dz=0
- When nz=1: all z-face operations are no-ops

**Updated tests:** All `tests/core/test_boundary_conditions_*.c` files (10+). New `tests/core/test_boundary_conditions_3d.c`.

**Done:** All BC types work for z-faces. All 4 backends produce identical results for 3D.

---

### Phase 5: SIMD Backends — AVX2/NEON (2 weeks)

**Files to modify (12):**

Linear solvers (8 files):

- `lib/src/solvers/linear/avx2/linear_solver_{jacobi,redblack,cg,bicgstab}_avx2.c`
- `lib/src/solvers/linear/neon/linear_solver_{jacobi,redblack,cg,bicgstab}_neon.c`

NS solvers (3 files):

- `lib/src/solvers/navier_stokes/avx2/solver_{projection,explicit_euler,rk2}_avx2.c`

Template:

- `lib/src/solvers/linear/simd_template/linear_solver_bicgstab_simd_template.h`

**Pattern:** Add k-loop around existing j-loop. SIMD still vectorizes on i (contiguous). 7-point stencil adds `p[idx +/- stride_z]` loads.

**Done:** AVX2/NEON match scalar CPU for both nz=1 and nz>1.

---

### Phase 6: OMP Backends (1 week)

**Files to modify (5):**

- `lib/src/solvers/linear/omp/linear_solver_{cg,redblack}_omp.c`
- `lib/src/solvers/navier_stokes/omp/solver_{projection,explicit_euler,rk2}_omp.c`

**Pattern:** Add k-loop, use `collapse(2)` on k,j for better parallelism. Convert k to `int` for OMP.

**Done:** OMP matches scalar CPU. No data races (ThreadSanitizer clean).

---

### Phase 7: CUDA Backend (1-2 weeks)

**Files to modify (2-3):**

- `lib/src/solvers/gpu/solver_projection_jacobi_gpu.cu` — 3D kernels, `dim3(16,8,4)` launch, d_w allocation
- `lib/src/boundary/gpu/boundary_conditions_gpu.cu` — z-face kernels

**Done:** GPU matches CPU for nz=1 and small 3D problem (32x32x32).

---

### Phase 8: I/O, Examples, Validation, Docs (2 weeks)

**I/O updates:**

- `lib/src/io/vtk_output.c` — 3D dimensions, z-coordinates, w-velocity in vector output
- `lib/src/io/csv_output.c` — w statistics columns

**Simulation API:**

- `lib/src/api/simulation_api.c` — `init_simulation_3d` implementation

**New examples:**

- `examples/lid_driven_cavity_3d.c`
- `examples/minimal_example_3d.c`

**New 3D validation tests:**

- `tests/validation/test_taylor_green_3d.c` — 3D Taylor-Green vortex (has exact solution)
- `tests/validation/test_poiseuille_3d.c` — 3D channel flow

**Existing test updates (~40 files):** All existing tests use nz=1 path via wrapper APIs. Verify all pass.

**Documentation:** README.md, docs/reference/api-reference.md, ROADMAP.md, docs/guides/examples.md

---

## Phase Dependency Graph

```text
Phase 0 (indexing macros)
    +-- Phase 1 (data structures)
            +-- Phase 2 (stencils + linear solvers)
            |       +-- Phase 3 (NS solvers)
            |               +-- Phase 5 (SIMD backends)
            |               +-- Phase 6 (OMP backends)
            |               +-- Phase 7 (CUDA backend)
            +-- Phase 4 (boundary conditions) --+
                                                 +-- Phase 8 (I/O, examples, validation)
```

Phases 5, 6, 7 can run in parallel after Phase 3+4 complete.

---

## Summary

| Phase | Name | Files | Duration |
| ----- | ---- | ----- | -------- |
| 0 | Indexing Macro | ~22 | 3-5 days |
| 1 | Core Data Structures | ~12 | 1-2 weeks |
| 2 | Stencils + Scalar Linear Solvers | ~10 | 1-2 weeks |
| 3 | Scalar CPU NS Solvers | ~5 | 1-2 weeks |
| 4 | Boundary Conditions (all backends) | ~18 | 2 weeks |
| 5 | SIMD Backends (AVX2/NEON) | ~12 | 2 weeks |
| 6 | OMP Backends | ~5 | 1 week |
| 7 | CUDA Backend | ~3 | 1-2 weeks |
| 8 | I/O, Examples, Validation, Docs | ~40+ | 2 weeks |
| **Total** | | **~90** | **12-18 weeks** |

---

## Verification

After each phase:

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug

# Run all tests (nz=1 regression)
ctest --test-dir build -C Debug --output-on-failure

# Verify no inline 2D indexing remains (Phase 0)
grep -rn "j \* nx + i" lib/src/

# Verify 3D tests pass (Phase 2+)
ctest --test-dir build -C Debug -R "3d|3D" --output-on-failure

# After Phase 8: run examples
cd build/Debug && ./minimal_example && ./lid_driven_cavity_3d
```

---

## Risks

1. **Phase 4 (BCs) is the riskiest** — largest API surface, 15+ backend files. Mitigate by updating internal API atomically, keeping old public API as wrappers.
2. **nz=1 performance** — branch-free design (stride_z=0, inv_dz2=0.0) eliminates runtime checks. The only overhead is: (a) w-array loads (zeros, cached after first access), (b) computing z-stencil terms that evaluate to 0.0. Benchmark after Phase 3 to confirm no regression.
3. **Cache pressure in 3D** — z-neighbor access (stride nx*ny) misses L1 for grids >64x64. Mitigated by k->j->i loop order (three-plane working set fits L2) and OMP static scheduling (contiguous k-slabs per thread).
