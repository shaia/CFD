# OpenMP vs Scalar: Why Threaded Kernels Lose, and How to Check

On the Windows/MSVC build, an OpenMP kernel is often *slower* than the scalar
reference it parallelizes. The cause is not compute speed. Entering a parallel
region costs about **12–15 µs** at 4 threads, and many regions in a CFD step do
far less work than that. This note explains the cost, where it shows up in this
codebase, and how to measure and validate a change that touches it.

Measurements: MSVC `/openmp:llvm`, 32-core Windows 11 machine, Release build,
`OMP_NUM_THREADS=4` (2026-09-12 for multigrid, 2026-09-26 for boundary conditions).

---

## 1. Why it happens

A `#pragma omp parallel for` wakes the worker threads, hands out the chunks, and
joins them at an implicit barrier. That fixed cost is paid per region, whatever
the loop does. A threaded loop wins only when

```text
serial_time(loop)  >  region_cost + serial_time(loop) / threads
```

With `region_cost ≈ 12 µs` and 4 threads, a loop needs about 16 µs of serial
work to break even. For a stream copy that is tens of thousands of doubles.

Six effects make this worse here:

| Effect | What happens | Consequence |
| --- | --- | --- |
| Region cost on Windows | Waking workers and syncing the barrier go through Windows thread/event primitives, not Linux futexes | ~12–15 µs per region at 4 threads (not measured on Linux for comparison) |
| Little work per region | A 2D edge is O(n): 258 copies on a 129×129 grid, ~0.1 µs serial | The region costs 100x the work it splits |
| Regions multiply | Kernels open one region per edge, per z-plane, per field, per multigrid level | 3D Neumann on 129³ opened 517 regions: ~5 ms vs 63 µs serial |
| `if(false)` is not free | libomp still enters a *serialized* parallel region, ~50 ns | Negligible once; ~13 µs over the 261 per-plane regions of a 65³ call |
| Two runtimes per process | The library builds with `/openmp:llvm` (`libomp140`); test and example executables inherit `/openmp` (`vcomp140`) | `omp_set_num_threads()` in a test never reaches the library |
| Busy cores | A barrier waits for its slowest thread; one preempted worker stalls the region | Threaded timings degrade more than serial ones when other builds run |

The fourth row matters for any fix. Gating with an `if()` clause removes the
thread wake-up, but it does not remove the region. Code that opens thousands of
tiny regions should hoist them (one region around the outer loop) or branch to a
plain serial loop.

## 2. Where it shows up in this codebase

| Area | Region pattern | Status |
| --- | --- | --- |
| Multigrid (`lib/src/solvers/linear/omp/linear_solver_multigrid_omp.c`) | One region per smoother / restriction / prolongation plane, at every level | **Gated** by `MG_OMP_MIN_POINTS` (32768) via `if(par)`. Before: a V-cycle ran 10x slower than scalar at 129×129 and was still slower at 513×513. After: 1.2x / 2.0x / 2.5x faster at 257 / 513 / 1025, and within ~0.01 ms of scalar below |
| Boundary conditions: OMP core (`boundary_conditions_core_impl.h`, `BC_OMP_FOR`) | One region per face pair per z-plane (Neumann, periodic, Dirichlet) | **Gated** by `BC_OMP_MIN_POINTS` (32768) via `if(bc_omp_worth_threading(points))` |
| Boundary conditions: OMP outlet (`omp/boundary_conditions_outlet_omp.c`) | One region per edge per z-plane | **Gated**, same threshold |
| Boundary conditions: SIMD templates (`boundary_conditions_simd_impl.h`, `boundary_conditions_outlet_simd.h`) | x-face loops ungated; y/z-face loops threaded from 256 points (`BC_SIMD_THRESHOLD`, now removed) | **Gated**, same threshold |
| Poisson solver walls (`poisson_solver_apply_bc` in `linear_solver.c`) | Runs once or twice per Krylov iteration | Serial on every backend, by design (comment in the source) |
| `apply_boundary_conditions()` in the NS solvers | Plain serial loop over all fields | Never threaded. Boundary work in an NS step does not go through the backend tables |

The backend BC tables (`bc_impl_omp`, `bc_impl_avx2`, `bc_impl_neon`) are
reached through the public `bc_apply_*` API. With `BC_BACKEND_AUTO` the
priority is SIMD > OMP > scalar. The examples call that API every step
(`lid_driven_cavity.c`, `pulsatile_inlet_flow.c`, and others), as do user
programs and multigrid OMP.

The threshold is on **points written by one region**: two per iteration for a
face pair (`2 * ny`, `2 * nx`, `2 * plane_size`), one for a single outlet edge.
In 2D no realistic grid reaches it, so every edge now runs serially. Only the
z-faces of 3D grids from 129×129 planes upward use the thread team.

## 3. Measured results (boundary conditions, A/B)

A = master (ungated), B = gated. Per-call time, median of 7 repeats, µs, 4 threads.

### 2D Neumann (`bc_apply_scalar_{cpu,omp,simd}`, n×n grid)

| n | cpu | omp A | omp B | simd A | simd B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 33 | 0.04 | 54.8 | 0.27 | 28.1 | 0.15 |
| 129 | 0.11 | 63.4 | 0.36 | 26.5 | 0.24 |
| 513 | 0.8–1.1 | 48.4 | 1.20 | 50.4 | 1.13 |
| 2049 | 4.5–6.3 | 48.3 | 4.81 | 57.6 | 5.06 |
| 8193 | 65–75 | 70.8–89.5 | 74.2 | 80.4–95.1 | 80.5 |
| 16385 | 171–276 | 111–159 | 81.3 | 112–149 | 87.9 |

Below an edge of 8193, B is within a fraction of a microsecond of scalar, where
A added about 50 µs per call. At 16385 (edge pair = 32770 points) B still
threads and stays ahead of scalar.

### 3D Neumann (n³)

| n | cpu | omp A | omp B | simd A | simd B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.4 | 753–964 | 4.1 | 537–572 | 2.7 |
| 33 | 2.5–5.7 | 1503–1692 | 9.4 | 726–797 | 6.7 |
| 65 | 7.4–9.6 | 2797–3154 | 22.7 | 1498–1500 | 21.7 |
| 129 | 63–76 | 4896–6275 | 169 | 3125–3292 | 168 |
| 257 | 426–815 | 12623–17091 | 818 | 12112–17010 | 732 |

B is 20–240x faster than A in 3D, but still **2–10x slower than scalar**. Two
causes remain:
1. The per-z-plane edge loops still open serialized `if(false)` regions (~50 ns each).
2. From 129³ upward the z-faces thread and lose to a cache-resident serial copy.

See the follow-ups in section 6.

### 2D outlet, one edge (`bc_apply_outlet_scalar_*`)

B tracks scalar within ~0.5 µs up to n = 4097; A added 20–30 µs per call. One
regression: the **right** (strided, column) edge at n = 16385 went from 52 µs
threaded (A) to 222 µs serial (B). A column copy touches one cache line per
point, so the "points" measure undercounts its cost. A 16385² 2D field is
2 GB, so this was left untuned.

### NS step time

`explicit_euler_omp`, `explicit_euler_optimized`, `rk2_omp`, `projection_omp`
and `projection_optimized` at 33–257 showed **no difference beyond noise**:
repeated A runs differed by up to 20%. This is expected, because
`apply_boundary_conditions()` never enters the backend BC tables (section 2).
The gain lands in code that calls `bc_apply_*` per step: the examples, user
programs, and multigrid.

## 4. How to check whether OpenMP is paying off

### Measure against scalar, not against the old OMP number

The question is always "does the threaded kernel beat the scalar reference at
this size?" Run cpu, omp and simd for the same input in the same process, back
to back.

### Benchmark recipe (Windows)

1. **Release build**, AVX2 on, in its own worktree so other sessions' builds
   and edits do not interfere:
   ```bash
   git worktree add ../cfd-bench -b perf/<topic> master
   cmake -S ../cfd-bench -B ../cfd-bench/build -DCFD_ENABLE_AVX2=ON
   cmake --build ../cfd-bench/build --config Release -j 16
   ```
   Keep the build directory short. MSBuild fails beyond 260-character paths,
   so a scratchpad under `%TEMP%` does not work.
2. **A standalone benchmark** linking the static libraries: include
   `lib/include` and `<build>/lib/include`, define `CFD_LIBRARY_STATIC_DEFINE`
   and `CFD_ENABLE_OPENMP`, compile with `/openmp:llvm`, and link
   `cfd_api cfd_omp cfd_simd cfd_scalar cfd_core` from `<build>/lib/Release`:
   ```cmake
   add_executable(bench_bc bench_bc.c)
   target_include_directories(bench_bc PRIVATE ${CFD}/lib/include ${CFD}/build/lib/include)
   target_compile_definitions(bench_bc PRIVATE CFD_LIBRARY_STATIC_DEFINE CFD_ENABLE_OPENMP)
   target_compile_options(bench_bc PRIVATE /openmp:llvm)
   target_link_directories(bench_bc PRIVATE ${CFD}/build/lib/Release)
   target_link_libraries(bench_bc PRIVATE cfd_api cfd_omp cfd_simd cfd_scalar cfd_core)
   ```
   Because the libraries are static, the executable carries the code. Copy it
   to `bench_A.exe` before changing anything, and to `bench_B.exe` after
   rebuilding.
3. **Time with `QueryPerformanceCounter`**, many calls per sample, and report
   the **median of 7** (or best of 5 for long NS steps). Sweep sizes across the
   expected crossover, e.g. n = 33 … 16385 in 2D and 17 … 257 in 3D.
4. **Set threads with `OMP_NUM_THREADS`**, never `omp_set_num_threads()`. The
   latter binds to `vcomp140` in the benchmark and does not reach the library's
   libomp regions.
   ```bash
   OMP_NUM_THREADS=4 ./bench_A.exe
   ```

### Reading the results

- **OMP time roughly flat across sizes, tens of µs** → region cost dominates;
  the work is too small to thread.
- **OMP cost grows with nz but not with nx·ny** → regions are opened per plane.
  Hoist them or branch to serial.
- **OMP a small constant above scalar even when gated** → serialized `if(false)`
  regions; count how many the call opens.
- **OMP beats scalar only past some n** → that is the crossover. Set the
  threshold at the work per region there.

Crossover for stream copies on this machine: an edge pair of 16386 points
(n = 8193) still lost, and 32770 points (n = 16385) won. That is where 32768
comes from. It matches the multigrid value measured independently.

## 5. How to validate a change

### Correctness

- The gate must not change results. Region gating (`if()` or a serial branch)
  runs the same loop body, and the BC kernels are pure copies and stores, so
  output is bit-identical. Reductions differ: with more than one thread, the
  combine order in a `reduction(+:...)` varies from run to run. Only
  single-thread runs are bit-reproducible against scalar.
- Run the fast suite with capped threads:
  ```bash
  OMP_NUM_THREADS=2 ctest --test-dir build -C Release -j 16 -LE "cross-arch|validation"
  ```
  Pay particular attention to `BoundaryConditions*Test`, `OMPConsistencyTest_1Thread`
  and `OMPConsistencyTest_4Threads`. The latter two set `OMP_NUM_THREADS` through
  ctest `ENVIRONMENT`, the only way thread counts reach the library.
- **Keep the threaded branch covered.** Once a threshold is raised, small test
  grids exercise only the serial path. At least one test should run above the
  threshold, e.g. a 3D case with 129×129 planes. `test_omp_consistency`
  deliberately keeps its multigrid configurations above `MG_OMP_MIN_POINTS`.
- On Clang/Linux, run OMP changes under ThreadSanitizer (`-DENABLE_TSAN=ON`).

### Performance (A/B protocol)

Timings from different moments are not comparable. Another session's build or
test suite shifts wall-clock by 3–5x, and threaded code suffers most.

1. Check the load before measuring (`Get-CimInstance Win32_Processor |
   Measure-Object LoadPercentage -Average`, running MSBuild/ctest processes).
2. Build A, freeze the executable, apply the change, build B, freeze it.
   Confirm B's build log recompiled the edited sources.
3. Run **interleaved**: A, B, A, B. Two A runs that disagree tell you the
   noise floor, and a difference smaller than that is no result.
4. If reverting a source from a backup to rebuild A, **`touch` it**. The
   restored file keeps the backup's older mtime, and MSBuild then silently keeps
   the other variant's objects.
5. Report the null result honestly. A change that is right but does not move
   end-to-end time is kept and stated as such, as the NS row in section 3 is.

## 6. Rules for writing OpenMP kernels here

- **Gate every region on its work.** Pass the points the region writes to a
  threshold predicate in an `if()` clause (`mg_omp_parallel`,
  `bc_omp_worth_threading`). Never leave an O(n) edge loop ungated.
- **Count regions, not just work.** One region around the outer loop beats one
  per plane. Branch to a plain serial loop for tiny work, because a gated region
  still costs ~50 ns.
- **Benchmark in Release against scalar at 4 threads** before claiming an OMP
  kernel is faster, on sizes on both sides of the threshold.
- **Keep the scalar reference fast.** It is the baseline users will fall back
  to, and it wins below the crossover.

Open follow-ups:
- The OMP and SIMD 3D BC paths are still 2–10x slower than scalar: per-plane
  serialized regions, and threaded z-face copies at 129³–257³ that lose to
  cached serial copies. Candidates: hoist the k-loop inside one region, and
  measure the z-face crossover separately (it is likely above 32768).
- Strided (column) loops cost more per point than contiguous ones. A single
  "points" threshold under-threads them at very large n.
