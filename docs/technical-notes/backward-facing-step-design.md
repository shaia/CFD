# Backward-Facing Step: Design and Validation Notes

This note covers the backward-facing step case added in commit `0ca44c7`:

- why the case was added;
- the one library change it needed (an inlet on part of an edge), how that change is
  designed, and what was rejected;
- how the case is set up on the projection solver;
- what the test measures and against which references;
- what the results show, including a gap to the references that is explained, not
  hidden;
- what the case costs to run, what it does not cover, and where it could go next.

The results page is [validation/backward-facing-step.md](../validation/backward-facing-step.md).
This note is the full reasoning behind it.

---

## 1. Why this case

Before this change, every validation case in the library was either **closed** (the
lid-driven cavity, natural convection) or **fully developed** (Poiseuille flow, turbulent
channel flow, Taylor-Green). None of them has flow separating from a wall and then
reattaching to it. Separation is the behaviour most engineering flows depend on, and where
numerical schemes and turbulence models are most likely to go wrong.

Two roadmap items asked for such a case:

- **ROADMAP 6.1, "Other benchmarks (P2)"** listed "Backward-facing step — compare to
  Armaly et al. (1983)".
- **ROADMAP Phase 7** lists a separated or adverse-pressure-gradient case as a
  **prerequisite** for the learned turbulence closure. In channel flow the momentum
  balance pins the shear stress, so a closure has little authority over the quantities the
  ship gate measures. A separated flow is where a closure can actually be told apart from
  another.

The backward-facing step is the standard answer to both. Its geometry is simple: a channel
that suddenly doubles in height. It has one clean scalar output, the reattachment length
`x_r`, and a long record of experiments and computations to compare against.

This change delivers the **laminar** step. The turbulent step that Phase 7 needs builds on
it; section 9 lists what is still missing for that.

---

## 2. The obstacle: an inlet always covered a whole edge

The library works on structured rectangular grids and has no solid mask or immersed
boundary. A step is still easy to represent without one, by using the geometry Gartling
(1990) made standard:

```text
  y = H  +--------------------------------------------------+
         |  inflow ->                                        |
         |  (parabola)                                       |  outlet
  y = h  +  <- step corner                                   |  (zero-gradient u, p = 0)
         |  wall (step face)                                 |
  y = 0  +--------------------------------------------------+
        x = 0                                              x = L
```

The domain starts **at** the step. The left edge is split in two: the upper half
`y in [h, H]` is the inflow, and the lower half `y in [0, h]` is the step face, a no-slip
wall. Nothing upstream of the step is modelled.

This needs only one thing the library lacked: an inlet that covers **part** of an edge.
Before this change `bc_apply_inlet()` always wrote every node of the edge. The profile
position ran from 0 at one corner to 1 at the other, so a parabolic inlet was always spread
over the full height `H`.

---

## 3. Library change: `bc_inlet_set_range()`

### 3.1 API

```c
/* lib/include/cfd/boundary/boundary_conditions.h */
typedef struct {
    ...
    struct {
        bool enabled;
        double start;   /* normalized edge position where the inlet begins */
        double end;     /* normalized edge position where it ends */
    } range;
} bc_inlet_config_t;

cfd_status_t bc_inlet_set_range(bc_inlet_config_t* config, double start, double end);
```

Usage for the step:

```c
bc_inlet_config_t inlet = bc_inlet_config_parabolic(1.5);   /* peak 1.5 -> mean 1.0 */
bc_inlet_set_range(&inlet, 0.5, 1.0);                       /* upper half of the left edge */

bc_apply_noslip(u, v, nx, ny);                              /* walls everywhere first */
bc_apply_outlet_velocity(u, v, nx, ny, &outlet);            /* then the outlet */
bc_apply_inlet(u, v, nx, ny, &inlet);                       /* the inlet last */
```

### 3.2 Semantics

1. **Positions are in the same coordinate as the profile.** Node `i` of `n` along the
   edge sits at `t = i/(n-1)`. That runs left to right on the bottom and top edges, and
   bottom to top on the left and right edges. The profile position already used this
   coordinate, so a range and a profile describe the edge the same way.

2. **Only nodes with `t` in `[start, end]` are written.** Every other node of the edge
   keeps whatever the previous boundary condition put there. That is why the order is wall
   first, inlet last.

3. **The profile spans the range, not the edge.** Inside the range, the position passed to
   the profile is rescaled: `s = (t - start) / (end - start)`, clamped to `[0, 1]`. A
   parabola `4s(1 - s)` therefore peaks midway through the range and is **zero at both
   ends**. At the step corner (`t = 0.5`) the inlet writes 0, which is also what the wall
   writes, so the corner has one consistent value whichever condition runs last.

4. **Validation.** `bc_inlet_set_range()` returns `CFD_ERROR_INVALID` for a NULL config
   or any range that is not `0 <= start < end <= 1`. The condition is written as the
   negation of the valid case, so a NaN bound is refused as well. A refused call leaves
   the config unchanged.

   The apply functions check the range again, because a caller can fill in `range` by
   hand instead of calling the setter. A range on a z-face (FRONT/BACK) is also
   `CFD_ERROR_INVALID`, since a whole xy-plane has no single edge position to restrict.

5. **Tolerance at the bounds.** Membership is tested with a slack of `1e-9`
   (`BC_INLET_RANGE_EPS`). A bound placed exactly on a node, such as 0.5 on an odd node
   count, keeps that node even when `i/(n-1)` rounds to a hair outside the bound.

6. **3D.** On the left, right, bottom and top faces of a 3D grid, the range applies in
   every z-plane, exactly as the unrestricted inlet does.

7. **Mass-flow specification.** `bc_inlet_config_mass_flow()` turns a flow rate into a
   mean velocity using the caller's `inlet_length`. With a range, the caller passes the
   length of the range. The library does not infer it.

### 3.3 Backward compatibility

The factory functions (`bc_inlet_config_uniform()`, `_parabolic()`, `_custom()`, the
time-varying ones) all start from `bc_inlet_config_t config = {0}`. A zero-initialised
`range` has `enabled = false`, which means the whole edge, exactly as before. Every
existing caller behaves identically.

A search of the repository and of the sibling `cfd-python` bindings found no code that
copies the struct field by field, which would silently drop the new member. The Python
bindings build their configs through the factories and compile against the header.

### 3.4 Implementation across backends

| Path | File | How the range is honoured |
| ---- | ---- | ------------------------- |
| Scalar, OpenMP, AVX2, NEON | `lib/src/boundary/cpu/boundary_conditions_inlet_scalar.c` | All four backend tables point `apply_inlet` at this one implementation, so a single change covers every CPU backend |
| Time-varying inlet | `lib/src/boundary/cpu/boundary_conditions_inlet_time_scalar.c` | Same loop, same helper |
| Shared helpers | `lib/src/boundary/boundary_conditions_inlet_common.h` | `bc_inlet_range_is_valid()` and `bc_inlet_node_position()` |
| GPU | `lib/src/boundary/gpu/boundary_conditions_inlet_gpu.cu` | Device function `inlet_node_position_gpu()` mirrors the CPU helper; each kernel takes `range_on, range_start, range_end`; the host wrapper validates the range the same way |

The GPU file cannot include the CPU common header, so the mapping is duplicated there, and
a comment points at the CPU original. This follows the repository's sibling rule: a
behaviour added to one backend goes into every backend in the same commit.

### 3.5 Alternatives considered

| Option | Why not |
| ------ | ------- |
| **A solid mask / immersed boundary** so a real step block can sit inside the domain | This is the general answer and would also allow an upstream inlet channel (section 6.2). But it touches every solver's stencil loops, the projection's divergence and the pressure operator, which is far too much for one validation case. It is recorded as possible future work. |
| **Index-based range** (`j_begin`, `j_end`) | Exact about which nodes are included, but it describes the edge differently from the profile, which is normalized. Two conventions on one struct invite mistakes, and the index form breaks whenever the grid is refined. |
| **A sentinel instead of `enabled`** (e.g. `end == 0` means the whole edge) | Relies on a magic value. The project's code style requires new options to use 0 as the backward-compatible default. An explicit `bool enabled` meets that without overloading a coordinate. |
| **The test writes the boundary nodes itself**, with no library change | Works for one test, but leaves every user who needs an inflow window, such as a jet, a slot or a step, writing the same loop by hand, and leaves the GPU path without it. |

### 3.6 Unit tests

These are in `tests/core/test_boundary_conditions_inlet.c` (56 tests in the file, all
passing):

| Test | What it pins down |
| ---- | ----------------- |
| `test_inlet_set_range_validation` | NULL, `start < 0`, `end > 1`, `start == end`, reversed bounds and NaN bounds are refused; a refused call leaves the config as the factory made it; a valid call stores the range |
| `test_inlet_range_parabolic_left_all_backends` | The step inflow on a 17-node edge through the CPU, OpenMP and SIMD entry points. The lower half is untouched (sentinel 999 kept), the upper half matches `2 · 4s(1-s)` node by node, the corner is 0, and the interior is untouched |
| `test_inlet_range_bottom_between_nodes` | Bounds that fall between nodes (0.25, 0.75 on 16 nodes) select exactly the nodes inside |
| `test_inlet_range_time_varying` | `bc_apply_inlet_time()` honours the range, checked with a ramp that has finished |
| `test_inlet_range_invalid_when_set_by_hand` | A hand-filled reversed range is refused by both apply paths, and nothing is written |
| `test_inlet_range_z_face_refused` | A range on a FRONT face is `CFD_ERROR_INVALID` |

The GPU kernels compile cleanly. **They are not exercised by any test**, and neither was
the GPU inlet before this change: the suite has no CUDA-level unit tests for boundary
kernels.

---

## 4. Running the case on the projection solver

### 4.1 How boundary values reach the solver

The projection solvers keep whatever boundary values the caller has set.
`solve_projection_method()` updates only interior nodes in the predictor and the
corrector. It copies the caller's boundary velocities into `u*` before the pressure solve,
and back into `u^{n+1}` afterwards (`copy_boundary_velocities_3d()` in
`lib/src/solvers/navier_stokes/boundary_copy_utils.h`). So the test drives the boundaries
itself, re-applying them after every step:

```c
static void apply_bfs_bcs(flow_field* field, size_t nx, size_t ny,
                          const bc_inlet_config_t* inlet,
                          const bc_outlet_config_t* outlet) {
    bc_apply_noslip(field->u, field->v, nx, ny);              /* all four edges = 0 */
    bc_apply_outlet_velocity(field->u, field->v, nx, ny, outlet);  /* right edge */
    bc_apply_inlet(field->u, field->v, nx, ny, inlet);        /* upper half of left */
}
```

Every step therefore sees no-slip on the bottom and top walls and on the step face, the
parabola over `y in [h, H]`, and a zero-gradient outlet on the right edge, one step
lagged. The lag does not matter at steady state.

### 4.2 Pressure boundary conditions

```c
params.pressure_bc.right        = POISSON_WALL_DIRICHLET;
params.pressure_bc.values.right = 0.0;
/* left, bottom, top stay POISSON_WALL_ZERO_GRADIENT (the default) */
```

Zero gradient on the walls and on the inflow is the usual choice for a projection method
with prescribed velocity. Fixing `p = 0` at the outlet does two things. It sets the
pressure level, and it makes the pressure operator nonsingular, so no compatibility
projection of the right-hand side is needed. The solver checks for that case with
`poisson_walls_are_singular()`, and it does not arise here.

This differs from the Poiseuille test, which prescribes pressure at **both** ends to drive
the flow. Here the inflow velocity drives the flow, so only the outlet pressure is fixed.

### 4.3 Pressure solver choice

Timed on the Re = 100 case (257×33 nodes, 200 steps, 4 OpenMP threads). These are
wall-clock figures on a machine shared with other builds, so only the ratios mean
anything:

| Solver | Pressure solve | ms / step |
| ------ | -------------- | --------- |
| `projection` (scalar) | CG (default) | 24 |
| `projection` (scalar) | MG-preconditioned CG (`NS_PRESSURE_SOLVER_PCG_MG`) | **15** |
| `projection_omp` | CG (default) | 112 |
| `projection_omp` | MG-preconditioned CG | 32 |
| `projection_omp` | plain multigrid | refused at init |

- **MG-preconditioned CG** is the fastest option on both backends. Both grid sizes used
  (257×33 and 513×33) satisfy multigrid's `2^k + 1` rule.
- **Plain multigrid is refused**, correctly. Per-face pressure walls are implemented only
  for CG, BiCGSTAB and GMRES, because the multigrid smoothers apply whole-domain walls
  inside their sweeps and could not hold `p = 0` on one face. The solver says so at init
  instead of solving a different problem.
- **OpenMP is slower than scalar** at these sizes on MSVC. This is the same per-region
  overhead measured before in this repository. It is why the quick Re = 100 case runs on
  the scalar solver.

### 4.4 Parameters

```c
params.dt       = 0.01 (Re = 100) / 0.005 (Re = 400);   /* fixed, CFL <= ~0.5 */
params.mu       = U * H / Re;                           /* U = 1, H = 1 */
params.max_iter = 1;                                    /* one projection per step */
```

The grid is uniform with `dx = dy = H/32` (33 nodes across `H`), which puts the step
corner on node 16. The domain is 8H at Re = 100 and 16H at Re = 400.

---

## 5. What is measured

### 5.1 Reattachment length

`x_r` is where the streamwise velocity on the **first row above the lower wall** (`j = 1`)
changes sign from negative (inside the bubble) to positive (after reattachment). It is
interpolated linearly between the two nodes that bracket the zero:

```text
x_r = (i + u_i / (u_i - u_{i+1})) * dx     where u_i < 0 <= u_{i+1}
```

On a no-slip wall, the sign of `u` at the first node above the wall is the sign of the
wall shear, so this is the zero-shear point to within the grid. The **last**
negative-to-positive crossing is used, not the first. At the foot of the step there is a
small counter-rotating corner eddy, and it must not be mistaken for the main bubble.
Results are reported in step heights, `x_r / h` with `h = 0.5`.

The same routine, run on the row below the top wall, looks for an upper-wall bubble.
`first_separation()` finds its start and `last_reattachment()` its end.

### 5.2 Steady state

Every 200 steps the test computes `max |u^n - u^{n-200}| / (200 dt)` over the field. The
run counts as steady once that drops below `1e-4` (in units of `U²/H`). The test fails if
this does not happen by `t_max`, so it cannot pass on a half-developed bubble.

### 5.3 Mass conservation

The volume flux through the inlet column and the outlet column is integrated with the
trapezoidal rule. Their relative difference must be below 1%. At steady state it is
3.5e-3 at 33 nodes across `H` and 7.0e-4 at 65.

### 5.4 Acceptance

```c
check_bfs(&c, &r, ref, 0.02 /* below */, 0.08 /* above */);
```

The test requires:

- the run reached steady state;
- `ref · 0.98 <= x_r/h <= ref · 1.08`;
- mass imbalance < 1%;
- at Re = 100, no upper-wall bubble.

Section 6 explains why the band is lopsided.

---

## 6. Reference data, and why the band is asymmetric

### 6.1 The values

`Re = U H / ν`, with `U` the mean inflow velocity and `H` the full channel height. This is
Armaly's definition, `Re = U D / ν` with `D = 2h`, since `2h = H` at expansion ratio 2.

| Re | `x_r / h` | Source | Inflow boundary |
| -- | --------- | ------ | --------------- |
| 100 | 3.00 | Katsoudas et al. (2025), 199×109 finite volume | channel, 1.35h upstream |
| ≈102 | 3.05 | Armaly et al. (1983), experiment, expansion ratio 1.94 | long channel |
| 200 | 4.95 | Katsoudas et al. | channel, 1.35h |
| ≈200 | ≈5.0 | Armaly et al., experiment (as quoted by Notus CFD) | long channel |
| ≈351 | 7.97 | Armaly et al., experiment | long channel |
| 400 | 8.10 | Katsoudas et al. | channel, 1.35h |
| 800 | 11.48 | Rogers & Kwak | — |
| 800 | 11.834 | Erturk (2008) | far upstream |
| 800 | 11.90 | Kim & Moin | — |
| 800 | 12.20 | Gartling (1990) | **at the step** |

Armaly's values come from a reproduction of his tables that gives step-height Reynolds
numbers `Re_S`. They are converted here with `Re = 2.13 Re_S`, which is the conversion at
his expansion ratio of 1.94. His experiment is two-dimensional only below about Re = 400;
above that, three-dimensional effects shorten the bubble below the 2D value.

The Re = 800 figures are as compiled by Katsoudas et al. and by the meshless-solver paper
in the sources. Of the Re = 800 values, only Gartling places the inflow at the step. The
"—" entries were not checked for their inlet placement.

### 6.2 The inflow position changes the answer

Most references do **not** use the inlet-at-the-step geometry this library can represent.
Barton (1997) computed the step both ways and found that **an upstream inlet channel
predicts shorter reattachment and separation lengths**, with the difference largest in
the expansion region and at low Re:

- With a channel, the stream feels the expansion before it reaches the corner and starts
  to turn downward early.
- With the inflow placed at the step, the parabola is forced to leave the corner parallel
  to the wall, and the bubble comes out longer.

The Re = 800 row shows the same thing: Gartling, with the inlet at the step, reports
12.20, against Erturk's 11.83 with a long inlet channel (+3%). So the expected behaviour
for this solver is to land **above** the channel-inlet references, by a margin that
shrinks as Re grows.

### 6.3 Evidence that the gap is not numerical error

At Re = 100, three independent checks show that the solver's excess over 3.00 is not
something that refinement would remove:

| Change | `x_r / h` | Effect |
| ------ | --------- | ------ |
| Baseline: 33 nodes across H, 8H, dt = 0.005 | 3.128 | — |
| dt 0.005 → 0.01 | 3.123 | −0.17% |
| Domain 8H → 16H | 3.128 | none in four digits |
| Grid 33 → 65 nodes across H | 3.177 | **+1.6%, away from 3.00** |

- The time step and the domain length do not move the answer.
- Refining the grid moves it **further** from the reference. A Richardson estimate from the
  two grids puts the converged value between about 3.19 (second order) and 3.23 (first
  order), roughly 6–8% above 3.00. A grid error would shrink toward the reference; this
  does the opposite.

The remaining gap therefore reflects what is being solved, not how coarsely: the geometry
differs from the reference's. Its direction and size (+6–8% at Re = 100, +4% at Re = 400,
+3% at Re = 800 between Gartling and Erturk) fit Barton's entrance effect.

### 6.4 The resulting band

The test accepts **2% below to 8% above** the channel-inlet reference:

- The **2% below** is measurement slack only. This geometry should never come out short.
- The **8% above** covers the entrance effect at the lowest Re tested, including the
  grid-converged Re = 100 estimate.

This is a bounded comparison, not a match, and the documentation says so. It still
catches real faults. A pressure boundary condition on the wrong face, an inlet over the
wrong span, a lost corner value, or a convection error would each move `x_r` well outside
the band, or stop the flow reaching steady state, or break mass balance.

---

## 7. Results

| Re | Solver | Nodes across H | Length | dt | Steps to steady | `x_r / h` | vs ref | Mass imbalance | Upper bubble |
| -- | ------ | -------------- | ------ | -- | --------------- | --------- | ------ | -------------- | ------------ |
| 100 | scalar | 33 | 8H | 0.01 | 2,200 (t = 22) | 3.123 | +4.1% | 3.5e-3 | none |
| 100 | scalar | 33 | 8H | 0.005 | 4,200 (t = 21) | 3.128 | +4.3% | 3.3e-3 | none |
| 100 | scalar | 33 | 16H | 0.005 | 4,200 (t = 21) | 3.128 | +4.3% | 3.3e-3 | none |
| 100 | scalar | 65 | 8H | 0.005 | 4,200 (t = 21) | 3.177 | +5.9% | 7.0e-4 | none |
| 400 | scalar | 33 | 16H | 0.005 | 26,200 (t = 131) | 8.440 | +4.2% | 3.5e-3 | none |
| 400 | OpenMP | 33 | 16H | 0.005 | 26,200 (t = 131) | 8.440 | +4.2% | 3.5e-3 | none |

- **The OpenMP projection reproduces the scalar result** at Re = 400: the same `x_r` to
  four decimals, the same step count and the same mass imbalance.
- **Re = 400 settles far more slowly than Re = 100.** Its bubble grows by a slowly
  decaying increment, about 5% less per unit time, and needs about six times as much
  physical time.
- **There is no upper-wall bubble at Re = 400.** Katsoudas et al. report one starting
  from Re = 400. It is marginal there, and 33 nodes across H do not resolve it, so the
  Re = 400 test does not check for it.

---

## 8. Test layout and cost

One executable, `tests/validation/test_backward_facing_step.c`, provides two ctest
entries. Both carry the `validation` label, so they are excluded from the fast
pre-commit run (`-LE "cross-arch|validation"`) and run in the validation jobs:

| ctest name | Command | Case | Wall time (Release, shared machine) |
| ---------- | ------- | ---- | ----------------------------------- |
| `BackwardFacingStepTest` | `test_backward_facing_step` | Re = 100, scalar | ≈ 1 min |
| `BackwardFacingStepRe400Test` | `test_backward_facing_step re400` | Re = 400, OpenMP | ≈ 39 min |

The ctest timeout for both is 7,200 s.

The split follows the project's backend policy. Scalar backends are reference
implementations, and may appear only in short tests. The long Re = 400 run therefore uses
an optimized backend, even though on MSVC at 17k nodes OpenMP is slower than scalar (39
min against 25 min). On Linux runners with a lower-overhead OpenMP runtime this gap is
expected to shrink or reverse. That has not been measured.

Define `BFS_TRACE` when compiling the test to print `x_r` and `max |du/dt|` every 200
steps, which is how the convergence behaviour above was observed.

---

## 9. Limitations and next steps

1. **The comparison is bounded, not a match**, until a low-Re reference with the inflow at
   the step is added. The most direct one is **Gartling's own Re = 800 case** (`x_r/h` =
   12.20, plus an upper-wall bubble). It needs a domain of about 30H and at least 65
   nodes across H, which is too slow for the suite today and is listed in ROADMAP 6.1. At
   Re = 800 the cell Péclet number `U·dx/ν` reaches about 19 at 65 nodes across H, so
   whether central differencing stays clean there needs checking first.
2. **An upstream inlet channel** would let the test compare directly with Katsoudas,
   Erturk and Armaly. It needs a solid block inside the domain, which is the mask /
   immersed-boundary work set aside in section 3.5.
3. **Turbulent backward-facing step** (the Phase 7 prerequisite, e.g. Driver & Seegmiller
   conditions). This still needs:
   - wall functions on the step face, which is a partial left edge;
   - turbulence inflow values (`k`, `ε` or `ν̃`) over part of an edge. The turbulence
     boundary conditions are per face today.
4. **GPU inlet test.** The GPU range kernels compile but nothing exercises them. A small
   CUDA test comparing `bc_apply_inlet_gpu()` against the CPU inlet node by node would
   close the gap for the whole GPU inlet, not only the range.
5. **Stretched grids.** The normalized edge position is based on node index, not
   coordinate. That was already true of every inlet profile. On a stretched grid, a range
   bound of 0.5 is therefore the middle node, not the middle of the edge, and the profile
   is not a true parabola in `y`. This is fine on the uniform grids used here.

---

## 10. Files

| File | Change |
| ---- | ------ |
| `lib/include/cfd/boundary/boundary_conditions.h` | `range` member on `bc_inlet_config_t`; `bc_inlet_set_range()` declaration and contract |
| `lib/src/boundary/boundary_conditions.c` | `bc_inlet_set_range()` |
| `lib/src/boundary/boundary_conditions_inlet_common.h` | `BC_INLET_RANGE_EPS`, `bc_inlet_range_is_valid()`, `bc_inlet_node_position()` |
| `lib/src/boundary/cpu/boundary_conditions_inlet_scalar.c` | Range check and per-node position (serves scalar, OpenMP, AVX2, NEON) |
| `lib/src/boundary/cpu/boundary_conditions_inlet_time_scalar.c` | Same, for time-varying inlets |
| `lib/src/boundary/gpu/boundary_conditions_inlet_gpu.cu` | `inlet_node_position_gpu()`, range arguments on the four kernels, host-side validation |
| `tests/core/test_boundary_conditions_inlet.c` | Six range tests |
| `tests/validation/test_backward_facing_step.c` | The validation case |
| `CMakeLists.txt` | Executable, two ctest entries, `validation` label, timeout |
| `docs/validation/backward-facing-step.md` | Results page |
| `docs/reference/solvers.md`, `docs/guides/examples.md` | Inlet range described |
| `ROADMAP.md`, `CHANGELOG.md` | Item closed; Re = 800 item and Phase 7 note added |

---

## Sources

- Armaly, B.F., Durst, F., Pereira, J.C.F., Schönung, B. (1983). Experimental and
  theoretical investigation of backward-facing step flow. *J. Fluid Mech.* 127, 473–496.
  [PDF](https://courses.washington.edu/me431/handouts/armaly-jfm-83.pdf) ·
  [Semantic Scholar](https://www.semanticscholar.org/paper/Experimental-and-theoretical-investigation-of-step-Armaly-Durst/b44391340c9775127e016e071e33c8fe43f98530)
- Armaly and Denham & Patrick reattachment data as tabulated in the Feel++ laminar
  backward-facing step benchmark:
  [docs.feelpp.org](https://docs.feelpp.org/toolboxes/latest/cfd/laminar_isothermal_backward_facing_step/index.html)
- Barton, I.E. (1997). The entrance effect of laminar flow over a backward-facing step
  geometry. *Int. J. Numer. Methods Fluids* 25(6), 633–644.
  [ADS abstract](https://ui.adsabs.harvard.edu/abs/1997IJNMF..25..633B/abstract) ·
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0363(19970930)25:6%3C633::AID-FLD551%3E3.0.CO;2-H)
- Erturk, E. (2008). Numerical solutions of 2-D steady incompressible flow over a
  backward-facing step, Part I: High Reynolds number solutions. *Computers & Fluids* 37(6),
  633–655.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045793007001545)
- Gartling, D.K. (1990). A test problem for outflow boundary conditions — flow over a
  backward-facing step. *Int. J. Numer. Methods Fluids* 11, 953–967.
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.1650110704)
- Katsoudas et al. (2025). A finite volume and Levenberg–Marquardt optimization framework
  for benchmarking MHD flows over backward-facing steps. arXiv:2507.16509. Source of the
  Re = 100–800 values and of the Re = 800 literature comparison.
  [arXiv](https://arxiv.org/html/2507.16509v1)
- Meshless point collocation solver for incompressible Navier-Stokes (2019), arXiv:1906.05387.
  Re = 800 comparisons against Gartling and other 2D computations.
  [arXiv](https://arxiv.org/pdf/1906.05387)
- Notus CFD, backward-facing step validation. Re = 200 comparison with Armaly's
  experimental `x_r/h ≈ 5`.
  [doc.notus-cfd.org](https://doc.notus-cfd.org/dc/d98/group__doc__validation__backward__facing__step.html)
