# Backward-Facing Step Validation

## Overview

Laminar flow over a backward-facing step is the standard test for separated flow: the
stream leaves the step corner, a recirculation bubble forms behind it, and the flow
reattaches to the lower wall a distance `x_r` downstream. Unlike the lid-driven cavity,
the flow enters and leaves the domain, so the case also exercises an inflow on part of an
edge, a zero-gradient outlet and a prescribed outlet pressure together.

The test is `tests/validation/test_backward_facing_step.c`. It measures the lower-wall
reattachment length against published two-dimensional results.

## Setup

The geometry is Gartling's (1990):

| Quantity | Value |
| -------- | ----- |
| Channel height `H` (outlet) | 1 |
| Step height `h` | `H/2` (expansion ratio 2) |
| Inflow | Parabola over the upper half of the left edge, `y in [h, H]`, mean `U = 1` |
| Walls | No-slip: bottom, top, and the step face `y in [0, h]` at `x = 0` |
| Outlet | Zero-gradient velocity, `p = 0` (the only prescribed pressure face) |
| Reynolds number | `Re = U H / nu`, which is Armaly's `U D / nu` with `D = 2h = H` |

The inflow uses `bc_inlet_set_range(&inlet, 0.5, 1.0)`, which restricts the inlet to part
of its edge and lays the profile over that part only. Each step applies no-slip on every
edge first, then the outlet, then the inlet, so the inlet owns the upper half of the left
edge and the wall owns the lower half. The step corner is a node, where both give zero.

`x_r` is where the streamwise velocity on the first row above the lower wall changes sign
from negative to positive, linearly interpolated between nodes. The last such crossing is
used, so a corner eddy at the foot of the step cannot be mistaken for it. A run counts as
steady when `max |du/dt| < 1e-4`, sampled every 200 steps.

## Reference Data

| Re | `x_r / h` | Source | Inlet |
| -- | --------- | ------ | ----- |
| 100 | 3.00 | Katsoudas et al. (2025), 199×109 finite volume | channel, 1.35h |
| ~102 | 3.05 | Armaly et al. (1983), experiment, expansion ratio 1.94 | channel |
| 400 | 8.10 | Katsoudas et al. | channel, 1.35h |
| ~351 | 7.97 | Armaly et al., experiment | channel |
| 800 | 11.48 – 12.20 | Rogers & Kwak 11.48, Erturk 11.834, Kim & Moin 11.90, Gartling 12.20 | mixed |

Armaly's Reynolds numbers are converted from the step-height values tabulated in
reproductions of his data (`Re = 2.13 Re_S` at expansion ratio 1.94). His experiment is
two-dimensional only below about Re = 400; above it, three-dimensional effects shorten the
bubble, so the Re = 800 comparison is against computations only.

### The inlet position matters

Where the inflow boundary sits changes the answer, and most references do not use this
geometry. Barton (1997) computed the step with the inlet at the step and with an upstream
inlet channel, and found that **an inlet channel gives shorter reattachment lengths**,
most of all at low Re. With a channel, the stream starts to turn down toward the step
before it reaches the corner; with the inlet at the step, the parabola is forced to leave
the corner parallel to the wall. The Re = 800 values show the same thing: Gartling (inlet
at the step) reports 12.20 against Erturk's 11.83 (long inlet channel).

So this solver, with the inlet at the step, is expected to come out **above** the
channel-inlet references, by an amount that shrinks as Re grows.

## Results

Scalar `projection` (the test runs Re = 400 on `projection_omp`, which follows the same
algorithm), pressure by MG-preconditioned CG (`NS_PRESSURE_SOLVER_PCG_MG`), uniform grid
with `dx = dy`:

| Re | Nodes across `H` | Length | dt | `x_r / h` | vs reference | Mass imbalance |
| -- | ---------------- | ------ | -- | --------- | ------------ | -------------- |
| 100 | 33 | 8H | 0.01 | 3.123 | +4.1% | 3.5e-3 |
| 100 | 33 | 8H | 0.005 | 3.128 | +4.3% | 3.3e-3 |
| 100 | 33 | 16H | 0.005 | 3.128 | +4.3% | 3.3e-3 |
| 100 | 65 | 8H | 0.005 | 3.177 | +5.9% | 7.0e-4 |
| 400 | 33 | 16H | 0.005 | 8.440 | +4.2% | 3.5e-3 |

The result does not depend on the time step (0.2% between dt = 0.01 and 0.005) or on the
domain length (identical to four digits at 8H and 16H). Refining the grid moves it **away**
from 3.00, toward about 3.2 by Richardson extrapolation. That rules out resolution as the
cause of the gap, and it points the way the inlet-position effect above predicts. No
upper-wall bubble forms at Re = 100, as in the references.

At Re = 400 the flow takes much longer to settle: 26,200 steps to t = 131 against 2,200 to
t = 22 at Re = 100. It lands 4.2% above the reference. Katsoudas et al. report an upper-wall
bubble starting at Re = 400; at 33 nodes across H none forms here. The bubble is marginal
at that Re, and this grid does not resolve it, so the test does not check for it.

The comparison is therefore bounded, not a match: the test accepts `x_r/h` from 2% below
to 8% above the channel-inlet reference. A reference computed with the inlet at the step
at low Re would tighten this. Gartling's Re = 800 case is one, but it needs a domain of
about 30H at 65 or more nodes across H, which is too slow for this suite.

## Running

```bash
# Re = 100 (scalar, about 2200 steps; a minute in Release)
ctest --test-dir build -C Release -R BackwardFacingStepTest --output-on-failure
# Re = 400 (OpenMP)
ctest --test-dir build -C Release -R BackwardFacingStepRe400Test --output-on-failure
```

Both carry the `validation` label. Define `BFS_TRACE` when compiling the test to print
`x_r` and `max |du/dt|` as the run converges.

## References

- Armaly, B.F., Durst, F., Pereira, J.C.F., Schönung, B. (1983). Experimental and
  theoretical investigation of backward-facing step flow. *J. Fluid Mech.* 127, 473–496.
- Barton, I.E. (1997). The entrance effect of laminar flow over a backward-facing step
  geometry. *Int. J. Numer. Methods Fluids* 25, 633–644.
- Erturk, E. (2008). Numerical solutions of 2-D steady incompressible flow over a
  backward-facing step, Part I: High Reynolds number solutions. *Computers & Fluids* 37,
  633–655.
- Gartling, D.K. (1990). A test problem for outflow boundary conditions — flow over a
  backward-facing step. *Int. J. Numer. Methods Fluids* 11, 953–967.
- Katsoudas et al. (2025). A finite volume and Levenberg–Marquardt optimization framework
  for benchmarking MHD flows over backward-facing steps. arXiv:2507.16509.
