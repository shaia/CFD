# Turbulent Channel DNS Validation

## Overview

Turbulent plane channel flow is the reference case for RANS closure validation: it is
statistically one-dimensional, fully developed, and has been resolved by direct numerical
simulation to high accuracy. That makes it the one flow in this library where a turbulence
model can be scored against *ground truth* rather than against another model or an
empirical correlation.

This document covers the DNS comparison added to `tests/validation/test_turbulent_channel.c`:
what it measures, where the reference data comes from, how to run it across Reynolds
numbers, and what the current results say about the k-epsilon and Spalart-Allmaras
implementations.

### Why the log-law check was not enough

The test already compared the computed velocity profile against the logarithmic law of the
wall. That check is worth keeping, but on its own it cannot measure closure error, because
**the wall-function boundary condition imposes log-law behaviour at the first interior
node**. The model is then graded on reproducing the thing it was constructed to reproduce.

Two observations show this concretely:

- k-epsilon and Spalart-Allmaras — a two-equation and a one-equation model, structurally
  very different — agreed on `u_tau` to **four significant figures** (0.9794 and 0.9834 at
  Re_tau >= 590). When the choice of closure does not move the metric, the metric is not
  measuring the closure.
- The residual deviation from the log law grew toward the outer edge of the probe window,
  which is exactly where the log law itself stops being valid and the wake region begins.
  Part of that "error" is real physics the log law does not describe, and cannot honestly
  be attributed to the model.

Against DNS the two models separate cleanly and consistently. That is the point of this
addition.

## Reference Data

`tests/validation/channel_dns_reference.h` carries the Moser-Kim-Mansour channel DNS
profiles, in the same style as the Ghia tables in `cavity_reference_data.h`: `static const
double` arrays, no data files under `tests/`.

**Source:** <https://turbulence.oden.utexas.edu/> (MKM data sets)
Moser, Kim & Mansour, *DNS of Turbulent Channel Flow up to Re_tau = 590*,
Physics of Fluids **11**(4), 943-945, 1999.
Numerical method: Kim, Moin & Moser, *J. Fluid Mech.* **177**, 133-166, 1987.
Files: `chan<N>/profiles/chan<N>.means` and `chan<N>.reystress`.

| Case | Actual Re_tau | Points | Quantities |
| ---- | ------------- | ------ | ---------- |
| chan395 | 392.24 | 129 | `y+`, `u+`, `k+`, `-<u'v'>+` |
| chan590 | 587.19 | 129 | `y+`, `u+`, `k+`, `-<u'v'>+` |

Normalization is by `u_tau` and the channel half-height `h`, so the tabulated mean velocity
**is** `u+` and needs no further scaling.

`k+` and `-<u'v'>+` are derived from the published Reynolds-stress components:

```text
k+     = 0.5 * (R_uu + R_vv + R_ww)
-uv+   = -R_uv                       (positive in the lower half of the channel)
```

The arrays were generated programmatically from the downloaded files rather than
transcribed by hand, and checked for monotonic `y+`, non-decreasing `u+`, zero wall values,
and a final `y+` equal to `Re_tau`. The derived peaks land on the textbook values —
`k+` peaks at 4.57 near `y+ = 16.9`, and `max(-uv+) = 0.84` — which is the cheapest
available confirmation that the derivation is right.

> **The case names are not the Reynolds numbers.** "chan395" is Re_tau = **392.24** and
> "chan590" is **587.19**. Driving the solver at 395 against 392.24 data bakes in a
> systematic offset. Pass the actual value.

## Running It

The default run is unchanged from before this addition — same grid, same time step, same
assertions, same results:

```bash
./test_turbulent_channel
# [channel] Re_tau = 395  ny = 21  y+_first = 39.5  ref = MKM DNS
```

An optional argument selects the friction Reynolds number, which makes the sweep
reproducible instead of requiring an edit-and-rebuild per data point:

```bash
./test_turbulent_channel 392.24     # matches chan395 DNS exactly
./test_turbulent_channel 587.19     # matches chan590 DNS exactly
./test_turbulent_channel 1000       # no DNS set; log-law check only
```

When no DNS set matches within a small tolerance the comparison is skipped and the header
line reports `ref = none (log-law only)`. The DNS block is **reported, never asserted** —
it is a measurement, and a model being 15% off in the Reynolds stresses is a finding about
the model, not a broken test.

### Grid and time step now follow Re_tau

Two constants in the test used to encode an assumption about the Reynolds number, and both
silently produced invalid results when it changed:

- **`CH_NY` (grid points across the channel).** On a uniform grid
  `y+_first = (CH_LY / (ny - 1)) * Re_tau`, so a fixed `ny` lets `y+` drift with `Re_tau`.
  At Re_tau = 180 the first node sat at `y+ = 18`, below the wall function's valid band; at
  Re_tau = 1000 it sat at `y+ = 100`, at the very top of it. `ny` now scales to hold
  `y+_first` at the stock grid's own value.
- **`CH_DT` (time step).** Refining the grid shrinks `dy`, so a fixed `dt` drives the
  convective CFL up with it — from ~0.3 at `ny = 21` to ~1.5 at `ny = 102`, where the
  explicit scheme simply fails. `dt` now scales with `dy`, and the step budgets scale
  inversely so physical time is preserved.
- **`CH_STEADY_TOL`** follows from the second: the steady-state test compares a *per-step*
  relative change in kinetic energy, so scaling `dt` would have made it easier to satisfy
  and stopped refined runs less converged. The threshold scales with `dt` to remain a bound
  on the rate of change.

All three reduce to exactly their previous values at the default Reynolds number, so the
stock run is bit-identical.

This matters beyond tidiness: before these were fixed, a naive `Re_tau` sweep produced an
apparent 3% -> 15% growth in closure error that was **entirely a numerical artifact** of
the drifting `y+` and the rising CFL.

## Current Results

Measured in Release with AVX2, each case driven at its data set's true `Re_tau`:

| Re_tau | model | `u+` RMS | `-uv+` RMS | `k+` RMS | nodes |
| ------ | ----- | -------- | ---------- | -------- | ----- |
| 392.24 | k-epsilon | 1.30% | 15.47% | 14.96% | 9 |
| 392.24 | SA | 1.76% | 16.41% | n/a | 9 |
| 587.19 | k-epsilon | 1.24% | 13.06% | 12.80% | 14 |
| 587.19 | SA | 1.60% | 12.48% | n/a | 14 |

> These figures were re-measured after #218 ("Fix the Krylov pressure operator,
> and refuse Poisson configuration that cannot be honoured"), which changed the
> converged velocity field and therefore the turbulence statistics. The effect is
> confined to Re_tau = 392.24; the 587.19 rows are unchanged. Most striking is
> Spalart-Allmaras at 392.24, whose shear-stress error fell from 28.73% to
> 16.41% -- that outlier was a symptom of the pressure-operator defect, not of
> the closure.

**Reynolds-stress error is roughly ten times the mean-velocity error.** That is the headline
and it is not a defect in this library — it is the well-known behaviour of linear
eddy-viscosity closures, which are calibrated on the channel mean profile and reproduce it
well while modelling the individual stresses poorly. The mean profile alone would suggest
these models are accurate to ~1%; the stresses show what they actually cost.

Spalart-Allmaras carries no turbulent kinetic energy, so `k+` is reported as `n/a` for it
rather than being synthesised from something else.

## Limitations

1. **The near-wall peak is never resolved.** The DNS `k+` peak sits at `y+ ~ 17`, below the
   first computational node at `y+ ~ 39`. Wall functions bridge that region by construction,
   so the classic near-wall closure failure is not measured at all — only outer-layer error
   is. A low-Reynolds-number model or a wall-resolved grid would be needed to see the rest,
   and would likely show a larger deficiency.
2. **The modelled shear stress uses a coarse derivative.** `-<u'v'>` is computed as
   `nu_t * dU/dy` with a central difference on a 21-point grid. At Re_tau = 392 one node
   reports `-uv+ = 1.105`, above the physical total stress there (~0.8), so individual shear
   values should be treated with caution even though the aggregate is meaningful.
3. **`k+` at the first node is pinned** at `3.33 = 1/sqrt(C_mu)` by the wall function's
   equilibrium assumption. That point measures the wall treatment, not the transport
   equations.
4. **Two Reynolds numbers.** Re_tau = 180 is published in the same archive but its first
   node falls below the wall function's valid band on any grid this coarse, so it is not
   usable here without a low-Re treatment.
5. **Cost grows steeply with Re_tau**, because refining the grid shrinks `dt` and multiplies
   the step count: roughly 9 s at Re_tau = 395 against several minutes by Re_tau = 1500. A
   routine sweep should stop at Re_tau = 1000.

## See Also

- `tests/validation/test_turbulent_channel.c` — the test
- `tests/validation/channel_dns_reference.h` — the reference data and its provenance
- [Lid-Driven Cavity Validation](lid-driven-cavity.md) — the laminar counterpart
- `docs/reference/solvers.md` — turbulence model descriptions and constants
