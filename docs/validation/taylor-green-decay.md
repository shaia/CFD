# Taylor-Green Vortex: Extended-Time Decay Rate

**Test:** `tests/validation/test_taylor_green_decay.c` (`TaylorGreenDecayTest`, label `validation`)
**ROADMAP:** §6.1 "Extended-time Taylor-Green decay-rate verification"

## What it checks

The short Taylor-Green test (`test_taylor_green_vortex.c`) stops at t = 0.2 with ν = 0.01.
By then the kinetic energy has fallen by 0.8%, so a 10% tolerance on the decay ratio cannot
tell a correct rate from a wrong one. This test runs until most of the energy is gone and
fits the decay rate over the whole run:

| | CI | Full validation |
|---|---|---|
| ν | 0.05 | 0.05 |
| t_end | 10 (KE down by e⁻² = 86%) | 20 (KE down by e⁻⁴ = 98%) |
| Grid (optimized backends) | 65×65 | 129×129 |
| dt | 0.2 h | 0.2 h |

- **Rate.** ln KE(t) is sampled about 400 times and fitted by least squares. The slope must
  be −4ν.
- **No drift.** The first and last thirds of the run are fitted separately and must agree.
  This catches a rate that drifts as the flow decays and would otherwise average out.
- **Order.** On 17/33/65 (or 33/65/129 in full validation), the rate error and the velocity
  L2 error must both converge at order > 1.7.

The scalar projection runs only a quick 33×33 case, per the scalar testing policy. AVX2,
OpenMP and CUDA run the fine grid.

## Configuration: one vortex cell with walls

The domain is [0, π]² with walls on the grid nodes, and the exact solution is

```
u =  sin x cos y e^{-2νt}
v = -cos x sin y e^{-2νt}
p = (cos 2x + cos 2y) e^{-4νt} / 4
```

The walls are streamlines (zero normal velocity), and ∂p/∂n = 0 there, so the projection's
zero-gradient pressure walls are exact for this flow. The tangential wall velocity decays
with the vortex, and the harness writes its exact value before every step, just as the
cavity harness writes the lid.

Because of those exact wall values, the test is less sensitive to a solver error than a
free periodic decay would be. A 1% viscosity error moves the fitted rate by 0.38%, not 1%.
The 0.2% rate tolerance still catches it, and every backend failed when this was checked
by running the solvers at 1.01ν.

## Results (dt = 0.2 h, t = 10)

| n | rate / 4ν | early third | late third | L2(u) |
|---|---|---|---|---|
| 17 | 1.00934 | 1.04488 | 0.99102 | 1.50e-2 |
| 33 | 1.00170 | 1.00819 | 0.99840 | 2.96e-3 |
| 65 | 1.00040 | 1.00170 | 0.99974 | 6.74e-4 |

Observed order: 2.45 and 2.08 for the rate error, 2.34 and 2.14 for L2(u). Scalar, AVX2,
OpenMP and CUDA give the same fitted rate to the digits shown.

## Why the periodic vortex is not the test

The textbook setup is the periodic vortex on [0, 2π]². Measured over the same run, none of
the solvers reproduces its decay, for reasons that come from the configuration or the model
rather than the discretization:

| Solver | rate / 4ν at n = 18 / 34 / 66 | Cause |
|---|---|---|
| projection (all backends) | 0.85 / 0.93 / 0.92 | The pressure Poisson solve has walls, not periodicity (`poisson_wall_t` offers zero-gradient or Dirichlet), so it imposes ∂p/∂n = 0 where the periodic flow has none. The error does not shrink with h. |
| rk2, rk4, explicit_euler | 0.42 / 0.42 / 0.42 | These solvers do not solve for pressure. They advance it with `dp/dt = −0.1 ρ ∇·u`, which cannot follow the e^{−4νt} decay of the vortex pressure. The result is the same on every grid and for RK2 and RK4 alike, which rules out both the spatial and the temporal discretization. |

Grid convention for a periodic vortex: the ghost copy `u[0] = u[n−2]`, `u[n−1] = u[1]` has
period (n−2)h, so the grid must span [−h, 2π] with h = 2π/(n−2). A grid spanning [0, 2π]
with n nodes has period 2π(n−2)/(n−1) and puts a kink at the seam. The periodic
cross-architecture cases in `test_solver_architecture.c` use the correct convention.
`taylor_green_reference.h`, used by the short test, still spans [0, 2π].

**Default forcing.** `ns_solver_params_default()` enables a sinusoidal momentum source
(`source_amplitude_u/v`). A decay test built on the defaults measures forced flow, not decay.
Both new tests set the amplitudes to zero.
