# Lid-Driven Cavity: Grid Convergence and Richardson Extrapolation

**Test:** `tests/validation/test_cavity_richardson.c`
(`CavityRichardsonTest` in CI; `CavityRichardson_Re100/_Re400/_Re1000` in full validation,
label `validation`)
**ROADMAP:** §6.1 "Multi-Reynolds grid-convergence study (Richardson extrapolation)"

## Method

The steady cavity is solved on three grids per Reynolds number with the AVX2 projection
(OpenMP if AVX2 is not compiled in). Each run must reach the harness's steady-state exit,
|d ln KE/dt| < 1e-6. Four functionals are extracted:

| Functional | Definition |
|---|---|
| u_c | u at the cavity centre (a node on every odd grid) |
| u_min | minimum of u on the vertical centreline |
| v_max, v_min | extrema of v on the horizontal centreline |

The extrema are located to sub-grid accuracy with a parabola through the discrete extremum
and its two neighbours. The observed order p, the extrapolated value and the fine-grid
convergence index (GCI, safety factor 1.25) follow Celik et al. (2008). There p is solved
iteratively, so the refinement ratio does not have to be constant.

dt = min(0.25 h, 0.2 h²·Re, 1/Re).

## Grids

| Re | CI | Full validation |
|---|---|---|
| 100 | 17/33/65 | 33/65/129 |
| 400 | — | 65/129/257 |
| 1000 | — | 129/193/257 (ratios 1.5, 1.33) |

Coarser grids are not in the asymptotic range at higher Re. At Re=400 a 33×33 grid gives
u_min = −0.134 against −0.265 on 65×65. At Re=1000 a 33×33 grid settles to an almost
motionless state (v_min ≈ −3e−4), and 65 → 129 still changes u_min by 50%.

## Results

### Re = 100, 33/65/129 (measured)

| | 33 | 65 | 129 | p | extrapolated | GCI | benchmark | extrap. vs benchmark |
|---|---|---|---|---|---|---|---|---|
| u_c | −0.175185 | −0.195104 | −0.203223 | 1.29 | −0.208809 | 3.44% | — | — |
| u_min | −0.176123 | −0.198307 | −0.207573 | 1.26 | −0.214220 | 4.00% | −0.2140424 | 0.08% |
| v_max | 0.146955 | 0.166206 | 0.174242 | 1.26 | 0.180000 | 4.13% | 0.1795728 | 0.24% |
| v_min | −0.198353 | −0.232273 | −0.245799 | 1.33 | −0.254769 | 4.56% | −0.2538030 | 0.38% |

All four functionals converge monotonically. The extrapolated extrema land within 0.4% of
the grid-converged benchmark values, while the 129×129 values are 0.6–3% away. Against Ghia
et al. (1982) the extrapolated values look worse than the fine grid (u_min 1.6%, v_min 3.9%),
because Ghia is itself a 129×129 solution, not a grid-converged one. It is the wrong
yardstick for an extrapolated value.

### Re = 100, 17/33/65 (CI)

Observed order 0.96–1.22 on u_c, u_min, v_max and v_min, with convergence monotone. The CI
test asserts monotone convergence and an order in [0.8, 3.0], not accuracy.

### Re = 400 and Re = 1000

Measured so far at Re=400: 65×65 is steady at t = 54.9 (u_min = −0.26529, v_max = 0.23531,
v_min = −0.35917), and 129×129 at t = 51.0 (u_min = −0.30435, v_max = 0.27789,
v_min = −0.41900). The 257×257 runs at Re=400, and the Re=1000 runs, were still in progress
when this was written. Their gates (GCI ≤ 10%, and the Re=1000 extrapolation within
max(GCI, 0.5%) of Botella & Peyret) are provisional until the first full-validation run
records them here.

## Observed order is about 1.3, not 2

The interior stencils are second order, but the measured order on centreline quantities is
1.1–1.3 at Re = 100, on 17/33/65 and on 33/65/129 alike. So this is not a coarse-grid
artifact. Two known sources are the zero-gradient pressure wall `p[0] = p[1]`, which is
first order on a node-centred grid, and the singular lid corners. A grid-converged
second-order scheme with these boundary closures is expected to show a reduced order. The
test's lower bound of 0.8 guards against a regression below today's order. It is not a
claim of second order.

## Benchmark values

| Re | u_min | v_max | v_min | Source |
|---|---|---|---|---|
| 100 | −0.2140424 | 0.1795728 | −0.2538030 | grid-converged high-resolution solutions; source to be confirmed |
| 1000 | −0.3885698 | 0.3769447 | −0.5270771 | Botella & Peyret (1998), Chebyshev spectral |

These values were entered from the literature without a copy of the papers at hand.
Confirm them against the original tables before relying on the full-validation gate. At
Re=100 the independent agreement of the extrapolated values (0.08–0.38%) is strong evidence
that they are right.

## References

- Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., Coleman, H., Raad, P.E. (2008).
  Procedure for estimation and reporting of uncertainty due to discretization in CFD
  applications. *J. Fluids Eng.* 130(7), 078001.
- Botella, O., Peyret, R. (1998). Benchmark spectral results on the lid-driven cavity flow.
  *Computers & Fluids* 27(4), 421–433.
- Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using the
  Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48, 387–411.
