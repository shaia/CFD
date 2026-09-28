# Lid-Driven Cavity: Grid Convergence and Richardson Extrapolation

**Test:** `tests/validation/test_cavity_richardson.c`
(`CavityRichardsonTest` in CI; `CavityRichardson_Re100/_Re400/_Re1000` in full validation,
label `validation`)
**ROADMAP:** §6.1 "Multi-Reynolds grid-convergence study (Richardson extrapolation)"

## Method

The steady cavity is solved on three grids per Reynolds number with the OpenMP projection
and the multigrid pressure solve (`NS_PRESSURE_SOLVER_MULTIGRID`). Each run must reach the
harness's steady-state exit, max |u^{n+1} − u^n| / (dt · U_lid) < 1e-6 over the whole field.

**Why multigrid.** On a 257×257 Re=1000 cavity it costs 13.7 ms/step from rest and
15.7 ms/step on a developed flow, against 135 and 140 ms/step for the AVX2 CG solve. After
120 steps the velocity fields agree to 5e-14. `NS_PRESSURE_SOLVER_PCG_MG` is not faster than
plain CG here (133 ms/step), because its V-cycle per CG iteration costs about what it saves.
Multigrid needs 2^k+1 points per side, which fixes the grids below. Four functionals are extracted:

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
| 1000 | — | 129/257/513 |

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

### Re = 400, 65/129/257 (measured)

| | 65 | 129 | 257 | p | extrapolated | GCI |
|---|---|---|---|---|---|---|
| u_c | −0.128752 | −0.117745 | −0.115296 | 2.17 | −0.114597 | 0.76% |
| u_min | −0.265294 | −0.304350 | −0.318569 | 1.46 | −0.326709 | 3.19% |
| v_max | 0.235312 | 0.277892 | 0.293162 | 1.48 | 0.301700 | 3.64% |
| v_min | −0.359166 | −0.418996 | −0.439914 | 1.52 | −0.451160 | 3.20% |

No benchmark gate: there is no grid-converged Re=400 reference in the repository. As a
sanity check, the extrapolated extrema are within 0.1–0.3% of Ghia et al. (1982).

### Re = 1000, 129/257/513 (measured)

| | 129 | 257 | 513 | p | extrapolated | GCI | Botella & Peyret | extrap. vs benchmark |
|---|---|---|---|---|---|---|---|---|
| u_c | −0.05734 | −0.05990 | −0.061083 | 1.11 | −0.062106 | 2.09% | — | — |
| u_min | −0.33280 | −0.36681 | −0.379945 | 1.37 | −0.388204 | 2.72% | −0.3885698 | 0.09% |
| v_max | 0.31938 | 0.35441 | 0.367975 | 1.37 | 0.376544 | 2.91% | 0.3769447 | 0.11% |
| v_min | −0.45538 | −0.49927 | −0.516088 | 1.38 | −0.526540 | 2.53% | −0.5270771 | 0.10% |

Extrapolated from three finite-difference grids, the extrema land within 0.11% of the
spectral benchmark. The grids reached steady state at t = 106.6, 94.9 and 107.5. Single-threaded
on the OpenMP multigrid projection, the whole Re=1000 case took about 4.4 hours, most of it on
513×513.

### u_c at Re=1000: a false steady state, not slow convergence

The first Re=1000 run reported u_c with an observed order of 0.62, below the gate. The cause
was the harness's steady-state exit at the time, `|d(ln KE)/dt| < 1e-6`. Kinetic energy
overshoots before it settles, and at the top of the overshoot its rate passes through zero. On
129×129 the test fired there, at t = 45.2, with u at the centre still moving:

| t | u_c (129×129) | KE rate |
|---|---|---|
| 45.2 | −0.058074 | 9.9e-7 (harness stopped here) |
| 60 | −0.057519 | 6.5e-5 |
| 100 | −0.057340 | 4.7e-6 |
| 200 | −0.057330 | 3.9e-9 |

The 7.4e-4 still to go was as large as the grid-to-grid differences, so the "order" measured
where each grid happened to stop. On 257×257 the KE approach was monotone and the stop at
t = 110 was genuine (u_c 2e-6 from settled). The lid-driven extrema had already settled on every
grid, which is why they converged cleanly throughout.

The exit is now a field residual, `max |u^{n+1} − u^n| / (dt · U_lid) < 1e-6`, which cannot
vanish while any part of the field still moves. With it, 129×129 runs to t = 106.6 and u_c
converges at p = 1.11, like the other functionals.

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
Confirm them against the original tables. The independent agreement of the extrapolated
values, 0.08–0.38% at Re=100 and 0.09–0.11% at Re=1000, is strong evidence that they are
right: three wrong numbers would not all be matched to a tenth of a percent.

## References

- Celik, I.B., Ghia, U., Roache, P.J., Freitas, C.J., Coleman, H., Raad, P.E. (2008).
  Procedure for estimation and reporting of uncertainty due to discretization in CFD
  applications. *J. Fluids Eng.* 130(7), 078001.
- Botella, O., Peyret, R. (1998). Benchmark spectral results on the lid-driven cavity flow.
  *Computers & Fluids* 27(4), 421–433.
- Ghia, U., Ghia, K.N., Shin, C.T. (1982). High-Re solutions for incompressible flow using the
  Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48, 387–411.
