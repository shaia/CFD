# Lid-Driven Cavity Validation

## Overview

The lid-driven cavity is universally recognized as the fundamental benchmark problem for validating incompressible flow solvers. It represents the perfect balance between geometric simplicity and physical complexity, making it the essential first test for any CFD code.

### Why This Benchmark Matters

Despite its deceptively simple geometry—a square cavity with a single moving boundary—the lid-driven cavity flow exhibits remarkably rich physics that tests every component of a CFD solver:

- **Recirculating flow**: Unlike channel flows with inlet/outlet boundaries, the cavity is a closed system where momentum must be conserved within the domain
- **Multiple length scales**: The primary vortex spans the entire cavity while secondary and tertiary vortices occupy progressively smaller corner regions
- **Boundary layer dynamics**: Thin shear layers develop along walls, requiring adequate grid resolution
- **Reynolds number sensitivity**: Flow topology changes dramatically from creeping flow (Re < 100) to turbulent transition (Re > 10,000)

The benchmark's enduring value comes from its ability to expose solver deficiencies—issues with pressure-velocity coupling, boundary condition treatment, or numerical diffusion will manifest as deviations from the well-established reference data.

This document describes the validation methodology, reference data, solver performance analysis, and practical guidance for achieving publication-quality results.

## Problem Description

### Geometry
- Square cavity: [0,1] × [0,1]
- Uniform grid spacing

### Boundary Conditions
| Boundary | Velocity | Pressure |
| ---------- | ---------- | ---------- |
| Top (lid) | u = U_lid, v = 0 | ∂p/∂n = 0 |
| Bottom | u = 0, v = 0 | ∂p/∂n = 0 |
| Left | u = 0, v = 0 | ∂p/∂n = 0 |
| Right | u = 0, v = 0 | ∂p/∂n = 0 |

### Reynolds Number

The Reynolds number characterizes the ratio of inertial to viscous forces and determines the flow regime:

```text
Re = (U_lid × L) / ν
```

Where:

- U_lid = lid velocity (typically 1 m/s)
- L = cavity length (1 m)
- ν = kinematic viscosity

**Physical Interpretation:**

- **Re < 100**: Viscous-dominated flow, single primary vortex with weak corner eddies
- **Re = 100-1000**: Transition regime, secondary vortices develop in bottom corners
- **Re = 1000-5000**: Inertia-dominated flow, tertiary vortices appear, primary vortex center shifts
- **Re > 5000**: Approaching unsteady/turbulent transition, possible Hopf bifurcation at Re ≈ 8000-10000

For validation purposes, Re = 100 provides an ideal starting point: the flow reaches a well-defined steady state, convergence is relatively fast, and reference data is abundant. Higher Reynolds numbers (Re = 400, 1000) test solver robustness under increasingly challenging conditions.

## Reference Data

### Ghia et al. (1982)

The primary reference data comes from:

> Ghia, U., Ghia, K.N., & Shin, C.T. (1982). "High-Re solutions for incompressible
> flow using the Navier-Stokes equations and a multigrid method."
> Journal of Computational Physics, 48, 387-411.

This seminal paper established the de facto standard for lid-driven cavity validation. Using a fine-grid multigrid solver on 129×129 grids, Ghia et al. computed highly accurate steady-state solutions for Re = 100, 400, 1000, 3200, 5000, 7500, and 10000.

**Why This Data Is Authoritative:**

- **High resolution**: 129×129 uniform grid provides excellent spatial accuracy
- **Advanced numerics**: Multigrid acceleration with streamfunction-vorticity formulation
- **Comprehensive coverage**: Seven Reynolds numbers spanning four decades
- **Widely validated**: Thousands of CFD codes have verified against this data
- **Tabulated values**: 17 discrete points per centerline allow quantitative comparison

The paper reports not just velocity profiles but also:

- Primary vortex center location (x, y coordinates)
- Secondary vortex characteristics (center, strength)
- Streamfunction extrema
- Vorticity profiles

For rapid validation at Re = 100, comparing centerline velocity profiles typically suffices. For comprehensive validation, all flow features should be verified.

### Validation Points

**Vertical Centerline (x = 0.5):** u-velocity at 17 y-locations
**Horizontal Centerline (y = 0.5):** v-velocity at 17 x-locations

Key values at Re = 100:
| Location | Ghia Value | Description |
| ---------- | ------------ | ------------- |
| u(0.5, 0.5) | -0.20581 | u at center |
| u_min | -0.21090 | Minimum u on centerline |
| v(0.5, 0.5) | 0.05454 | v at center |

## Validation Methodology

### Test Structure

The validation tests are organized into four test files:

1. **test_cavity_setup.c** - Basic setup and boundary condition tests
2. **test_cavity_flow.c** - Flow development and stability tests
3. **test_cavity_validation.c** - Conservation and Ghia comparison
4. **test_cavity_reference.c** - Reference-based regression tests

### Error Metrics

**RMS Error Against Ghia:**
```
RMS = sqrt( (1/N) × Σ (u_computed - u_ghia)² )
```

Computed by:
1. Interpolating computed solution at Ghia's 17 sample points
2. Computing squared difference at each point
3. Taking root-mean-square

### Tolerance Standards

| RMS Error | Quality | Interpretation |
| ----------- | --------- | ---------------- |
| < 0.05 | Excellent | Publication quality |
| < 0.10 | Acceptable | Engineering use |
| 0.10 - 0.20 | Marginal | May need refinement |
| > 0.20 | Poor | Solver needs improvement |

### Test Configuration

Two modes are available:

**Fast Mode (CI):**
- Iterations: 1500-4000 steps
- Time step: 0.0005
- Purpose: Quick regression testing

**Full Validation Mode:**
- Iterations: 3000-10000 steps
- Time step: 0.0005
- Purpose: Comprehensive validation

Enable full mode with:
```c
#define CAVITY_FULL_VALIDATION 1
```

## Current Solver Performance

### Status: NEEDS IMPROVEMENT

The current projection solver produces:
- **u-centerline RMS: ~0.38** (target: < 0.10)
- **v-centerline RMS: ~0.13** (target: < 0.10)

### Observed Issues

1. **Slow Convergence:** Even with 4000 iterations, the flow hasn't reached steady state
2. **u_min Discrepancy:** Computing -0.15 vs Ghia's -0.21
3. **v_center Discrepancy:** Computing 0.13 vs Ghia's 0.05

### Root Cause Analysis

The observed discrepancies stem from several interconnected factors:

#### 1. Insufficient Time Stepping

The lid-driven cavity flow at Re = 100 requires ~10-20 time units to reach steady state. With dt = 0.0005 and 4000 steps, we're only simulating t = 2.0 time units—far too short for full development.

**Physical timeline:**

- t < 1: Initial transient, lid drives fluid downward
- t = 1-5: Primary vortex develops, reaches ~60% of final strength
- t = 5-10: Secondary vortices form in bottom corners
- t = 10-20: Flow approaches steady state (residuals < 1e-6)
- t > 20: Fully converged steady state

**Solution:** Increase to 20000-50000 steps or use adaptive time stepping with residual monitoring.

#### 2. Pressure Solver Convergence

The Jacobi iterative solver (currently 1000 iterations, tolerance 1e-6) may not fully converge the Poisson equation at each time step, accumulating pressure errors over time.

**Why this matters:**

- Projection method requires **accurate** pressure to enforce incompressibility
- Even 1% pressure error → 5-10% velocity error after 4000 steps
- Insufficient Poisson convergence → spurious divergence → incorrect vortex structure

**Solution:** Use more robust iterative solvers (Conjugate Gradient, Red-Black SOR with ω = 1.9) or multigrid methods.

#### 3. Time Step Selection

dt = 0.0005 may violate CFL or diffusion stability for the coarse grids being tested.

**Stability criteria:**

- **CFL condition:** dt ≤ CFL × h /  | u | _max, where CFL ≈ 0.5-1.0 for explicit schemes
- **Diffusion limit:** dt ≤ h²/(4ν) for explicit diffusion
- At Re = 100 with h = 1/32: u_max ≈ 1, ν = 0.01 → dt_max ≈ 0.0003 (CFL) or dt_max ≈ 0.002 (diffusion)

**Solution:** Reduce dt to 0.0001 or use adaptive time stepping with CFL monitoring.

#### 4. Boundary Condition Treatment

The lid-driven cavity has a **corner singularity** at the top corners where the moving lid meets the stationary walls. The velocity discontinuity creates infinite vorticity in theory, requiring special treatment.

**Regularization strategies:**

- Smooth lid velocity near corners: u(x) = U_lid × sin²(πx/2)
- Use finer grid near corners (stretched mesh)
- Apply special stencils at corner cells

Without regularization, local numerical errors at corners propagate into the domain, degrading global accuracy.

#### 5. Numerical Dissipation

First-order upwinding for advection introduces artificial viscosity, effectively increasing Re beyond the physical value. This smears out vortex structures and reduces peak velocities.

**Solution:** Use second-order centered differences or QUICK scheme for advection terms (requires additional stabilization).

## Test Files Reference

### cavity_reference_data.h
Contains:
- Ghia et al. reference data arrays
- Tolerance definitions
- Reference value structs for regression testing

### lid_driven_cavity_common.h
Contains:
- Test configuration (step counts, time steps)
- Simulation context management
- Boundary condition application
- Simulation runner utility

### Expected Test Output

```
========================================
REFERENCE-BASED VALIDATION TESTS
========================================

Target tolerance (scientific): RMS < 0.10
Current solver baseline:       RMS < 0.40

[Ghia et al. Comparison]
    u-centerline vs Ghia Re=100:
      RMS error:    0.3836
      [WARNING] RMS 0.3836 > target 0.10 - solver needs improvement
```

## Practical Guide: Achieving RMS < 0.10

This section provides a systematic approach to achieving publication-quality validation results.

### Step 1: Verify Basic Setup

Before tuning parameters, ensure the fundamentals are correct:

```c
// Check that boundary conditions are applied correctly
bc_dirichlet_values_t u_bc = {
    .left = 0.0,
    .right = 0.0,
    .top = 1.0,      // Lid velocity
    .bottom = 0.0
};

bc_dirichlet_values_t v_bc = {
    .left = 0.0,
    .right = 0.0,
    .top = 0.0,      // No vertical velocity at lid
    .bottom = 0.0
};

// Verify Reynolds number
double nu = 0.01;  // Kinematic viscosity
double Re = (U_lid * L) / nu;  // Should be 100
```

**Checkpoint:** Run 1000 steps and verify that:

- Velocity at lid is u ≈ 1.0, v ≈ 0.0
- Velocity at walls is u ≈ 0.0, v ≈ 0.0
- Flow field shows clockwise circulation

### Step 2: Grid Resolution Study

Test multiple grids to ensure spatial convergence:

| Grid | h | Expected RMS | Computational Cost |
| ------ | --- | -------------- | ------------------- |
| 17×17 | 0.0625 | ~0.30 | 1x (baseline) |
| 33×33 | 0.03125 | ~0.15 | 4x |
| 65×65 | 0.015625 | ~0.08 | 16x |
| 129×129 | 0.0078125 | ~0.05 | 64x |

**Rule of thumb:** For second-order schemes, doubling grid resolution should reduce error by ~4×.

**Recommendation:** Start with 65×65 for validation (good balance of accuracy and speed), use 129×129 for final verification.

### Step 3: Time Integration Tuning

#### A. Determine Required Simulation Time

Monitor residuals to detect steady state:

```c
double residual = 0.0;
for (size_t i = 0; i < nx*ny; i++) {
    double du = u_new[i] - u_old[i];
    double dv = v_new[i] - v_old[i];
    residual += sqrt(du*du + dv*dv);
}
residual /= (nx * ny);

// Continue until residual < 1e-6
if (residual < 1e-6) {
    printf("Steady state reached at t = %.2f\n", current_time);
}
```

**Expected timeline:**

- Re = 100: 10-15 time units
- Re = 400: 20-30 time units
- Re = 1000: 40-60 time units

#### B. Select Appropriate Time Step

Use CFL condition to guide dt selection:

```c
// CFL = (u*dt)/dx should be < 0.5 for stability
double u_max = 1.0;  // Lid velocity
double dx = 1.0 / (nx - 1);
double dt_cfl = 0.5 * dx / u_max;

// Diffusion stability: dt < dx²/(4*nu)
double dt_diff = 0.25 * dx * dx / nu;

// Use the more restrictive
double dt = fmin(dt_cfl, dt_diff);
printf("Recommended dt: %.6f\n", dt);
```

For 65×65 grid at Re = 100:

- dx = 0.015625
- dt_cfl ≈ 0.0078
- dt_diff ≈ 0.00061
- **Use dt = 0.0005** (safe margin below diffusion limit)

### Step 4: Pressure Solver Optimization

The Poisson equation must be solved accurately at each time step.

#### A. Iterative Solver Selection

| Solver | Iterations for 1e-6 | Pros | Cons |
| -------- | --------------------- | ------ | ------ |
| Jacobi | ~5000 | Simple | Very slow |
| SOR (ω=1.9) | ~800 | Fast, easy | Requires tuning ω |
| Conjugate Gradient | ~150 | Robust | More complex |
| Multigrid | ~10 | Fastest | Implementation overhead |

**Recommendation:** Use Conjugate Gradient with diagonal preconditioning for production runs.

#### B. Convergence Tolerance

```c
poisson_solver_params_t params = {
    .method = POISSON_CG,
    .tolerance = 1e-6,      // Absolute residual
    .max_iterations = 5000,
    .backend = BACKEND_SIMD  // Use AVX2/NEON if available
};
```

**Important:** Pressure tolerance should be 100× tighter than target velocity accuracy. For RMS < 0.10 velocity error, use pressure tolerance ≤ 1e-6.

### Step 5: Convergence Verification

After implementing improvements, systematically verify:

#### A. Temporal Convergence

Run with multiple time steps:

- dt = 0.001: RMS_1
- dt = 0.0005: RMS_2
- dt = 0.00025: RMS_3

Expect: RMS_1 / RMS_2 ≈ 4 (second-order temporal accuracy)

#### B. Spatial Convergence

Run with multiple grids (same dt, same final time):

- 33×33: RMS_33
- 65×65: RMS_65
- 129×129: RMS_129

Expect: RMS_33 / RMS_65 ≈ 4 (second-order spatial accuracy)

#### C. Steady-State Verification

Verify residuals have plateaued:

```bash
# Plot residual history
step  residual
1000  1.5e-3
2000  8.2e-4
5000  2.1e-5
10000 3.7e-6  ← Steady state reached
15000 3.9e-6  ← No further improvement
```

### Step 6: Ghia Comparison

Extract centerline profiles and compute RMS error:

```c
// Interpolate solution at Ghia's 17 y-locations
double u_computed[17];
for (int k = 0; k < 17; k++) {
    double y = ghia_y_locations[k];
    size_t j = (size_t)(y * (ny - 1));
    size_t i = nx / 2;  // Vertical centerline
    u_computed[k] = u[i + j*nx];
}

// Compute RMS error
double rms = 0.0;
for (int k = 0; k < 17; k++) {
    double diff = u_computed[k] - ghia_u_values[k];
    rms += diff * diff;
}
rms = sqrt(rms / 17.0);

printf("RMS error vs Ghia: %.4f\n", rms);
```

**Success criteria:**

- RMS < 0.05: Excellent (publication quality)
- RMS < 0.10: Acceptable (engineering use)
- RMS > 0.10: Needs improvement

### Troubleshooting Common Issues

**Problem: RMS stagnates at ~0.20-0.30**

Likely causes:

- Grid too coarse (try 65×65 minimum)
- Insufficient simulation time (check residuals)
- First-order numerical scheme (adds artificial diffusion)

**Problem: Solution diverges (NaN values)**

Likely causes:

- Time step too large (reduce by 2×)
- Pressure solver failing to converge (check Poisson iterations)
- Boundary conditions incorrectly applied

**Problem: Secondary vortices missing**

Likely causes:

- Grid too coarse in corners (need stretched mesh or 129×129)
- Not converged to steady state (run longer)
- Excessive numerical dissipation (check advection scheme)

## Running the Tests

```bash
# Build tests
./build.sh build-tests

# Run individual test suites
./build/Debug/test_cavity_setup.exe
./build/Debug/test_cavity_flow.exe
./build/Debug/test_cavity_validation.exe
./build/Debug/test_cavity_reference.exe

# Run with CTest
cd build && ctest -R Cavity
```

## References

1. Ghia, U., Ghia, K.N., & Shin, C.T. (1982). Journal of Computational Physics, 48, 387-411.
2. Botella, O., & Peyret, R. (1998). Computers & Fluids, 27(4), 421-433.
3. Erturk, E., Corke, T.C., & Gökçöl, C. (2005). International Journal for Numerical Methods in Fluids, 48(7), 747-774.

---


The lid-driven cavity is universally used as the first test case for CFD codes because it offers a perfect balance of simplicity and physical richness.

## Why It's Special

### 1. Geometric Simplicity

- **Square domain**: No complex meshing required
- **Single moving boundary**: Top wall (lid) moves at constant velocity
- **All other walls stationary**: Simple no-slip conditions
- **No inlet/outlet**: Closed system eliminates open boundary complexity

```
    u = U (moving lid)
    ─────────────────────
    │                   │
    │                   │
u=0 │     Cavity        │ u=0
v=0 │                   │ v=0
    │                   │
    ─────────────────────
          u=0, v=0
```

### 2. Rich Physics Despite Simplicity

The lid motion drives surprisingly complex flow patterns that reveal fundamental fluid dynamics principles:

**Vortex Hierarchy:**

The cavity develops a hierarchy of nested vortices driven by the shear at boundaries:

- **Primary vortex**: Large clockwise circulation filling ~80% of the cavity, driven directly by the moving lid
- **Secondary vortices**: Counter-rotating (anti-clockwise) eddies trapped in bottom corners where primary circulation meets the no-slip wall
- **Tertiary vortices**: At high Re, additional clockwise vortices appear within secondary vortices, demonstrating self-similar flow structures

**Physical mechanism:** As the lid drags fluid to the right, continuity forces downward flow at the right wall. This fluid reaches the bottom, travels left along the bottom wall, then rises at the left wall to complete the primary vortex. In the bottom corners, the primary vortex flow (moving left) collides with the no-slip boundary, creating regions of reverse flow—the secondary vortices.

**Reynolds Number Scaling:**

The flow topology changes dramatically with Re, making this a single test that validates solver performance across multiple flow regimes:

| Reynolds Number | Flow Character | Physical Regime | Numerical Challenge |
| ----------------- | ---------------- | ----------------- | --------------------- |
| Re = 100 | Single primary vortex, weak corner eddies | Viscous-dominated, steady | Good first test, fast convergence |
| Re = 400 | Stronger corner vortices, primary vortex shifts toward bottom-right | Transitional regime | Requires adequate resolution |
| Re = 1000 | Well-developed secondary vortices, primary vortex elongates | Inertia-dominated, steady | Tests robustness, may need finer grids |
| Re = 5000+ | Tertiary vortices, primary vortex becomes asymmetric | Approaching unsteady transition | Challenging, possible time-dependent instabilities |

**Why this matters for validation:**

- Low Re verifies basic solver correctness (momentum balance, pressure-velocity coupling)
- Medium Re tests robustness under increasing nonlinearity
- High Re stresses spatial resolution and numerical stability

### 3. Well-Documented Reference Data

The seminal paper by **Ghia, Ghia, and Shin (1982)** provides tabulated velocity profiles at various Reynolds numbers. This data has become the de facto standard for CFD validation:

- Centerline u-velocity profile (vertical cut through cavity center)
- Centerline v-velocity profile (horizontal cut through cavity center)
- Primary vortex center location
- Secondary vortex characteristics

**Citation**: Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

### 4. Tests All CFD Components

A working lid-driven cavity simulation validates:

| Component | How It's Tested |
| ----------- | ----------------- |
| **Boundary conditions** | Dirichlet (velocity), Neumann (pressure) |
| **Pressure solver** | Must converge to enforce incompressibility |
| **Advection scheme** | Transports momentum throughout domain |
| **Diffusion treatment** | Viscous effects smooth velocity gradients |
| **Grid handling** | Staggered or collocated arrangement |
| **Time stepping** | Stability with moving boundary |

### 5. Scalable Complexity

The lid-driven cavity serves as a progression path for solver development:

- **Low Re (100)**: Easy to solve, quick validation—ideal for debugging new features
- **Medium Re (1000)**: Tests solver robustness—requires good pressure-velocity coupling
- **High Re (5000+)**: Challenges numerical stability and resolution—may reveal hidden bugs
- **3D extension**: Same geometry, much richer physics—Taylor-Görtler vortices along spanwise direction
- **Unsteady variants**: Oscillating lid, impulsive start—tests time-dependent algorithms

**Pedagogical value:** A single geometry can teach viscous flow (Re = 1), Stokes flow (Re < 1), laminar recirculation (Re = 100-1000), and transitional instabilities (Re > 5000).

## Connection to Real-World Applications

While the lid-driven cavity is a canonical benchmark, its physics appear in numerous engineering contexts:

### Industrial Mixing

**Stirred tank reactors** exhibit similar recirculating flow patterns:

- Impeller (analogous to lid) drives primary circulation
- Tank walls (analogous to cavity walls) create secondary eddies
- Understanding vortex structure → optimize mixing efficiency

**Design insight:** Secondary vortices are "dead zones" with poor mixing. CFD helps position baffles to eliminate these regions.

### Materials Processing

**Czochralski crystal growth** uses a rotating crucible to melt semiconductor material:

- Rotation (similar to lid motion) drives melt convection
- Thermal gradients + recirculation → crystal quality
- CFD predicts melt flow patterns to avoid defects

**Coating flows** (spin coating, blade coating) involve shear-driven flow in confined geometries—similar to lid-driven cavity but with free surfaces.

### Microfluidics

**Lab-on-a-chip devices** use electroosmotic flow (EO flow) to drive fluid in microchannels:

- Channel walls with moving fluid layer (analogous to moving lid)
- Very low Re (<< 1) → purely viscous, predictable flow
- Validation at low Re ensures solver accuracy for microfluidic simulations

### Biological Flows

**Heart valves** create recirculation zones during diastole:

- Leaflet motion → shear-driven flow
- Stagnant regions → risk of thrombus formation
- CFD identifies high-risk geometries

**Blood pumps** (ventricular assist devices) must avoid recirculation zones that cause hemolysis. Lid-driven cavity validation at Re = 100-1000 builds confidence for these critical simulations.

### Atmospheric and Oceanic Flows

**Atmospheric eddies** in street canyons (urban flow):

- Wind over building tops → shear-driven vortex in canyon
- Pollutant dispersion controlled by recirculation
- Lid-driven cavity captures essential physics at much lower cost than full urban simulation

**Ocean gyres** driven by wind stress exhibit multi-scale vortex structures similar to cavity flow.

### Why Validation Transfers to Real Applications

The lid-driven cavity validates the **fundamental numerical building blocks** that all applications rely on:

1. **Momentum advection**: If solver gets vortex center location wrong, it will also mispredict mixing in a stirred tank
2. **Pressure-velocity coupling**: If projection method fails to enforce incompressibility in cavity, it will fail in blood pump simulations
3. **Boundary layer resolution**: If solver can't resolve corner vortices, it can't predict separation in airfoils
4. **Time integration**: If solver shows spurious oscillations at Re = 1000, it will be unstable for turbulent flows

**Bottom line:** A solver that achieves RMS < 0.10 on the lid-driven cavity benchmark has demonstrated the numerical accuracy and robustness required for production CFD.

## Implementation in This Library

The `examples/lid_driven_cavity.c` demonstrates:

```c
// Lid-driven cavity boundary conditions using Dirichlet BCs
bc_dirichlet_values_t u_bc = {
    .left = 0.0,          // Left wall: no-slip
    .right = 0.0,         // Right wall: no-slip
    .top = lid_velocity,  // Moving lid
    .bottom = 0.0         // Bottom wall: no-slip
};

bc_dirichlet_values_t v_bc = {
    .left = 0.0,
    .right = 0.0,
    .top = 0.0,           // No vertical velocity at lid
    .bottom = 0.0
};

bc_apply_dirichlet_velocity(field->u, field->v, nx, ny, &u_bc, &v_bc);
```

## Visualization Tips

When viewing results in ParaView or similar:

1. **Streamlines**: Show the vortex structure clearly
2. **Velocity magnitude contours**: Highlight high-velocity regions near lid
3. **Vector glyphs**: Display flow direction throughout domain
4. **Centerline plots**: Compare quantitatively to Ghia et al. data

## Further Reading

- Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.*, 48(3), 387-411.
- Erturk, E., Corke, T. C., & Gökçöl, C. (2005). Numerical solutions of 2-D steady incompressible driven cavity flow at high Reynolds numbers. *Int. J. Numer. Methods Fluids*, 48(7), 747-774.
- Bruneau, C. H., & Saad, M. (2006). The 2D lid-driven cavity problem revisited. *Computers & Fluids*, 35(3), 326-348.
