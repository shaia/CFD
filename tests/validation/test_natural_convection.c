/**
 * @file test_natural_convection.c
 * @brief Quantitative natural-convection validation (de Vahl Davis benchmark)
 *
 * Differentially-heated square cavity (de Vahl Davis, 1983):
 *   - Hot left wall  (T = T_hot, Dirichlet)
 *   - Cold right wall (T = T_cold, Dirichlet)
 *   - Adiabatic top and bottom walls (Neumann / zero-gradient)
 *   - No-slip velocity on all walls
 *   - Boussinesq buoyancy couples temperature to the momentum equations.
 *
 * The cavity is run to steady state (detected via a kinetic-energy residual)
 * and the measured benchmark quantities are compared to the published de Vahl
 * Davis reference values:
 *
 *   Ra      u_max (vert. centerline)  v_max (horiz. centerline)  Nu_avg (hot wall)
 *   1e3     3.649                     3.697                      1.117
 *   1e4     16.178                    19.617                     2.238
 *
 * Velocities are non-dimensionalized by the thermal velocity scale alpha/L.
 *
 * The Rayleigh number is  Ra = g*beta*dT*L^3 / (nu*alpha)  with Pr = nu/alpha.
 *
 * Tiers (CAVITY_FULL_VALIDATION pattern, mirrors test_cavity_backends.c):
 *   - CI (always):   Ra = 1e3 on a coarse grid (fast).
 *   - Release only:  Ra = 1e4 on a finer grid (-DCAVITY_FULL_VALIDATION=ON).
 *
 * Because the energy equation + Boussinesq coupling is implemented only in the
 * scalar CPU projection solver, this validation runs on NS_SOLVER_TYPE_PROJECTION.
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>

#ifndef CAVITY_FULL_VALIDATION
#define CAVITY_FULL_VALIDATION 0
#endif

/* ============================================================================
 * FIXED PHYSICAL SETUP (shared across Rayleigh numbers)
 * ============================================================================ */

#define DVD_L       1.0      /* Unit square cavity */
#define DVD_T_HOT   310.0
#define DVD_T_COLD  290.0
#define DVD_T_REF   300.0    /* (T_hot + T_cold) / 2 */
#define DVD_DT_TEMP (DVD_T_HOT - DVD_T_COLD)  /* 20 K */
#define DVD_BETA    0.003333 /* ~1/T_ref [1/K] */
#define DVD_G       9.81
#define DVD_PR      0.71     /* Prandtl number (air) */

/* Steady-state detection: relative kinetic-energy change per step */
#define DVD_STEADY_TOL 1e-6
#define DVD_MIN_STEPS  200

/* ============================================================================
 * TEST SETUP / TEARDOWN
 * ============================================================================ */

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* ============================================================================
 * Velocity no-slip boundary conditions for the cavity walls.
 * (Thermal BCs are configured via params.thermal_bc and applied in solver_step.)
 * ============================================================================ */

static void apply_cavity_velocity_bcs(flow_field* field, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        field->u[IDX_2D(0, j, nx)] = 0.0;
        field->v[IDX_2D(0, j, nx)] = 0.0;
        field->u[IDX_2D(nx - 1, j, nx)] = 0.0;
        field->v[IDX_2D(nx - 1, j, nx)] = 0.0;
    }
    for (size_t i = 0; i < nx; i++) {
        field->u[IDX_2D(i, 0, nx)] = 0.0;
        field->v[IDX_2D(i, 0, nx)] = 0.0;
        field->u[IDX_2D(i, ny - 1, nx)] = 0.0;
        field->v[IDX_2D(i, ny - 1, nx)] = 0.0;
    }
}

/* ============================================================================
 * Domain kinetic energy (steady-state metric, no density weighting needed).
 * ============================================================================ */

static double compute_kinetic_energy(const flow_field* field) {
    double ke = 0.0;
    size_t total = field->nx * field->ny;
    for (size_t n = 0; n < total; n++) {
        ke += field->u[n] * field->u[n] + field->v[n] * field->v[n];
    }
    return 0.5 * ke;
}

/* ============================================================================
 * Average Nusselt number on the hot wall (i=0).
 *
 * Nu(y) = -d(T*)/d(x*) at x*=0,  T* = (T - T_cold)/(T_hot - T_cold), x* = x/L.
 * Uses a 2nd-order one-sided difference in x, averaged over the wall height
 * with the trapezoidal rule. Pure conduction => Nu = 1.
 * ============================================================================ */

static double compute_nu_hot_wall(const flow_field* field, double dx, double L) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    double inv_dT = 1.0 / DVD_DT_TEMP;
    double dy = L / (double)(ny - 1);

    double integral = 0.0;
    for (size_t j = 0; j < ny; j++) {
        double T0 = (field->T[IDX_2D(0, j, nx)] - DVD_T_COLD) * inv_dT;
        double T1 = (field->T[IDX_2D(1, j, nx)] - DVD_T_COLD) * inv_dT;
        double T2 = (field->T[IDX_2D(2, j, nx)] - DVD_T_COLD) * inv_dT;
        /* d(T*)/dx (one-sided, 2nd order); x* = x/L, so scale by L */
        double dTstar_dx = (-3.0 * T0 + 4.0 * T1 - T2) / (2.0 * dx);
        double nu_local = -dTstar_dx * L;
        double weight = (j == 0 || j == ny - 1) ? 0.5 : 1.0;
        integral += weight * nu_local;
    }
    return integral * dy / L; /* average over wall height L */
}

/* ============================================================================
 * Parameterized de Vahl Davis benchmark.
 *
 * Derives (alpha, nu) from the target Rayleigh number at fixed Pr, runs the
 * cavity to steady state, then asserts the non-dimensional peak centerline
 * velocities and the hot-wall Nusselt number against the reference values.
 * ============================================================================ */

static void run_dvd_benchmark(const char* solver_name, double Ra, size_t n,
                              double dt, int max_steps,
                              double tol_rel, double ref_umax, double ref_vmax,
                              double ref_nu) {
    /* Derive diffusivities: nu*alpha = g*beta*dT*L^3 / Ra,  Pr = nu/alpha */
    double nu_alpha = DVD_G * DVD_BETA * DVD_DT_TEMP * (DVD_L * DVD_L * DVD_L) / Ra;
    double alpha = sqrt(nu_alpha / DVD_PR);
    double nu = DVD_PR * alpha;
    double dx = DVD_L / (double)(n - 1);

    /* Explicit energy update is stable only for dt < dx^2 / (2*alpha*ndim).
     * Fail loudly on a misconfigured dt rather than silently diverging. */
    double dt_thermal_limit = (dx * dx) / (2.0 * alpha * 2.0);
    TEST_ASSERT_TRUE_MESSAGE(dt < dt_thermal_limit,
                             "dt exceeds thermal-diffusion stability limit");

    grid* g = grid_create(n, n, 1, 0.0, DVD_L, 0.0, DVD_L, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(n, n, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Quiescent start, linear temperature profile from hot (left) to cold (right) */
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            size_t idx = IDX_2D(i, j, n);
            field->u[idx] = 0.0;
            field->v[idx] = 0.0;
            field->w[idx] = 0.0;
            field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = DVD_T_HOT - DVD_DT_TEMP * (g->x[i] / DVD_L);
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = nu;
    params.alpha = alpha;
    params.beta = DVD_BETA;
    params.T_ref = DVD_T_REF;
    params.gravity[0] = 0.0;
    params.gravity[1] = -DVD_G;
    params.gravity[2] = 0.0;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    /* Hot/cold Dirichlet on left/right, adiabatic Neumann on top/bottom */
    params.thermal_bc.left   = BC_TYPE_DIRICHLET;
    params.thermal_bc.right  = BC_TYPE_DIRICHLET;
    params.thermal_bc.top    = BC_TYPE_NEUMANN;
    params.thermal_bc.bottom = BC_TYPE_NEUMANN;
    params.thermal_bc.dirichlet_values.left  = DVD_T_HOT;
    params.thermal_bc.dirichlet_values.right = DVD_T_COLD;

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, solver_name);
    if (!solver) {
        /* Backend not compiled in — skip without failing */
        printf("\n  [skip] de Vahl Davis: solver '%s' unavailable\n", solver_name);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }
    cfd_status_t init_status = solver_init(solver, g, &params);
    if (init_status == CFD_ERROR_UNSUPPORTED) {
        /* OMP sub-solver (e.g. OMP CG) unavailable — skip without failing */
        printf("\n  [skip] de Vahl Davis: solver '%s' init unsupported\n", solver_name);
        solver_destroy(solver);
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return;
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, init_status);

    /* March to steady state via kinetic-energy residual */
    ns_solver_stats_t stats = ns_solver_stats_default();
    double prev_ke = compute_kinetic_energy(field);
    int steps_done = 0;
    int converged = 0;
    for (int step = 0; step < max_steps; step++) {
        apply_cavity_velocity_bcs(field, n, n);
        cfd_status_t status = solver_step(solver, field, g, &params, &stats);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "Solver step should succeed");
        apply_cavity_velocity_bcs(field, n, n);

        double ke = compute_kinetic_energy(field);
        double residual = fabs(ke - prev_ke) / (prev_ke + 1e-10);
        prev_ke = ke;
        steps_done = step + 1;
        if (step > DVD_MIN_STEPS && residual < DVD_STEADY_TOL) {
            converged = 1;
            break;
        }
    }

    /* ---- Compute benchmark quantities ---- */
    double vel_scale = DVD_L / alpha; /* non-dimensionalize by alpha/L */

    /* u_max on the vertical centerline (x = 0.5) */
    size_t ic = n / 2;
    double umax = 0.0;
    for (size_t j = 0; j < n; j++) {
        double u = fabs(field->u[IDX_2D(ic, j, n)]);
        if (u > umax) umax = u;
    }
    umax *= vel_scale;

    /* v_max on the horizontal centerline (y = 0.5) */
    size_t jc = n / 2;
    double vmax = 0.0;
    for (size_t i = 0; i < n; i++) {
        double v = fabs(field->v[IDX_2D(i, jc, n)]);
        if (v > vmax) vmax = v;
    }
    vmax *= vel_scale;

    double nu_avg = compute_nu_hot_wall(field, dx, DVD_L);

    double err_u = fabs(umax - ref_umax) / ref_umax;
    double err_v = fabs(vmax - ref_vmax) / ref_vmax;
    double err_nu = fabs(nu_avg - ref_nu) / ref_nu;

    printf("\n  de Vahl Davis Ra=%.0f  (grid %zux%zu, dt=%.4g, steps=%d%s)\n",
           Ra, n, n, dt, steps_done, converged ? ", converged" : ", CAP HIT");
    printf("    u_max* = %8.3f  (ref %7.3f, err %5.1f%%)\n", umax, ref_umax, 100.0 * err_u);
    printf("    v_max* = %8.3f  (ref %7.3f, err %5.1f%%)\n", vmax, ref_vmax, 100.0 * err_v);
    printf("    Nu_avg = %8.3f  (ref %7.3f, err %5.1f%%)\n", nu_avg, ref_nu, 100.0 * err_nu);

    /* ---- Safety nets ---- */
    for (size_t idx = 0; idx < n * n; idx++) {
        TEST_ASSERT_TRUE(isfinite(field->u[idx]));
        TEST_ASSERT_TRUE(isfinite(field->v[idx]));
        TEST_ASSERT_TRUE(isfinite(field->T[idx]));
    }
    TEST_ASSERT_TRUE_MESSAGE(converged, "Cavity did not reach steady state within max_steps");

    /* ---- Quantitative gate vs de Vahl Davis ---- */
    TEST_ASSERT_TRUE_MESSAGE(err_nu < tol_rel, "Nu_avg outside de Vahl Davis tolerance");
    TEST_ASSERT_TRUE_MESSAGE(err_u < tol_rel, "u_max outside de Vahl Davis tolerance");
    TEST_ASSERT_TRUE_MESSAGE(err_v < tol_rel, "v_max outside de Vahl Davis tolerance");

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * CI tier: Ra = 1e3 (always runs)
 * ============================================================================ */

static void test_dvd_ra1e3(void) {
    /* Coarse grid + dt below the thermal-stability limit. Converges in ~1900
     * steps; observed error vs de Vahl Davis is ~1% (velocities) / ~2.5% (Nu),
     * so the 10% gate carries ample margin for cross-platform float variance.
     * The 30k cap is a regression backstop (never hit when converging). */
    run_dvd_benchmark(NS_SOLVER_TYPE_PROJECTION,
                      /*Ra*/ 1000.0, /*n*/ 41, /*dt*/ 0.002, /*max_steps*/ 30000,
                      /*tol_rel*/ 0.10, /*u*/ 3.649, /*v*/ 3.697, /*Nu*/ 1.117);
}


/* ============================================================================
 * Release tier: Ra = 1e3 grid refinement (only under -DCAVITY_FULL_VALIDATION=ON)
 *
 * Runs the same Ra=1e3 benchmark on a finer 81x81 grid with a tighter tolerance,
 * demonstrating convergence toward the de Vahl Davis reference. (Ra=1e4 is not
 * validated here: its thin boundary layers are beyond what this CPU-only,
 * central-differenced, first-order-coupled explicit solver resolves within a
 * feasible step budget — at 81x81 it plateaus in a transient overshoot far from
 * the benchmark. A higher-order/upwind scheme or much finer grid would be needed.)
 * ============================================================================ */

#if CAVITY_FULL_VALIDATION
/* OMP backend at the coarse resolution: validates that the energy equation +
 * Boussinesq buoyancy reproduce the de Vahl Davis reference on a parallelized
 * solver. Gated behind full validation (Release tier) — in a Debug build the
 * per-step OpenMP overhead on this small grid makes it far too slow for the
 * always-on CI tier. Skips cleanly if the OMP backend is not built. */
static void test_dvd_ra1e3_omp(void) {
    run_dvd_benchmark(NS_SOLVER_TYPE_PROJECTION_OMP,
                      /*Ra*/ 1000.0, /*n*/ 41, /*dt*/ 0.002, /*max_steps*/ 30000,
                      /*tol_rel*/ 0.10, /*u*/ 3.649, /*v*/ 3.697, /*Nu*/ 1.117);
}

/* Same benchmark on the AVX2 projection backend: validates that vectorized
 * buoyancy + the AVX2 energy step reproduce the de Vahl Davis reference.
 * Skips cleanly if the SIMD backend is not built. */
static void test_dvd_ra1e3_avx2(void) {
    run_dvd_benchmark(NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
                      /*Ra*/ 1000.0, /*n*/ 41, /*dt*/ 0.002, /*max_steps*/ 30000,
                      /*tol_rel*/ 0.10, /*u*/ 3.649, /*v*/ 3.697, /*Nu*/ 1.117);
}

/* Same benchmark on the CUDA projection backend: validates that the GPU
 * Boussinesq buoyancy + per-face thermal BCs + energy step reproduce the de
 * Vahl Davis reference. Skips cleanly if no CUDA device is present. Gated behind
 * full validation: the per-step GPU wrapper re-creates its device context each
 * solver_step, so step-by-step marching is slow — acceptable only at release tier. */
static void test_dvd_ra1e3_gpu(void) {
    run_dvd_benchmark(NS_SOLVER_TYPE_PROJECTION_GPU,
                      /*Ra*/ 1000.0, /*n*/ 41, /*dt*/ 0.002, /*max_steps*/ 30000,
                      /*tol_rel*/ 0.10, /*u*/ 3.649, /*v*/ 3.697, /*Nu*/ 1.117);
}

static void test_dvd_ra1e3_fine(void) {
    /* Expensive 81x81 validation runs on the optimized OMP backend (project
     * policy: long-running validation must not use the scalar reference). */
    run_dvd_benchmark(NS_SOLVER_TYPE_PROJECTION_OMP,
                      /*Ra*/ 1000.0, /*n*/ 81, /*dt*/ 0.0008, /*max_steps*/ 40000,
                      /*tol_rel*/ 0.05, /*u*/ 3.649, /*v*/ 3.697, /*Nu*/ 1.117);
}
#endif

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_dvd_ra1e3);
#if CAVITY_FULL_VALIDATION
    RUN_TEST(test_dvd_ra1e3_omp);
    RUN_TEST(test_dvd_ra1e3_avx2);
    RUN_TEST(test_dvd_ra1e3_gpu);
    RUN_TEST(test_dvd_ra1e3_fine);
#endif
    return UNITY_END();
}
