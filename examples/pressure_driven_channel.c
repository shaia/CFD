/**
 * Pressure-Driven Channel Flow
 *
 * Fluid at rest in a channel between two plates starts to move because the
 * pressure at the inlet is held above the pressure at the outlet -- no inlet
 * velocity is prescribed anywhere. It settles to Poiseuille flow, whose profile
 * is known exactly:
 *
 *   u(y) = (G / 2 mu) y (H - y),   G = -dp/dx = (p_in - p_out) / L
 *
 * The run is made twice, with the viscous term advanced explicitly and then
 * implicitly (Crank-Nicolson). With a viscous fluid on a fine grid the explicit
 * scheme is held to the diffusion limit dt < h^2 / (4 nu); the implicit one is
 * not, and reaches steady state in far fewer steps.
 *
 * The larger step has a price here. The projection method is non-incremental, so
 * its steady state carries a splitting error that grows with dt: the implicit run
 * settles faster but a little further from the exact profile. Implicit viscous
 * stepping buys stability, not accuracy.
 *
 * This example demonstrates:
 *   - Driving a flow by pressure: Dirichlet pressure faces (params.pressure_bc)
 *   - Zero-gradient velocity at the open ends, no-slip on the plates
 *   - Implicit viscous time integration (params.viscous_scheme)
 *   - Checking a computed profile against an analytical solution
 *
 * Usage: pressure_driven_channel [ny]
 *   ny = grid points across the channel (default: 33); the channel is 4x longer
 */

#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H          1.0 /* channel height */
#define LENGTH     4.0 /* channel length */
#define NU         0.1 /* kinematic viscosity: Re = U_max H / nu = 10 */
#define U_MAX      1.0 /* centerline velocity the pressure drop is chosen to give */
#define STEADY_TOL 1e-6

/* The example fails unless the profiles match Poiseuille this closely. Measured at
 * the default 33 points across: 2.0e-3 explicit, 1.24e-2 Crank-Nicolson (whose
 * larger dt carries a larger projection splitting error). About 5x margin. */
#define MAX_L2_EXPLICIT 1e-2
#define MAX_L2_IMPLICIT 5e-2

/* Pressure gradient that gives centerline velocity U_MAX: u_max = G H^2 / (8 nu) */
#define GRAD_P (8.0 * NU * U_MAX / (H * H))
#define P_IN   (GRAD_P * LENGTH)
#define P_OUT  0.0

static double wall_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (1e-9 * (double)ts.tv_nsec);
}

static double poiseuille_u(double y) {
    return GRAD_P / (2.0 * NU) * y * (H - y);
}

/* No-slip on the plates; zero-gradient velocity at the open ends, so the only
 * thing driving the flow is the pressure difference. */
static void apply_velocity_bc(flow_field* f) {
    size_t nx = f->nx, ny = f->ny;
    for (size_t j = 0; j < ny; j++) {
        f->u[IDX_2D(0, j, nx)] = f->u[IDX_2D(1, j, nx)];
        f->v[IDX_2D(0, j, nx)] = f->v[IDX_2D(1, j, nx)];
        f->u[IDX_2D(nx - 1, j, nx)] = f->u[IDX_2D(nx - 2, j, nx)];
        f->v[IDX_2D(nx - 1, j, nx)] = f->v[IDX_2D(nx - 2, j, nx)];
    }
    for (size_t i = 0; i < nx; i++) {
        f->u[IDX_2D(i, 0, nx)] = f->v[IDX_2D(i, 0, nx)] = 0.0;
        f->u[IDX_2D(i, ny - 1, nx)] = f->v[IDX_2D(i, ny - 1, nx)] = 0.0;
    }
}

typedef struct {
    int ok;
    int steps;
    double seconds;
    double t_end;
    double u_center; /* centerline velocity at mid-length */
    double l2_error; /* relative L2 error of the mid-length profile */
} run_result_t;

static run_result_t run(size_t nx, size_t ny, ns_viscous_scheme_t scheme, double dt) {
    run_result_t r;
    memset(&r, 0, sizeof(r));

    grid* g = grid_create(nx, ny, 1, 0.0, LENGTH, 0.0, H, 0.0, 0.0);
    flow_field* f = flow_field_create(nx, ny, 1);
    size_t total = nx * ny;
    double* u_prev = malloc(total * sizeof(double));
    double* v_prev = malloc(total * sizeof(double));
    ns_solver_registry_t* registry = cfd_registry_create();
    ns_solver_t* solver = NULL;
    if (!g || !f || !u_prev || !v_prev || !registry) {
        goto cleanup;
    }
    grid_initialize_uniform(g);

    /* At rest, with the pressure already falling linearly along the channel */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = IDX_2D(i, j, nx);
            f->u[idx] = f->v[idx] = f->w[idx] = 0.0;
            f->p[idx] = P_IN + (P_OUT - P_IN) * g->x[i] / LENGTH;
            f->rho[idx] = 1.0;
            f->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = NU;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0; /* the pressure drop is the only forcing */
    params.source_amplitude_v = 0.0;

    /* Prescribe the pressure at the inlet and outlet faces. The plates keep the
     * default zero-gradient pressure wall. */
    params.pressure_bc.left = POISSON_WALL_DIRICHLET;
    params.pressure_bc.right = POISSON_WALL_DIRICHLET;
    params.pressure_bc.values.left = P_IN;
    params.pressure_bc.values.right = P_OUT;

    params.viscous_scheme = scheme;

    cfd_registry_register_defaults(registry);
    solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    if (!solver || solver_init(solver, g, &params) != CFD_SUCCESS) {
        fprintf(stderr, "Solver init failed\n");
        goto cleanup;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();
    double residual = 1.0;
    double t0 = wall_seconds();
    int max_steps = 1000000;
    for (r.steps = 1; r.steps <= max_steps && residual >= STEADY_TOL; r.steps++) {
        apply_velocity_bc(f);
        memcpy(u_prev, f->u, total * sizeof(double));
        memcpy(v_prev, f->v, total * sizeof(double));
        if (solver_step(solver, f, g, &params, &stats) != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d\n", r.steps);
            goto cleanup;
        }
        double change = 0.0;
        for (size_t k = 0; k < total; k++) {
            change = fmax(change, fabs(f->u[k] - u_prev[k]));
            change = fmax(change, fabs(f->v[k] - v_prev[k]));
        }
        residual = change / (dt * U_MAX);
    }
    r.steps--;
    r.seconds = wall_seconds() - t0;
    r.t_end = r.steps * dt;
    apply_velocity_bc(f);

    /* Compare the mid-length profile with Poiseuille */
    size_t im = nx / 2;
    double err2 = 0.0, ref2 = 0.0;
    for (size_t j = 0; j < ny; j++) {
        double exact = poiseuille_u(g->y[j]);
        double d = f->u[IDX_2D(im, j, nx)] - exact;
        err2 += d * d;
        ref2 += exact * exact;
    }
    r.l2_error = sqrt(err2 / ref2);
    r.u_center = f->u[IDX_2D(im, ny / 2, nx)];
    r.ok = (residual < STEADY_TOL);

cleanup:
    free(u_prev);
    free(v_prev);
    if (solver)
        solver_destroy(solver);
    if (registry)
        cfd_registry_destroy(registry);
    if (f)
        flow_field_destroy(f);
    if (g)
        grid_destroy(g);
    return r;
}

int main(int argc, char* argv[]) {
    size_t ny = (argc > 1 && atoi(argv[1]) > 8) ? (size_t)atoi(argv[1]) : 33;
    size_t nx = 4 * (ny - 1) + 1; /* square cells */
    double h = H / (double)(ny - 1);

    double dt_diffusive = h * h / (4.0 * NU); /* explicit viscous limit, 2D */
    double dt_advective = 0.25 * h / U_MAX;

    printf("Pressure-Driven Channel Flow\n");
    printf("============================\n");
    printf("Channel %.0f x %.0f, grid %zu x %zu, nu = %.2f\n", LENGTH, H, nx, ny, NU);
    printf("Pressure %.2f at the inlet, %.2f at the outlet (dp/dx = %.3f)\n", P_IN, P_OUT, -GRAD_P);
    printf("Exact centerline velocity: %.4f\n\n", poiseuille_u(0.5 * H));
    printf("dt limits: diffusive (explicit viscous only) %.2e, advective %.2e\n\n", dt_diffusive,
           dt_advective);

    double dt_explicit = 0.9 * dt_diffusive;
    double dt_implicit = dt_advective; /* no diffusion limit: the advective one is next */

    printf("Explicit viscous term, dt = %.2e ...\n", dt_explicit);
    run_result_t ex = run(nx, ny, NS_VISCOUS_SCHEME_EXPLICIT, dt_explicit);
    printf("Crank-Nicolson viscous term, dt = %.2e ...\n\n", dt_implicit);
    run_result_t cn = run(nx, ny, NS_VISCOUS_SCHEME_CRANK_NICOLSON, dt_implicit);

    printf("  %-16s %8s %8s %9s %12s %12s\n", "viscous term", "steps", "t", "wall [s]", "u_center",
           "L2 error");
    printf("  %-16s %8d %8.2f %9.2f %12.5f %12.2e%s\n", "explicit", ex.steps, ex.t_end, ex.seconds,
           ex.u_center, ex.l2_error, ex.ok ? "" : "  (not steady)");
    printf("  %-16s %8d %8.2f %9.2f %12.5f %12.2e%s\n", "Crank-Nicolson", cn.steps, cn.t_end,
           cn.seconds, cn.u_center, cn.l2_error, cn.ok ? "" : "  (not steady)");
    if (ex.ok && cn.ok && cn.steps > 0) {
        printf("\nThe implicit run reached steady state in %.1fx fewer steps. Its profile is\n"
               "further from Poiseuille because the projection's steady state depends on dt,\n"
               "and its dt is %.1fx larger.\n",
               (double)ex.steps / cn.steps, dt_implicit / dt_explicit);
    }

    /* Self-check: both runs settled, both profiles are Poiseuille, and the implicit
     * run needed fewer steps -- the claims above, not just that the runs stopped. */
    int pass = ex.ok && cn.ok && ex.l2_error < MAX_L2_EXPLICIT && cn.l2_error < MAX_L2_IMPLICIT &&
               cn.steps < ex.steps;
    if (!pass) {
        fprintf(stderr,
                "\nFAILED: expected L2 < %.0e (explicit) and < %.0e (implicit), and fewer "
                "implicit steps\n",
                MAX_L2_EXPLICIT, MAX_L2_IMPLICIT);
    }
    return pass ? 0 : 1;
}
