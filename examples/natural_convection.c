/**
 * Natural Convection in a Heated Cavity
 *
 * The de Vahl Davis (1983) benchmark: a square cavity with a hot left wall, a
 * cold right wall and insulated top and bottom. There is no lid and no inlet --
 * the flow is driven entirely by buoyancy, through the Boussinesq term
 * -beta (T - T_ref) g in the momentum equations.
 *
 * This example demonstrates:
 *   - The energy equation (params.alpha, the thermal diffusivity)
 *   - Boussinesq buoyancy (params.beta, params.T_ref, params.gravity)
 *   - Per-face thermal boundary conditions (params.thermal_bc):
 *     Dirichlet on the heated walls, Neumann (adiabatic) on the others
 *   - Choosing diffusivities from the Rayleigh and Prandtl numbers
 *   - Running to steady state on a residual that covers every field
 *   - Checking the result against the published benchmark: the peak
 *     centerline velocities and the hot-wall Nusselt number
 *
 * Usage: natural_convection [Ra] [n]
 *   Ra = Rayleigh number (default: 1000; the benchmark also tabulates 1e4)
 *   n  = grid points per side (default: 41)
 */

#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/io/vtk_output.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define L      1.0           /* cavity side */
#define T_HOT  310.0         /* left wall */
#define T_COLD 290.0         /* right wall */
#define T_REF  300.0         /* Boussinesq reference: the mean wall temperature */
#define BETA   (1.0 / T_REF) /* thermal expansion of an ideal gas */
#define G      9.81
#define PR     0.71 /* Prandtl number of air */

/* Steady when nothing changes faster than this, in units of the diffusive time
 * L^2/alpha, with velocity scaled by alpha/L and temperature by T_HOT - T_COLD. */
#define STEADY_TOL 1e-4

/* No-slip on all four walls; the solver applies the thermal BCs itself. */
static void apply_wall_velocity(flow_field* f) {
    size_t nx = f->nx, ny = f->ny;
    for (size_t j = 0; j < ny; j++) {
        f->u[IDX_2D(0, j, nx)] = f->v[IDX_2D(0, j, nx)] = 0.0;
        f->u[IDX_2D(nx - 1, j, nx)] = f->v[IDX_2D(nx - 1, j, nx)] = 0.0;
    }
    for (size_t i = 0; i < nx; i++) {
        f->u[IDX_2D(i, 0, nx)] = f->v[IDX_2D(i, 0, nx)] = 0.0;
        f->u[IDX_2D(i, ny - 1, nx)] = f->v[IDX_2D(i, ny - 1, nx)] = 0.0;
    }
}

/* Mean Nusselt number on the hot wall: minus the dimensionless wall temperature
 * gradient, averaged over its height, with a second-order one-sided difference.
 * Pure conduction gives exactly 1. */
static double hot_wall_nusselt(const flow_field* f, double h) {
    size_t nx = f->nx, ny = f->ny;
    double sum = 0.0;
    for (size_t j = 0; j < ny; j++) {
        double t0 = (f->T[IDX_2D(0, j, nx)] - T_COLD) / (T_HOT - T_COLD);
        double t1 = (f->T[IDX_2D(1, j, nx)] - T_COLD) / (T_HOT - T_COLD);
        double t2 = (f->T[IDX_2D(2, j, nx)] - T_COLD) / (T_HOT - T_COLD);
        double nu_local = -(-3.0 * t0 + 4.0 * t1 - t2) / (2.0 * h) * L;
        sum += ((j == 0 || j == ny - 1) ? 0.5 : 1.0) * nu_local;
    }
    return sum * h / L;
}

int main(int argc, char* argv[]) {
    double Ra = (argc > 1 && atof(argv[1]) > 0) ? atof(argv[1]) : 1000.0;
    size_t n = (argc > 2 && atoi(argv[2]) > 4) ? (size_t)atoi(argv[2]) : 41;

    /* Ra = g beta dT L^3 / (nu alpha) and Pr = nu / alpha fix both diffusivities */
    double nu_alpha = G * BETA * (T_HOT - T_COLD) * L * L * L / Ra;
    double alpha = sqrt(nu_alpha / PR);
    double nu = PR * alpha;
    double h = L / (double)(n - 1);

    /* The energy equation is explicit, so dt must respect its diffusion limit
     * h^2 / (4 alpha); the momentum limit h^2 / (4 nu) is looser since Pr < 1. */
    double dt = 0.5 * h * h / (4.0 * alpha);
    double t_scale = L * L / alpha; /* diffusive time */
    double u_scale = alpha / L;     /* thermal velocity */

    printf("Natural Convection (de Vahl Davis cavity)\n");
    printf("=========================================\n");
    printf("Grid:        %zu x %zu\n", n, n);
    printf("Ra:          %.0f   Pr: %.2f\n", Ra, PR);
    printf("nu = %.4e, alpha = %.4e, dt = %.4e\n\n", nu, alpha, dt);

    grid* g = grid_create(n, n, 1, 0.0, L, 0.0, L, 0.0, 0.0);
    flow_field* field = flow_field_create(n, n, 1);
    if (!g || !field) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    grid_initialize_uniform(g);

    /* Fluid at rest, with the conduction profile between the walls */
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            size_t idx = IDX_2D(i, j, n);
            field->u[idx] = field->v[idx] = field->w[idx] = field->p[idx] = 0.0;
            field->rho[idx] = 1.0;
            field->T[idx] = T_HOT - (T_HOT - T_COLD) * g->x[i] / L;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = nu;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0; /* no artificial forcing: buoyancy drives it */
    params.source_amplitude_v = 0.0;

    /* Energy equation and Boussinesq buoyancy */
    params.alpha = alpha;
    params.beta = BETA;
    params.T_ref = T_REF;
    params.gravity[0] = 0.0;
    params.gravity[1] = -G;
    params.gravity[2] = 0.0;

    /* Heated walls fixed, the others adiabatic */
    params.thermal_bc.left = BC_TYPE_DIRICHLET;
    params.thermal_bc.right = BC_TYPE_DIRICHLET;
    params.thermal_bc.top = BC_TYPE_NEUMANN;
    params.thermal_bc.bottom = BC_TYPE_NEUMANN;
    params.thermal_bc.dirichlet_values.left = T_HOT;
    params.thermal_bc.dirichlet_values.right = T_COLD;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    cfd_status_t status = solver ? solver_init(solver, g, &params) : CFD_ERROR;
    if (status != CFD_SUCCESS) {
        fprintf(stderr, "Solver init failed (status=%d)\n", status);
        return 1;
    }

    /* Steady state: every field has stopped changing, measured per unit of
     * diffusive time. A test on kinetic energy alone can pass at a turning
     * point of the energy while the flow is still evolving. */
    size_t total = n * n;
    double* u_prev = malloc(total * sizeof(double));
    double* v_prev = malloc(total * sizeof(double));
    double* T_prev = malloc(total * sizeof(double));
    if (!u_prev || !v_prev || !T_prev) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    int max_steps = 200000;
    int step = 0;
    double residual = 0.0;
    ns_solver_stats_t stats = ns_solver_stats_default();
    printf("Running to steady state...\n");
    for (step = 1; step <= max_steps; step++) {
        apply_wall_velocity(field);
        memcpy(u_prev, field->u, total * sizeof(double));
        memcpy(v_prev, field->v, total * sizeof(double));
        memcpy(T_prev, field->T, total * sizeof(double));

        status = solver_step(solver, field, g, &params, &stats);
        if (status != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d (status=%d)\n", step, status);
            break;
        }

        double change = 0.0;
        for (size_t k = 0; k < total; k++) {
            change = fmax(change, fabs(field->u[k] - u_prev[k]) / u_scale);
            change = fmax(change, fabs(field->v[k] - v_prev[k]) / u_scale);
            change = fmax(change, fabs(field->T[k] - T_prev[k]) / (T_HOT - T_COLD));
        }
        residual = change / (dt / t_scale);

        if (step % 2000 == 0) {
            printf("  step %6d  t* = %6.3f  residual = %.2e  Nu = %.4f\n", step,
                   step * dt / t_scale, residual, hot_wall_nusselt(field, h));
        }
        if (residual < STEADY_TOL) {
            break;
        }
    }
    apply_wall_velocity(field);

    /* Benchmark quantities, velocities scaled by alpha/L */
    double u_max = 0.0, v_max = 0.0;
    for (size_t j = 0; j < n; j++) {
        u_max = fmax(u_max, fabs(field->u[IDX_2D(n / 2, j, n)]));
    }
    for (size_t i = 0; i < n; i++) {
        v_max = fmax(v_max, fabs(field->v[IDX_2D(i, n / 2, n)]));
    }
    u_max /= u_scale;
    v_max /= u_scale;
    double nusselt = hot_wall_nusselt(field, h);

    printf("\n%s after %d steps (t* = %.3f, residual %.2e)\n\n",
           residual < STEADY_TOL ? "Steady" : "NOT steady", step, step * dt / t_scale, residual);

    /* de Vahl Davis (1983): peak centerline velocities and Nu_0, the mean Nusselt
     * number on the hot wall -- the quantity computed above */
    const double ref_ra[2] = {1e3, 1e4};
    const double ref[2][3] = {{3.649, 3.697, 1.117}, {16.178, 19.617, 2.238}};
    int r = (fabs(Ra - ref_ra[0]) < 1.0) ? 0 : (fabs(Ra - ref_ra[1]) < 1.0) ? 1 : -1;
    printf("  %-26s %10s %12s\n", "quantity", "computed", "de Vahl Davis");
    printf("  %-26s %10.3f %12s\n", "u_max (vertical centerline)", u_max, "");
    printf("  %-26s %10.3f %12s\n", "v_max (horiz. centerline)", v_max, "");
    printf("  %-26s %10.3f %12s\n", "Nu (hot wall)", nusselt, "");
    if (r >= 0) {
        printf("\n  Reference at Ra = %.0f: u_max %.3f, v_max %.3f, Nu %.3f\n", ref_ra[r],
               ref[r][0], ref[r][1], ref[r][2]);
        printf("  Differences: %.1f%%, %.1f%%, %.1f%%\n",
               100.0 * fabs(u_max - ref[r][0]) / ref[r][0],
               100.0 * fabs(v_max - ref[r][1]) / ref[r][1],
               100.0 * fabs(nusselt - ref[r][2]) / ref[r][2]);
    }

    /* Temperature and velocity for ParaView */
    cfd_set_output_base_dir("output");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "natural_convection", n, n);
    if (run_dir[0] != '\0') {
        char path[600];
#ifdef _WIN32
        snprintf(path, sizeof(path), "%s\\natural_convection.vtk", run_dir);
#else
        snprintf(path, sizeof(path), "%s/natural_convection.vtk", run_dir);
#endif
        write_vtk_flow_field(path, field, n, n, 1, 0.0, L, 0.0, L, 0.0, 0.0);
        printf("\nOutput: %s (includes the temperature field)\n", path);
    }

    free(u_prev);
    free(v_prev);
    free(T_prev);
    solver_destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
    return 0;
}
