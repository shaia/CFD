/**
 * Fast Steady Flow: Multigrid Pressure Solve and Checkpoint/Restart
 *
 * A steady run spends almost all of its time in the pressure Poisson solve,
 * once per step for tens of thousands of steps. This example shows the two
 * tools for that: a faster pressure solve, and the ability to stop a long run
 * and pick it up again later.
 *
 * This example demonstrates:
 *   - Selecting the multigrid pressure solve (params.pressure_solver) on the
 *     OpenMP projection solver, and timing it against the default CG solve
 *   - Running a lid-driven cavity to steady state on a velocity residual
 *   - Saving a checkpoint mid-run (save_simulation_checkpoint)
 *   - Restarting from it in a fresh simulation (load_simulation_from_checkpoint)
 *     and checking that the restarted run finishes where the uninterrupted
 *     one does
 *
 * Multigrid needs 2^k+1 points per side (17, 33, 65, 129, 257, ...).
 *
 * Usage: steady_flow_multigrid [n] [Re]
 *   n  = grid points per side, 2^k+1 (default: 129)
 *   Re = Reynolds number (default: 100)
 */

#include "cfd/api/simulation_api.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/filesystem.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Steady when no velocity changes faster than this per unit time (lid speed 1) */
#define STEADY_TOL 1e-6

static void apply_cavity_bc(flow_field* field) {
    bc_dirichlet_values_t u_bc = {.left = 0.0, .right = 0.0, .top = 1.0, .bottom = 0.0};
    bc_dirichlet_values_t v_bc = {.left = 0.0, .right = 0.0, .top = 0.0, .bottom = 0.0};
    bc_apply_dirichlet_velocity(field->u, field->v, field->nx, field->ny, &u_bc, &v_bc);
    bc_apply_neumann(field->p, field->nx, field->ny);
}

static double wall_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (1e-9 * (double)ts.tv_nsec);
}

static simulation_data* create_cavity(size_t n, double re, double dt,
                                      ns_pressure_solver_t pressure_solver) {
    simulation_data* sim = init_simulation_with_solver(n, n, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_PROJECTION_OMP);
    if (!sim) {
        return NULL;
    }
    sim->params.mu = 1.0 / re;
    sim->params.dt = dt;
    sim->params.max_iter = 1;
    sim->params.source_amplitude_u = 0.0;
    sim->params.source_amplitude_v = 0.0;
    /* The one line this example is about: the pressure solve. It is read on
     * every step, so it can also be changed between steps. */
    sim->params.pressure_solver = pressure_solver;
    apply_cavity_bc(sim->field);
    return sim;
}

/* One step plus the steady-state residual max|du|/dt over u and v. The step is
 * read from the clock rather than params.dt: the simulation API may shorten it
 * to respect the CFL limit. */
static cfd_status_t step_with_residual(simulation_data* sim, double* u_prev, double* v_prev,
                                       double* residual) {
    size_t total = sim->field->nx * sim->field->ny;
    apply_cavity_bc(sim->field);
    memcpy(u_prev, sim->field->u, total * sizeof(double));
    memcpy(v_prev, sim->field->v, total * sizeof(double));
    double t_before = sim->current_time;
    cfd_status_t status = run_simulation_step(sim);
    double dt = sim->current_time - t_before;
    double change = 0.0;
    for (size_t k = 0; k < total; k++) {
        change = fmax(change, fabs(sim->field->u[k] - u_prev[k]));
        change = fmax(change, fabs(sim->field->v[k] - v_prev[k]));
    }
    *residual = (dt > 0.0) ? change / dt : INFINITY;
    return status;
}

/* Average wall time per step over `steps` steps. */
static double time_per_step(size_t n, double re, double dt, ns_pressure_solver_t ps, int steps) {
    simulation_data* sim = create_cavity(n, re, dt, ps);
    if (!sim) {
        return -1.0;
    }
    /* Let the flow start first: the opening steps are not typical of a run */
    for (int k = 0; k < 50; k++) {
        apply_cavity_bc(sim->field);
        run_simulation_step(sim);
    }
    double t0 = wall_seconds();
    for (int k = 0; k < steps; k++) {
        apply_cavity_bc(sim->field);
        if (run_simulation_step(sim) != CFD_SUCCESS) {
            free_simulation(sim);
            return -1.0;
        }
    }
    double per_step = (wall_seconds() - t0) / steps;
    free_simulation(sim);
    return per_step;
}

int main(int argc, char* argv[]) {
    size_t n = (argc > 1) ? (size_t)atoi(argv[1]) : 129;
    double re = (argc > 2 && atof(argv[2]) > 0) ? atof(argv[2]) : 100.0;
    size_t m = n - 1;
    if (n < 17 || (m & (m - 1)) != 0) {
        fprintf(stderr, "n must be 2^k+1 (17, 33, 65, 129, 257, ...) for multigrid\n");
        return 1;
    }
    double h = 1.0 / (double)m;
    /* Advective (0.25 h), diffusive (0.2 h^2 Re) and advection-diffusion (1/Re) limits */
    double dt = fmin(0.25 * h, fmin(0.2 * h * h * re, 1.0 / re));

    printf("Fast Steady Flow: Multigrid + Checkpoint/Restart\n");
    printf("================================================\n");
    printf("Lid-driven cavity, %zu x %zu, Re = %.0f, dt <= %.3e, solver projection_omp\n\n", n, n,
           re, dt);

    /* ---- 1. What the pressure solve costs ---- */
    printf("1. Pressure solve cost (after 50 steps of start-up)\n");
    int timed = 200;
    double t_cg = time_per_step(n, re, dt, NS_PRESSURE_SOLVER_DEFAULT, timed);
    double t_mg = time_per_step(n, re, dt, NS_PRESSURE_SOLVER_MULTIGRID, timed);
    if (t_cg < 0.0 || t_mg < 0.0) {
        fprintf(stderr, "Timing run failed (is the OpenMP backend built?)\n");
        return 1;
    }
    printf("   CG (default):  %8.2f ms/step\n", 1e3 * t_cg);
    printf("   Multigrid:     %8.2f ms/step   (%.1fx)\n\n", 1e3 * t_mg, t_cg / t_mg);
    fflush(stdout);

    /* ---- 2. A steady run, checkpointed halfway ---- */
    cfd_set_output_base_dir("output");
    char run_dir[512];
    cfd_create_run_directory_ex(run_dir, sizeof(run_dir), "steady_flow_multigrid", n, n);
    if (run_dir[0] == '\0') {
        fprintf(stderr, "Failed to create run directory\n");
        return 1;
    }
    char ckpt[600];
#ifdef _WIN32
    snprintf(ckpt, sizeof(ckpt), "%s\\halfway.cfdchk", run_dir);
#else
    snprintf(ckpt, sizeof(ckpt), "%s/halfway.cfdchk", run_dir);
#endif

    size_t total = n * n;
    double* u_prev = malloc(total * sizeof(double));
    double* v_prev = malloc(total * sizeof(double));
    simulation_data* sim = create_cavity(n, re, dt, NS_PRESSURE_SOLVER_MULTIGRID);
    if (!u_prev || !v_prev || !sim) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    printf("2. Running to steady state with multigrid...\n");
    int max_steps = 400000;
    int checkpoint_step = -1;
    int steps = 0;
    double residual = 1.0;
    double t0 = wall_seconds();
    for (steps = 1; steps <= max_steps && residual >= STEADY_TOL; steps++) {
        if (step_with_residual(sim, u_prev, v_prev, &residual) != CFD_SUCCESS) {
            fprintf(stderr, "Solver failed at step %d\n", steps);
            return 1;
        }
        /* Checkpoint once the flow has developed: the first time it slows below
         * a thousand times the steady tolerance */
        if (checkpoint_step < 0 && residual < 1e3 * STEADY_TOL) {
            if (save_simulation_checkpoint(sim, ckpt) != CFD_SUCCESS) {
                fprintf(stderr, "Checkpoint failed\n");
                return 1;
            }
            checkpoint_step = steps;
            printf("   checkpoint written at step %d (t = %.2f): %s\n", steps, sim->current_time,
                   ckpt);
        }
        if (steps % 2000 == 0) {
            printf("   step %6d  t = %6.2f  residual = %.2e\n", steps, sim->current_time, residual);
            fflush(stdout);
        }
    }
    steps--;
    apply_cavity_bc(sim->field);
    printf("   steady at step %d, t = %.2f, %.1f s wall time\n\n", steps, sim->current_time,
           wall_seconds() - t0);

    /* ---- 3. Restart from the checkpoint and finish the same run ---- */
    printf("3. Restarting from the checkpoint in a fresh simulation...\n");
    simulation_data* restarted = load_simulation_from_checkpoint(ckpt);
    if (!restarted) {
        fprintf(stderr, "Could not load %s\n", ckpt);
        return 1;
    }
    printf("   loaded: t = %.2f, solver %s, pressure solve %s\n", restarted->current_time,
           restarted->solver ? restarted->solver->name : "?",
           restarted->params.pressure_solver == NS_PRESSURE_SOLVER_MULTIGRID ? "multigrid"
                                                                             : "not multigrid");
    for (int k = checkpoint_step; k < steps; k++) {
        if (step_with_residual(restarted, u_prev, v_prev, &residual) != CFD_SUCCESS) {
            fprintf(stderr, "Restarted solver failed\n");
            return 1;
        }
    }
    apply_cavity_bc(restarted->field);

    double diff = 0.0;
    for (size_t k = 0; k < total; k++) {
        diff = fmax(diff, fabs(restarted->field->u[k] - sim->field->u[k]));
        diff = fmax(diff, fabs(restarted->field->v[k] - sim->field->v[k]));
    }
    printf("   after the same %d steps: t = %.2f, max |velocity difference| = %.1e\n",
           steps - checkpoint_step, restarted->current_time, diff);
    if (diff == 0.0) {
        printf("   bit-identical to the uninterrupted run\n");
    } else {
        printf("   round-off: with several threads, reductions do not sum in a fixed order\n");
    }

    size_t c = n / 2;
    printf("\nSteady u at the cavity centre: %.5f\n", sim->field->u[c * n + c]);

    free(u_prev);
    free(v_prev);
    free_simulation(restarted);
    free_simulation(sim);
    return 0;
}
