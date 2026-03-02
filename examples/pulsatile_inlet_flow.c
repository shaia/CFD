/**
 * Pulsatile Inlet Flow Example
 *
 * Demonstrates time-varying boundary conditions for channel flow with
 * three different inlet time profiles: sinusoidal, ramp, and step change.
 *
 * This example demonstrates:
 *   - bc_inlet_config_time_sinusoidal() for pulsatile flow
 *   - bc_inlet_config_time_ramp() for smooth start-up
 *   - bc_inlet_config_time_step() for sudden changes
 *   - BC_TIME_CONTEXT() macro for time context
 *   - bc_apply_inlet_time() for time-varying application
 *   - bc_apply_outlet_velocity() and bc_apply_noslip() for walls/outlet
 */

#include "cfd/api/simulation_api.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdio.h>

static void apply_walls_and_outlet(flow_field* field, size_t nx, size_t ny) {
    /* Zero-gradient outlet on right */
    bc_outlet_config_t outlet = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&outlet, BC_EDGE_RIGHT);
    bc_apply_outlet_velocity(field->u, field->v, nx, ny, &outlet);

    /* No-slip walls on top and bottom */
    for (size_t i = 0; i < nx; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->u[(ny - 1) * nx + i] = 0.0;
        field->v[(ny - 1) * nx + i] = 0.0;
    }

    /* Neumann pressure */
    bc_apply_neumann(field->p, nx, ny);
}

static void run_time_varying_case(const char* label,
                                   bc_inlet_config_t* inlet,
                                   size_t nx, size_t ny,
                                   double dt, int num_steps,
                                   int print_interval) {
    printf("  Case: %s\n", label);

    simulation_data* sim = init_simulation_with_solver(
        nx, ny, 1,
        0.0, 4.0, 0.0, 1.0, 0.0, 0.0,
        NS_SOLVER_TYPE_PROJECTION);

    if (!sim) {
        printf("    Failed to initialize simulation\n");
        return;
    }

    sim->params.dt = dt;
    sim->params.mu = 0.01;
    sim->params.source_amplitude_u = 0.0;
    sim->params.source_amplitude_v = 0.0;

    double time = 0.0;
    for (int step = 0; step < num_steps; step++) {
        /* Apply time-varying inlet BC */
        bc_time_context_t tctx = BC_TIME_CONTEXT(time, dt);
        bc_apply_inlet_time(sim->field->u, sim->field->v, nx, ny, inlet, &tctx);

        /* Apply outlet + wall BCs */
        apply_walls_and_outlet(sim->field, nx, ny);

        /* Step the solver */
        run_simulation_step(sim);
        time += dt;

        if (step % print_interval == 0) {
            /* Sample inlet velocity at mid-height */
            size_t mid_j = ny / 2;
            double u_inlet = sim->field->u[mid_j * nx + 0];
            printf("    t=%.3f: inlet u_mid = %.4f\n", time, u_inlet);
        }
    }

    free_simulation(sim);
    printf("\n");
}

int main(void) {
    printf("Time-Varying Boundary Conditions Example\n");
    printf("=========================================\n\n");

    size_t nx = 80, ny = 20;
    double dt = 0.001;
    int steps = 500;
    int print_every = 50;

    printf("Channel: 4.0 x 1.0, Grid: %zu x %zu\n", nx, ny);
    printf("Solver: projection, nu = 0.01, dt = %.3f\n\n", dt);

    /* Case 1: Sinusoidal pulsation
     * Base velocity (1.0, 0.0) modulated by (1.0 + 0.3*sin(2*pi*2.0*t))
     * i.e. oscillates between u=0.7 and u=1.3 at 2 Hz */
    bc_inlet_config_t inlet_sin = bc_inlet_config_time_sinusoidal(
        1.0, 0.0,   /* base velocity (u, v) */
        2.0,         /* frequency: 2 Hz */
        0.3,         /* amplitude: 30% modulation */
        0.0,         /* phase: 0 radians */
        1.0          /* offset: 1.0 (oscillates around base) */
    );
    bc_inlet_set_edge(&inlet_sin, BC_EDGE_LEFT);
    run_time_varying_case("Sinusoidal (freq=2Hz, amp=30%)",
                          &inlet_sin, nx, ny, dt, steps, print_every);

    /* Case 2: Ramp start-up
     * Velocity ramps from 0 to full over t=[0, 0.25] */
    bc_inlet_config_t inlet_ramp = bc_inlet_config_time_ramp(
        1.0, 0.0,    /* target velocity (u, v) */
        0.0, 0.25,   /* ramp from t=0 to t=0.25 */
        0.0, 1.0     /* multiplier: 0 -> 1 */
    );
    bc_inlet_set_edge(&inlet_ramp, BC_EDGE_LEFT);
    run_time_varying_case("Ramp Start-up (0 -> 1.0 over t=[0, 0.25])",
                          &inlet_ramp, nx, ny, dt, steps, print_every);

    /* Case 3: Step change
     * Velocity jumps from 0.5 to 1.5 at t=0.2 */
    bc_inlet_config_t inlet_step = bc_inlet_config_time_step(
        1.0, 0.0,    /* base velocity (u, v) */
        0.2,          /* step time */
        0.5,          /* multiplier before step */
        1.5           /* multiplier after step */
    );
    bc_inlet_set_edge(&inlet_step, BC_EDGE_LEFT);
    run_time_varying_case("Step Change (0.5 -> 1.5 at t=0.2)",
                          &inlet_step, nx, ny, dt, steps, print_every);

    printf("All time-varying BC cases completed.\n");
    return 0;
}
