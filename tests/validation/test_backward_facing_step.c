/**
 * @file test_backward_facing_step.c
 * @brief Laminar backward-facing step: lower-wall reattachment length
 *
 * Geometry (Gartling 1990, the standard 2D benchmark): a channel of height
 * H = 1 with the step at the inlet plane. Fluid enters through the upper half
 * of the left edge, y in [h, H] with step height h = H/2 (expansion ratio 2),
 * with a parabolic profile of mean velocity U = 1. Every other wall is no-slip;
 * the right edge is a zero-gradient outlet with p = 0 prescribed there.
 * Re = U H / nu, the definition Armaly et al. (1983) use (their D = 2h = H).
 *
 * The flow separates at the step corner and reattaches on the lower wall at
 * x_r. Reference x_r / h from 2D steady computations and experiment:
 *
 *     Re    x_r/h
 *     100   3.00          Katsoudas et al. 2025 (199x109 FV); Armaly 3.05 at ~102
 *     200   4.95          same; Armaly's experiment gives ~5.0
 *     400   8.10          same
 *     800   11.48-12.20   Rogers & Kwak 11.48, Erturk 11.834, Kim & Moin 11.90,
 *                         Gartling 12.20
 *
 * Only Gartling puts the inlet at the step as here; the others start the flow
 * in an upstream channel, which shortens x_r (Barton 1997), most at low Re.
 * So this geometry is expected to land above the Re = 100 and 400 values --
 * see the band in check_bfs() and docs/validation/backward-facing-step.md.
 * Armaly's experiment is 2D only below Re ~ 400.
 *
 * What is measured: x_r is where the streamwise velocity on the first row
 * above the lower wall changes sign from negative (recirculation) to positive,
 * linearly interpolated between nodes. It is taken from the last such crossing,
 * so a corner eddy at the foot of the step cannot be mistaken for it.
 */

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * CASE DEFINITION
 * ============================================================================ */

#define BFS_H        1.0           /* Channel height (outlet) */
#define BFS_STEP     (0.5 * BFS_H) /* Step height h */
#define BFS_U_MEAN   1.0           /* Mean inlet velocity */

typedef struct {
    double re;
    double length;       /* Domain length in units of H */
    size_t ny;           /* Nodes across H; the step corner must sit on a node */
    double dt;
    double t_max;        /* Give up on steadiness after this time */
    double steady_tol;   /* max |du/dt| below which the flow counts as steady */
    const char* solver;
    ns_pressure_solver_t pressure_solver;
} bfs_case_t;

typedef struct {
    int ok;
    char msg[256];
    double xr_over_h;       /* Lower-wall reattachment length / h */
    double upper_sep;       /* Upper-wall bubble start / h, or 0 without one */
    double upper_reatt;     /* Upper-wall bubble end / h, or 0 without one */
    double mass_imbalance;  /* |outflow - inflow| / inflow */
    double t_end;
    double dudt;            /* max |du/dt| when the run stopped */
    int steps;
} bfs_result_t;

/* ============================================================================
 * MEASUREMENT
 * ============================================================================ */

/* Last negative-to-positive crossing of u along row j, as x (from the step) of
 * the interpolated zero. Returns 0 when the row has no reversed flow. */
static double last_reattachment(const double* u, size_t nx, size_t j, double dx) {
    size_t last_neg = 0;
    for (size_t i = 1; i < nx - 1; i++) {
        if (u[IDX_2D(i, j, nx)] < 0.0) {
            last_neg = i;
        }
    }
    if (last_neg == 0) {
        return 0.0;
    }
    double a = u[IDX_2D(last_neg, j, nx)];
    double b = u[IDX_2D(last_neg + 1, j, nx)];
    return ((double)last_neg + a / (a - b)) * dx;
}

/* First positive-to-negative crossing along row j (start of a bubble), or 0 */
static double first_separation(const double* u, size_t nx, size_t j, double dx) {
    for (size_t i = 1; i < nx - 1; i++) {
        double a = u[IDX_2D(i, j, nx)];
        double b = u[IDX_2D(i + 1, j, nx)];
        if (a >= 0.0 && b < 0.0) {
            return ((double)i + a / (a - b)) * dx;
        }
    }
    return 0.0;
}

/* ============================================================================
 * RUNNER
 * ============================================================================ */

static void apply_bfs_bcs(flow_field* field, size_t nx, size_t ny,
                          const bc_inlet_config_t* inlet,
                          const bc_outlet_config_t* outlet) {
    /* Walls first; the outlet and the inlet then overwrite their own nodes */
    bc_apply_noslip(field->u, field->v, nx, ny);
    bc_apply_outlet_velocity(field->u, field->v, nx, ny, outlet);
    bc_apply_inlet(field->u, field->v, nx, ny, inlet);
}

static bfs_result_t run_bfs(const bfs_case_t* c) {
    bfs_result_t r = {0};
    size_t ny = c->ny;
    double dy = BFS_H / (double)(ny - 1);
    size_t nx = (size_t)(c->length / dy + 0.5) + 1;
    double dx = c->length / (double)(nx - 1);

    grid* g = grid_create(nx, ny, 1, 0.0, c->length, 0.0, BFS_H, 0.0, 0.0);
    flow_field* field = flow_field_create(nx, ny, 1);
    ns_solver_registry_t* registry = cfd_registry_create();
    ns_solver_t* solver = NULL;
    double* u_prev = (double*)cfd_calloc(nx * ny, sizeof(double));
    if (!g || !field || !registry || !u_prev) {
        snprintf(r.msg, sizeof(r.msg), "allocation failed");
        goto cleanup;
    }
    grid_initialize_uniform(g);
    cfd_registry_register_defaults(registry);

    for (size_t n = 0; n < nx * ny; n++) {
        field->rho[n] = 1.0;
        field->T[n] = 300.0;
    }

    /* Parabola over the upper half, peak 1.5 U for a mean of U */
    bc_inlet_config_t inlet = bc_inlet_config_parabolic(1.5 * BFS_U_MEAN);
    if (bc_inlet_set_range(&inlet, BFS_STEP / BFS_H, 1.0) != CFD_SUCCESS) {
        snprintf(r.msg, sizeof(r.msg), "bc_inlet_set_range refused the step");
        goto cleanup;
    }
    bc_outlet_config_t outlet = bc_outlet_config_zero_gradient();
    apply_bfs_bcs(field, nx, ny, &inlet, &outlet);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = c->dt;
    params.mu = BFS_U_MEAN * BFS_H / c->re;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;
    params.pressure_solver = c->pressure_solver;
    /* Walls and inlet zero-gradient; the outlet fixes the pressure level */
    params.pressure_bc.right = POISSON_WALL_DIRICHLET;
    params.pressure_bc.values.right = 0.0;

    solver = cfd_solver_create(registry, c->solver);
    if (!solver) {
        snprintf(r.msg, sizeof(r.msg), "could not create %s", c->solver);
        goto cleanup;
    }
    cfd_status_t status = solver_init(solver, g, &params);
    if (status != CFD_SUCCESS) {
        r.ok = (status == CFD_ERROR_UNSUPPORTED) ? -1 : 0;
        snprintf(r.msg, sizeof(r.msg), "%s init: %s", c->solver, cfd_get_last_error());
        goto cleanup;
    }
    ns_solver_stats_t stats = ns_solver_stats_default();

    const int check_every = 200;
    int max_steps = (int)(c->t_max / c->dt + 0.5);
    memcpy(u_prev, field->u, nx * ny * sizeof(double));
    for (int step = 1; step <= max_steps; step++) {
        status = solver_step(solver, field, g, &params, &stats);
        if (status != CFD_SUCCESS) {
            snprintf(r.msg, sizeof(r.msg), "step %d: %s", step, cfd_get_last_error());
            goto cleanup;
        }
        apply_bfs_bcs(field, nx, ny, &inlet, &outlet);
        r.steps = step;

        if (step % check_every == 0) {
            double dmax = 0.0;
            for (size_t n = 0; n < nx * ny; n++) {
                dmax = fmax(dmax, fabs(field->u[n] - u_prev[n]));
            }
            memcpy(u_prev, field->u, nx * ny * sizeof(double));
            r.dudt = dmax / (check_every * c->dt);
#ifdef BFS_TRACE
            printf("    t=%7.2f  x_r/h=%7.4f  max|du/dt|=%.3e\n", step * c->dt,
                   last_reattachment(field->u, nx, 1, dx) / BFS_STEP, r.dudt);
            fflush(stdout);
#endif
            if (!isfinite(r.dudt)) {
                snprintf(r.msg, sizeof(r.msg), "diverged by step %d", step);
                goto cleanup;
            }
            if (r.dudt < c->steady_tol) {
                break;
            }
        }
    }
    r.t_end = r.steps * c->dt;

    r.xr_over_h = last_reattachment(field->u, nx, 1, dx) / BFS_STEP;
    double sep = first_separation(field->u, nx, ny - 2, dx);
    if (sep > 0.0) {
        r.upper_sep = sep / BFS_STEP;
        r.upper_reatt = last_reattachment(field->u, nx, ny - 2, dx) / BFS_STEP;
    }

    /* Trapezoidal flux through the inlet and outlet columns */
    double q_in = 0.0, q_out = 0.0;
    for (size_t j = 0; j + 1 < ny; j++) {
        q_in += 0.5 * (field->u[IDX_2D(0, j, nx)] + field->u[IDX_2D(0, j + 1, nx)]) * dy;
        q_out += 0.5 * (field->u[IDX_2D(nx - 1, j, nx)] +
                        field->u[IDX_2D(nx - 1, j + 1, nx)]) * dy;
    }
    r.mass_imbalance = fabs(q_out - q_in) / q_in;
    r.ok = 1;

cleanup:
    if (solver) solver_destroy(solver);
    if (registry) cfd_registry_destroy(registry);
    if (field) flow_field_destroy(field);
    if (g) grid_destroy(g);
    cfd_free(u_prev);
    return r;
}

static void report(const char* name, const bfs_case_t* c, const bfs_result_t* r) {
    printf("\n  %s: Re=%.0f, %s, L=%.0fH, ny=%zu, dt=%g\n", name, c->re, c->solver,
           c->length, c->ny, c->dt);
    printf("    t=%.1f (%d steps), max|du/dt|=%.2e\n", r->t_end, r->steps, r->dudt);
    printf("    x_r/h=%.4f  upper bubble %.3f..%.3f  mass imbalance %.2e\n",
           r->xr_over_h, r->upper_sep, r->upper_reatt, r->mass_imbalance);
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}


/* Common acceptance checks. The band is [ref*(1-below), ref*(1+above)]; see
 * the reference notes at each call for why it is asymmetric. */
static void check_bfs(const bfs_case_t* c, const bfs_result_t* r, double ref,
                      double below, double above) {
    if (r->ok < 0) {
        TEST_IGNORE_MESSAGE(r->msg);
    }
    TEST_ASSERT_TRUE_MESSAGE(r->ok, r->msg);
    TEST_ASSERT_TRUE_MESSAGE(r->dudt < c->steady_tol, "did not reach steady state by t_max");
    TEST_ASSERT_TRUE_MESSAGE(r->xr_over_h >= ref * (1.0 - below),
                             "reattachment length below the reference band");
    TEST_ASSERT_TRUE_MESSAGE(r->xr_over_h <= ref * (1.0 + above),
                             "reattachment length above the reference band");
    TEST_ASSERT_TRUE_MESSAGE(r->mass_imbalance < 1e-2, "outflow does not match inflow");
}

/*
 * Re = 100, the quick case (scalar solver, ~2200 steps).
 *
 * References place the inlet on an upstream channel and give 3.00 (Katsoudas
 * et al.) and 3.05 (Armaly's experiment at Re ~ 102). With the inlet at the
 * step, as here, the reattachment length comes out longer -- Barton (1997)
 * finds an inlet channel shortens it -- and this solver converges to about
 * 3.2: 3.128 at 33 nodes across H, 3.177 at 65. Hence the band sits above the
 * reference: 2% below it for measurement slack, 8% above it for the entrance
 * effect. dt = 0.01 moves x_r by 0.2% against 0.005; a domain of 16H instead
 * of 8H changes nothing in the fourth digit.
 */
void test_bfs_re100(void) {
    bfs_case_t c = {.re = 100.0, .length = 8.0, .ny = 33, .dt = 0.01,
                    .t_max = 60.0, .steady_tol = 1e-4,
                    .solver = NS_SOLVER_TYPE_PROJECTION,
                    .pressure_solver = NS_PRESSURE_SOLVER_PCG_MG};
    bfs_result_t r = run_bfs(&c);
    report("Re=100", &c, &r);
    check_bfs(&c, &r, 3.00, 0.02, 0.08);
    /* Katsoudas et al. find no upper-wall bubble below Re = 400 */
    TEST_ASSERT_EQUAL_DOUBLE(0.0, r.upper_sep);
}

/*
 * Re = 400 (OpenMP solver, a domain of 16H, ~26000 steps).
 *
 * Katsoudas et al. give 8.10 with a short inlet channel; the scalar solver
 * reaches 8.44 here (+4.2%), again above it, and the same band applies. They
 * also report an upper-wall bubble from Re = 400 on; 33 nodes across H do not
 * resolve that marginal bubble, so it is not checked.
 */
void test_bfs_re400(void) {
    bfs_case_t c = {.re = 400.0, .length = 16.0, .ny = 33, .dt = 0.005,
                    .t_max = 400.0, .steady_tol = 1e-4,
                    .solver = NS_SOLVER_TYPE_PROJECTION_OMP,
                    .pressure_solver = NS_PRESSURE_SOLVER_PCG_MG};
    bfs_result_t r = run_bfs(&c);
    report("Re=400", &c, &r);
    check_bfs(&c, &r, 8.10, 0.02, 0.08);
}

int main(int argc, char** argv) {
    UNITY_BEGIN();
    if (argc > 1 && strcmp(argv[1], "re400") == 0) {
        RUN_TEST(test_bfs_re400);
    } else {
        RUN_TEST(test_bfs_re100);
    }
    return UNITY_END();
}
