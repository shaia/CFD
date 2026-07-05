/**
 * @file test_turbulent_channel.c
 * @brief Turbulent channel flow validation for the RANS models (k-eps and SA)
 *
 * Fully-developed turbulent channel flow at Re_tau = 395:
 *   - half-height delta = 1, channel y in [0, 2], x in [0, 4] (x-uniform flow)
 *   - rho = 1, nu = 1/Re_tau, constant streamwise body force f_x = u_tau^2/delta
 *   - exact steady force balance: u_tau = sqrt(f_x * delta) = 1
 *
 * The flow is streamwise-uniform, so this exercises the wall-normal RANS
 * balance: 0 = d/dy[(nu + nu_t) du/dy] + f_x with log-law wall functions.
 *
 * Assertions (for BOTH k-epsilon and Spalart-Allmaras, projection solver):
 *   1. First-node y+ lies in the wall-function validity window [30, 100]
 *      (guards against silent grid/parameter drift).
 *   2. Recovered friction velocity within 10% of the exact value 1.0
 *      (steady momentum balance — the strongest check).
 *   3. u+ matches the log law ln(y+)/kappa + B within 15% for nodes with
 *      30 < y+ < 0.3*Re_tau.
 *   4. Velocity profile symmetric about the centerline within 2%.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* Channel configuration */
#define CH_RE_TAU 395.0
#define CH_DELTA  1.0
#define CH_NX     16
#define CH_NY     21
#define CH_LX     4.0
#define CH_LY     2.0
#define CH_DT     0.002
#define CH_MIN_STEPS   5000
#define CH_MAX_STEPS   40000
#define CH_STEADY_TOL  1e-6

/* Log-law constants (must match turbulence_solver_internal.h) */
#define CH_KAPPA 0.41
#define CH_B     5.2

/* Constant streamwise body force f_x = u_tau^2/delta = 1 */
static void channel_body_force(double x, double y, double z, double t, void* ctx,
                               double* su, double* sv, double* sw) {
    (void)x; (void)y; (void)z; (void)t; (void)ctx;
    *su = 1.0;
    *sv = 0.0;
    *sw = 0.0;
}

/* Impose channel BCs directly on the boundary nodes: periodic in x,
 * no-slip walls at y=0 and y=Ly (the projection solver preserves
 * caller-set boundary values). */
static void apply_channel_bc(flow_field* field) {
    size_t nx = field->nx, ny = field->ny;
    for (size_t j = 0; j < ny; j++) {
        field->u[j * nx] = field->u[j * nx + (nx - 2)];
        field->v[j * nx] = field->v[j * nx + (nx - 2)];
        field->u[j * nx + (nx - 1)] = field->u[j * nx + 1];
        field->v[j * nx + (nx - 1)] = field->v[j * nx + 1];
    }
    for (size_t i = 0; i < nx; i++) {
        field->u[i] = 0.0;
        field->v[i] = 0.0;
        field->u[(ny - 1) * nx + i] = 0.0;
        field->v[(ny - 1) * nx + i] = 0.0;
    }
}

static double compute_ke(const flow_field* field) {
    double ke = 0.0;
    size_t total = field->nx * field->ny;
    for (size_t n = 0; n < total; n++) {
        ke += field->u[n] * field->u[n] + field->v[n] * field->v[n];
    }
    return 0.5 * ke;
}

static void run_channel(turbulence_model_t model, const char* label) {
    const double nu = 1.0 / CH_RE_TAU;

    grid* g = grid_create(CH_NX, CH_NY, 1, 0.0, CH_LX, 0.0, CH_LY, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(CH_NX, CH_NY, 1);
    TEST_ASSERT_NOT_NULL(field);

    const double y_p = g->y[1] - g->y[0];
    const double yplus_first = y_p / nu; /* u_tau = 1 */
    /* Assertion 1: wall-function validity window */
    TEST_ASSERT_TRUE_MESSAGE(yplus_first >= 30.0 && yplus_first <= 100.0,
                             "first-node y+ outside [30, 100]");

    /* IC: plug profile near the expected turbulent bulk velocity, zero at walls */
    const double u_bulk0 = 15.0;
    for (size_t j = 0; j < CH_NY; j++) {
        for (size_t i = 0; i < CH_NX; i++) {
            size_t idx = j * CH_NX + i;
            double y = g->y[j];
            int at_wall = (j == 0 || j == CH_NY - 1);
            field->u[idx] = at_wall ? 0.0 : u_bulk0 * (1.0 - pow(fabs(y - CH_DELTA), 8.0));
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = CH_DT;
    params.max_iter = 1;
    params.mu = nu; /* rho = 1: dynamic == kinematic */
    params.source_func = channel_body_force;
    params.turb_model = model;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.top = BC_TYPE_NOSLIP;
    /* left/right stay PERIODIC (zero-init) */

    /* Turbulence IC: ~5% intensity of the expected bulk velocity */
    double k0 = 1.5 * pow(0.05 * u_bulk0, 2.0);
    double eps0 = pow(0.09, 0.75) * pow(k0, 1.5) / (0.07 * CH_DELTA);
    double nu_tilde0 = 3.0 * nu;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      turbulence_init_uniform(field, &params, k0, eps0, nu_tilde0));

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);
    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(slv);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solver_init(slv, g, &params));

    /* March to steady state (kinetic-energy residual) */
    double prev_ke = compute_ke(field);
    int converged = 0;
    int step = 0;
    for (step = 0; step < CH_MAX_STEPS; step++) {
        apply_channel_bc(field);
        ns_solver_stats_t stats;
        cfd_status_t status = solver_step(slv, field, g, &params, &stats);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "solver step failed");

        double ke = compute_ke(field);
        double residual = fabs(ke - prev_ke) / (prev_ke + 1e-10);
        prev_ke = ke;
        if (residual < CH_STEADY_TOL && step > CH_MIN_STEPS) {
            converged = 1;
            break;
        }
    }
    apply_channel_bc(field);

    /* Assertion 2: recovered u_tau from both walls within 10% of exact 1.0 */
    size_t i_mid = CH_NX / 2;
    double u_p_bot = fabs(field->u[1 * CH_NX + i_mid]);
    double u_p_top = fabs(field->u[(CH_NY - 2) * CH_NX + i_mid]);
    double ut_bot = turbulence_wall_u_tau(u_p_bot, y_p, nu);
    double ut_top = turbulence_wall_u_tau(u_p_top, y_p, nu);

    printf("[%s] steps=%d converged=%d u_tau_bot=%.4f u_tau_top=%.4f "
           "u_p=%.3f y+=%.1f\n",
           label, step, converged, ut_bot, ut_top, u_p_bot, yplus_first);

    TEST_ASSERT_TRUE_MESSAGE(fabs(ut_bot - 1.0) < 0.10,
                             "bottom-wall u_tau deviates >10% from force balance");
    TEST_ASSERT_TRUE_MESSAGE(fabs(ut_top - 1.0) < 0.10,
                             "top-wall u_tau deviates >10% from force balance");

    /* Assertion 3: log-law profile for 30 < y+ < 0.3*Re_tau (bottom half) */
    for (size_t j = 1; j < CH_NY / 2; j++) {
        double y = g->y[j];
        double yplus = y / nu; /* u_tau = 1 */
        if (yplus <= 30.0 || yplus >= 0.3 * CH_RE_TAU) {
            continue;
        }
        double u_plus = field->u[j * CH_NX + i_mid] / ut_bot;
        double u_plus_log = log(yplus) / CH_KAPPA + CH_B;
        double rel_err = fabs(u_plus - u_plus_log) / u_plus_log;
        printf("[%s] y+=%.1f u+=%.2f log-law=%.2f err=%.1f%%\n",
               label, yplus, u_plus, u_plus_log, 100.0 * rel_err);
        TEST_ASSERT_TRUE_MESSAGE(rel_err < 0.15,
                                 "u+ deviates >15% from the log law");
    }

    /* Assertion 4: symmetry about the centerline within 2% */
    double u_max = 0.0;
    for (size_t j = 0; j < CH_NY; j++) {
        double u = fabs(field->u[j * CH_NX + i_mid]);
        if (u > u_max) u_max = u;
    }
    for (size_t j = 1; j < CH_NY / 2; j++) {
        double u_lo = field->u[j * CH_NX + i_mid];
        double u_hi = field->u[(CH_NY - 1 - j) * CH_NX + i_mid];
        TEST_ASSERT_TRUE_MESSAGE(fabs(u_lo - u_hi) / u_max < 0.02,
                                 "velocity profile asymmetric >2%");
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

static void test_channel_kepsilon(void) {
    run_channel(TURB_MODEL_K_EPSILON, "k-epsilon");
}

static void test_channel_spalart_allmaras(void) {
    run_channel(TURB_MODEL_SPALART_ALLMARAS, "SA");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_channel_kepsilon);
    RUN_TEST(test_channel_spalart_allmaras);
    return UNITY_END();
}
