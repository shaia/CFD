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
#include "channel_dns_reference.h"

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* Channel configuration */
#define CH_RE_TAU 395.0

/* Friction Reynolds number actually used. Defaults to CH_RE_TAU; an optional
 * argv[1] overrides it so the closure error can be swept across Re_tau
 * without an edit-and-rebuild cycle per point. */
static double g_re_tau = CH_RE_TAU;
#define CH_DELTA  1.0
#define CH_NX     16
#define CH_NY     21
#define CH_LX     4.0
#define CH_LY     2.0
#define CH_DT     0.002
#define CH_MIN_STEPS   5000
#define CH_MAX_STEPS   40000
#define CH_STEADY_TOL  1e-6

/* Wall-function validity requires the first interior node to sit in 30 < y+ < 100.
 * On a uniform grid y+_first = (CH_LY / (ny - 1)) * Re_tau, so a FIXED ny makes y+
 * drift with Re_tau -- which is why Re_tau=180 (y+=18) and Re_tau=1000 (y+=100)
 * both fall outside the usable band on the stock 21-point grid. Scaling ny with
 * Re_tau holds y+ put and makes several Reynolds numbers valid at once, which is
 * what a train/holdout split needs.
 *
 * The target is the stock grid's own y+ so the default is unchanged:
 * 1 + round(2.0 * 395 / 39.5) == 21 exactly. */
#define CH_YPLUS_TARGET 39.5

static size_t g_ny = CH_NY;

/* Points across the channel that put the first node at CH_YPLUS_TARGET. */
static size_t channel_ny_for(double re_tau) {
    long n = lround(CH_LY * re_tau / CH_YPLUS_TARGET);
    if (n < 8) {
        n = 8; /* floor: fewer than this cannot resolve the profile at all */
    }
    return (size_t)n + 1;
}

/* DNS table matching the current Re_tau, or none. The tolerance is wide enough
 * to accept the nominal label (395) for the actual value (392.24), since users
 * will reach for the round number. */
static const double* g_dns_yplus      = NULL;
static const double* g_dns_uplus      = NULL;
static const double* g_dns_KPLUS_ptr  = NULL;
static const double* g_dns_UVPLUS_ptr = NULL;
static size_t        g_dns_n          = 0;

static void channel_select_dns(double re_tau) {
    if (fabs(re_tau - CHAN_DNS_395_RETAU) <= 3.0) {
        g_dns_yplus      = CHAN_DNS_395_YPLUS;
        g_dns_uplus      = CHAN_DNS_395_UPLUS;
        g_dns_KPLUS_ptr  = CHAN_DNS_395_KPLUS;
        g_dns_UVPLUS_ptr = CHAN_DNS_395_UVPLUS;
        g_dns_n          = CHAN_DNS_395_N;
    } else if (fabs(re_tau - CHAN_DNS_590_RETAU) <= 3.0) {
        g_dns_yplus      = CHAN_DNS_590_YPLUS;
        g_dns_uplus      = CHAN_DNS_590_UPLUS;
        g_dns_KPLUS_ptr  = CHAN_DNS_590_KPLUS;
        g_dns_UVPLUS_ptr = CHAN_DNS_590_UVPLUS;
        g_dns_n          = CHAN_DNS_590_N;
    } else {
        g_dns_yplus      = NULL;
        g_dns_uplus      = NULL;
        g_dns_KPLUS_ptr  = NULL;
        g_dns_UVPLUS_ptr = NULL;
        g_dns_n          = 0;
    }
}

/* CH_DT is the stable step for the stock 21-point grid (convective CFL ~0.3 at
 * u_bulk ~15). dy shrinks as ny grows with Re_tau, so a FIXED dt drives CFL up
 * with it -- 0.77 at ny=52, 1.14 at ny=77, 1.5 at ny=102, where the explicit
 * scheme simply fails. Scaling dt with dy holds CFL at the value the stock grid
 * was tuned for. Step budgets scale inversely so physical time is preserved.
 *
 * At the default ny the ratio is exactly 1, so dt and both step counts are
 * unchanged. */
static double channel_dy(void) {
    return CH_LY / (double)(g_ny - 1);
}

static double channel_dt(void) {
    const double dy_default = CH_LY / (double)(CH_NY - 1);
    return CH_DT * (channel_dy() / dy_default);
}

/* The steady-state test compares a PER-STEP relative change in kinetic energy
 * against CH_STEADY_TOL, so its meaning is tied to dt. Scaling dt with the grid
 * (above) shrinks the per-step change proportionally, which would make the
 * criterion easier -- a refined run would stop LESS converged than the stock
 * one, silently. Scaling the threshold the same way keeps it a bound on the
 * rate of change rather than on the change per step. At the default dt the
 * ratio is exactly 1, so the stock run is unaffected. */
static double channel_steady_tol(void) {
    return CH_STEADY_TOL * (channel_dt() / CH_DT);
}

static int channel_scale_steps(int steps) {
    const double dy_default = CH_LY / (double)(CH_NY - 1);
    double f = dy_default / channel_dy(); /* >= 1 when the grid is refined */
    double scaled = (double)steps * f;
    if (scaled > 2.0e6) {
        scaled = 2.0e6; /* hard cap so a bad Re_tau cannot run unbounded */
    }
    return (int)scaled;
}

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
    const double nu = 1.0 / g_re_tau;

    grid* g = grid_create(CH_NX, g_ny, 1, 0.0, CH_LX, 0.0, CH_LY, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(CH_NX, g_ny, 1);
    TEST_ASSERT_NOT_NULL(field);

    const double y_p = g->y[1] - g->y[0];
    const double yplus_first = y_p / nu; /* u_tau = 1 */
    /* Assertion 1: wall-function validity window */
    TEST_ASSERT_TRUE_MESSAGE(yplus_first >= 30.0 && yplus_first <= 100.0,
                             "first-node y+ outside [30, 100]");

    /* IC: plug profile near the expected turbulent bulk velocity, zero at walls */
    const double u_bulk0 = 15.0;
    for (size_t j = 0; j < g_ny; j++) {
        for (size_t i = 0; i < CH_NX; i++) {
            size_t idx = j * CH_NX + i;
            double y = g->y[j];
            int at_wall = (j == 0 || j == g_ny - 1);
            field->u[idx] = at_wall ? 0.0 : u_bulk0 * (1.0 - pow(fabs(y - CH_DELTA), 8.0));
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = channel_dt();
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
    const double steady_tol = channel_steady_tol();
    const int max_steps = channel_scale_steps(CH_MAX_STEPS);
    const int min_steps = channel_scale_steps(CH_MIN_STEPS);
    for (step = 0; step < max_steps; step++) {
        apply_channel_bc(field);
        ns_solver_stats_t stats;
        cfd_status_t status = solver_step(slv, field, g, &params, &stats);
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, "solver step failed");

        double ke = compute_ke(field);
        double residual = fabs(ke - prev_ke) / (prev_ke + 1e-10);
        prev_ke = ke;
        if (residual < steady_tol && step > min_steps) {
            converged = 1;
            break;
        }
    }
    apply_channel_bc(field);

    /* Assertion 2: recovered u_tau from both walls within 10% of exact 1.0 */
    size_t i_mid = CH_NX / 2;
    double u_p_bot = fabs(field->u[1 * CH_NX + i_mid]);
    double u_p_top = fabs(field->u[(g_ny - 2) * CH_NX + i_mid]);
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
    for (size_t j = 1; j < g_ny / 2; j++) {
        double y = g->y[j];
        double yplus = y / nu; /* u_tau = 1 */
        if (yplus <= 30.0 || yplus >= 0.3 * g_re_tau) {
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

    /* Closure error against DNS ground truth.
     *
     * Reported, not asserted. The log-law check above cannot measure closure
     * error on its own -- the wall function imposes the log law at the first
     * node, so the model is graded on reproducing what it is built to
     * reproduce. DNS is external truth and breaks that circularity.
     *
     * Three quantities, in increasing order of how hard they are for an
     * eddy-viscosity closure:
     *   u+     mean velocity        (the calibration target; models do well)
     *   -uv+   turbulent shear      (nu_t * dU/dy, the Boussinesq assumption)
     *   k+     turbulent energy     (k-epsilon only; SA carries no TKE)
     *
     * Everything is normalised by the RECOVERED u_tau, not the nominal 1.0, so
     * a model that gets the friction wrong is not also penalised twice here.
     *
     * Note the DNS k+ peak sits near y+ ~ 17, below the first computational
     * node at y+ ~ 39. Wall functions bridge that region by construction, so
     * the peak itself is never resolved and only the log layer is compared. */
    if (g_dns_n > 0) {
        const double ut2 = ut_bot * ut_bot;
        const double dy = channel_dy();
        double su = 0.0, sk = 0.0, sv = 0.0;
        int nu_cnt = 0, nk_cnt = 0, nv_cnt = 0;

        for (size_t j = 1; j < g_ny / 2; j++) {
            double yplus = g->y[j] / nu; /* u_tau = 1 in the forcing balance */
            double u_dns = channel_dns_interp(g_dns_yplus, g_dns_uplus, g_dns_n, yplus);
            if (u_dns <= 0.0) {
                continue; /* outside the tabulated range */
            }
            size_t idx = j * CH_NX + i_mid;

            double u_plus = field->u[idx] / ut_bot;
            double eu = fabs(u_plus - u_dns) / u_dns;
            su += eu * eu;
            nu_cnt++;

            /* Modelled turbulent shear stress: -<u'v'> = nu_t * dU/dy. */
            double uv_dns = channel_dns_interp(g_dns_yplus, g_dns_UVPLUS_ptr, g_dns_n, yplus);
            double dudy = (field->u[(j + 1) * CH_NX + i_mid] -
                           field->u[(j - 1) * CH_NX + i_mid]) / (2.0 * dy);
            double uv_mod = field->nu_t[idx] * dudy / ut2;
            double euv = 0.0;
            if (uv_dns > 0.01) {
                euv = fabs(uv_mod - uv_dns) / uv_dns;
                sv += euv * euv;
                nv_cnt++;
            }

            /* Turbulent kinetic energy (k-epsilon only). */
            double ek = 0.0;
            double k_dns = channel_dns_interp(g_dns_yplus, g_dns_KPLUS_ptr, g_dns_n, yplus);
            double k_mod = 0.0;
            if (model == TURB_MODEL_K_EPSILON && k_dns > 0.01) {
                k_mod = field->turb_k[idx] / ut2;
                ek = fabs(k_mod - k_dns) / k_dns;
                sk += ek * ek;
                nk_cnt++;
            }


            /* Machine-readable dump for the locality study: can a correction
             * built ONLY from local invariants reproduce the DNS target, or
             * does it need the non-local coordinate y/delta? Columns are
             * deliberately split into local and non-local groups. */
            if (model == TURB_MODEL_K_EPSILON && k_dns > 0.01 && k_mod > 1e-12) {
                double eps_c = fmax(field->turb_eps[idx], 1e-30);
                double s_star = fabs(dudy) * field->turb_k[idx] / eps_c;
                double re_t   = field->turb_k[idx] * field->turb_k[idx] / (nu * eps_c);
                double nut_p  = field->nu_t[idx] / nu;
                printf("CSV,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                       g_re_tau,
                       yplus,                      /* local  */
                       s_star,                     /* local  */
                       re_t,                       /* local  */
                       nut_p,                      /* local  */
                       g->y[j] / CH_DELTA,         /* NON-local */
                       k_mod,                      /* model k+ */
                       k_dns);                     /* target k+ */
            }
            printf("[%s] DNS y+=%6.1f  u+=%5.2f/%5.2f (%4.1f%%)"
                   "  -uv+=%5.3f/%5.3f (%5.1f%%)"
                   "  k+=%5.2f/%5.2f (%5.1f%%)\n",
                   label, yplus, u_plus, u_dns, 100.0 * eu,
                   uv_mod, uv_dns, 100.0 * euv,
                   k_mod, k_dns, 100.0 * ek);
        }
        if (nu_cnt > 0) {
            printf("[%s] DNS-RMS  u+=%.2f%%  -uv+=%.2f%%  k+=%s  (%d nodes)\n",
                   label, 100.0 * sqrt(su / nu_cnt),
                   nv_cnt ? 100.0 * sqrt(sv / nv_cnt) : 0.0,
                   nk_cnt ? "see below" : "n/a (no TKE in this model)", nu_cnt);
            if (nk_cnt) {
                printf("[%s] DNS-RMS  k+=%.2f%% over %d nodes\n",
                       label, 100.0 * sqrt(sk / nk_cnt), nk_cnt);
            }
        }
    }

    /* Assertion 4: symmetry about the centerline within 2% */
    double u_max = 0.0;
    for (size_t j = 0; j < g_ny; j++) {
        double u = fabs(field->u[j * CH_NX + i_mid]);
        if (u > u_max) u_max = u;
    }
    for (size_t j = 1; j < g_ny / 2; j++) {
        double u_lo = field->u[j * CH_NX + i_mid];
        double u_hi = field->u[(g_ny - 1 - j) * CH_NX + i_mid];
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

int main(int argc, char** argv) {
    if (argc > 1) {
        double re = atof(argv[1]);
        if (re <= 0.0) {
            printf("Invalid Re_tau '%s'; must be positive\n", argv[1]);
            return 1;
        }
        g_re_tau = re;
        /* Hold first-node y+ fixed as Re_tau moves; without this the wall
         * function silently leaves its valid band. */
        g_ny = channel_ny_for(g_re_tau);
    }
    channel_select_dns(g_re_tau);
    printf("[channel] Re_tau = %.0f  ny = %zu  y+_first = %.1f  ref = %s\n",
           g_re_tau, g_ny, (CH_LY / (double)(g_ny - 1)) * g_re_tau,
           g_dns_n > 0 ? "MKM DNS" : "none (log-law only)");

    UNITY_BEGIN();
    RUN_TEST(test_channel_kepsilon);
    RUN_TEST(test_channel_spalart_allmaras);
    return UNITY_END();
}
