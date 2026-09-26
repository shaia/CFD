/**
 * @file test_taylor_green_decay.c
 * @brief Extended-time Taylor-Green decay-rate verification (ROADMAP 6.1)
 *
 * test_taylor_green_vortex.c stops at t = 0.2 with nu = 0.01, where the kinetic
 * energy has decayed by 0.8% -- too little for a 10% tolerance to tell a right
 * decay rate from a wrong one. This test runs until most of the energy is gone
 * (nu*t = 0.5, KE down by e^-2 = 86%; e^-4 = 98% in full validation) and fits the
 * rate instead of reading one ratio:
 *
 *   - ln KE(t) is fitted by least squares over ~400 samples; the slope must be
 *     -4 nu. Separate fits over the first and last thirds must agree, so a rate
 *     that drifts as the flow decays cannot hide in the average.
 *   - The rate error and the velocity L2 error must fall at second order under
 *     grid refinement: the scheme is O(h^2) in space and dt is tied to h.
 *
 * Configuration: one cell of the vortex array, [0, pi]^2, with walls on nodes.
 *
 *   u =  sin x cos y e^{-2 nu t},  v = -cos x sin y e^{-2 nu t},
 *   p =  (cos 2x + cos 2y) e^{-4 nu t} / 4
 *
 * The walls are streamlines (normal velocity zero) and dp/dn = 0 there, so the
 * projection's zero-gradient pressure walls are exact for this flow. The
 * tangential wall velocity decays with the vortex; the harness writes its exact
 * value before every step, as the cavity harness writes the lid.
 *
 * Why not the periodic vortex: the projection pressure solve has walls, not
 * periodicity, so on a periodic domain it imposes dp/dn = 0 where none exists.
 * Measured there, the rate settles near 0.92 of 4 nu and does not improve with
 * refinement -- a configuration error, not a discretization error. The
 * pseudo-compressible Euler/RK2/RK4 solvers are periodic-only and reach about
 * 0.42 of 4 nu on every grid: their pressure update dp/dt = -0.1 div(u) cannot
 * follow the e^{-4 nu t} decay of the vortex pressure. See
 * docs/validation/taylor-green-decay.md.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef CAVITY_FULL_VALIDATION
#define CAVITY_FULL_VALIDATION 0
#endif

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define TGD_NU          0.05
#define TGD_CFL         0.2 /* dt = 0.2 h: advective CFL 0.2, diffusive nu dt/h^2 < 0.1 */
#define TGD_SAMPLES     400
#define TGD_MAX_SAMPLES 1024

#if CAVITY_FULL_VALIDATION
#define TGD_T_END  20.0 /* KE down by e^-4 */
#define TGD_N_FINE 129
#else
#define TGD_T_END  10.0 /* KE down by e^-2 */
#define TGD_N_FINE 65
#endif

/* Measured with dt = 0.2 h, t = 10 (rate error / late-early / L2u):
 *   n=17: 0.93% / 5.4% / 1.5e-2   n=33: 0.17% / 1.0% / 3.0e-3
 *   n=65: 0.041% / 0.20% / 6.7e-4
 * Tolerances sit about 4x above the n=65 values. */
#define TGD_RATE_TOL  2e-3 /* |fitted rate / 4 nu - 1| on the fine grid */
#define TGD_DRIFT_TOL 5e-3 /* |late-third rate - early-third rate| / 4 nu */
#define TGD_L2U_TOL   3e-3 /* relative L2 error of u at t_end */
#define TGD_MIN_ORDER 1.7  /* observed order of rate and L2u errors */

/* The scalar reference runs only a small, quick case (scalar testing policy). */
#define TGD_N_SCALAR         33
#define TGD_RATE_TOL_SCALAR  5e-3
#define TGD_DRIFT_TOL_SCALAR 2e-2

/* ============================================================================
 * EXACT SOLUTION
 * ============================================================================ */

static void tgd_exact(double x, double y, double t, double* u, double* v, double* p) {
    double d = exp(-2.0 * TGD_NU * t);
    *u = sin(x) * cos(y) * d;
    *v = -cos(x) * sin(y) * d;
    *p = 0.25 * (cos(2.0 * x) + cos(2.0 * y)) * d * d;
}

static void tgd_set_walls(flow_field* f, const grid* g, double t) {
    size_t n = f->nx;
    double p;
    for (size_t k = 0; k < n; k++) {
        size_t edges[4] = {IDX_2D(k, 0, n), IDX_2D(k, n - 1, n), IDX_2D(0, k, n),
                           IDX_2D(n - 1, k, n)};
        double xs[4] = {g->x[k], g->x[k], g->x[0], g->x[n - 1]};
        double ys[4] = {g->y[0], g->y[n - 1], g->y[k], g->y[k]};
        for (int e = 0; e < 4; e++) {
            tgd_exact(xs[e], ys[e], t, &f->u[edges[e]], &f->v[edges[e]], &p);
        }
    }
}

/* Trapezoidal kinetic energy over the whole box, walls included. */
static double tgd_kinetic_energy(const flow_field* f, double h) {
    size_t n = f->nx;
    double ke = 0.0;
    for (size_t j = 0; j < n; j++) {
        double wj = (j == 0 || j == n - 1) ? 0.5 : 1.0;
        for (size_t i = 0; i < n; i++) {
            double wi = (i == 0 || i == n - 1) ? 0.5 : 1.0;
            size_t idx = IDX_2D(i, j, n);
            ke += wi * wj * 0.5 * (f->u[idx] * f->u[idx] + f->v[idx] * f->v[idx]);
        }
    }
    return ke * h * h;
}

/* Least-squares slope of y against t over samples [a, b). */
static double tgd_slope(const double* t, const double* y, int a, int b) {
    double st = 0.0, sy = 0.0, stt = 0.0, sty = 0.0;
    int m = b - a;
    for (int k = a; k < b; k++) {
        st += t[k];
        sy += y[k];
        stt += t[k] * t[k];
        sty += t[k] * y[k];
    }
    return (m * sty - st * sy) / (m * stt - st * st);
}

/* ============================================================================
 * RUNNER
 * ============================================================================ */

typedef struct {
    int unavailable;
    char error_msg[256];
    double rate;       /* fitted d(ln KE)/dt over the run, divided by -4 nu */
    double rate_early; /* same, first third */
    double rate_late;  /* same, last third */
    double l2_u;       /* relative L2 error of u at t_end */
    double t_end;
    int steps;
} tgd_result_t;

static tgd_result_t tgd_run(const char* solver_type, size_t n, double t_end) {
    tgd_result_t r;
    memset(&r, 0, sizeof(r));

    double h = M_PI / (double)(n - 1);
    double dt = TGD_CFL * h;
    grid* g = grid_create(n, n, 1, 0.0, M_PI, 0.0, M_PI, 0.0, 0.0);
    flow_field* f = flow_field_create(n, n, 1);
    ns_solver_registry_t* registry = cfd_registry_create();
    ns_solver_t* solver = NULL;
    if (!g || !f || !registry) {
        snprintf(r.error_msg, sizeof(r.error_msg), "allocation failed");
        goto cleanup;
    }
    grid_initialize_uniform(g);
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            size_t idx = IDX_2D(i, j, n);
            tgd_exact(g->x[i], g->y[j], 0.0, &f->u[idx], &f->v[idx], &f->p[idx]);
            f->rho[idx] = 1.0;
            f->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = TGD_NU;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0; /* the default forcing would drive the vortex */
    params.source_amplitude_v = 0.0;

    cfd_registry_register_defaults(registry);
    solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        r.unavailable = 1;
        snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' not available", solver_type);
        goto cleanup;
    }
    cfd_status_t status = solver_init(solver, g, &params);
    if (status != CFD_SUCCESS) {
        r.unavailable = (status == CFD_ERROR_UNSUPPORTED);
        snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' init returned %d", solver_type,
                 status);
        goto cleanup;
    }

    int total_steps = (int)ceil(t_end / dt - 1e-9);
    int stride = total_steps / TGD_SAMPLES;
    if (stride < 1)
        stride = 1;

    double ts[TGD_MAX_SAMPLES], ys[TGD_MAX_SAMPLES];
    int ns = 0;
    ts[ns] = 0.0;
    ys[ns++] = log(tgd_kinetic_energy(f, h));

    ns_solver_stats_t stats = ns_solver_stats_default();
    double t = 0.0;
    for (int step = 1; step <= total_steps; step++) {
        tgd_set_walls(f, g, t);
        status = solver_step(solver, f, g, &params, &stats);
        if (status != CFD_SUCCESS) {
            snprintf(r.error_msg, sizeof(r.error_msg), "step %d returned %d", step, status);
            goto cleanup;
        }
        t = step * dt;
        tgd_set_walls(f, g, t);
        if ((step % stride == 0 || step == total_steps) && ns < TGD_MAX_SAMPLES) {
            double ke = tgd_kinetic_energy(f, h);
            if (!(ke > 0.0) || !isfinite(ke)) {
                snprintf(r.error_msg, sizeof(r.error_msg), "KE not positive at step %d", step);
                goto cleanup;
            }
            ts[ns] = t;
            ys[ns++] = log(ke);
        }
    }

    double sum_err = 0.0, sum_ref = 0.0;
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            double ue, ve, pe;
            tgd_exact(g->x[i], g->y[j], t, &ue, &ve, &pe);
            double d = f->u[IDX_2D(i, j, n)] - ue;
            sum_err += d * d;
            sum_ref += ue * ue;
        }
    }

    double scale = -1.0 / (4.0 * TGD_NU);
    r.rate = scale * tgd_slope(ts, ys, 0, ns);
    r.rate_early = scale * tgd_slope(ts, ys, 0, ns / 3);
    r.rate_late = scale * tgd_slope(ts, ys, 2 * ns / 3, ns);
    r.l2_u = sqrt(sum_err / sum_ref);
    r.t_end = t;
    r.steps = total_steps;

    printf("      %-22s n=%3zu t=%.1f  rate/4nu=%.5f (early %.5f, late %.5f)  L2u=%.2e\n",
           solver_type, n, r.t_end, r.rate, r.rate_early, r.rate_late, r.l2_u);

cleanup:
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

/* ============================================================================
 * CHECKS
 * ============================================================================ */

static void tgd_check_backend(const char* solver_type, size_t n, double rate_tol,
                              double drift_tol) {
    printf("\n    %s: decay rate over t = %.0f (KE down by e^-%.0f)\n", solver_type, TGD_T_END,
           4.0 * TGD_NU * TGD_T_END);
    tgd_result_t r = tgd_run(solver_type, n, TGD_T_END);
    if (r.unavailable) {
        TEST_IGNORE_MESSAGE(r.error_msg);
    }
    TEST_ASSERT_TRUE_MESSAGE(r.error_msg[0] == '\0', r.error_msg);

    TEST_ASSERT_TRUE_MESSAGE(fabs(r.rate - 1.0) < rate_tol,
                             "Fitted KE decay rate differs from 4 nu");
    TEST_ASSERT_TRUE_MESSAGE(fabs(r.rate_late - r.rate_early) < drift_tol,
                             "Decay rate drifts between the early and late thirds of the run");
    if (n >= TGD_N_FINE) {
        TEST_ASSERT_TRUE_MESSAGE(r.l2_u < TGD_L2U_TOL, "Velocity error at t_end exceeds tolerance");
    }
}

void test_decay_rate_scalar(void) {
    tgd_check_backend(NS_SOLVER_TYPE_PROJECTION, TGD_N_SCALAR, TGD_RATE_TOL_SCALAR,
                      TGD_DRIFT_TOL_SCALAR);
}

void test_decay_rate_avx2(void) {
    tgd_check_backend(NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, TGD_N_FINE, TGD_RATE_TOL, TGD_DRIFT_TOL);
}

void test_decay_rate_omp(void) {
    tgd_check_backend(NS_SOLVER_TYPE_PROJECTION_OMP, TGD_N_FINE, TGD_RATE_TOL, TGD_DRIFT_TOL);
}

void test_decay_rate_gpu(void) {
    tgd_check_backend(NS_SOLVER_TYPE_PROJECTION_GPU, TGD_N_FINE, TGD_RATE_TOL, TGD_DRIFT_TOL);
}

/**
 * The rate error and the velocity error must both fall at second order.
 * Runs on the fastest backend available: AVX2, else OpenMP.
 */
void test_decay_rate_grid_convergence(void) {
    const char* candidates[] = {NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_OMP};
    size_t sizes[] = {TGD_N_FINE / 4 + 1, TGD_N_FINE / 2 + 1, TGD_N_FINE};
    printf("\n    Grid convergence of the decay rate, t = %.0f\n", TGD_T_END);

    for (int c = 0; c < 2; c++) {
        tgd_result_t r[3];
        int unavailable = 0;
        for (int k = 0; k < 3; k++) {
            r[k] = tgd_run(candidates[c], sizes[k], TGD_T_END);
            if (r[k].unavailable) {
                unavailable = 1;
                break;
            }
            TEST_ASSERT_TRUE_MESSAGE(r[k].error_msg[0] == '\0', r[k].error_msg);
        }
        if (unavailable)
            continue;

        for (int k = 0; k < 2; k++) {
            double rate_order = log(fabs(r[k].rate - 1.0) / fabs(r[k + 1].rate - 1.0)) / log(2.0);
            double l2_order = log(r[k].l2_u / r[k + 1].l2_u) / log(2.0);
            printf("      %zu -> %zu: rate-error order %.2f, L2u order %.2f\n", sizes[k],
                   sizes[k + 1], rate_order, l2_order);
            TEST_ASSERT_TRUE_MESSAGE(rate_order > TGD_MIN_ORDER,
                                     "Decay-rate error does not converge at second order");
            TEST_ASSERT_TRUE_MESSAGE(l2_order > TGD_MIN_ORDER,
                                     "Velocity error does not converge at second order");
        }
        return;
    }
    TEST_IGNORE_MESSAGE("Neither the AVX2 nor the OpenMP projection is compiled in");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("TAYLOR-GREEN EXTENDED-TIME DECAY RATE\n");
    printf("========================================\n");
    printf("Box [0,pi]^2, nu = %.2f, dt = %.1f h, t_end = %.0f%s\n", TGD_NU, TGD_CFL, TGD_T_END,
           CAVITY_FULL_VALIDATION ? " (full validation)" : "");

#if !CAVITY_FULL_VALIDATION
    RUN_TEST(test_decay_rate_scalar);
#endif
    RUN_TEST(test_decay_rate_avx2);
    RUN_TEST(test_decay_rate_omp);
    RUN_TEST(test_decay_rate_gpu);
    RUN_TEST(test_decay_rate_grid_convergence);

    return UNITY_END();
}
