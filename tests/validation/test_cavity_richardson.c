/**
 * @file test_cavity_richardson.c
 * @brief Multi-Reynolds grid-convergence study of the lid-driven cavity with
 *        Richardson extrapolation (ROADMAP 6.1)
 *
 * For each Reynolds number the steady cavity is solved on three grids, and four
 * standard functionals are extracted:
 *
 *   u_c    u at the cavity centre (a node on every odd grid)
 *   u_min  minimum of u on the vertical centreline
 *   v_max  maximum of v on the horizontal centreline
 *   v_min  minimum of v on the horizontal centreline
 *
 * The extrema are located to sub-grid accuracy by a parabola through the
 * discrete extremum and its two neighbours.
 *
 * From the three values the study computes, per Celik et al. (2008), the
 * observed order of accuracy p (solved iteratively, so the refinement ratio need
 * not be constant), the Richardson-extrapolated value, and the fine-grid
 * convergence index GCI (safety factor 1.25). It then requires:
 *
 *   - every grid reached steady state (the harness's |d ln KE/dt| < 1e-6 exit);
 *   - monotone convergence: successive changes have the same sign and shrink;
 *   - the observed order inside a band around what the scheme delivers;
 *   - (full validation) a fine-grid GCI below RICH_GCI_MAX, and the extrapolated
 *     value within max(GCI, benchmark tolerance) of the benchmark.
 *
 * Observed order: the interior stencils are O(h^2), but the zero-gradient
 * pressure wall p[0] = p[1] is O(h) and the lid corners are singular; the
 * measured order on centreline quantities is 1.1-1.3 at Re=100. The band
 * guards against a regression below that; it is not a claim of second order.
 *
 * Grids, each triple in the asymptotic range for its Re (measured -- see
 * docs/validation/cavity-grid-convergence.md):
 *
 *   CI:    Re=100   17/33/65
 *   Full:  Re=100   33/65/129
 *          Re=400   65/129/257
 *          Re=1000  129/257/513
 *
 * Coarser grids are not in the asymptotic range at high Re: at Re=1000 a 33x33
 * grid settles to an almost motionless state, and 65 -> 129 still changes u_min
 * by 50%.
 *
 * Runs on the OpenMP projection with the multigrid pressure solve, which needs
 * 2^k+1 points per side -- hence the grids above. On a 257x257 Re=1000 cavity it
 * costs 16 ms/step against 140 ms for the AVX2 CG solve, with velocity fields
 * that agree to 5e-14. The scalar projection also has multigrid but is excluded
 * by the scalar testing policy (no scalar solver in a long-running test).
 */

#include "cavity_reference_data.h"
#include "lid_driven_cavity_common.h"

#include "cfd/core/indexing.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

#define RICH_N_FUNCTIONALS 4
#define RICH_GCI_SAFETY    1.25

/* Observed-order band (see file comment). */
#define RICH_P_MIN 0.8
#define RICH_P_MAX 3.0

/* Upper bound on the fine-grid GCI in full validation. */
#define RICH_GCI_MAX 0.10

/* Physical time budget; the run stops earlier at steady state. */
#define RICH_T_MAX_RE100  40.0
#define RICH_T_MAX_HIGHRE 120.0

typedef struct {
    const char* name;
    double re;
    size_t n[3]; /* coarse -> fine */
    double t_max;
    double ref[RICH_N_FUNCTIONALS]; /* benchmark u_c, u_min, v_max, v_min; 0 = none */
    double ref_tol;                 /* relative accuracy credited to the benchmark */
    const char* ref_source;
} rich_case_t;

static const char* const RICH_NAMES[RICH_N_FUNCTIONALS] = {"u_c", "u_min", "v_max", "v_min"};

/* ============================================================================
 * FUNCTIONALS
 * ============================================================================ */

/* Extremum of the parabola through the discrete extremum and its neighbours. */
static double rich_parabolic_extremum(const double* f, size_t n, int want_min) {
    size_t k = 1;
    for (size_t i = 1; i + 1 < n; i++) {
        if (want_min ? (f[i] < f[k]) : (f[i] > f[k])) {
            k = i;
        }
    }
    double a = f[k - 1], b = f[k], c = f[k + 1];
    double curvature = a - (2.0 * b) + c;
    if (fabs(curvature) < 1e-300) {
        return b;
    }
    double s = 0.5 * (a - c) / curvature;
    return b - (0.25 * (a - c) * s);
}

typedef struct {
    int ok;
    int unavailable;
    char error_msg[256];
    double f[RICH_N_FUNCTIONALS];
    double sim_time;
} rich_grid_result_t;

static rich_grid_result_t rich_solve(size_t n, double re, double t_max) {
    rich_grid_result_t r;
    memset(&r, 0, sizeof(r));

    /* dt: advective CFL 0.25; diffusive nu dt/h^2 = 0.2; and dt <= 1/Re, inside
     * the central-difference advection-diffusion limit 2 nu / U^2. */
    double h = 1.0 / (double)(n - 1);
    double dt = fmin(0.25 * h, fmin(0.2 * h * h * re, 1.0 / re));
    int max_steps = (int)(t_max / dt);

    cavity_context_t* ctx = NULL;
    cavity_sim_result_t sim =
        cavity_run_with_pressure_solver_ctx(NS_SOLVER_TYPE_PROJECTION_OMP, n, n, re, 1.0, max_steps,
                                            dt, NS_PRESSURE_SOLVER_MULTIGRID, &ctx);
    if (!sim.success) {
        r.unavailable = sim.solver_unavailable;
        snprintf(r.error_msg, sizeof(r.error_msg), "%s", sim.error_msg);
        return r;
    }
    if (!sim.converged) {
        snprintf(r.error_msg, sizeof(r.error_msg),
                 "%zux%zu Re=%.0f not steady by t=%.1f (d ln KE/dt = %.2e)", n, n, re, sim.sim_time,
                 sim.final_residual);
        cavity_context_destroy(ctx);
        return r;
    }

    double* col = malloc(n * sizeof(double));
    double* row = malloc(n * sizeof(double));
    if (!col || !row) {
        free(col);
        free(row);
        cavity_context_destroy(ctx);
        snprintf(r.error_msg, sizeof(r.error_msg), "allocation failed");
        return r;
    }
    size_t m = n / 2;
    for (size_t j = 0; j < n; j++)
        col[j] = ctx->field->u[IDX_2D(m, j, n)];
    for (size_t i = 0; i < n; i++)
        row[i] = ctx->field->v[IDX_2D(i, m, n)];

    r.f[0] = col[m];
    r.f[1] = rich_parabolic_extremum(col, n, 1);
    r.f[2] = rich_parabolic_extremum(row, n, 0);
    r.f[3] = rich_parabolic_extremum(row, n, 1);
    r.sim_time = sim.sim_time;
    r.ok = 1;

    printf("      %3zux%-3zu steady at t=%5.1f  u_c=%.5f u_min=%.5f v_max=%.5f v_min=%.5f\n", n, n,
           r.sim_time, r.f[0], r.f[1], r.f[2], r.f[3]);

    free(col);
    free(row);
    cavity_context_destroy(ctx);
    return r;
}

/* ============================================================================
 * RICHARDSON EXTRAPOLATION (Celik et al. 2008)
 * ============================================================================ */

typedef struct {
    int monotone;
    double p;     /* observed order */
    double f_ext; /* extrapolated value */
    double gci;   /* fine-grid convergence index, relative */
} rich_estimate_t;

/* f1 fine, f3 coarse; r21 = h2/h1, r32 = h3/h2. */
static rich_estimate_t rich_estimate(double f1, double f2, double f3, double r21, double r32) {
    rich_estimate_t e = {0};
    double e21 = f2 - f1;
    double e32 = f3 - f2;
    /* Monotone: the two changes have the same sign and the fine-grid one is smaller */
    e.monotone = (e21 != 0.0) && (e32 / e21 > 0.0) && (fabs(e21) < fabs(e32));
    if (!e.monotone) {
        return e;
    }

    /* p = |ln|e32/e21| + q(p)| / ln r21, q(p) = ln((r21^p - 1) / (r32^p - 1)) */
    double ln_ratio = log(e32 / e21);
    double p = ln_ratio / log(r21);
    for (int it = 0; it < 200; it++) {
        double q = log((pow(r21, p) - 1.0) / (pow(r32, p) - 1.0));
        double p_next = fabs(ln_ratio + q) / log(r21);
        if (fabs(p_next - p) < 1e-12) {
            p = p_next;
            break;
        }
        p = p_next;
    }

    double r21p = pow(r21, p);
    e.p = p;
    e.f_ext = ((r21p * f1) - f2) / (r21p - 1.0);
    e.gci = RICH_GCI_SAFETY * fabs(e21 / f1) / (r21p - 1.0);
    return e;
}

/* ============================================================================
 * STUDY
 * ============================================================================ */

static void rich_run_case(const rich_case_t* c, int full) {
    rich_grid_result_t g[3];

    printf("\n    %s on %zu/%zu/%zu\n", c->name, c->n[0], c->n[1], c->n[2]);

    g[0] = rich_solve(c->n[0], c->re, c->t_max);
    if (g[0].unavailable) {
        TEST_IGNORE_MESSAGE("The OpenMP projection is not compiled in");
    }
    TEST_ASSERT_TRUE_MESSAGE(g[0].ok, g[0].error_msg);
    for (int k = 1; k < 3; k++) {
        g[k] = rich_solve(c->n[k], c->re, c->t_max);
        TEST_ASSERT_TRUE_MESSAGE(g[k].ok, g[k].error_msg);
    }

    /* h = 1/(n-1) */
    double r21 = (double)(c->n[2] - 1) / (double)(c->n[1] - 1);
    double r32 = (double)(c->n[1] - 1) / (double)(c->n[0] - 1);

    int failed = 0;
    char msg[256] = "";
    printf("      %-6s %10s %10s  %5s  %6s  %10s  %9s\n", "", "fine", "extrap", "p", "GCI",
           "benchmark", "|ext-ref|");
    for (int q = 0; q < RICH_N_FUNCTIONALS; q++) {
        double f1 = g[2].f[q], f2 = g[1].f[q], f3 = g[0].f[q];
        rich_estimate_t e = rich_estimate(f1, f2, f3, r21, r32);

        if (!e.monotone) {
            printf("      %-6s not monotone: %.6f -> %.6f -> %.6f\n", RICH_NAMES[q], f3, f2, f1);
            snprintf(msg, sizeof(msg), "%s: convergence is not monotone", RICH_NAMES[q]);
            failed = 1;
            continue;
        }

        double ref = c->ref[q];
        double dev = (ref != 0.0) ? fabs(e.f_ext - ref) / fabs(ref) : 0.0;
        double allowed = fmax(e.gci, c->ref_tol);
        if (ref != 0.0) {
            printf("      %-6s %10.6f %10.6f  %5.2f  %5.2f%%  %10.6f  %8.2f%%\n", RICH_NAMES[q], f1,
                   e.f_ext, e.p, 100.0 * e.gci, ref, 100.0 * dev);
        } else {
            printf("      %-6s %10.6f %10.6f  %5.2f  %5.2f%%\n", RICH_NAMES[q], f1, e.f_ext, e.p,
                   100.0 * e.gci);
        }

        if (e.p < RICH_P_MIN || e.p > RICH_P_MAX) {
            snprintf(msg, sizeof(msg), "%s: observed order %.2f outside [%.1f, %.1f]",
                     RICH_NAMES[q], e.p, RICH_P_MIN, RICH_P_MAX);
            failed = 1;
        }
        if (full && e.gci > RICH_GCI_MAX) {
            snprintf(msg, sizeof(msg), "%s: fine-grid GCI %.1f%% exceeds %.0f%%", RICH_NAMES[q],
                     100.0 * e.gci, 100.0 * RICH_GCI_MAX);
            failed = 1;
        }
        if (full && ref != 0.0 && dev > allowed) {
            snprintf(msg, sizeof(msg),
                     "%s: extrapolated %.6f is %.2f%% from benchmark %.6f (allowed %.2f%%)",
                     RICH_NAMES[q], e.f_ext, 100.0 * dev, ref, 100.0 * allowed);
            failed = 1;
        }
    }
    if (c->ref_source) {
        printf("      benchmark: %s\n", c->ref_source);
    }
    if (failed) {
        TEST_FAIL_MESSAGE(msg);
    }
}

/* ============================================================================
 * CASES
 * ============================================================================ */

#if CAVITY_FULL_VALIDATION

/* Benchmark values and their sources: docs/validation/cavity-grid-convergence.md */
static const rich_case_t CASE_RE100 = {"Re=100",
                                       100.0,
                                       {33, 65, 129},
                                       RICH_T_MAX_RE100,
                                       {0.0, -0.2140424, 0.1795728, -0.2538030},
                                       0.005,
                                       "grid-converged literature extrema"};

static const rich_case_t CASE_RE400 = {
    "Re=400", 400.0, {65, 129, 257}, RICH_T_MAX_HIGHRE, {0.0, 0.0, 0.0, 0.0}, 0.0, NULL};

static const rich_case_t CASE_RE1000 = {"Re=1000",
                                        1000.0,
                                        {129, 257, 513},
                                        RICH_T_MAX_HIGHRE,
                                        {0.0, -0.3885698, 0.3769447, -0.5270771},
                                        0.005,
                                        "Botella & Peyret (1998), spectral"};

void test_richardson_re100(void) {
    rich_run_case(&CASE_RE100, 1);
}
void test_richardson_re400(void) {
    rich_run_case(&CASE_RE400, 1);
}
void test_richardson_re1000(void) {
    rich_run_case(&CASE_RE1000, 1);
}

#else

/* CI: the Re=100 study on coarse grids -- the machinery, monotone convergence
 * and the observed order, without the full-validation accuracy gates. */
static const rich_case_t CASE_RE100_CI = {
    "Re=100 (CI)", 100.0, {17, 33, 65}, RICH_T_MAX_RE100, {0.0, 0.0, 0.0, 0.0}, 0.0, NULL};

void test_richardson_re100_ci(void) {
    rich_run_case(&CASE_RE100_CI, 0);
}

#endif

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : NULL;

    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("CAVITY GRID CONVERGENCE (RICHARDSON)\n");
    printf("========================================\n");

#if CAVITY_FULL_VALIDATION
    if (!filter || strcmp(filter, "re100") == 0)
        RUN_TEST(test_richardson_re100);
    if (!filter || strcmp(filter, "re400") == 0)
        RUN_TEST(test_richardson_re400);
    if (!filter || strcmp(filter, "re1000") == 0)
        RUN_TEST(test_richardson_re1000);
#else
    (void)filter;
    RUN_TEST(test_richardson_re100_ci);
#endif

    return UNITY_END();
}
