/**
 * @file test_solver_architecture.c
 * @brief Cross-architecture consistency: every backend of a solver family must
 *        reproduce the scalar reference within 0.1% (ROADMAP 6.1)
 *
 * Each case runs the scalar solver and every other backend of the same family
 * (AVX2/SIMD, OpenMP, CUDA) from the same initial state, then compares the WHOLE
 * field, not a probe point:
 *
 *   max|u_b - u_ref| / max|u_ref|   and the same for v      < 0.1%
 *   max|p_b - p_ref| / range(p_ref), pressure mean removed  < 0.1%
 *
 * Pressure is compared with its mean removed because the zero-gradient pressure
 * Poisson operator fixes it only up to a constant. The four corner nodes are
 * left out: no five-point stencil reads them, so no backend is obliged to agree
 * on them.
 *
 * Each family is run on the problem its solvers are built for:
 *
 *   - Projection: the 33x33 lid-driven cavity. Its pressure solve has walls, not
 *     periodicity, so a periodic problem is outside its contract.
 *   - Explicit Euler: the same cavity (its stencil reads the ghost cells and it
 *     keeps the caller's velocity boundaries), and a periodic Taylor-Green vortex.
 *   - RK2 / RK4: a periodic Taylor-Green vortex only. They wrap the stencil and
 *     make every field periodic after the step, so a lid never enters them.
 *
 * The Taylor-Green grid spans [-h, 2pi] with h = 2pi/(n-2): the periodic ghost
 * copy u[0] = u[n-2] then has period exactly 2pi.
 *
 * A backend that is not compiled in is skipped; a family with no backend to
 * compare against is ignored. A mismatch fails the test: fix the backend, not
 * the tolerance.
 */

#include "../test_macros.h"
#include "lid_driven_cavity_common.h"

#include "cfd/core/indexing.h"

#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/** Backends must agree with the scalar reference to 0.1% (ROADMAP 6.1). */
#define ARCH_CONSISTENCY_TOL 1e-3

#define ARCH_CAVITY_N  33
#define ARCH_CAVITY_RE 100.0
/* t = 1.0: the primary vortex has formed, so the whole field is in play */
#define ARCH_PROJ_STEPS 2000
#define ARCH_PROJ_DT    0.0005
/* Explicit Euler caps its own step at 1e-4 */
#define ARCH_EULER_STEPS 5000
#define ARCH_EULER_DT    0.0001

#define ARCH_TG_N        34
#define ARCH_TG_NU       0.05
#define ARCH_TG_STEPS    400
#define ARCH_TG_DT_RK    0.02 /* t = 8: KE has decayed by e^-1.6 */
#define ARCH_TG_DT_EULER 0.0001

typedef enum { ARCH_CASE_CAVITY, ARCH_CASE_TAYLOR_GREEN } arch_case_t;

typedef struct {
    double du; /* max|du| / max|u_ref| */
    double dv; /* max|dv| / max|v_ref| */
    double dp; /* max|dp| / range(p_ref), both means removed */
} field_diff_t;

/* ============================================================================
 * FIELD COMPARISON
 * ============================================================================ */

static int is_corner(size_t i, size_t j, size_t nx, size_t ny) {
    return (i == 0 || i == nx - 1) && (j == 0 || j == ny - 1);
}

static field_diff_t compare_fields(const flow_field* ref, const flow_field* b) {
    size_t nx = ref->nx, ny = ref->ny;
    double p_ref_mean = 0.0, p_b_mean = 0.0;
    size_t count = 0;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (is_corner(i, j, nx, ny))
                continue;
            p_ref_mean += ref->p[IDX_2D(i, j, nx)];
            p_b_mean += b->p[IDX_2D(i, j, nx)];
            count++;
        }
    }
    p_ref_mean /= (double)count;
    p_b_mean /= (double)count;

    double u_max = 0.0, v_max = 0.0, du = 0.0, dv = 0.0, dp = 0.0;
    double p_lo = INFINITY, p_hi = -INFINITY;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            if (is_corner(i, j, nx, ny))
                continue;
            size_t idx = IDX_2D(i, j, nx);
            double p_ref = ref->p[idx] - p_ref_mean;
            u_max = fmax(u_max, fabs(ref->u[idx]));
            v_max = fmax(v_max, fabs(ref->v[idx]));
            du = fmax(du, fabs(b->u[idx] - ref->u[idx]));
            dv = fmax(dv, fabs(b->v[idx] - ref->v[idx]));
            dp = fmax(dp, fabs((b->p[idx] - p_b_mean) - p_ref));
            p_lo = fmin(p_lo, p_ref);
            p_hi = fmax(p_hi, p_ref);
        }
    }

    field_diff_t d;
    d.du = du / u_max;
    d.dv = dv / v_max;
    d.dp = dp / (p_hi - p_lo);
    return d;
}

/* ============================================================================
 * RUNNERS
 * ============================================================================ */

typedef struct {
    flow_field* field; /* owned by ctx or standalone; release with arch_run_free */
    cavity_context_t* ctx;
    grid* g;
    int unavailable;
    char error_msg[256];
} arch_run_t;

static void arch_run_free(arch_run_t* r) {
    if (r->ctx) {
        cavity_context_destroy(r->ctx);
    } else {
        if (r->field)
            flow_field_destroy(r->field);
        if (r->g)
            grid_destroy(r->g);
    }
    memset(r, 0, sizeof(*r));
}

static arch_run_t run_cavity(const char* type, int steps, double dt) {
    arch_run_t r = {0};
    cavity_sim_result_t sim = cavity_run_with_solver_ctx(type, ARCH_CAVITY_N, ARCH_CAVITY_N,
                                                         ARCH_CAVITY_RE, 1.0, steps, dt, &r.ctx);
    r.unavailable = sim.solver_unavailable;
    if (!sim.success) {
        snprintf(r.error_msg, sizeof(r.error_msg), "%s", sim.error_msg);
        return r;
    }
    r.field = r.ctx->field;
    return r;
}

/* Periodic Taylor-Green vortex, u = cos x sin y, v = -sin x cos y, advanced with
 * the default momentum source switched off. */
static arch_run_t run_taylor_green(const char* type, int steps, double dt) {
    arch_run_t r = {0};
    size_t n = ARCH_TG_N;
    double h = 2.0 * M_PI / (double)(n - 2);

    r.g = grid_create(n, n, 1, -h, 2.0 * M_PI, -h, 2.0 * M_PI, 0.0, 0.0);
    r.field = flow_field_create(n, n, 1);
    if (!r.g || !r.field) {
        snprintf(r.error_msg, sizeof(r.error_msg), "Failed to allocate grid/field");
        return r;
    }
    grid_initialize_uniform(r.g);
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            size_t idx = IDX_2D(i, j, n);
            double x = r.g->x[i], y = r.g->y[j];
            r.field->u[idx] = cos(x) * sin(y);
            r.field->v[idx] = -sin(x) * cos(y);
            r.field->p[idx] = -0.25 * (cos(2.0 * x) + cos(2.0 * y));
            r.field->rho[idx] = 1.0;
            r.field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = ARCH_TG_NU;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* solver = cfd_solver_create(registry, type);
    if (!solver) {
        r.unavailable = 1;
        snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' not available", type);
        cfd_registry_destroy(registry);
        return r;
    }

    cfd_status_t status = solver_init(solver, r.g, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        r.unavailable = 1;
        snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' backend not compiled", type);
    } else if (status != CFD_SUCCESS) {
        snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' init failed (%d)", type, status);
    } else {
        ns_solver_stats_t stats = ns_solver_stats_default();
        for (int step = 0; step < steps; step++) {
            bc_apply_periodic(r.field->u, n, n);
            bc_apply_periodic(r.field->v, n, n);
            bc_apply_periodic(r.field->p, n, n);
            status = solver_step(solver, r.field, r.g, &params, &stats);
            if (status != CFD_SUCCESS) {
                snprintf(r.error_msg, sizeof(r.error_msg), "Solver '%s' step %d failed (%d)", type,
                         step, status);
                break;
            }
        }
    }

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    return r;
}

static arch_run_t run_case(arch_case_t c, const char* type, int steps, double dt) {
    return (c == ARCH_CASE_CAVITY) ? run_cavity(type, steps, dt)
                                   : run_taylor_green(type, steps, dt);
}

/* ============================================================================
 * FAMILY COMPARISON
 * ============================================================================ */

/* types[0] is the scalar reference; the rest are compared against it. */
static void check_family(arch_case_t c, const char* const* types, int n_types, int steps,
                         double dt) {
    arch_run_t ref = run_case(c, types[0], steps, dt);
    if (ref.error_msg[0]) {
        char msg[sizeof(ref.error_msg)];
        snprintf(msg, sizeof(msg), "%s", ref.error_msg);
        arch_run_free(&ref);
        TEST_FAIL_PRINTF("Scalar reference %s failed: %s", types[0], msg);
    }

    int compared = 0;
    int failed = 0;
    for (int k = 1; k < n_types; k++) {
        arch_run_t b = run_case(c, types[k], steps, dt);
        if (b.unavailable) {
            printf("      %-26s SKIPPED (%s)\n", types[k], b.error_msg);
            arch_run_free(&b);
            continue;
        }
        if (b.error_msg[0]) {
            printf("      %-26s FAILED: %s\n", types[k], b.error_msg);
            failed = 1;
            arch_run_free(&b);
            continue;
        }

        field_diff_t d = compare_fields(ref.field, b.field);
        int ok = d.du < ARCH_CONSISTENCY_TOL && d.dv < ARCH_CONSISTENCY_TOL &&
                 d.dp < ARCH_CONSISTENCY_TOL;
        printf("      %-26s du=%.2e dv=%.2e dp=%.2e  %s\n", types[k], d.du, d.dv, d.dp,
               ok ? "ok" : "EXCEEDS 0.1%");
        failed |= !ok;
        compared++;
        arch_run_free(&b);
    }
    arch_run_free(&ref);

    if (failed) {
        TEST_FAIL_MESSAGE("A backend differs from the scalar reference by more than 0.1%");
    }
    if (compared == 0) {
        TEST_IGNORE_MESSAGE("No optimized backend compiled in to compare against");
    }
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

static const char* const PROJECTION_TYPES[] = {
    NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_OMP,
    NS_SOLVER_TYPE_PROJECTION_GPU};
static const char* const EULER_TYPES[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
    NS_SOLVER_TYPE_EXPLICIT_EULER_OMP, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU};
static const char* const RK2_TYPES[] = {NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OPTIMIZED,
                                        NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_GPU};
static const char* const RK4_TYPES[] = {NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OPTIMIZED,
                                        NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_GPU};

#define N_TYPES(a) ((int)(sizeof(a) / sizeof((a)[0])))

void test_projection_cavity(void) {
    printf("\n    Projection, 33x33 cavity Re=100, t=%.1f\n", ARCH_PROJ_STEPS * ARCH_PROJ_DT);
    check_family(ARCH_CASE_CAVITY, PROJECTION_TYPES, N_TYPES(PROJECTION_TYPES), ARCH_PROJ_STEPS,
                 ARCH_PROJ_DT);
}

void test_euler_cavity(void) {
    printf("\n    Explicit Euler, 33x33 cavity Re=100, t=%.1f\n", ARCH_EULER_STEPS * ARCH_EULER_DT);
    check_family(ARCH_CASE_CAVITY, EULER_TYPES, N_TYPES(EULER_TYPES), ARCH_EULER_STEPS,
                 ARCH_EULER_DT);
}

void test_euler_taylor_green(void) {
    printf("\n    Explicit Euler, periodic Taylor-Green\n");
    check_family(ARCH_CASE_TAYLOR_GREEN, EULER_TYPES, N_TYPES(EULER_TYPES), ARCH_TG_STEPS,
                 ARCH_TG_DT_EULER);
}

void test_rk2_taylor_green(void) {
    printf("\n    RK2, periodic Taylor-Green, t=%.0f\n", ARCH_TG_STEPS * ARCH_TG_DT_RK);
    check_family(ARCH_CASE_TAYLOR_GREEN, RK2_TYPES, N_TYPES(RK2_TYPES), ARCH_TG_STEPS,
                 ARCH_TG_DT_RK);
}

void test_rk4_taylor_green(void) {
    printf("\n    RK4, periodic Taylor-Green, t=%.0f\n", ARCH_TG_STEPS * ARCH_TG_DT_RK);
    check_family(ARCH_CASE_TAYLOR_GREEN, RK4_TYPES, N_TYPES(RK4_TYPES), ARCH_TG_STEPS,
                 ARCH_TG_DT_RK);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("CROSS-ARCHITECTURE CONSISTENCY\n");
    printf("========================================\n");
    printf("Every backend vs the scalar reference, whole field, tolerance %.1f%%\n",
           ARCH_CONSISTENCY_TOL * 100.0);

    RUN_TEST(test_projection_cavity);
    RUN_TEST(test_euler_cavity);
    RUN_TEST(test_euler_taylor_green);
    RUN_TEST(test_rk2_taylor_green);
    RUN_TEST(test_rk4_taylor_green);

    return UNITY_END();
}
