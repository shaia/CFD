/**
 * Viscous-scheme selection tests (ns_solver_params_t.viscous_scheme)
 *
 * - Zero-initialized and default params select the explicit scheme, and only
 *   the scalar and OpenMP projection solvers advertise the implicit capability
 * - Every other solver rejects an implicit scheme with CFD_ERROR_UNSUPPORTED at
 *   init and at step; an unknown value is CFD_ERROR_INVALID everywhere; an
 *   implicit scheme with a turbulence model is CFD_ERROR_UNSUPPORTED; so do
 *   the exported GPU entry points that bypass solver_step()
 * - One step on a discrete Laplacian eigenmode multiplies it by exactly the
 *   theta-method amplification factor, on both backends, across a change of dt
 *   and scheme on one solver
 * - Crank-Nicolson converges at second order in time and backward Euler at
 *   first on that mode, at time steps all above the explicit limit
 * - A viscous-bound cavity at 20x the explicit limit stays bounded under both
 *   implicit schemes while the explicit scheme does not
 * - At a small dt the implicit schemes agree with the explicit one, and differ
 *   from it (the option is not ignored)
 * - The OpenMP projection reproduces the scalar one, in 2D and 3D
 * - compute_time_step() drops the viscous limit for an implicit scheme
 */

#include "test_solver_helpers.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

/* Solvers that must refuse an implicit scheme */
static const char* const EXPLICIT_ONLY_SOLVERS[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
    NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_OPTIMIZED,
    NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_OPTIMIZED,
    NS_SOLVER_TYPE_EXPLICIT_EULER_GPU, NS_SOLVER_TYPE_PROJECTION_GPU,
    NS_SOLVER_TYPE_RK2_GPU, NS_SOLVER_TYPE_RK4_GPU,
};
#define NUM_EXPLICIT_ONLY (sizeof(EXPLICIT_ONLY_SOLVERS) / sizeof(EXPLICIT_ONLY_SOLVERS[0]))

/* Solvers that implement it */
static const char* const IMPLICIT_SOLVERS[] = {
    NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OMP,
};
#define NUM_IMPLICIT (sizeof(IMPLICIT_SOLVERS) / sizeof(IMPLICIT_SOLVERS[0]))

static const ns_viscous_scheme_t IMPLICIT_SCHEMES[] = {
    NS_VISCOUS_SCHEME_BACKWARD_EULER, NS_VISCOUS_SCHEME_CRANK_NICOLSON,
};

/* Eigenmode runs: w = sin(pi x) sin(pi y) on a 17x17 unit grid */
#define MODE_N   17
#define MODE_NU  0.1
/* The CG viscous solve runs at relative tolerance 1e-10 */
#define AMPLIFICATION_TOL 1e-9

/* Cavity runs: nu = 1 makes the viscous limit bind, not the convective one */
#define CAVITY_N      33
#define CAVITY_NU     1.0
#define CAVITY_STEPS  100
/* OpenMP differs from scalar only by reduction order in the CG solves */
#define BACKEND_MATCH_TOL 1e-10

static ns_solver_registry_t* create_registry(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);
    return registry;
}

static ns_solver_params_t make_params(ns_viscous_scheme_t scheme, double nu, double dt) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = nu;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;
    params.viscous_scheme = scheme;
    return params;
}

/** The explicit viscous limit dt < h^2 / (2 * nu * ndim), without the CFL factor */
static double explicit_viscous_limit(double h, double nu, int ndim) {
    return h * h / (2.0 * nu * ndim);
}

/**
 * Create and init a solver. Returns NULL when the solver is not registered or
 * its init reports CFD_ERROR_UNSUPPORTED (this build lacks the backend).
 */
static ns_solver_t* create_solver(ns_solver_registry_t* registry, const char* type,
                                  const grid* g, const ns_solver_params_t* params) {
    ns_solver_t* slv = cfd_solver_create(registry, type);
    if (!slv) {
        return NULL;
    }
    cfd_status_t status = solver_init(slv, g, params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(slv);
        return NULL;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, type);
    return slv;
}

//=============================================================================
// DEFAULTS AND CAPABILITIES
//=============================================================================

void test_default_scheme_is_explicit(void) {
    TEST_ASSERT_EQUAL_INT(0, NS_VISCOUS_SCHEME_EXPLICIT);
    TEST_ASSERT_EQUAL_INT(NS_VISCOUS_SCHEME_EXPLICIT,
                          ns_solver_params_default().viscous_scheme);

    ns_solver_params_t zeroed;
    memset(&zeroed, 0, sizeof(zeroed));
    TEST_ASSERT_EQUAL_INT(NS_VISCOUS_SCHEME_EXPLICIT, zeroed.viscous_scheme);
}

void test_capability_flag(void) {
    ns_solver_registry_t* registry = create_registry();

    for (size_t s = 0; s < NUM_IMPLICIT; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, IMPLICIT_SOLVERS[s]);
        if (!slv) {
            continue;
        }
        TEST_ASSERT_TRUE_MESSAGE(slv->capabilities & NS_SOLVER_CAP_IMPLICIT_VISCOUS,
                                 IMPLICIT_SOLVERS[s]);
        solver_destroy(slv);
    }
    for (size_t s = 0; s < NUM_EXPLICIT_ONLY; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, EXPLICIT_ONLY_SOLVERS[s]);
        if (!slv) {
            continue;
        }
        TEST_ASSERT_FALSE_MESSAGE(slv->capabilities & NS_SOLVER_CAP_IMPLICIT_VISCOUS,
                                  EXPLICIT_ONLY_SOLVERS[s]);
        solver_destroy(slv);
    }
    cfd_registry_destroy(registry);
}

//=============================================================================
// REJECTION
//=============================================================================

void test_explicit_only_solvers_reject_implicit_at_init(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    int checked = 0;
    for (size_t s = 0; s < NUM_EXPLICIT_ONLY; s++) {
        for (size_t k = 0; k < 2; k++) {
            ns_solver_t* slv = cfd_solver_create(registry, EXPLICIT_ONLY_SOLVERS[s]);
            if (!slv) {
                continue;
            }
            ns_solver_params_t params = make_params(IMPLICIT_SCHEMES[k], MODE_NU, 1e-3);
            cfd_status_t status = solver_init(slv, g, &params);
            solver_destroy(slv);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
                                          EXPLICIT_ONLY_SOLVERS[s]);
            checked++;
        }
    }
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No solver was checked");
}

void test_explicit_only_solvers_reject_implicit_at_step(void) {
    /* A scheme set on params after a successful init must still be refused:
     * running explicit at a dt chosen for an implicit scheme would blow up. */
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(MODE_N, MODE_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    int checked = 0;
    for (size_t s = 0; s < NUM_EXPLICIT_ONLY; s++) {
        ns_solver_params_t params = make_params(NS_VISCOUS_SCHEME_EXPLICIT, MODE_NU, 1e-3);
        ns_solver_t* slv = create_solver(registry, EXPLICIT_ONLY_SOLVERS[s], g, &params);
        if (!slv) {
            continue;
        }
        params.viscous_scheme = NS_VISCOUS_SCHEME_BACKWARD_EULER;
        ns_solver_stats_t stats = ns_solver_stats_default();
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED,
                                      solver_step(slv, field, g, &params, &stats),
                                      EXPLICIT_ONLY_SOLVERS[s]);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED,
                                      solver_solve(slv, field, g, &params, &stats),
                                      EXPLICIT_ONLY_SOLVERS[s]);
        solver_destroy(slv);
        checked++;
    }
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No solver was checked");
}

void test_exported_gpu_entry_points_reject_implicit(void) {
    /* These are exported and reach the kernels without solver_step(), so the
     * capability check there does not cover them. Without CUDA the stubs refuse
     * everything, which is the same answer for a different reason. */
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(MODE_N, MODE_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);
    gpu_config_t config = gpu_config_default();

    for (size_t k = 0; k < 2; k++) {
        ns_solver_params_t params = make_params(IMPLICIT_SCHEMES[k], MODE_NU, 1e-3);
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED,
                              solve_navier_stokes_gpu(field, g, &params, &config));
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED,
                              solve_projection_method_gpu(field, g, &params, &config));
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED,
                              solve_rk2_method_gpu(field, g, &params, &config));
        TEST_ASSERT_EQUAL_INT(CFD_ERROR_UNSUPPORTED,
                              solve_rk4_method_gpu(field, g, &params, &config));
    }
    flow_field_destroy(field);
    grid_destroy(g);
}

void test_unknown_scheme_rejected(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_params_t params = make_params((ns_viscous_scheme_t)7, MODE_NU, 1e-3);

    const char* const* lists[] = {EXPLICIT_ONLY_SOLVERS, IMPLICIT_SOLVERS};
    const size_t sizes[] = {NUM_EXPLICIT_ONLY, NUM_IMPLICIT};
    int checked = 0;
    for (size_t l = 0; l < 2; l++) {
        for (size_t s = 0; s < sizes[l]; s++) {
            ns_solver_t* slv = cfd_solver_create(registry, lists[l][s]);
            if (!slv) {
                continue;
            }
            cfd_status_t status = solver_init(slv, g, &params);
            solver_destroy(slv);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID, status, lists[l][s]);
            checked++;
        }
    }
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No solver was checked");
}

void test_turbulence_with_implicit_rejected(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(MODE_N, MODE_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    int checked = 0;
    for (size_t s = 0; s < NUM_IMPLICIT; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, IMPLICIT_SOLVERS[s]);
        if (!slv) {
            continue;
        }
        ns_solver_params_t params =
            make_params(NS_VISCOUS_SCHEME_BACKWARD_EULER, MODE_NU, 1e-3);
        params.turb_model = TURB_MODEL_K_EPSILON;
        cfd_status_t status = solver_init(slv, g, &params);
        solver_destroy(slv);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, status, IMPLICIT_SOLVERS[s]);

        /* And at step, when the model is switched on after an implicit init */
        params.turb_model = TURB_MODEL_NONE;
        slv = create_solver(registry, IMPLICIT_SOLVERS[s], g, &params);
        if (!slv) {
            continue;
        }
        params.turb_model = TURB_MODEL_K_EPSILON;
        ns_solver_stats_t stats = ns_solver_stats_default();
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED,
                                      solver_step(slv, field, g, &params, &stats),
                                      IMPLICIT_SOLVERS[s]);
        solver_destroy(slv);
        checked++;
    }
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No implicit solver was checked");
}

//=============================================================================
// EXACT AMPLIFICATION ON A DISCRETE EIGENMODE
//
// u = v = 0, w = sin(pi x) sin(pi y) on a 2D grid. w enters neither the
// convective terms nor the 2D divergence, so the pressure solve sees a zero
// right-hand side and the step reduces to the viscous update alone. w vanishes
// on every boundary node and is an eigenvector of the discrete Laplacian with
//
//     lam = -(8 / h^2) sin^2(pi h / 2),
//
// so one theta-step multiplies it by exactly
//
//     g = (1 + (1 - theta) nu dt lam) / (1 - theta nu dt lam).
//
// This pins theta, the sign of the Helmholtz shift and the scaling of its
// right-hand side at once: getting any of them wrong changes g.
//=============================================================================

static double mode_lambda(double h) {
    double s = sin(M_PI * h / 2.0);
    return -(8.0 / (h * h)) * s * s;
}

static double theta_of(ns_viscous_scheme_t scheme) {
    return scheme == NS_VISCOUS_SCHEME_CRANK_NICOLSON ? 0.5 : 1.0;
}

static double amplification(ns_viscous_scheme_t scheme, double nu, double dt, double lam) {
    double theta = theta_of(scheme);
    return (1.0 + (1.0 - theta) * nu * dt * lam) / (1.0 - theta * nu * dt * lam);
}

static void init_mode(flow_field* field, const grid* g) {
    size_t n = field->nx * field->ny;
    memset(field->u, 0, n * sizeof(double));
    memset(field->v, 0, n * sizeof(double));
    memset(field->p, 0, n * sizeof(double));
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            field->w[IDX_2D(i, j, field->nx)] = sin(M_PI * g->x[i]) * sin(M_PI * g->y[j]);
            field->rho[IDX_2D(i, j, field->nx)] = 1.0;
        }
    }
}

/** max over the interior of |w - factor * w0| / max|w0| */
static double mode_error(const flow_field* field, const grid* g, double factor) {
    double worst = 0.0;
    for (size_t j = 1; j < field->ny - 1; j++) {
        for (size_t i = 1; i < field->nx - 1; i++) {
            double w0 = sin(M_PI * g->x[i]) * sin(M_PI * g->y[j]);
            double e = fabs(field->w[IDX_2D(i, j, field->nx)] - factor * w0);
            if (e > worst || isnan(e)) {
                worst = e;
            }
        }
    }
    return worst;
}

void test_one_step_matches_theta_amplification(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(MODE_N, MODE_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    double h = 1.0 / (double)(MODE_N - 1);
    double lam = mode_lambda(h);
    double limit = explicit_viscous_limit(h, MODE_NU, 2);

    /* Below, near and far above the explicit limit */
    const double dts[] = {0.2 * limit, 3.0 * limit, 50.0 * limit};

    int checked = 0;
    for (size_t s = 0; s < NUM_IMPLICIT; s++) {
        /* One solver across every (scheme, dt) pair: the owned viscous solver
         * has to be rebuilt each time the shift moves, not reused stale. */
        ns_solver_params_t params =
            make_params(NS_VISCOUS_SCHEME_BACKWARD_EULER, MODE_NU, dts[0]);
        ns_solver_t* slv = create_solver(registry, IMPLICIT_SOLVERS[s], g, &params);
        if (!slv) {
            printf("  %s unavailable (skipping)\n", IMPLICIT_SOLVERS[s]);
            continue;
        }
        for (size_t k = 0; k < 2; k++) {
            for (size_t d = 0; d < sizeof(dts) / sizeof(dts[0]); d++) {
                params.viscous_scheme = IMPLICIT_SCHEMES[k];
                params.dt = dts[d];
                init_mode(field, g);

                ns_solver_stats_t stats = ns_solver_stats_default();
                TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS,
                                              solver_step(slv, field, g, &params, &stats),
                                              IMPLICIT_SOLVERS[s]);

                double gfac = amplification(IMPLICIT_SCHEMES[k], MODE_NU, dts[d], lam);
                double err = mode_error(field, g, gfac);
                printf("  %-16s %s dt=%5.1fx limit  g=%+.6f  err=%.2e\n",
                       IMPLICIT_SOLVERS[s], k == 0 ? "BE" : "CN", dts[d] / limit, gfac, err);
                TEST_ASSERT_TRUE_MESSAGE(err < AMPLIFICATION_TOL,
                                         "one step did not apply the theta amplification");
                checked++;
            }
        }
        solver_destroy(slv);
    }
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No implicit solver was checked");
}

/**
 * Error at T against the exact semi-discrete decay exp(nu lam T), for dt, dt/2
 * and dt/4 -- all above the explicit limit.
 */
static void measure_order(ns_viscous_scheme_t scheme, double* order_out) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(MODE_N, MODE_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(MODE_N, MODE_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    double h = 1.0 / (double)(MODE_N - 1);
    double lam = mode_lambda(h);
    const double T = 0.4;
    const int base_steps = 8;  /* dt = 0.05, 5x the explicit limit */
    double errors[3];

    for (int r = 0; r < 3; r++) {
        int steps = base_steps << r;
        double dt = T / steps;
        ns_solver_params_t params = make_params(scheme, MODE_NU, dt);
        ns_solver_t* slv = create_solver(registry, NS_SOLVER_TYPE_PROJECTION, g, &params);
        TEST_ASSERT_NOT_NULL(slv);
        init_mode(field, g);

        ns_solver_stats_t stats = ns_solver_stats_default();
        for (int n = 0; n < steps; n++) {
            TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, solver_step(slv, field, g, &params, &stats));
        }
        solver_destroy(slv);
        errors[r] = mode_error(field, g, exp(MODE_NU * lam * T));
    }

    double o1 = log2(errors[0] / errors[1]);
    double o2 = log2(errors[1] / errors[2]);
    printf("  %s errors %.3e %.3e %.3e  orders %.3f %.3f\n",
           scheme == NS_VISCOUS_SCHEME_CRANK_NICOLSON ? "CN" : "BE",
           errors[0], errors[1], errors[2], o1, o2);
    *order_out = o2;

    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
}

void test_crank_nicolson_is_second_order(void) {
    double order = 0.0;
    measure_order(NS_VISCOUS_SCHEME_CRANK_NICOLSON, &order);
    TEST_ASSERT_DOUBLE_WITHIN(0.15, 2.0, order);
}

void test_backward_euler_is_first_order(void) {
    double order = 0.0;
    measure_order(NS_VISCOUS_SCHEME_BACKWARD_EULER, &order);
    TEST_ASSERT_DOUBLE_WITHIN(0.15, 1.0, order);
}

//=============================================================================
// CAVITY: STABILITY, CONSISTENCY, BACKENDS
//=============================================================================

typedef struct {
    cfd_status_t status;  /**< First failing step status, or CFD_SUCCESS */
    double max_speed;     /**< max |u|, |v| at the end (or when it failed) */
    flow_field* field;    /**< Final field; caller destroys */
} cavity_run;

/**
 * Lid-driven cavity at nu = CAVITY_NU: u = 1 on the top row, held there by the
 * projection's boundary handling, everything else at rest.
 */
static cavity_run run_cavity(ns_solver_registry_t* registry, const char* type,
                             ns_viscous_scheme_t scheme, double dt, int steps, size_t nz) {
    cavity_run run = {CFD_SUCCESS, 0.0, NULL};
    size_t n = (nz > 1) ? 9 : CAVITY_N;
    grid* g = grid_create(n, n, nz, 0.0, 1.0, 0.0, 1.0, 0.0, nz > 1 ? 1.0 : 0.0);
    flow_field* field = flow_field_create(n, n, nz);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    size_t plane = n * n;
    size_t total = plane * nz;
    for (size_t idx = 0; idx < total; idx++) {
        field->u[idx] = field->v[idx] = field->w[idx] = field->p[idx] = 0.0;
        field->rho[idx] = 1.0;
    }
    for (size_t k = 0; k < nz; k++) {
        for (size_t i = 0; i < n; i++) {
            field->u[k * plane + IDX_2D(i, n - 1, n)] = 1.0;
        }
    }

    ns_solver_params_t params = make_params(scheme, CAVITY_NU, dt);
    ns_solver_t* slv = create_solver(registry, type, g, &params);
    if (!slv) {
        flow_field_destroy(field);
        grid_destroy(g);
        run.status = CFD_ERROR_UNSUPPORTED;
        return run;
    }

    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int s = 0; s < steps && run.status == CFD_SUCCESS; s++) {
        run.status = solver_step(slv, field, g, &params, &stats);
    }
    solver_destroy(slv);
    grid_destroy(g);

    for (size_t idx = 0; idx < total; idx++) {
        double m = fmax(fabs(field->u[idx]), fabs(field->v[idx]));
        if (m > run.max_speed || isnan(m)) {
            run.max_speed = m;
        }
    }
    run.field = field;
    return run;
}

static double cavity_limit(void) {
    return explicit_viscous_limit(1.0 / (double)(CAVITY_N - 1), CAVITY_NU, 2);
}

void test_implicit_is_stable_past_the_explicit_limit(void) {
    ns_solver_registry_t* registry = create_registry();
    double dt = 20.0 * cavity_limit();

    cavity_run exp_run = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION,
                                    NS_VISCOUS_SCHEME_EXPLICIT, dt, CAVITY_STEPS, 1);
    printf("  explicit at 20x: status=%d max speed=%.3e\n", exp_run.status, exp_run.max_speed);
    TEST_ASSERT_TRUE_MESSAGE(exp_run.status != CFD_SUCCESS || exp_run.max_speed > 10.0,
                             "explicit run at 20x the viscous limit should not stay bounded "
                             "(if it does, this test proves nothing)");
    flow_field_destroy(exp_run.field);

    for (size_t k = 0; k < 2; k++) {
        cavity_run run = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION,
                                    IMPLICIT_SCHEMES[k], dt, CAVITY_STEPS, 1);
        printf("  %s at 20x: status=%d max speed=%.3e\n", k == 0 ? "BE" : "CN",
               run.status, run.max_speed);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run.status);
        TEST_ASSERT_TRUE(test_flow_field_is_valid(run.field));
        /* The lid is the only forcing; a stable run stays bounded by it */
        TEST_ASSERT_TRUE_MESSAGE(run.max_speed <= 1.0 + 1e-6,
                                 "implicit run exceeded the lid speed");
        flow_field_destroy(run.field);
    }
    cfd_registry_destroy(registry);
}

/** max over u and v of ||a - b||_2 / ||a||_2 */
static double relative_difference(const flow_field* a, const flow_field* b) {
    size_t n = a->nx * a->ny * a->nz;
    double u_rel = test_compute_l2_error(a->u, b->u, n) / test_compute_l2_norm(a->u, n);
    double v_rel = test_compute_l2_error(a->v, b->v, n) / test_compute_l2_norm(a->v, n);
    return fmax(u_rel, v_rel);
}

/**
 * Relative difference between an implicit and the explicit run to the same
 * time, at dt = fraction * the explicit limit.
 */
static double difference_from_explicit(ns_solver_registry_t* registry,
                                       ns_viscous_scheme_t scheme, double fraction,
                                       int steps) {
    double dt = fraction * cavity_limit();
    cavity_run ref = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION,
                                NS_VISCOUS_SCHEME_EXPLICIT, dt, steps, 1);
    cavity_run run = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION, scheme, dt, steps, 1);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, ref.status);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run.status);
    double diff = relative_difference(ref.field, run.field);
    flow_field_destroy(ref.field);
    flow_field_destroy(run.field);
    return diff;
}

void test_implicit_agrees_with_explicit_at_small_dt(void) {
    /* Each scheme is a consistent O(dt) discretization of the same semi-discrete
     * system, so two of them differ by O(dt) at a fixed time: halving dt must
     * halve the difference. A difference that did not shrink would mean the
     * implicit step solves a different problem; one that vanished would mean
     * the option was accepted and ignored. */
    ns_solver_registry_t* registry = create_registry();

    for (size_t k = 0; k < 2; k++) {
        double coarse = difference_from_explicit(registry, IMPLICIT_SCHEMES[k], 0.1, 50);
        double fine = difference_from_explicit(registry, IMPLICIT_SCHEMES[k], 0.05, 100);
        printf("  %s vs explicit at t = 5x limit: dt 0.1x %.3e, dt 0.05x %.3e, ratio %.2f\n",
               k == 0 ? "BE" : "CN", coarse, fine, coarse / fine);
        TEST_ASSERT_TRUE_MESSAGE(fine > 1e-8, "implicit scheme accepted but not applied");
        TEST_ASSERT_TRUE_MESSAGE(coarse < 2e-2, "implicit strays from explicit at small dt");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(0.3, 2.0, coarse / fine,
                                          "difference from explicit is not O(dt)");
    }
    cfd_registry_destroy(registry);
}

static void check_omp_matches_scalar(size_t nz) {
    ns_solver_registry_t* registry = create_registry();
    double h = (nz > 1) ? 1.0 / 8.0 : 1.0 / (double)(CAVITY_N - 1);
    double dt = 20.0 * explicit_viscous_limit(h, CAVITY_NU, nz > 1 ? 3 : 2);
    const int steps = 20;

    for (size_t k = 0; k < 2; k++) {
        cavity_run omp = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION_OMP,
                                    IMPLICIT_SCHEMES[k], dt, steps, nz);
        if (omp.status == CFD_ERROR_UNSUPPORTED && omp.field == NULL) {
            cfd_registry_destroy(registry);
            TEST_IGNORE_MESSAGE("projection_omp not available");
        }
        cavity_run ref = run_cavity(registry, NS_SOLVER_TYPE_PROJECTION,
                                    IMPLICIT_SCHEMES[k], dt, steps, nz);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, ref.status);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, omp.status);
        TEST_ASSERT_TRUE(ref.max_speed <= 1.0 + 1e-6);

        double diff = relative_difference(ref.field, omp.field);
        printf("  %s %s: OMP vs scalar relative difference %.3e\n",
               nz > 1 ? "3D" : "2D", k == 0 ? "BE" : "CN", diff);
        TEST_ASSERT_TRUE_MESSAGE(diff < BACKEND_MATCH_TOL,
                                 "OpenMP implicit projection differs from scalar");
        flow_field_destroy(ref.field);
        flow_field_destroy(omp.field);
    }
    cfd_registry_destroy(registry);
}

void test_omp_matches_scalar_2d(void) {
    check_omp_matches_scalar(1);
}

void test_omp_matches_scalar_3d(void) {
    check_omp_matches_scalar(9);
}

//=============================================================================
// TIME STEP
//=============================================================================

void test_compute_time_step_drops_viscous_limit(void) {
    grid* g = grid_create(CAVITY_N, CAVITY_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(CAVITY_N, CAVITY_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);
    size_t total = (size_t)CAVITY_N * CAVITY_N;
    for (size_t idx = 0; idx < total; idx++) {
        field->u[idx] = 1.0;
        field->v[idx] = field->w[idx] = field->p[idx] = 0.0;
        field->rho[idx] = 1.0;
    }

    ns_solver_params_t exp_params = make_params(NS_VISCOUS_SCHEME_EXPLICIT, CAVITY_NU, 0.0);
    compute_time_step(field, g, &exp_params);

    for (size_t k = 0; k < 2; k++) {
        ns_solver_params_t imp_params = make_params(IMPLICIT_SCHEMES[k], CAVITY_NU, 0.0);
        compute_time_step(field, g, &imp_params);
        printf("  compute_time_step: explicit %.3e, %s %.3e\n", exp_params.dt,
               k == 0 ? "BE" : "CN", imp_params.dt);
        /* Viscous-bound here: dropping the diffusion limit must raise dt */
        TEST_ASSERT_TRUE(imp_params.dt > 5.0 * exp_params.dt);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_default_scheme_is_explicit);
    RUN_TEST(test_capability_flag);

    RUN_TEST(test_explicit_only_solvers_reject_implicit_at_init);
    RUN_TEST(test_explicit_only_solvers_reject_implicit_at_step);
    RUN_TEST(test_exported_gpu_entry_points_reject_implicit);
    RUN_TEST(test_unknown_scheme_rejected);
    RUN_TEST(test_turbulence_with_implicit_rejected);

    RUN_TEST(test_one_step_matches_theta_amplification);
    RUN_TEST(test_crank_nicolson_is_second_order);
    RUN_TEST(test_backward_euler_is_first_order);

    RUN_TEST(test_implicit_is_stable_past_the_explicit_limit);
    RUN_TEST(test_implicit_agrees_with_explicit_at_small_dt);
    RUN_TEST(test_omp_matches_scalar_2d);
    RUN_TEST(test_omp_matches_scalar_3d);

    RUN_TEST(test_compute_time_step_drops_viscous_limit);

    return UNITY_END();
}
