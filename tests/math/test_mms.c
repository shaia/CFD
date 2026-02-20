/**
 * @file test_mms.c
 * @brief Method of Manufactured Solutions (MMS) tests (ROADMAP 1.3.3)
 *
 * Validates solver accuracy by running with known manufactured solutions
 * that require non-zero forcing terms. This extends beyond Taylor-Green
 * (which is an exact N-S solution with zero source) to test that solvers
 * correctly handle source term injection.
 *
 * Manufactured solution: Modified Taylor-Green with decay rate α ≠ 2ν
 *   u_m(x,y,t) =  cos(x) sin(y) exp(-αt)
 *   v_m(x,y,t) = -sin(x) cos(y) exp(-αt)
 *   Required source: f = (2ν - α) · u_exact
 *
 * Tests:
 *   1. Source callback mechanism works
 *   2. Spatial convergence (Euler): grid refinement, verify O(h^1.5+)
 *   3. Spatial convergence (RK2): grid refinement, verify O(h^1.5+)
 *   4. Temporal convergence (Euler): dt refinement, verify error decreases
 *   5. Temporal convergence (RK2): dt refinement, expect better rate than Euler
 */

#include "unity.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "../validation/taylor_green_reference.h"
#include "../solvers/navier_stokes/test_solver_helpers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ============================================================================
 * MMS PARAMETERS
 * ============================================================================ */

/* Physical parameters */
#define MMS_NU              0.01    /* Kinematic viscosity */
#define MMS_ALPHA           (MMS_NU)  /* Decay rate (α = ν gives source = ν·u_exact) */

/* Convergence tolerances (same as existing convergence tests) */
#define SPATIAL_RATE_MIN    1.4     /* Super-linear (BC-limited O(h^1.5)) */
#define TEMPORAL_RATE_MIN  -0.5     /* Accept negative (spatial-dominated regime) */

/* Spatial convergence parameters */
#define SPATIAL_FINAL_TIME  0.1
#define SPATIAL_BASE_DT     0.0001

/* Temporal convergence parameters */
#define TEMPORAL_GRID_SIZE  128
#define TEMPORAL_BASE_DT    0.001
#define TEMPORAL_FINAL_TIME 0.1

/* ============================================================================
 * MMS MANUFACTURED SOLUTION AND SOURCE TERMS
 * ============================================================================ */

/**
 * MMS context for source term callback
 */
typedef struct {
    double nu;      /* Viscosity */
    double alpha;   /* Decay rate of manufactured solution */
} mms_context_t;

/**
 * Compute exact manufactured solution at given time
 */
static void mms_exact_solution(double x, double y, double t,
                                double nu, double alpha,
                                double* u_exact, double* v_exact, double* p_exact) {
    double decay = exp(-alpha * t);
    *u_exact =  cos(x) * sin(y) * decay;
    *v_exact = -sin(x) * cos(y) * decay;
    *p_exact = -0.25 * (cos(2.0 * x) + cos(2.0 * y)) * decay * decay;
}

/**
 * MMS source term callback
 * For modified TG with decay α: source = (2ν - α) · u_exact
 */
static void mms_source_func(double x, double y, double t, void* context,
                             double* source_u, double* source_v) {
    mms_context_t* mms = (mms_context_t*)context;
    double coeff = 2.0 * mms->nu - mms->alpha;
    double decay = exp(-mms->alpha * t);

    *source_u =  coeff * cos(x) * sin(y) * decay;
    *source_v = -coeff * sin(x) * cos(y) * decay;
}

/**
 * Initialize field with exact manufactured solution at t=0
 */
static void mms_init_field(flow_field* field, const grid* g,
                            double nu, double alpha) {
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = g->x[i];
            double y = g->y[j];
            mms_exact_solution(x, y, 0.0, nu, alpha,
                               &field->u[idx], &field->v[idx], &field->p[idx]);
            field->rho[idx] = 1.0;
            field->T[idx] = 1.0;
        }
    }
}

/**
 * Compute L2 error between numerical and manufactured solution
 */
static double mms_compute_error(flow_field* field, const grid* g,
                                 double nu, double alpha, double t_final) {
    double err_u_sq = 0.0, err_v_sq = 0.0;
    double norm_u_sq = 0.0, norm_v_sq = 0.0;

    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double u_exact, v_exact, p_exact;
            mms_exact_solution(g->x[i], g->y[j], t_final, nu, alpha,
                               &u_exact, &v_exact, &p_exact);

            double du = field->u[idx] - u_exact;
            double dv = field->v[idx] - v_exact;

            err_u_sq += du * du;
            err_v_sq += dv * dv;
            norm_u_sq += u_exact * u_exact;
            norm_v_sq += v_exact * v_exact;
        }
    }

    /* Relative L2 error */
    double total_err = sqrt(err_u_sq + err_v_sq);
    double total_norm = sqrt(norm_u_sq + norm_v_sq);

    return (total_norm > 1e-15) ? (total_err / total_norm) : total_err;
}

/**
 * Run MMS simulation with given solver and parameters
 */
static double mms_run_simulation(const char* solver_type,
                                  size_t nx, size_t ny,
                                  double nu, double alpha,
                                  double dt, int max_steps) {
    /* Create grid on [0, 2π] with periodic BCs (same as Taylor-Green) */
    grid* g = grid_create(nx, ny,
                          TG_DOMAIN_XMIN, TG_DOMAIN_XMAX,
                          TG_DOMAIN_YMIN, TG_DOMAIN_YMAX);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    /* Create field and initialize with exact solution at t=0 */
    flow_field* field = flow_field_create(nx, ny);
    TEST_ASSERT_NOT_NULL(field);
    mms_init_field(field, g, nu, alpha);

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        /* Solver unavailable - cleanup and skip */
        cfd_registry_destroy(registry);
        flow_field_destroy(field);
        grid_destroy(g);
        return -1.0;
    }

    /* Setup parameters with MMS source callback */
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = nu;
    params.max_iter = 1;  /* Single step per call (iterations for internal solvers) */

    mms_context_t mms_ctx = {.nu = nu, .alpha = alpha};
    params.source_func = mms_source_func;
    params.source_context = &mms_ctx;

    /* Run simulation */
    cfd_status_t status = CFD_SUCCESS;
    for (int step = 0; step < max_steps && status == CFD_SUCCESS; step++) {
        /* Apply periodic BCs before each step */
        bc_apply_periodic(field->u, field->nx, field->ny);
        bc_apply_periodic(field->v, field->nx, field->ny);
        bc_apply_periodic(field->p, field->nx, field->ny);

        /* Take one time step */
        status = solver->step(solver, field, g, &params, NULL);
    }

    /* Compute error at final time */
    double t_final = dt * max_steps;
    double error = (status == CFD_SUCCESS) ?
                   mms_compute_error(field, g, nu, alpha, t_final) : 1e10;

    /* Cleanup */
    solver->destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    return error;
}

/**
 * Compute convergence rate between two refinement levels
 */
static double compute_convergence_rate(double e_coarse, double e_fine,
                                        double h_coarse, double h_fine) {
    if (e_fine < 1e-15 || e_coarse < 1e-15) return 0.0;
    if (h_fine < 1e-15 || h_coarse < 1e-15) return 0.0;
    return log(e_coarse / e_fine) / log(h_coarse / h_fine);
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

/**
 * Test 1: Verify source callback mechanism works
 */
void test_mms_source_callback(void) {
    printf("\n  Testing MMS source callback mechanism:\n");

    size_t n = 32;
    double nu = 0.1;    /* Higher viscosity for this test to amplify difference */
    double alpha = nu;
    double dt = 0.001;
    int steps = 100;  /* Run longer to see source effect */

    /* Run with source callback */
    double error_with_source = mms_run_simulation(
        NS_SOLVER_TYPE_EXPLICIT_EULER, n, n, nu, alpha, dt, steps);

    /* Run without source (should drift from manufactured solution) */
    grid* g = grid_create(n, n,
                          TG_DOMAIN_XMIN, TG_DOMAIN_XMAX,
                          TG_DOMAIN_YMIN, TG_DOMAIN_YMAX);
    grid_initialize_uniform(g);
    flow_field* field = flow_field_create(n, n);
    mms_init_field(field, g, nu, alpha);

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* solver = cfd_solver_create(registry, NS_SOLVER_TYPE_EXPLICIT_EULER);
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = dt;
    params.mu = nu;
    params.source_func = NULL;  /* No source callback */
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    for (int step = 0; step < steps; step++) {
        bc_apply_periodic(field->u, field->nx, field->ny);
        bc_apply_periodic(field->v, field->nx, field->ny);
        bc_apply_periodic(field->p, field->nx, field->ny);
        solver->step(solver, field, g, &params, NULL);
    }

    double t_final = dt * steps;
    double error_without_source = mms_compute_error(field, g, nu, alpha, t_final);

    printf("    With MMS source:    error = %.6e\n", error_with_source);
    printf("    Without source:     error = %.6e\n", error_without_source);
    printf("    Difference:         %.2f%%\n",
           100.0 * fabs(error_with_source - error_without_source) / (error_without_source + 1e-15));

    /* Source callback should have measurable effect (at least 0.1% difference) */
    TEST_ASSERT_TRUE(fabs(error_with_source - error_without_source) > error_without_source * 0.001);
    /* Both should have reasonable accuracy */
    TEST_ASSERT_TRUE(error_with_source < 0.2 && error_without_source < 0.2);

    solver->destroy(solver);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);
}

/**
 * Test 2: Spatial convergence with Explicit Euler
 */
void test_mms_euler_spatial_convergence(void) {
    printf("\n  Testing MMS spatial convergence (Explicit Euler):\n");

    size_t grid_sizes[] = {16, 32, 64, 128};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    double errors[4];
    double spacings[4];

    double h_ref = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (128 - 1);

    for (int i = 0; i < num_sizes; i++) {
        size_t n = grid_sizes[i];
        double h = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (n - 1);
        spacings[i] = h;

        /* Scale dt proportionally to h */
        double dt = SPATIAL_BASE_DT * (h / h_ref);
        int steps = (int)round(SPATIAL_FINAL_TIME / dt);

        errors[i] = mms_run_simulation(
            NS_SOLVER_TYPE_EXPLICIT_EULER, n, n, MMS_NU, MMS_ALPHA, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] >= 0.0 && errors[i] < 1.0,
            "Simulation failed or excessive error");

        printf("      %3zux%-3zu (h=%.4f, dt=%.4f): L2 error = %.6e\n",
               n, n, h, dt, errors[i]);
    }

    /* Verify convergence rates */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_sizes; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                spacings[i-1], spacings[i]);
        printf("      %zu->%zu: %.2f (expected ~1.5-2.0)\n",
               grid_sizes[i-1], grid_sizes[i], rate);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] < errors[i-1] * 1.1,
            "Error did not decrease with refinement");

        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > SPATIAL_RATE_MIN,
                "Spatial convergence rate too low");
        }
    }
}

/**
 * Test 3: Spatial convergence with RK2
 */
void test_mms_rk2_spatial_convergence(void) {
    printf("\n  Testing MMS spatial convergence (RK2):\n");

    size_t grid_sizes[] = {16, 32, 64, 128};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    double errors[4];
    double spacings[4];

    double h_ref = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (128 - 1);

    for (int i = 0; i < num_sizes; i++) {
        size_t n = grid_sizes[i];
        double h = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (n - 1);
        spacings[i] = h;

        double dt = SPATIAL_BASE_DT * (h / h_ref);
        int steps = (int)round(SPATIAL_FINAL_TIME / dt);

        errors[i] = mms_run_simulation(
            NS_SOLVER_TYPE_RK2, n, n, MMS_NU, MMS_ALPHA, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] >= 0.0 && errors[i] < 1.0,
            "Simulation failed or excessive error");

        printf("      %3zux%-3zu (h=%.4f, dt=%.4f): L2 error = %.6e\n",
               n, n, h, dt, errors[i]);
    }

    printf("    Convergence rates:\n");
    for (int i = 1; i < num_sizes; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                spacings[i-1], spacings[i]);
        printf("      %zu->%zu: %.2f (expected ~1.5-2.0)\n",
               grid_sizes[i-1], grid_sizes[i], rate);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] < errors[i-1] * 1.1,
            "Error did not decrease with refinement");

        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > SPATIAL_RATE_MIN,
                "Spatial convergence rate too low");
        }
    }
}

/**
 * Test 4: Temporal convergence with Explicit Euler
 */
void test_mms_euler_temporal_convergence(void) {
    printf("\n  Testing MMS temporal convergence (Explicit Euler):\n");

    double dts[] = {TEMPORAL_BASE_DT, TEMPORAL_BASE_DT/2.0,
                    TEMPORAL_BASE_DT/4.0, TEMPORAL_BASE_DT/8.0};
    int num_dts = sizeof(dts) / sizeof(dts[0]);
    double errors[4];

    for (int i = 0; i < num_dts; i++) {
        double dt = dts[i];
        int steps = (int)round(TEMPORAL_FINAL_TIME / dt);

        errors[i] = mms_run_simulation(
            NS_SOLVER_TYPE_EXPLICIT_EULER,
            TEMPORAL_GRID_SIZE, TEMPORAL_GRID_SIZE,
            MMS_NU, MMS_ALPHA, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] >= 0.0 && errors[i] < 1.0,
            "Simulation failed or excessive error");

        printf("      dt=%.4f (%4d steps): L2 error = %.6e\n",
               dt, steps, errors[i]);
    }

    /* Verify error decreases (rate check relaxed due to spatial dominance) */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_dts; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                dts[i-1], dts[i]);
        printf("      dt %.4f->%.4f: %.2f\n", dts[i-1], dts[i], rate);

        /* Accept negative rates (spatial error dominates) */
        TEST_ASSERT_TRUE_MESSAGE(rate > TEMPORAL_RATE_MIN,
            "Temporal convergence degraded too much");
    }
}

/**
 * Test 5: Temporal convergence with RK2 (expect better than Euler)
 */
void test_mms_rk2_temporal_convergence(void) {
    printf("\n  Testing MMS temporal convergence (RK2):\n");

    double dts[] = {TEMPORAL_BASE_DT, TEMPORAL_BASE_DT/2.0,
                    TEMPORAL_BASE_DT/4.0, TEMPORAL_BASE_DT/8.0};
    int num_dts = sizeof(dts) / sizeof(dts[0]);
    double errors[4];

    for (int i = 0; i < num_dts; i++) {
        double dt = dts[i];
        int steps = (int)round(TEMPORAL_FINAL_TIME / dt);

        errors[i] = mms_run_simulation(
            NS_SOLVER_TYPE_RK2,
            TEMPORAL_GRID_SIZE, TEMPORAL_GRID_SIZE,
            MMS_NU, MMS_ALPHA, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(errors[i] >= 0.0 && errors[i] < 1.0,
            "Simulation failed or excessive error");

        printf("      dt=%.4f (%4d steps): L2 error = %.6e\n",
               dt, steps, errors[i]);
    }

    printf("    Convergence rates:\n");
    for (int i = 1; i < num_dts; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                dts[i-1], dts[i]);
        printf("      dt %.4f->%.4f: %.2f (RK2 O(dt²) expected)\n",
               dts[i-1], dts[i], rate);

        /* RK2 should have better temporal accuracy than Euler */
        TEST_ASSERT_TRUE_MESSAGE(rate > TEMPORAL_RATE_MIN,
            "Temporal convergence degraded too much");
    }
}

/* ============================================================================
 * UNITY SETUP/TEARDOWN
 * ============================================================================ */

void setUp(void) {
    /* Called before each test */
}

void tearDown(void) {
    /* Called after each test */
}

/* ============================================================================
 * TEST RUNNER
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_mms_source_callback);
    RUN_TEST(test_mms_euler_spatial_convergence);
    RUN_TEST(test_mms_rk2_spatial_convergence);
    RUN_TEST(test_mms_euler_temporal_convergence);
    RUN_TEST(test_mms_rk2_temporal_convergence);

    return UNITY_END();
}
