/**
 * Convection-scheme selection tests (ns_solver_params_t.convection_scheme)
 *
 * - Zero-initialized and default params select central differencing
 * - Every CPU solver rejects an unknown scheme with CFD_ERROR_INVALID at init
 * - GPU solvers reject upwind with CFD_ERROR_UNSUPPORTED at init and at step
 *   (skipped when no GPU solver is registered)
 * - Upwind keeps an advected profile within its initial range on the scalar
 *   solvers, while central differencing overshoots it
 * - The OpenMP and AVX2 upwind kernels reproduce the scalar ones, and each
 *   differs from its own central result
 */

#include "test_solver_helpers.h"
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

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

static const char* const CPU_SOLVERS[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
    NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
    NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OMP,
    NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_OPTIMIZED,
    NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_OPTIMIZED,
};
#define NUM_CPU_SOLVERS (sizeof(CPU_SOLVERS) / sizeof(CPU_SOLVERS[0]))

static const char* const GPU_SOLVERS[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER_GPU, NS_SOLVER_TYPE_PROJECTION_GPU,
    NS_SOLVER_TYPE_RK2_GPU, NS_SOLVER_TYPE_RK4_GPU,
};
#define NUM_GPU_SOLVERS (sizeof(GPU_SOLVERS) / sizeof(GPU_SOLVERS[0]))

/* Scalar reference and its OpenMP / AVX2 counterparts, per algorithm */
typedef struct {
    const char* scalar;
    const char* omp;
    const char* optimized;
} backend_family;

static const backend_family FAMILIES[] = {
    {NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
     NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED},
    {NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OMP,
     NS_SOLVER_TYPE_PROJECTION_OPTIMIZED},
    {NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_OPTIMIZED},
    {NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_OPTIMIZED},
};
#define NUM_FAMILIES (sizeof(FAMILIES) / sizeof(FAMILIES[0]))

/* Cross-backend runs: 19 points leave 17 interior columns, so the AVX2 kernels
 * exercise their scalar remainder columns. */
#define CONSISTENCY_N     19
#define CONSISTENCY_STEPS 20
/* OpenMP/AVX2 differ from scalar only by floating-point contraction and
 * reciprocal-vs-division rounding */
#define BACKEND_MATCH_TOL 1e-10
/* A backend that ignored the scheme would match its central run exactly */
#define SCHEME_SEPARATION_MIN 1e-8

/* Boundedness runs: one wavelength of a smoothed top-hat in v(x) across
 * BOUNDED_N interior points, advected by a uniform u. */
#define BOUNDED_N      32
#define BOUNDED_U      50.0
#define BOUNDED_STEPS  80
#define BOUNDED_WIDTH  0.02   /* tanh edge width; keeps derivatives under the clamp */
/* Upwind updates are convex combinations of neighbors at CFL <= 1 for every
 * integrator here: explicit Euler and Heun are SSP, and on this linear problem
 * classical RK4's stability polynomial keeps nonnegative coefficients up to
 * CFL 1. The tolerance only absorbs round-off. */
#define BOUNDS_TOL     1e-12
/* Central differencing over- and undershoots the edges by O(0.1-1) here */
#define CENTRAL_OVERSHOOT_MIN 0.1

static ns_solver_registry_t* create_registry(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);
    return registry;
}

static ns_solver_params_t make_params(ns_convection_scheme_t scheme) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 5e-4;
    params.mu = 0.01;
    params.max_iter = 1;
    /* The AVX2 explicit Euler lanes omit source terms, and AVX2 projection
     * passes a different iteration to them; zero them for backend comparisons. */
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;
    params.convection_scheme = scheme;
    return params;
}

//=============================================================================
// DEFAULTS
//=============================================================================

void test_default_scheme_is_central(void) {
    TEST_ASSERT_EQUAL_INT(0, NS_CONVECTION_SCHEME_CENTRAL);
    TEST_ASSERT_EQUAL_INT(NS_CONVECTION_SCHEME_CENTRAL,
                          ns_solver_params_default().convection_scheme);

    ns_solver_params_t zeroed;
    memset(&zeroed, 0, sizeof(zeroed));
    TEST_ASSERT_EQUAL_INT(NS_CONVECTION_SCHEME_CENTRAL, zeroed.convection_scheme);
}

/**
 * Run a solver for CONSISTENCY_STEPS on a Taylor-Green field. Returns the
 * field (caller destroys it), or NULL when the solver is not registered or its
 * init reports CFD_ERROR_UNSUPPORTED (this build lacks the backend).
 */
static flow_field* run_taylor_green(ns_solver_registry_t* registry, const char* type,
                                    const ns_solver_params_t* params) {
    if (!test_solver_available(registry, type)) {
        return NULL;
    }
    grid* g = grid_create(CONSISTENCY_N, CONSISTENCY_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(CONSISTENCY_N, CONSISTENCY_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);
    test_init_taylor_green(field, g);

    ns_solver_t* slv = cfd_solver_create(registry, type);
    TEST_ASSERT_NOT_NULL(slv);
    cfd_status_t status = solver_init(slv, g, params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(slv);
        flow_field_destroy(field);
        grid_destroy(g);
        return NULL;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, type);

    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int step = 0; step < CONSISTENCY_STEPS && status == CFD_SUCCESS; step++) {
        status = solver_step(slv, field, g, params, &stats);
    }
    solver_destroy(slv);
    grid_destroy(g);
    TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, type);
    TEST_ASSERT_TRUE_MESSAGE(test_flow_field_is_valid(field), type);
    return field;
}

/** max over u and v of ||a - b||_2 / ||a||_2 */
static double relative_difference(const flow_field* a, const flow_field* b) {
    size_t n = a->nx * a->ny * a->nz;
    double u_rel = test_compute_l2_error(a->u, b->u, n) / test_compute_l2_norm(a->u, n);
    double v_rel = test_compute_l2_error(a->v, b->v, n) / test_compute_l2_norm(a->v, n);
    return fmax(u_rel, v_rel);
}

//=============================================================================
// REJECTION
//=============================================================================

void test_unknown_scheme_rejected(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(CONSISTENCY_N, CONSISTENCY_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_params_t params = make_params(NS_CONVECTION_SCHEME_CENTRAL);
    params.convection_scheme = (ns_convection_scheme_t)99;

    int checked = 0;
    for (size_t s = 0; s < NUM_CPU_SOLVERS; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, CPU_SOLVERS[s]);
        if (!slv) {
            printf("  %s not registered (skipping)\n", CPU_SOLVERS[s]);
            continue;
        }
        cfd_status_t status = solver_init(slv, g, &params);
        solver_destroy(slv);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID, status, CPU_SOLVERS[s]);
        checked++;
    }
    grid_destroy(g);
    cfd_registry_destroy(registry);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No CPU solver was checked");
}

void test_gpu_rejects_upwind(void) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(CONSISTENCY_N, CONSISTENCY_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    flow_field* field = flow_field_create(CONSISTENCY_N, CONSISTENCY_N, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);
    test_init_taylor_green(field, g);

    ns_solver_params_t central = make_params(NS_CONVECTION_SCHEME_CENTRAL);
    ns_solver_params_t upwind = make_params(NS_CONVECTION_SCHEME_UPWIND);

    int checked = 0;
    for (size_t s = 0; s < NUM_GPU_SOLVERS; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, GPU_SOLVERS[s]);
        if (!slv) {
            continue;
        }
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, solver_init(slv, g, &upwind),
                                      GPU_SOLVERS[s]);

        /* Params changed after a successful init must not run central silently */
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, solver_init(slv, g, &central),
                                      GPU_SOLVERS[s]);
        ns_solver_stats_t stats = ns_solver_stats_default();
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED,
                                      solver_step(slv, field, g, &upwind, &stats),
                                      GPU_SOLVERS[s]);
        solver_destroy(slv);
        checked++;
    }
    flow_field_destroy(field);
    grid_destroy(g);
    cfd_registry_destroy(registry);
    if (checked == 0) {
        TEST_IGNORE_MESSAGE("No GPU solver registered");
    }
}

//=============================================================================
// BOUNDEDNESS
//=============================================================================

/**
 * Advect a smoothed top-hat in v(x) by uniform u with nu = 0 on an nx x 3 grid
 * with periodic boundaries. Reports the largest excursion of v above the
 * initial maximum or below the initial minimum over all steps.
 *
 * A single interior row keeps the problem one-dimensional for every solver:
 * the ghost rows are periodic copies of it, so dv/dy = 0 and the divergence
 * (and therefore the projection pressure) stays exactly zero.
 */
static double run_bounded_advection(ns_solver_registry_t* registry, const char* type,
                                    ns_convection_scheme_t scheme) {
    size_t nx = BOUNDED_N + 2;
    size_t ny = 3;
    double dx = 1.0 / BOUNDED_N;
    /* Interior points 1..BOUNDED_N span exactly one unit period */
    grid* g = grid_create(nx, ny, 1, 0.0, (double)(nx - 1) * dx, 0.0, 2.0 * dx, 0.0, 0.0);
    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_NOT_NULL(field);
    grid_initialize_uniform(g);

    double vmin = INFINITY;
    double vmax = -INFINITY;
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double x = g->x[i];
            field->u[idx] = BOUNDED_U;
            field->v[idx] = 0.5 * (tanh((x - 0.25) / BOUNDED_WIDTH) -
                                   tanh((x - 0.75) / BOUNDED_WIDTH));
            field->rho[idx] = 1.0;
            vmin = fmin(vmin, field->v[idx]);
            vmax = fmax(vmax, field->v[idx]);
        }
    }

    ns_solver_params_t params = make_params(scheme);
    params.mu = 0.0;
    params.dt = 0.5 * dx / BOUNDED_U;  /* CFL 0.5; explicit Euler caps its dt lower */

    ns_solver_t* slv = cfd_solver_create(registry, type);
    TEST_ASSERT_NOT_NULL(slv);
    TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, solver_init(slv, g, &params), type);

    ns_solver_stats_t stats = ns_solver_stats_default();
    double excursion = 0.0;
    for (int step = 0; step < BOUNDED_STEPS; step++) {
        /* Explicit Euler and projection restore the caller's ghost cells */
        apply_boundary_conditions(field, g);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, solver_step(slv, field, g, &params, &stats),
                                      type);
        for (size_t i = 1; i < nx - 1; i++) {
            double v = field->v[IDX_2D(i, 1, nx)];
            excursion = fmax(excursion, fmax(v - vmax, vmin - v));
        }
    }

    solver_destroy(slv);
    flow_field_destroy(field);
    grid_destroy(g);
    return excursion;
}

void test_upwind_bounded_central_overshoots(void) {
    ns_solver_registry_t* registry = create_registry();

    for (size_t f = 0; f < NUM_FAMILIES; f++) {
        const char* type = FAMILIES[f].scalar;
        double upwind = run_bounded_advection(registry, type, NS_CONVECTION_SCHEME_UPWIND);
        double central = run_bounded_advection(registry, type, NS_CONVECTION_SCHEME_CENTRAL);
        printf("  %-16s excursion beyond initial range: upwind %.3e, central %.3e\n",
               type, upwind, central);
        TEST_ASSERT_TRUE_MESSAGE(upwind <= BOUNDS_TOL, type);
        TEST_ASSERT_TRUE_MESSAGE(central >= CENTRAL_OVERSHOOT_MIN, type);
    }

    cfd_registry_destroy(registry);
}

//=============================================================================
// CROSS-BACKEND CONSISTENCY
//=============================================================================

void test_upwind_backends_match_scalar(void) {
    ns_solver_registry_t* registry = create_registry();
    ns_solver_params_t central = make_params(NS_CONVECTION_SCHEME_CENTRAL);
    ns_solver_params_t upwind = make_params(NS_CONVECTION_SCHEME_UPWIND);

    int compared = 0;
    for (size_t f = 0; f < NUM_FAMILIES; f++) {
        flow_field* reference = run_taylor_green(registry, FAMILIES[f].scalar, &upwind);
        TEST_ASSERT_NOT_NULL(reference);

        const char* others[] = {FAMILIES[f].omp, FAMILIES[f].optimized};
        for (size_t o = 0; o < 2; o++) {
            flow_field* up = run_taylor_green(registry, others[o], &upwind);
            if (!up) {
                printf("  %s not available in this build (skipping)\n", others[o]);
                continue;
            }
            flow_field* ce = run_taylor_green(registry, others[o], &central);
            TEST_ASSERT_NOT_NULL(ce);

            double match = relative_difference(reference, up);
            double separation = relative_difference(ce, up);
            printf("  %-26s upwind vs scalar upwind %.2e, upwind vs own central %.2e\n",
                   others[o], match, separation);
            TEST_ASSERT_TRUE_MESSAGE(match <= BACKEND_MATCH_TOL, others[o]);
            TEST_ASSERT_TRUE_MESSAGE(separation >= SCHEME_SEPARATION_MIN, others[o]);

            flow_field_destroy(ce);
            flow_field_destroy(up);
            compared++;
        }
        flow_field_destroy(reference);
    }
    cfd_registry_destroy(registry);
    if (compared == 0) {
        TEST_IGNORE_MESSAGE("No OpenMP or AVX2 solver available");
    }
}

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_default_scheme_is_central);
    RUN_TEST(test_unknown_scheme_rejected);
    RUN_TEST(test_gpu_rejects_upwind);
    RUN_TEST(test_upwind_bounded_central_overshoots);
    RUN_TEST(test_upwind_backends_match_scalar);

    return UNITY_END();
}
