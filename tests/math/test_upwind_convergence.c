/**
 * @file test_upwind_convergence.c
 * @brief Spatial convergence order of the NS convection schemes in a full solver
 *
 * A sine wave in v(x) is advected by a uniform u = U through a periodic box:
 *
 *     v(x, t) = sin(2*pi*(x - U*t)),  u = U,  w = 0,  p = 0,  nu = 0
 *
 * u.grad(u) and the divergence vanish, so u and p stay fixed and v obeys pure
 * 1D advection with this exact solution. The scalar RK2 solver wraps its
 * stencil periodically over interior points 1..nx-2, so the grid places one
 * wavelength across exactly those points.
 *
 * The CFL number is held fixed while the grid is refined, so RK2's O(dt^2)
 * error scales like O(h^2): central differencing converges at second order and
 * first-order upwind at first order. The final time is a quarter of the advection
 * period; over a full period the upwind amplitude decay leaves the asymptotic
 * range on the coarsest grid.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define ADVECTION_U   1.0
#define CFL           0.5
#define FINAL_TIME    0.25  /* quarter of the unit advection period */

#define NUM_GRIDS 3
static const size_t INTERIOR_POINTS[NUM_GRIDS] = {16, 32, 64};

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

/**
 * Advect the sine wave with the scalar RK2 solver on n interior points and
 * return the L2 error of v against the exact solution. Returns a negative
 * value on failure (after reporting it through Unity).
 */
static double run_advection(ns_convection_scheme_t scheme, size_t n) {
    size_t nx = n + 2;
    size_t ny = 3;

    /* dx = 1/n, so interior points 1..n span exactly one unit period */
    double dx = 1.0 / (double)n;
    grid* g = grid_create(nx, ny, 1, 0.0, (double)(nx - 1) * dx, 0.0, 2.0 * dx, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = ADVECTION_U;
            field->v[idx] = sin(2.0 * M_PI * g->x[i]);
            field->rho[idx] = 1.0;
        }
    }

    /* An integer step count lands exactly on FINAL_TIME */
    int steps = (int)lround(FINAL_TIME * ADVECTION_U / (CFL * dx));
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = FINAL_TIME / steps;
    params.mu = 0.0;
    params.max_iter = 1;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;
    params.convection_scheme = scheme;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);
    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_RK2);
    TEST_ASSERT_NOT_NULL(slv);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, solver_init(slv, g, &params));

    ns_solver_stats_t stats = ns_solver_stats_default();
    double error = -1.0;
    int ok = 1;
    for (int s = 0; s < steps; s++) {
        if (solver_step(slv, field, g, &params, &stats) != CFD_SUCCESS) {
            ok = 0;
            break;
        }
    }

    if (ok) {
        double t = steps * params.dt;
        double sum_sq = 0.0;
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = nx + i; /* interior row j = 1 */
            double exact = sin(2.0 * M_PI * (g->x[i] - ADVECTION_U * t));
            double err = field->v[idx] - exact;
            sum_sq += err * err;
        }
        error = sqrt(sum_sq / (double)n);
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    flow_field_destroy(field);
    grid_destroy(g);

    TEST_ASSERT_TRUE_MESSAGE(ok, "RK2 step failed during advection");
    return error;
}

/** Run the refinement study and assert every observed rate lies in [min_rate, max_rate] */
static void assert_convergence_rate(ns_convection_scheme_t scheme, const char* label,
                                    double min_rate, double max_rate) {
    double errors[NUM_GRIDS];
    for (int k = 0; k < NUM_GRIDS; k++) {
        errors[k] = run_advection(scheme, INTERIOR_POINTS[k]);
        printf("    %s n=%zu: L2 error = %.6e\n", label, INTERIOR_POINTS[k], errors[k]);
    }
    for (int k = 1; k < NUM_GRIDS; k++) {
        double rate = log(errors[k - 1] / errors[k]) / log(2.0);
        printf("    %s rate %zu->%zu: %.3f\n", label, INTERIOR_POINTS[k - 1],
               INTERIOR_POINTS[k], rate);
        TEST_ASSERT_TRUE_MESSAGE(rate >= min_rate && rate <= max_rate,
                                 "Observed convergence rate outside the expected range");
    }
}

void test_upwind_converges_first_order(void) {
    printf("\n  Upwind convection (expected rate ~1):\n");
    assert_convergence_rate(NS_CONVECTION_SCHEME_UPWIND, "upwind", 0.8, 1.2);
}

void test_central_converges_second_order(void) {
    printf("\n  Central convection (expected rate ~2):\n");
    assert_convergence_rate(NS_CONVECTION_SCHEME_CENTRAL, "central", 1.7, 2.3);
}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("CONVECTION SCHEME CONVERGENCE ORDER (RK2)\n");
    printf("========================================\n");

    RUN_TEST(test_upwind_converges_first_order);
    RUN_TEST(test_central_converges_second_order);

    return UNITY_END();
}
