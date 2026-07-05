/**
 * @file test_turbulence_disabled_regression.c
 * @brief Laminar solves must be bitwise unaffected by the turbulence fields
 *
 * With params.turb_model == TURB_MODEL_NONE (the default), the momentum
 * kernels must take the original laminar code path and never read the
 * turbulence arrays. Verified by poisoning the turbulence arrays with garbage
 * in one field, running identical laminar solves on both, and comparing
 * u/v/p bitwise.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

#define REG_NX 17
#define REG_NY 17
#define REG_STEPS 20

static void init_taylor_green(flow_field* field, const grid* g) {
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = j * field->nx + i;
            double x = g->x[i], y = g->y[j];
            field->u[idx] = cos(2.0 * M_PI * x) * sin(2.0 * M_PI * y);
            field->v[idx] = -sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
}

static void run_laminar_regression(const char* solver_type) {
    grid* g = grid_create(REG_NX, REG_NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field_a = flow_field_create(REG_NX, REG_NY, 1);
    flow_field* field_b = flow_field_create(REG_NX, REG_NY, 1);
    TEST_ASSERT_NOT_NULL(field_a);
    TEST_ASSERT_NOT_NULL(field_b);
    init_taylor_green(field_a, g);
    init_taylor_green(field_b, g);

    /* Poison field_b's turbulence arrays: a laminar solve must never read them */
    size_t total = REG_NX * REG_NY;
    for (size_t n = 0; n < total; n++) {
        field_b->turb_k[n] = 1e30;
        field_b->turb_eps[n] = -1e30;
        field_b->turb_nu_tilde[n] = 1e30;
        field_b->nu_t[n] = 1e30;
    }

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv_a = cfd_solver_create(registry, solver_type);
    ns_solver_t* slv_b = cfd_solver_create(registry, solver_type);
    TEST_ASSERT_NOT_NULL_MESSAGE(slv_a, "solver not available");
    TEST_ASSERT_NOT_NULL_MESSAGE(slv_b, "solver not available");

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 1e-4;
    params.max_iter = 1;
    TEST_ASSERT_EQUAL(TURB_MODEL_NONE, params.turb_model);

    TEST_ASSERT_EQUAL(CFD_SUCCESS, solver_init(slv_a, g, &params));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solver_init(slv_b, g, &params));

    for (int s = 0; s < REG_STEPS; s++) {
        ns_solver_stats_t stats_a, stats_b;
        TEST_ASSERT_EQUAL(CFD_SUCCESS, solver_step(slv_a, field_a, g, &params, &stats_a));
        TEST_ASSERT_EQUAL(CFD_SUCCESS, solver_step(slv_b, field_b, g, &params, &stats_b));
    }

    /* Bitwise identical velocity and pressure */
    TEST_ASSERT_EQUAL_MESSAGE(0, memcmp(field_a->u, field_b->u, total * sizeof(double)),
                              "u differs: laminar path read turbulence arrays");
    TEST_ASSERT_EQUAL_MESSAGE(0, memcmp(field_a->v, field_b->v, total * sizeof(double)),
                              "v differs: laminar path read turbulence arrays");
    TEST_ASSERT_EQUAL_MESSAGE(0, memcmp(field_a->p, field_b->p, total * sizeof(double)),
                              "p differs: laminar path read turbulence arrays");

    solver_destroy(slv_a);
    solver_destroy(slv_b);
    cfd_registry_destroy(registry);
    flow_field_destroy(field_a);
    flow_field_destroy(field_b);
    grid_destroy(g);
}

static void test_projection_laminar_unaffected(void) {
    run_laminar_regression(NS_SOLVER_TYPE_PROJECTION);
}

static void test_rk2_laminar_unaffected(void) {
    run_laminar_regression(NS_SOLVER_TYPE_RK2);
}

static void test_explicit_euler_laminar_unaffected(void) {
    run_laminar_regression(NS_SOLVER_TYPE_EXPLICIT_EULER);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_projection_laminar_unaffected);
    RUN_TEST(test_rk2_laminar_unaffected);
    RUN_TEST(test_explicit_euler_laminar_unaffected);
    return UNITY_END();
}
