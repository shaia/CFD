/**
 * 3D Navier-Stokes Solver Tests
 *
 * Validates that the three scalar CPU NS solvers (Explicit Euler, Projection,
 * RK2) correctly handle nz>1 grids and produce identical results when nz=1.
 *
 * Test approach:
 * - Quiescent flow (u=v=w=0, uniform p/rho) on small 3D grids verifies
 *   solvers accept nz>1 and remain stable.
 * - Backward compatibility tests verify nz=1 produces identical results
 *   to the 2D code path.
 */

#include "../test_solver_helpers.h"
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

/* ========================================================================
 * Helper: Create a quiescent 3D flow field (u=v=w=0, uniform p and rho)
 * ======================================================================== */
static void init_quiescent_3d(flow_field* field) {
    size_t total = field->nx * field->ny * field->nz;
    memset(field->u, 0, total * sizeof(double));
    memset(field->v, 0, total * sizeof(double));
    memset(field->w, 0, total * sizeof(double));
    for (size_t i = 0; i < total; i++) {
        field->p[i] = 1.0;
        field->rho[i] = 1.0;
        field->T[i] = 300.0;
    }
}

/* Helper: run a solver by name for a given number of steps, return status */
static cfd_status_t run_solver_steps(const char* solver_name, flow_field* field,
                                     grid* g, ns_solver_params_t* params, int steps) {
    ns_solver_registry_t* registry = cfd_registry_create();
    if (!registry) {
        return CFD_ERROR_NOMEM;
    }
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, solver_name);
    if (!slv) {
        cfd_registry_destroy(registry);
        return CFD_ERROR_NOT_FOUND;
    }

    cfd_status_t init_status = solver_init(slv, g, params);
    if (init_status != CFD_SUCCESS) {
        solver_destroy(slv);
        cfd_registry_destroy(registry);
        return init_status;
    }

    params->max_iter = steps;
    ns_solver_stats_t stats = ns_solver_stats_default();
    cfd_status_t status = solver_step(slv, field, g, params, &stats);

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    return status;
}

/* ========================================================================
 * 3D QUIESCENT TESTS
 * Verify solvers accept nz>1 and maintain quiescent state (no spurious motion)
 * ======================================================================== */

void test_3d_explicit_euler_quiescent(void) {
    printf("\n=== Test: 3D Explicit Euler Quiescent ===\n");

    size_t nx = 8, ny = 8, nz = 8;
    grid* g = grid_create(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, nz);
    TEST_ASSERT_NOT_NULL(field);
    init_quiescent_3d(field);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 1e-4;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    cfd_status_t status = run_solver_steps(NS_SOLVER_TYPE_EXPLICIT_EULER, field, g, &params, 5);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Velocity should stay near zero */
    size_t total = nx * ny * nz;
    double max_vel = 0.0;
    for (size_t i = 0; i < total; i++) {
        double v2 = field->u[i] * field->u[i] + field->v[i] * field->v[i] +
                    field->w[i] * field->w[i];
        if (v2 > max_vel) max_vel = v2;
    }
    max_vel = sqrt(max_vel);
    printf("  Max velocity magnitude: %.2e (expect ~0)\n", max_vel);
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, 0.0, max_vel);

    flow_field_destroy(field);
    grid_destroy(g);
    printf("PASSED\n");
}

void test_3d_projection_quiescent(void) {
    printf("\n=== Test: 3D Projection Quiescent ===\n");

    size_t nx = 8, ny = 8, nz = 8;
    grid* g = grid_create(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, nz);
    TEST_ASSERT_NOT_NULL(field);
    init_quiescent_3d(field);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 1e-4;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    cfd_status_t status = run_solver_steps(NS_SOLVER_TYPE_PROJECTION, field, g, &params, 3);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify divergence-free: div = du/dx + dv/dy + dw/dz should be near zero */
    size_t plane = nx * ny;
    size_t stride_z = plane;
    double max_div = 0.0;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);
    double dz = 1.0 / (nz - 1);

    for (size_t k = 1; k < nz - 1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = k * stride_z + IDX_2D(i, j, nx);
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);
                double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) / (2.0 * dz);
                double div = fabs(du_dx + dv_dy + dw_dz);
                if (div > max_div) max_div = div;
            }
        }
    }
    printf("  Max divergence: %.2e\n", max_div);
    TEST_ASSERT_DOUBLE_WITHIN(1e-8, 0.0, max_div);

    flow_field_destroy(field);
    grid_destroy(g);
    printf("PASSED\n");
}

void test_3d_rk2_quiescent(void) {
    printf("\n=== Test: 3D RK2 Quiescent ===\n");

    size_t nx = 8, ny = 8, nz = 8;
    grid* g = grid_create(nx, ny, nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, nz);
    TEST_ASSERT_NOT_NULL(field);
    init_quiescent_3d(field);

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 1e-4;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    cfd_status_t status = run_solver_steps(NS_SOLVER_TYPE_RK2, field, g, &params, 5);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    size_t total = nx * ny * nz;
    double max_vel = 0.0;
    for (size_t i = 0; i < total; i++) {
        double v2 = field->u[i] * field->u[i] + field->v[i] * field->v[i] +
                    field->w[i] * field->w[i];
        if (v2 > max_vel) max_vel = v2;
    }
    max_vel = sqrt(max_vel);
    printf("  Max velocity magnitude: %.2e (expect ~0)\n", max_vel);
    TEST_ASSERT_DOUBLE_WITHIN(1e-6, 0.0, max_vel);

    flow_field_destroy(field);
    grid_destroy(g);
    printf("PASSED\n");
}

/* ========================================================================
 * BACKWARD COMPATIBILITY TESTS
 * Verify nz=1 produces identical results to existing 2D code
 * ======================================================================== */

/**
 * Helper: compute RMS (L2 norm) of an array.
 */
static double compute_l2_norm(const double* arr, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrt(sum / n);
}

/**
 * Helper: run solver on 2D (nz=1) grid and verify results match golden L2
 * norms. This guards the "nz=1 produces bit-identical 2D results" invariant.
 * Sets a simple sinusoidal initial condition and runs 3 steps.
 */
static void run_backward_compat_test(const char* solver_name,
                                     double golden_l2_u, double golden_l2_v,
                                     double golden_l2_p) {
    size_t nx = 16, ny = 16;

    /* Create nz=1 grid (the 3D code path with nz==1 should produce 2D results) */
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field = flow_field_create(nx, ny, 1);
    TEST_ASSERT_NOT_NULL(field);

    /* Set a simple initial condition */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = IDX_2D(i, j, nx);
            double x = g->x[i];
            double y = g->y[j];
            field->u[idx] = 0.1 * sin(M_PI * y);
            field->v[idx] = 0.05 * sin(2.0 * M_PI * x);
            field->w[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 1e-4;
    params.source_amplitude_u = 0.0;
    params.source_amplitude_v = 0.0;

    cfd_status_t status = run_solver_steps(solver_name, field, g, &params, 3);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    size_t total = nx * ny;

    /* w should remain identically zero when nz=1 (no z-momentum) */
    double max_w = 0.0;
    for (size_t i = 0; i < total; i++) {
        double aw = fabs(field->w[i]);
        if (aw > max_w) max_w = aw;
    }
    printf("  Max |w| with nz=1: %.2e (expect 0)\n", max_w);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 0.0, max_w);

    /* All values should be finite */
    for (size_t i = 0; i < total; i++) {
        TEST_ASSERT_TRUE(isfinite(field->u[i]));
        TEST_ASSERT_TRUE(isfinite(field->v[i]));
        TEST_ASSERT_TRUE(isfinite(field->p[i]));
    }

    /* Verify L2 norms match golden values (guards bit-identical 2D output) */
    double l2_u = compute_l2_norm(field->u, total);
    double l2_v = compute_l2_norm(field->v, total);
    double l2_p = compute_l2_norm(field->p, total);
    printf("  L2(u)=%.17e  L2(v)=%.17e  L2(p)=%.17e\n", l2_u, l2_v, l2_p);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, golden_l2_u, l2_u);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, golden_l2_v, l2_v);
    TEST_ASSERT_DOUBLE_WITHIN(1e-12, golden_l2_p, l2_p);

    flow_field_destroy(field);
    grid_destroy(g);
}

void test_3d_explicit_euler_backward_compat(void) {
    printf("\n=== Test: 3D Explicit Euler Backward Compat (nz=1) ===\n");
    run_backward_compat_test(NS_SOLVER_TYPE_EXPLICIT_EULER,
                             6.84647305901105868e-02,
                             3.42314945425199885e-02,
                             1.00000000000000000e+00);
    printf("PASSED\n");
}

void test_3d_projection_backward_compat(void) {
    printf("\n=== Test: 3D Projection Backward Compat (nz=1) ===\n");
    run_backward_compat_test(NS_SOLVER_TYPE_PROJECTION,
                             6.84647639323831686e-02,
                             3.42315494726977212e-02,
                             1.00000039251590289e+00);
    printf("PASSED\n");
}

void test_3d_rk2_backward_compat(void) {
    printf("\n=== Test: 3D RK2 Backward Compat (nz=1) ===\n");
    run_backward_compat_test(NS_SOLVER_TYPE_RK2,
                             6.88584742267375205e-02,
                             3.49775875753182836e-02,
                             1.00000000000000044e+00);
    printf("PASSED\n");
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    UNITY_BEGIN();

    /* 3D quiescent tests */
    RUN_TEST(test_3d_explicit_euler_quiescent);
    RUN_TEST(test_3d_projection_quiescent);
    RUN_TEST(test_3d_rk2_quiescent);

    /* Backward compatibility (nz=1) */
    RUN_TEST(test_3d_explicit_euler_backward_compat);
    RUN_TEST(test_3d_projection_backward_compat);
    RUN_TEST(test_3d_rk2_backward_compat);

    return UNITY_END();
}
