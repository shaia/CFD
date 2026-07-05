/**
 * @file test_turbulence_cross_backend.c
 * @brief Cross-backend consistency for the turbulence transport kernels
 *
 * Part 1 — kernel-level: advances the k-epsilon and Spalart-Allmaras
 * transport equations with the scalar, OMP, and AVX2 backend entry points on
 * identical field states (frozen shear velocity) and requires Linf agreement
 * within 1e-10 (the OMP kernel is numerically identical to scalar; the AVX2
 * kernel mirrors the scalar operation order).
 *
 * Part 2 — solver-level: runs the full RK2 solver family (scalar vs OMP vs
 * AVX2) with each turbulence model on a wall-bounded channel-like setup and
 * requires 1% relative agreement on u (the tolerance existing cross-arch
 * solver tests use), plus finite positive turbulence fields.
 *
 * Unavailable backends (CFD_ERROR_UNSUPPORTED / missing solver) are skipped
 * without failing.
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
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Internal backend entry points (exported from the static library; declared
 * in turbulence_solver_internal.h which is not on the test include path). */
extern cfd_status_t turbulence_step_explicit_with_workspace(
    flow_field* field, const grid* grid, const ns_solver_params_t* params,
    double dt, double time, double* workspace, size_t workspace_size);
extern cfd_status_t turbulence_step_explicit_omp_with_workspace(
    flow_field* field, const grid* grid, const ns_solver_params_t* params,
    double dt, double time, double* workspace, size_t workspace_size);
extern cfd_status_t turbulence_step_explicit_avx2_with_workspace(
    flow_field* field, const grid* grid, const ns_solver_params_t* params,
    double dt, double time, double* workspace, size_t workspace_size);

typedef cfd_status_t (*turb_step_fn)(flow_field*, const grid*,
                                     const ns_solver_params_t*,
                                     double, double, double*, size_t);

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

#define XB_NX 34
#define XB_NY 18
#define XB_STEPS 20
#define XB_DT 5e-4
#define XB_KERNEL_TOL 1e-10

/* ============================================================================
 * Part 1: kernel-level comparison
 * ============================================================================ */

static void init_kernel_case(flow_field* field, const grid* g,
                             const ns_solver_params_t* params) {
    for (size_t j = 0; j < XB_NY; j++) {
        for (size_t i = 0; i < XB_NX; i++) {
            size_t idx = j * XB_NX + i;
            double x = g->x[i], y = g->y[j];
            field->rho[idx] = 1.0;
            field->u[idx] = 4.0 * y * (2.0 - y) + 0.2 * sin(2.0 * M_PI * x / 4.0);
            field->v[idx] = 0.1 * sin(2.0 * M_PI * x / 4.0) * cos(M_PI * y);
        }
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      turbulence_init_uniform(field, params, 0.05, 0.1, 3e-3));
}

static double linf_diff(const double* a, const double* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void run_kernel_backend(turb_step_fn step, flow_field* field, const grid* g,
                               const ns_solver_params_t* params, int* unavailable) {
    for (int s = 0; s < XB_STEPS; s++) {
        cfd_status_t status = step(field, g, params, XB_DT, s * XB_DT, NULL, 0);
        if (status == CFD_ERROR_UNSUPPORTED) {
            *unavailable = 1;
            return;
        }
        TEST_ASSERT_EQUAL(CFD_SUCCESS, status);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, params));
    }
}

static void compare_kernel_backends(turbulence_model_t model, const char* label) {
    grid* g = grid_create(XB_NX, XB_NY, 1, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-3;
    params.turb_model = model;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.top = BC_TYPE_NOSLIP;

    turb_step_fn backends[3] = {
        turbulence_step_explicit_with_workspace,
        turbulence_step_explicit_omp_with_workspace,
        turbulence_step_explicit_avx2_with_workspace,
    };
    const char* names[3] = {"scalar", "omp", "avx2"};
    flow_field* fields[3] = {NULL, NULL, NULL};
    int available[3] = {1, 1, 1};

    for (int b = 0; b < 3; b++) {
        fields[b] = flow_field_create(XB_NX, XB_NY, 1);
        TEST_ASSERT_NOT_NULL(fields[b]);
        init_kernel_case(fields[b], g, &params);
        int unavailable = 0;
        run_kernel_backend(backends[b], fields[b], g, &params, &unavailable);
        if (unavailable) {
            available[b] = 0;
            printf("[%s] %s backend unavailable, skipping\n", label, names[b]);
        }
    }

    size_t total = XB_NX * XB_NY;
    for (int b = 1; b < 3; b++) {
        if (!available[b]) continue;
        double d_k = linf_diff(fields[0]->turb_k, fields[b]->turb_k, total);
        double d_e = linf_diff(fields[0]->turb_eps, fields[b]->turb_eps, total);
        double d_nt = linf_diff(fields[0]->turb_nu_tilde, fields[b]->turb_nu_tilde, total);
        double d_nut = linf_diff(fields[0]->nu_t, fields[b]->nu_t, total);
        printf("[%s] scalar vs %s: Linf k=%.2e eps=%.2e nt=%.2e nu_t=%.2e\n",
               label, names[b], d_k, d_e, d_nt, d_nut);
        TEST_ASSERT_TRUE_MESSAGE(d_k < XB_KERNEL_TOL, "k differs across backends");
        TEST_ASSERT_TRUE_MESSAGE(d_e < XB_KERNEL_TOL, "eps differs across backends");
        TEST_ASSERT_TRUE_MESSAGE(d_nt < XB_KERNEL_TOL, "nu_tilde differs across backends");
        TEST_ASSERT_TRUE_MESSAGE(d_nut < XB_KERNEL_TOL, "nu_t differs across backends");
    }

    for (int b = 0; b < 3; b++) {
        flow_field_destroy(fields[b]);
    }
    grid_destroy(g);
}

static void test_kernel_consistency_kepsilon(void) {
    compare_kernel_backends(TURB_MODEL_K_EPSILON, "k-epsilon kernels");
}

static void test_kernel_consistency_sa(void) {
    compare_kernel_backends(TURB_MODEL_SPALART_ALLMARAS, "SA kernels");
}

/* ============================================================================
 * Part 2: full-solver comparison (RK2 family)
 * ============================================================================ */

static void channel_force(double x, double y, double z, double t, void* ctx,
                          double* su, double* sv, double* sw) {
    (void)x; (void)y; (void)z; (void)t; (void)ctx;
    *su = 1.0;
    *sv = 0.0;
    *sw = 0.0;
}

static void init_solver_case(flow_field* field, const grid* g,
                             const ns_solver_params_t* params) {
    for (size_t j = 0; j < XB_NY; j++) {
        for (size_t i = 0; i < XB_NX; i++) {
            size_t idx = j * XB_NX + i;
            double y = g->y[j];
            int at_wall = (j == 0 || j == XB_NY - 1);
            field->u[idx] = at_wall ? 0.0 : 10.0 * y * (2.0 - y) / 2.0;
            field->v[idx] = 0.0;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      turbulence_init_uniform(field, params, 0.05, 0.1, 3e-3));
}

/* Run one solver for XB_STEPS; returns 0 if solver/backend unavailable. */
static int run_solver_case(const char* solver_type, turbulence_model_t model,
                           flow_field* field, const grid* g) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = XB_DT;
    params.max_iter = 1;
    params.mu = 1e-3;
    params.source_func = channel_force;
    params.turb_model = model;
    params.turb_bc.bottom = BC_TYPE_NOSLIP;
    params.turb_bc.top = BC_TYPE_NOSLIP;

    init_solver_case(field, g, &params);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);
    ns_solver_t* slv = cfd_solver_create(registry, solver_type);
    if (!slv) {
        cfd_registry_destroy(registry);
        return 0;
    }
    cfd_status_t status = solver_init(slv, g, &params);
    if (status == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(slv);
        cfd_registry_destroy(registry);
        return 0;
    }
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    for (int s = 0; s < XB_STEPS; s++) {
        ns_solver_stats_t stats;
        status = solver_step(slv, field, g, &params, &stats);
        if (status == CFD_ERROR_UNSUPPORTED) {
            solver_destroy(slv);
            cfd_registry_destroy(registry);
            return 0;
        }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, solver_type);
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    return 1;
}

static void compare_solver_backends(turbulence_model_t model, const char* label) {
    grid* g = grid_create(XB_NX, XB_NY, 1, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    const char* solvers[3] = {NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP,
                              NS_SOLVER_TYPE_RK2_OPTIMIZED};
    const char* names[3] = {"scalar", "omp", "avx2"};
    flow_field* fields[3] = {NULL, NULL, NULL};
    int available[3] = {0, 0, 0};

    for (int b = 0; b < 3; b++) {
        fields[b] = flow_field_create(XB_NX, XB_NY, 1);
        TEST_ASSERT_NOT_NULL(fields[b]);
        available[b] = run_solver_case(solvers[b], model, fields[b], g);
        if (!available[b]) {
            printf("[%s] %s solver unavailable, skipping\n", label, names[b]);
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(available[0], "scalar reference solver unavailable");

    size_t total = XB_NX * XB_NY;
    double u_max = 0.0;
    for (size_t n = 0; n < total; n++) {
        double a = fabs(fields[0]->u[n]);
        if (a > u_max) u_max = a;
    }
    TEST_ASSERT_TRUE(u_max > 0.0);

    for (int b = 1; b < 3; b++) {
        if (!available[b]) continue;
        double d_u = linf_diff(fields[0]->u, fields[b]->u, total) / u_max;
        double d_nut = linf_diff(fields[0]->nu_t, fields[b]->nu_t, total);
        printf("[%s] scalar vs %s: rel Linf u=%.2e, Linf nu_t=%.2e\n",
               label, names[b], d_u, d_nut);
        /* 1% relative tolerance, matching existing cross-arch solver tests */
        TEST_ASSERT_TRUE_MESSAGE(d_u < 0.01, "u differs >1% across backends");
        for (size_t n = 0; n < total; n++) {
            TEST_ASSERT_TRUE(isfinite(fields[b]->nu_t[n]));
            TEST_ASSERT_TRUE(fields[b]->nu_t[n] >= 0.0);
        }
    }

    for (int b = 0; b < 3; b++) {
        flow_field_destroy(fields[b]);
    }
    grid_destroy(g);
}

static void test_solver_consistency_kepsilon(void) {
    compare_solver_backends(TURB_MODEL_K_EPSILON, "k-epsilon rk2");
}

static void test_solver_consistency_sa(void) {
    compare_solver_backends(TURB_MODEL_SPALART_ALLMARAS, "SA rk2");
}

/* ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_kernel_consistency_kepsilon);
    RUN_TEST(test_kernel_consistency_sa);
    RUN_TEST(test_solver_consistency_kepsilon);
    RUN_TEST(test_solver_consistency_sa);
    return UNITY_END();
}
