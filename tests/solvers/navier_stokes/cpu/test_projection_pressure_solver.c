/**
 * Pressure-solver selection tests for the projection method
 *
 * Verifies the ns_solver_params_t.pressure_solver field:
 * - NS_PRESSURE_SOLVER_MULTIGRID and NS_PRESSURE_SOLVER_PCG_MG produce
 *   divergence-free fields on the scalar "projection" solver
 * - The MG modes agree with the default CG pressure solve within solver
 *   tolerance
 * - Non-2^k+1 grids are rejected at init with CFD_ERROR_UNSUPPORTED
 * - Degenerate grids are rejected with CFD_ERROR_INVALID instead, since no
 *   pressure solver choice can rescue them
 * - projection_optimized / projection_omp / projection_gpu reject any
 *   non-default selection with CFD_ERROR_UNSUPPORTED (no silent fallbacks)
 * - Zero-initialized params keep the existing CG behavior bit-for-bit
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

#define GRID_N 33          /* 2^5+1: multigrid-conforming */
#define GRID_N_BAD 30      /* not 2^k+1, but large enough to be a valid grid */
#define GRID_N_DEGENERATE 2 /* too small to hold an interior cell */
#define NUM_STEPS 5
#define TEST_DT 1e-3

static ns_solver_params_t make_params(ns_pressure_solver_t pressure_solver) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = TEST_DT;
    params.mu = 0.01;
    params.max_iter = 1;
    params.pressure_solver = pressure_solver;
    return params;
}

/**
 * Run the scalar "projection" solver for NUM_STEPS on a Taylor-Green field.
 * Takes a fully populated params struct so callers can exercise exactly how
 * the fields were initialized (default() vs zero-init). Returns the init
 * status; on CFD_SUCCESS the field holds the final state.
 */
static cfd_status_t run_projection_taylor_green(const ns_solver_params_t* params_in,
                                                flow_field* field, const grid* g) {
    ns_solver_params_t params = *params_in;

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL_MESSAGE(slv, "projection solver not available");

    cfd_status_t init_status = solver_init(slv, g, &params);
    if (init_status != CFD_SUCCESS) {
        solver_destroy(slv);
        cfd_registry_destroy(registry);
        return init_status;
    }

    test_init_taylor_green(field, g);

    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int step = 0; step < NUM_STEPS; step++) {
        cfd_status_t step_status = solver_step(slv, field, g, &params, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, step_status,
                                      "projection step failed");
    }

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    return CFD_SUCCESS;
}

//=============================================================================
// TEST: MG PRESSURE SOLVES PRODUCE DIVERGENCE-FREE FIELDS
//=============================================================================

static void run_divergence_free_check(ns_pressure_solver_t pressure_solver,
                                      const char* label) {
    ns_solver_params_t params = make_params(pressure_solver);

    /* Divergence tolerance is relaxed (matching test_solver_projection.c):
     * the projection solver's simple iterative method may not fully converge
     * in a few steps. The meaningful assertion is that each step reduces
     * the divergence. */
    test_result result = test_run_divergence_free(
        NS_SOLVER_TYPE_PROJECTION, GRID_N, GRID_N, &params, 10, 1.0);

    if (result.solver_unavailable) {
        printf("      %s: solver unavailable, skipping\n", label);
        return;
    }

    printf("      %s: %s\n", label, result.message);
    TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
    TEST_ASSERT_TRUE_MESSAGE(result.final_divergence < result.initial_divergence,
        "Projection with an MG pressure solve must reduce divergence");
}

void test_projection_mg_divergence_free(void) {
    printf("\n    Testing projection + multigrid pressure solve...\n");
    run_divergence_free_check(NS_PRESSURE_SOLVER_MULTIGRID, "MULTIGRID");
}

void test_projection_pcg_mg_divergence_free(void) {
    printf("\n    Testing projection + MG-preconditioned CG pressure solve...\n");
    run_divergence_free_check(NS_PRESSURE_SOLVER_PCG_MG, "PCG_MG");
}

//=============================================================================
// TEST: MG PRESSURE SOLVES AGREE WITH THE DEFAULT CG SOLVE
//=============================================================================

void test_projection_mg_matches_cg(void) {
    printf("\n    Testing MG pressure solves agree with default CG...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field_cg = flow_field_create(GRID_N, GRID_N, 1);
    flow_field* field_mg = flow_field_create(GRID_N, GRID_N, 1);
    flow_field* field_pcg = flow_field_create(GRID_N, GRID_N, 1);
    TEST_ASSERT_NOT_NULL(field_cg);
    TEST_ASSERT_NOT_NULL(field_mg);
    TEST_ASSERT_NOT_NULL(field_pcg);

    ns_solver_params_t params_cg = make_params(NS_PRESSURE_SOLVER_DEFAULT);
    ns_solver_params_t params_mg = make_params(NS_PRESSURE_SOLVER_MULTIGRID);
    ns_solver_params_t params_pcg = make_params(NS_PRESSURE_SOLVER_PCG_MG);
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_cg, field_cg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_mg, field_mg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_pcg, field_pcg, g));

    size_t n = (size_t)GRID_N * GRID_N;
    double u_norm = test_compute_l2_norm(field_cg->u, n);
    double v_norm = test_compute_l2_norm(field_cg->v, n);
    TEST_ASSERT_TRUE_MESSAGE(u_norm > 0.0 && v_norm > 0.0,
                             "CG reference run produced a zero field");

    double u_err_mg = test_compute_l2_error(field_cg->u, field_mg->u, n) / u_norm;
    double v_err_mg = test_compute_l2_error(field_cg->v, field_mg->v, n) / v_norm;
    double u_err_pcg = test_compute_l2_error(field_cg->u, field_pcg->u, n) / u_norm;
    double v_err_pcg = test_compute_l2_error(field_cg->v, field_pcg->v, n) / v_norm;

    printf("      MULTIGRID vs CG: rel L2 u=%.2e v=%.2e\n", u_err_mg, v_err_mg);
    printf("      PCG_MG    vs CG: rel L2 u=%.2e v=%.2e\n", u_err_pcg, v_err_pcg);

    /* Pressure is skipped: Neumann solutions differ by an additive constant.
     *
     * PCG_MG shares CG's effective operator (interior-only Krylov updates),
     * so it matches CG almost exactly. Standalone multigrid solves the true
     * Neumann system with a mean-compatible RHS — a slightly different
     * boundary treatment of the pressure solve — so velocity agreement is
     * O(1e-3) after a few steps (measured 3e-3 on 33x33), not bitwise. */
    TEST_ASSERT_TRUE_MESSAGE(u_err_mg < TOLERANCE_RELAXED && v_err_mg < TOLERANCE_RELAXED,
        "Multigrid pressure solve should agree with CG within solver tolerance");
    TEST_ASSERT_TRUE_MESSAGE(u_err_pcg < TOLERANCE_MODERATE && v_err_pcg < TOLERANCE_MODERATE,
        "MG-PCG pressure solve should agree with CG within solver tolerance");

    flow_field_destroy(field_cg);
    flow_field_destroy(field_mg);
    flow_field_destroy(field_pcg);
    grid_destroy(g);
}

//=============================================================================
// TEST: DEGENERATE GRID IS REJECTED AS INVALID, NOT UNSUPPORTED
//=============================================================================

/**
 * A grid too small to hold an interior cell is a caller error (INVALID), not
 * a configuration this solver merely lacks support for (UNSUPPORTED). The two
 * statuses drive different recovery: UNSUPPORTED tells the caller to pick a
 * different pressure solver, which would be useless advice here since no
 * pressure solver can work on a 2x2 grid.
 */
void test_projection_mg_rejects_degenerate_grid_as_invalid(void) {
    printf("\n    Testing MG pressure solver rejects degenerate grids as INVALID...\n");

    grid* g = grid_create(GRID_N_DEGENERATE, GRID_N_DEGENERATE, 1,
                          0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_pressure_solver_t modes[] = {
        NS_PRESSURE_SOLVER_MULTIGRID,
        NS_PRESSURE_SOLVER_PCG_MG,
    };
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
        TEST_ASSERT_NOT_NULL_MESSAGE(slv, "projection solver not available");

        ns_solver_params_t params = make_params(modes[m]);
        cfd_status_t init_status = solver_init(slv, g, &params);
        printf("      mode %d on %dx%d: init status %d\n",
               (int)modes[m], GRID_N_DEGENERATE, GRID_N_DEGENERATE, (int)init_status);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID, init_status,
            "MG pressure solver on a degenerate grid must fail init with INVALID");

        solver_destroy(slv);
    }

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

//=============================================================================
// TEST: NON-2^K+1 GRID IS REJECTED AT INIT
//=============================================================================

void test_projection_mg_rejects_non_pow2_grid(void) {
    printf("\n    Testing MG pressure solver rejects non-2^k+1 grids at init...\n");

    grid* g = grid_create(GRID_N_BAD, GRID_N_BAD, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_pressure_solver_t modes[] = {
        NS_PRESSURE_SOLVER_MULTIGRID,
        NS_PRESSURE_SOLVER_PCG_MG,
    };
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
        TEST_ASSERT_NOT_NULL_MESSAGE(slv, "projection solver not available");

        ns_solver_params_t params = make_params(modes[m]);
        cfd_status_t init_status = solver_init(slv, g, &params);
        printf("      mode %d on %dx%d: init status %d\n",
               (int)modes[m], GRID_N_BAD, GRID_N_BAD, (int)init_status);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, init_status,
            "MG pressure solver on a non-2^k+1 grid must fail init with UNSUPPORTED");

        solver_destroy(slv);
    }

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

//=============================================================================
// TEST: NON-SCALAR PROJECTION BACKENDS REJECT MG SELECTION
//=============================================================================

void test_projection_backends_reject_mg(void) {
    printf("\n    Testing non-scalar projection backends reject MG selection...\n");

    /* Conforming 33x33 grid: proves the rejection is an explicit policy,
     * not a dimensional failure. */
    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    const char* backends[] = {
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        NS_SOLVER_TYPE_PROJECTION_OMP,
        NS_SOLVER_TYPE_PROJECTION_GPU,
    };
    ns_pressure_solver_t modes[] = {
        NS_PRESSURE_SOLVER_MULTIGRID,
        NS_PRESSURE_SOLVER_PCG_MG,
    };

    for (size_t b = 0; b < sizeof(backends) / sizeof(backends[0]); b++) {
        for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
            ns_solver_t* slv = cfd_solver_create(registry, backends[b]);
            if (!slv) {
                printf("      %s: not available (skipping)\n", backends[b]);
                break;
            }

            ns_solver_params_t params = make_params(modes[m]);
            cfd_status_t init_status = solver_init(slv, g, &params);
            printf("      %s, mode %d: init status %d\n",
                   backends[b], (int)modes[m], (int)init_status);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, init_status,
                "Non-scalar projection backends must reject MG pressure solvers");

            solver_destroy(slv);
        }
    }

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

//=============================================================================
// TEST: ZERO-INITIALIZED PARAMS KEEP THE EXISTING CG BEHAVIOR
//=============================================================================

void test_projection_zero_init_backward_compat(void) {
    printf("\n    Testing zero-init params preserve default CG behavior...\n");

    /* Backward-compat contract: a caller that predates the pressure_solver
     * field - one that zero-initializes ns_solver_params_t and never touches
     * pressure_solver - must still get the original CG pressure solve. That
     * only holds because the CG path is selected by enum value 0. */
    ns_solver_params_t defaults = ns_solver_params_default();
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, (int)NS_PRESSURE_SOLVER_DEFAULT,
        "Zero-init safety requires the default CG path to map to enum value 0");
    TEST_ASSERT_EQUAL_INT_MESSAGE(NS_PRESSURE_SOLVER_DEFAULT, defaults.pressure_solver,
        "ns_solver_params_default must leave pressure_solver at the zero default");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* field_zero = flow_field_create(GRID_N, GRID_N, 1);
    flow_field* field_cg = flow_field_create(GRID_N, GRID_N, 1);
    flow_field* field_mg = flow_field_create(GRID_N, GRID_N, 1);
    TEST_ASSERT_NOT_NULL(field_zero);
    TEST_ASSERT_NOT_NULL(field_cg);
    TEST_ASSERT_NOT_NULL(field_mg);

    /* Build three configs from one genuinely zero-initialized struct (the
     * legacy caller pattern) that differ ONLY in pressure_solver. Anchoring on
     * a common zero base keeps every other field identical, so any difference
     * in the result is attributable solely to the pressure-solver selection -
     * not to default()'s nonzero source amplitudes or other tuning fields. */
    ns_solver_params_t base;
    memset(&base, 0, sizeof(base));
    base.dt = TEST_DT;
    base.mu = 0.01;
    base.max_iter = 1;
    TEST_ASSERT_EQUAL_INT_MESSAGE(NS_PRESSURE_SOLVER_DEFAULT, base.pressure_solver,
        "A zero-initialized params struct must leave pressure_solver at the CG default");

    ns_solver_params_t params_zero = base; /* pressure_solver left at zero-init 0 */
    ns_solver_params_t params_cg = base;
    params_cg.pressure_solver = NS_PRESSURE_SOLVER_DEFAULT; /* explicit CG */
    ns_solver_params_t params_mg = base;
    params_mg.pressure_solver = NS_PRESSURE_SOLVER_MULTIGRID; /* explicit MG */

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_zero, field_zero, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_cg, field_cg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(&params_mg, field_mg, g));

    /* The zero-init run must take the CG path: bitwise identical to the
     * explicit-CG run (scalar solve is deterministic), and NOT the multigrid
     * run. The second check guards against a regression that maps the zero
     * value to some other pressure solver. */
    size_t bytes = (size_t)GRID_N * GRID_N * sizeof(double);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, memcmp(field_zero->u, field_cg->u, bytes),
                                  "zero-init u field must match explicit CG bitwise");
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, memcmp(field_zero->v, field_cg->v, bytes),
                                  "zero-init v field must match explicit CG bitwise");
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, memcmp(field_zero->p, field_cg->p, bytes),
                                  "zero-init p field must match explicit CG bitwise");
    TEST_ASSERT_TRUE_MESSAGE(memcmp(field_zero->u, field_mg->u, bytes) != 0,
        "zero-init must select CG, not the multigrid pressure solver");

    flow_field_destroy(field_zero);
    flow_field_destroy(field_cg);
    flow_field_destroy(field_mg);
    grid_destroy(g);
}

//=============================================================================
// MAIN
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("Projection Pressure-Solver Selection Tests\n");
    printf("========================================\n");

    RUN_TEST(test_projection_mg_divergence_free);
    RUN_TEST(test_projection_pcg_mg_divergence_free);
    RUN_TEST(test_projection_mg_matches_cg);
    RUN_TEST(test_projection_mg_rejects_degenerate_grid_as_invalid);
    RUN_TEST(test_projection_mg_rejects_non_pow2_grid);
    RUN_TEST(test_projection_backends_reject_mg);
    RUN_TEST(test_projection_zero_init_backward_compat);

    printf("\n========================================\n");
    return UNITY_END();
}
