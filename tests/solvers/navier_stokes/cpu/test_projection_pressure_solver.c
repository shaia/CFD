/**
 * Pressure-solver selection tests for the projection method
 *
 * Verifies the ns_solver_params_t.pressure_solver field on the solvers that
 * implement the multigrid modes, "projection" and "projection_omp" (OpenMP
 * cases are skipped when the build has no OpenMP backend):
 * - NS_PRESSURE_SOLVER_MULTIGRID and NS_PRESSURE_SOLVER_PCG_MG produce
 *   divergence-free fields
 * - The MG modes agree with the default CG pressure solve within solver
 *   tolerance
 * - Non-2^k+1 grids are rejected at init with CFD_ERROR_UNSUPPORTED
 * - Degenerate grids are rejected with CFD_ERROR_INVALID instead, since no
 *   pressure solver choice can rescue them
 * - projection_omp reproduces the scalar projection in each MG mode on a grid
 *   large enough to run the threaded multigrid kernels
 * - projection_optimized / projection_gpu reject any non-default selection
 *   with CFD_ERROR_UNSUPPORTED (no silent fallbacks)
 * - Zero-initialized params keep the existing CG behavior bit-for-bit
 */

#include "../test_solver_helpers.h"
#include "../../../../lib/src/solvers/linear/multigrid_internal.h"
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

/* Smallest square 2^k+1 grid whose finest multigrid plane (255^2 interior
 * points) reaches MG_OMP_MIN_POINTS, so the OpenMP multigrid kernels use threads */
#define GRID_N_LARGE 257
/* nu*dt*(2/dx^2 + 2/dy^2)/2 = 0.13 at h = 1/256 with mu = 0.01; TEST_DT would
 * exceed the explicit diffusion limit there */
#define TEST_DT_LARGE 1e-4

/* OMP vs scalar multigrid projection: no parallel reductions on this path, so
 * the bound only absorbs floating-point contraction differences (same bound as
 * MG_L2_TOL in test_omp_consistency.c) */
#define OMP_MG_MATCH_TOL 1e-10
/* OMP PCG_MG must be at least this much closer to scalar PCG_MG than to scalar CG */
#define PCG_MG_SEPARATION 10.0

/* Projection solvers that implement the multigrid pressure modes */
static const char* const MG_BACKENDS[] = {
    NS_SOLVER_TYPE_PROJECTION,
    NS_SOLVER_TYPE_PROJECTION_OMP,
};
#define NUM_MG_BACKENDS (sizeof(MG_BACKENDS) / sizeof(MG_BACKENDS[0]))

static const ns_pressure_solver_t MG_MODES[] = {
    NS_PRESSURE_SOLVER_MULTIGRID,
    NS_PRESSURE_SOLVER_PCG_MG,
};
#define NUM_MG_MODES (sizeof(MG_MODES) / sizeof(MG_MODES[0]))

static ns_solver_params_t make_params(ns_pressure_solver_t pressure_solver) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = TEST_DT;
    params.mu = 0.01;
    params.max_iter = 1;
    params.pressure_solver = pressure_solver;
    return params;
}

/** Nonzero when the solver type is registered in this build */
static int projection_available(const char* solver_type) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);
    int available = test_solver_available(registry, solver_type);
    cfd_registry_destroy(registry);
    return available;
}

/**
 * Run a projection solver for NUM_STEPS on a Taylor-Green field.
 * Takes a fully populated params struct so callers can exercise exactly how
 * the fields were initialized (default() vs zero-init). Returns the init
 * status; on CFD_SUCCESS the field holds the final state.
 */
static cfd_status_t run_projection_taylor_green(const char* solver_type,
                                                const ns_solver_params_t* params_in,
                                                flow_field* field, const grid* g) {
    ns_solver_params_t params = *params_in;

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, solver_type);
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

/** Relative L2 distance between the velocity fields of b and reference a (max of u, v) */
static double velocity_distance(const flow_field* a, const flow_field* b, size_t n) {
    double u_norm = test_compute_l2_norm(a->u, n);
    double v_norm = test_compute_l2_norm(a->v, n);
    TEST_ASSERT_TRUE_MESSAGE(u_norm > 0.0 && v_norm > 0.0,
                             "Reference run produced a zero field");

    double u_err = test_compute_l2_error(a->u, b->u, n) / u_norm;
    double v_err = test_compute_l2_error(a->v, b->v, n) / v_norm;
    return (u_err > v_err) ? u_err : v_err;
}

//=============================================================================
// TEST: MG PRESSURE SOLVES PRODUCE DIVERGENCE-FREE FIELDS
//=============================================================================

static void run_divergence_free_check(ns_pressure_solver_t pressure_solver,
                                      const char* label) {
    for (size_t b = 0; b < NUM_MG_BACKENDS; b++) {
        const char* solver_type = MG_BACKENDS[b];
        if (!projection_available(solver_type)) {
            printf("      %s %s: not available (skipping)\n", solver_type, label);
            continue;
        }

        ns_solver_params_t params = make_params(pressure_solver);

        /* Divergence tolerance is relaxed (matching test_solver_projection.c):
         * the projection solver's simple iterative method may not fully converge
         * in a few steps. The meaningful assertion is that each step reduces
         * the divergence. */
        test_result result = test_run_divergence_free(
            solver_type, GRID_N, GRID_N, &params, 10, 1.0);

        if (result.solver_unavailable) {
            printf("      %s %s: solver unavailable, skipping\n", solver_type, label);
            continue;
        }

        printf("      %s %s: %s\n", solver_type, label, result.message);
        TEST_ASSERT_TRUE_MESSAGE(result.passed, result.message);
        TEST_ASSERT_TRUE_MESSAGE(result.final_divergence < result.initial_divergence,
            "Projection with an MG pressure solve must reduce divergence");
    }
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

static void check_mg_matches_cg(const char* solver_type) {
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
        run_projection_taylor_green(solver_type, &params_cg, field_cg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(solver_type, &params_mg, field_mg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS,
        run_projection_taylor_green(solver_type, &params_pcg, field_pcg, g));

    size_t n = (size_t)GRID_N * GRID_N;
    double u_norm = test_compute_l2_norm(field_cg->u, n);
    double v_norm = test_compute_l2_norm(field_cg->v, n);
    TEST_ASSERT_TRUE_MESSAGE(u_norm > 0.0 && v_norm > 0.0,
                             "CG reference run produced a zero field");

    double u_err_mg = test_compute_l2_error(field_cg->u, field_mg->u, n) / u_norm;
    double v_err_mg = test_compute_l2_error(field_cg->v, field_mg->v, n) / v_norm;
    double u_err_pcg = test_compute_l2_error(field_cg->u, field_pcg->u, n) / u_norm;
    double v_err_pcg = test_compute_l2_error(field_cg->v, field_pcg->v, n) / v_norm;

    printf("      %s MULTIGRID vs CG: rel L2 u=%.2e v=%.2e\n", solver_type, u_err_mg, v_err_mg);
    printf("      %s PCG_MG    vs CG: rel L2 u=%.2e v=%.2e\n", solver_type, u_err_pcg, v_err_pcg);

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

void test_projection_mg_matches_cg(void) {
    printf("\n    Testing MG pressure solves agree with default CG...\n");

    for (size_t b = 0; b < NUM_MG_BACKENDS; b++) {
        if (!projection_available(MG_BACKENDS[b])) {
            printf("      %s: not available (skipping)\n", MG_BACKENDS[b]);
            continue;
        }
        check_mg_matches_cg(MG_BACKENDS[b]);
    }
}

//=============================================================================
// TEST: OPENMP MG PRESSURE SOLVES MATCH THE SCALAR ONES
//=============================================================================

/**
 * projection_omp in each MG mode must reproduce the scalar projection in the
 * same mode, on a grid where the OpenMP multigrid kernels run threaded.
 *
 * MULTIGRID: the OMP predictor, RHS, multigrid cycle and corrector are all
 * element-wise with serial sums, so the fields agree to rounding.
 *
 * PCG_MG: OMP CG's dot products are parallel reductions, and at this dt an
 * MG-preconditioned and a plain CG pressure solve give nearly the same
 * velocity, so no absolute bound could tell PCG_MG from a slip to plain CG.
 * Instead the OMP PCG_MG run must be much closer to scalar PCG_MG than to
 * scalar CG.
 */
void test_projection_omp_mg_matches_scalar(void) {
    printf("\n    Testing projection_omp MG pressure solves match the scalar projection...\n");

    if (!projection_available(NS_SOLVER_TYPE_PROJECTION_OMP)) {
        printf("      %s: not available (skipping)\n", NS_SOLVER_TYPE_PROJECTION_OMP);
        return;
    }

    TEST_ASSERT_TRUE_MESSAGE(
        (size_t)(GRID_N_LARGE - 2) * (GRID_N_LARGE - 2) >= MG_OMP_MIN_POINTS,
        "GRID_N_LARGE must put the finest multigrid plane over MG_OMP_MIN_POINTS");

    grid* g = grid_create(GRID_N_LARGE, GRID_N_LARGE, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    flow_field* scalar_cg = flow_field_create(GRID_N_LARGE, GRID_N_LARGE, 1);
    flow_field* scalar_mg = flow_field_create(GRID_N_LARGE, GRID_N_LARGE, 1);
    flow_field* scalar_pcg = flow_field_create(GRID_N_LARGE, GRID_N_LARGE, 1);
    flow_field* omp_mg = flow_field_create(GRID_N_LARGE, GRID_N_LARGE, 1);
    flow_field* omp_pcg = flow_field_create(GRID_N_LARGE, GRID_N_LARGE, 1);
    TEST_ASSERT_NOT_NULL(scalar_cg);
    TEST_ASSERT_NOT_NULL(scalar_mg);
    TEST_ASSERT_NOT_NULL(scalar_pcg);
    TEST_ASSERT_NOT_NULL(omp_mg);
    TEST_ASSERT_NOT_NULL(omp_pcg);

    ns_solver_params_t params_cg = make_params(NS_PRESSURE_SOLVER_DEFAULT);
    ns_solver_params_t params_mg = make_params(NS_PRESSURE_SOLVER_MULTIGRID);
    ns_solver_params_t params_pcg = make_params(NS_PRESSURE_SOLVER_PCG_MG);
    params_cg.dt = TEST_DT_LARGE;
    params_mg.dt = TEST_DT_LARGE;
    params_pcg.dt = TEST_DT_LARGE;

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_cg, scalar_cg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_mg, scalar_mg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_pcg, scalar_pcg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION_OMP, &params_mg, omp_mg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION_OMP, &params_pcg, omp_pcg, g));

    size_t n = (size_t)GRID_N_LARGE * GRID_N_LARGE;
    double mg_dist = velocity_distance(scalar_mg, omp_mg, n);
    double pcg_to_pcg = velocity_distance(scalar_pcg, omp_pcg, n);
    double pcg_to_cg = velocity_distance(scalar_cg, omp_pcg, n);

    printf("      MULTIGRID omp vs scalar MULTIGRID: rel L2 %.2e\n", mg_dist);
    printf("      PCG_MG    omp vs scalar PCG_MG:    rel L2 %.2e\n", pcg_to_pcg);
    printf("      PCG_MG    omp vs scalar CG:        rel L2 %.2e\n", pcg_to_cg);

    TEST_ASSERT_TRUE_MESSAGE(mg_dist < OMP_MG_MATCH_TOL,
        "projection_omp MULTIGRID must reproduce the scalar MULTIGRID projection");
    TEST_ASSERT_TRUE_MESSAGE(pcg_to_cg > 0.0 && pcg_to_pcg * PCG_MG_SEPARATION <= pcg_to_cg,
        "projection_omp PCG_MG must match scalar PCG_MG, not plain CG");

    flow_field_destroy(scalar_cg);
    flow_field_destroy(scalar_mg);
    flow_field_destroy(scalar_pcg);
    flow_field_destroy(omp_mg);
    flow_field_destroy(omp_pcg);
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

    for (size_t b = 0; b < NUM_MG_BACKENDS; b++) {
        if (!test_solver_available(registry, MG_BACKENDS[b])) {
            printf("      %s: not available (skipping)\n", MG_BACKENDS[b]);
            continue;
        }

        for (size_t m = 0; m < NUM_MG_MODES; m++) {
            ns_solver_t* slv = cfd_solver_create(registry, MG_BACKENDS[b]);
            TEST_ASSERT_NOT_NULL_MESSAGE(slv, "projection solver not available");

            ns_solver_params_t params = make_params(MG_MODES[m]);
            cfd_status_t init_status = solver_init(slv, g, &params);
            printf("      %s, mode %d on %dx%d: init status %d\n", MG_BACKENDS[b],
                   (int)MG_MODES[m], GRID_N_DEGENERATE, GRID_N_DEGENERATE, (int)init_status);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_INVALID, init_status,
                "MG pressure solver on a degenerate grid must fail init with INVALID");

            solver_destroy(slv);
        }
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

    for (size_t b = 0; b < NUM_MG_BACKENDS; b++) {
        if (!test_solver_available(registry, MG_BACKENDS[b])) {
            printf("      %s: not available (skipping)\n", MG_BACKENDS[b]);
            continue;
        }

        for (size_t m = 0; m < NUM_MG_MODES; m++) {
            ns_solver_t* slv = cfd_solver_create(registry, MG_BACKENDS[b]);
            TEST_ASSERT_NOT_NULL_MESSAGE(slv, "projection solver not available");

            ns_solver_params_t params = make_params(MG_MODES[m]);
            cfd_status_t init_status = solver_init(slv, g, &params);
            printf("      %s, mode %d on %dx%d: init status %d\n", MG_BACKENDS[b],
                   (int)MG_MODES[m], GRID_N_BAD, GRID_N_BAD, (int)init_status);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, init_status,
                "MG pressure solver on a non-2^k+1 grid must fail init with UNSUPPORTED");

            solver_destroy(slv);
        }
    }

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

//=============================================================================
// TEST: OTHER PROJECTION BACKENDS REJECT MG SELECTION
//=============================================================================

void test_projection_backends_reject_mg(void) {
    printf("\n    Testing SIMD and GPU projection backends reject MG selection...\n");

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
        NS_SOLVER_TYPE_PROJECTION_GPU,
    };

    for (size_t b = 0; b < sizeof(backends) / sizeof(backends[0]); b++) {
        for (size_t m = 0; m < NUM_MG_MODES; m++) {
            ns_solver_t* slv = cfd_solver_create(registry, backends[b]);
            if (!slv) {
                printf("      %s: not available (skipping)\n", backends[b]);
                break;
            }

            ns_solver_params_t params = make_params(MG_MODES[m]);
            cfd_status_t init_status = solver_init(slv, g, &params);
            printf("      %s, mode %d: init status %d\n",
                   backends[b], (int)MG_MODES[m], (int)init_status);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, init_status,
                "Projection backends without a multigrid pressure solve must reject it");

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

    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_zero, field_zero, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_cg, field_cg, g));
    TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, run_projection_taylor_green(
        NS_SOLVER_TYPE_PROJECTION, &params_mg, field_mg, g));

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
// TEST: A SOLVER THAT RUNS NO PRESSURE SOLVE REFUSES A PRESSURE-SOLVER CHOICE
//=============================================================================

/**
 * ns_solver_params_t.pressure_solver was validated by no time integrator at
 * all: rk4 or explicit_euler accepted NS_PRESSURE_SOLVER_MULTIGRID and then
 * ignored it, because those solvers run no Poisson solve for it to select.
 */
void test_time_integrators_reject_a_pressure_solver_choice(void) {
    printf("\n    Testing the time integrators refuse a pressure-solver choice...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    /* Every registered solver that integrates in time without projecting. */
    const char* integrators[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,
        NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OPTIMIZED,
        NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_GPU,
        NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OPTIMIZED,
        NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_GPU,
    };

    int checked = 0;
    for (size_t i = 0; i < sizeof(integrators) / sizeof(integrators[0]); i++) {
        for (size_t m = 0; m < NUM_MG_MODES; m++) {
            ns_solver_t* slv = cfd_solver_create(registry, integrators[i]);
            if (!slv) {
                break;  /* backend absent in this build */
            }
            ns_solver_params_t params = make_params(MG_MODES[m]);
            cfd_status_t status = solver_init(slv, g, &params);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
                "a solver that runs no pressure solve must refuse a pressure-solver choice");
            solver_destroy(slv);
            checked++;
        }
    }
    printf("      refused on %d (solver, mode) pairs\n", checked);
    TEST_ASSERT_GREATER_THAN_INT(0, checked);

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

/** A value outside the enum is a caller error, not an unsupported backend. */
void test_unknown_pressure_solver_is_invalid(void) {
    printf("\n    Testing an out-of-range pressure_solver is INVALID...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(slv);

    ns_solver_params_t params = make_params((ns_pressure_solver_t)99);
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, solver_init(slv, g, &params));

    solver_destroy(slv);
    cfd_registry_destroy(registry);
    grid_destroy(g);
}

//=============================================================================
// TEST: THE GPU BACKENDS REFUSE A TURBULENCE MODEL AT INIT
//=============================================================================

/**
 * The GPU backends have no RANS kernels. They used to report that per step,
 * from inside solve_projection_method_gpu -- after init had already returned
 * success, which is the one moment a caller could still pick another backend.
 * The entry points keep their own check: they are exported, so a direct caller
 * reaches them without passing through any init.
 */
void test_gpu_rejects_turbulence_at_init(void) {
    printf("\n    Testing the GPU backends refuse a turbulence model at init...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    const char* gpu_solvers[] = {
        NS_SOLVER_TYPE_PROJECTION_GPU, NS_SOLVER_TYPE_EXPLICIT_EULER_GPU,
        NS_SOLVER_TYPE_RK2_GPU, NS_SOLVER_TYPE_RK4_GPU,
    };
    const turbulence_model_t models[] = {
        TURB_MODEL_K_EPSILON, TURB_MODEL_SPALART_ALLMARAS,
    };

    int checked = 0;
    for (size_t s = 0; s < sizeof(gpu_solvers) / sizeof(gpu_solvers[0]); s++) {
        for (size_t m = 0; m < sizeof(models) / sizeof(models[0]); m++) {
            ns_solver_t* slv = cfd_solver_create(registry, gpu_solvers[s]);
            if (!slv) {
                printf("      %s: not available (skipping)\n", gpu_solvers[s]);
                break;  /* no CUDA in this build */
            }
            ns_solver_params_t params = make_params(NS_PRESSURE_SOLVER_DEFAULT);
            params.turb_model = models[m];
            cfd_status_t status = solver_init(slv, g, &params);
            TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
                "a GPU solver must refuse a turbulence model at init, not at step time");
            solver_destroy(slv);
            checked++;
        }
    }
    printf("      refused on %d (solver, model) pairs\n", checked);

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

/** The CPU solvers do implement RANS and still accept a model. */
void test_cpu_accepts_turbulence(void) {
    printf("\n    Testing the CPU solvers still accept a turbulence model...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    const char* cpu_solvers[] = {
        NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        NS_SOLVER_TYPE_PROJECTION_OMP, NS_SOLVER_TYPE_EXPLICIT_EULER,
        NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK4,
    };
    for (size_t s = 0; s < sizeof(cpu_solvers) / sizeof(cpu_solvers[0]); s++) {
        ns_solver_t* slv = cfd_solver_create(registry, cpu_solvers[s]);
        if (!slv) {
            continue;
        }
        ns_solver_params_t params = make_params(NS_PRESSURE_SOLVER_DEFAULT);
        params.turb_model = TURB_MODEL_K_EPSILON;
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, solver_init(slv, g, &params),
            "a solver that implements RANS must accept a turbulence model");
        solver_destroy(slv);
    }

    cfd_registry_destroy(registry);
    grid_destroy(g);
}

/** An out-of-range turb_model is a caller error on every backend. */
void test_unknown_turbulence_model_is_invalid(void) {
    printf("\n    Testing an out-of-range turb_model is INVALID...\n");

    grid* g = grid_create(GRID_N, GRID_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL(registry);
    cfd_registry_register_defaults(registry);

    ns_solver_t* slv = cfd_solver_create(registry, NS_SOLVER_TYPE_PROJECTION);
    TEST_ASSERT_NOT_NULL(slv);

    ns_solver_params_t params = make_params(NS_PRESSURE_SOLVER_DEFAULT);
    params.turb_model = (turbulence_model_t)77;
    TEST_ASSERT_EQUAL_INT(CFD_ERROR_INVALID, solver_init(slv, g, &params));

    solver_destroy(slv);
    cfd_registry_destroy(registry);
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
    RUN_TEST(test_projection_omp_mg_matches_scalar);
    RUN_TEST(test_projection_mg_rejects_degenerate_grid_as_invalid);
    RUN_TEST(test_projection_mg_rejects_non_pow2_grid);
    RUN_TEST(test_projection_backends_reject_mg);
    RUN_TEST(test_projection_zero_init_backward_compat);
    RUN_TEST(test_time_integrators_reject_a_pressure_solver_choice);
    RUN_TEST(test_unknown_pressure_solver_is_invalid);
    RUN_TEST(test_gpu_rejects_turbulence_at_init);
    RUN_TEST(test_cpu_accepts_turbulence);
    RUN_TEST(test_unknown_turbulence_model_is_invalid);

    printf("\n========================================\n");
    return UNITY_END();
}
