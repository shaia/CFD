/**
 * @file test_poisson_config.c
 * @brief poisson_solver_init refuses configuration it cannot honour.
 *
 * An audit found eleven places where a parameter could be set and then silently
 * ignored: BiCGSTAB dropped params.krylov.preconditioner on every backend, three
 * GPU solvers never called a caller's apply_bc, multigrid overwrote it, and
 * ns_solver_params_t.pressure_solver was validated by no time integrator at all.
 * The repo forbids exactly this -- see .claude/CLAUDE.md on silent fallbacks --
 * but had no single place to enforce it.
 *
 * poisson_solver_check_config is that place. These tests pin its decisions, one
 * per rule, so a future parameter added to the wrong group or a backend that
 * quietly stops honouring one fails here rather than in someone's results.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/memory.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

#define CFG_N 17
static const double CFG_H = 1.0 / (double)(CFG_N - 1);

/** Init with these params, returning the status. Skips if the backend is absent. */
static cfd_status_t init_with(poisson_solver_method_t method,
                              poisson_solver_backend_t backend,
                              const poisson_solver_params_t* params,
                              poisson_solver_apply_bc_func hook,
                              int* created) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        *created = 0;
        return CFD_ERROR_UNSUPPORTED;
    }
    *created = 1;
    if (hook) {
        solver->apply_bc = hook;
    }
    cfd_status_t status =
        poisson_solver_init(solver, CFG_N, CFG_N, 1, CFG_H, CFG_H, 0.0, params);
    poisson_solver_destroy(solver);
    return status;
}

static void noop_hook(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

/* ============================================================================
 * A GROUP THE METHOD DOES NOT READ
 * ============================================================================ */

/**
 * The grouping is what makes this checkable at all: with a flat struct, "the
 * caller set an SOR knob on a CG solve" and "the caller left it alone" are the
 * same bytes.
 */
void test_group_not_owned_is_refused(void) {
    int created;

    /* sor on a Krylov method */
    poisson_solver_params_t p = poisson_solver_params_default();
    p.sor.omega = 1.7;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &p, NULL, &created),
        "CG does not read params.sor");

    /* sor on Jacobi, which reads no group at all */
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR, &p, NULL, &created),
        "Jacobi does not read params.sor");

    /* krylov on a stationary method */
    poisson_solver_params_t k = poisson_solver_params_default();
    k.krylov.preconditioner = POISSON_PRECOND_JACOBI;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR, &k, NULL, &created),
        "Red-Black SOR does not read params.krylov");

    /* multigrid on a Krylov method that is not preconditioned by one */
    poisson_solver_params_t m = poisson_solver_params_default();
    m.multigrid.max_levels = 2;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &m, NULL, &created),
        "plain CG does not read params.multigrid");
}

/** The same groups, on the methods that do read them, are accepted. */
void test_group_owned_is_accepted(void) {
    int created;

    poisson_solver_params_t p = poisson_solver_params_default();
    p.sor.omega = 1.7;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR, &p, NULL, &created));

    poisson_solver_params_t k = poisson_solver_params_default();
    k.krylov.preconditioner = POISSON_PRECOND_JACOBI;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &k, NULL, &created));

    poisson_solver_params_t m = poisson_solver_params_default();
    m.multigrid.max_levels = 2;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR, &m, NULL, &created));

    /* CG reads the multigrid group when preconditioned by one. */
    poisson_solver_params_t pcg = poisson_solver_params_default();
    pcg.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
    pcg.multigrid.max_levels = 2;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &pcg, NULL, &created));
}

/* ============================================================================
 * EXCEPTIONS WITHIN AN OWNED GROUP
 * ============================================================================ */

/** BiCGSTAB reads params.krylov but has no preconditioner on any backend. */
void test_bicgstab_refuses_a_preconditioner(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.krylov.preconditioner = POISSON_PRECOND_JACOBI;

    const poisson_solver_backend_t backends[] = {
        POISSON_BACKEND_SCALAR, POISSON_BACKEND_OMP, POISSON_BACKEND_SIMD
    };
    for (size_t b = 0; b < sizeof(backends) / sizeof(backends[0]); b++) {
        cfd_status_t status =
            init_with(POISSON_METHOD_BICGSTAB, backends[b], &p, NULL, &created);
        if (!created) {
            continue;
        }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
            "BiCGSTAB must refuse a preconditioner rather than ignore it");
    }
}

/** restart is the GMRES basis size; no other method reads it. */
void test_restart_outside_gmres_is_refused(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.krylov.restart = 10;

    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &p, NULL, &created));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR, &p, NULL, &created));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR, &p, NULL, &created));
}

/** The multigrid preconditioner is scalar and OpenMP CG only. */
void test_mg_preconditioner_backend_limits(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &p, NULL, &created));

    cfd_status_t simd = init_with(POISSON_METHOD_CG, POISSON_BACKEND_SIMD, &p, NULL, &created);
    if (created) {
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, simd,
            "SIMD CG has no multigrid preconditioner");
    }
    cfd_status_t gmres = init_with(POISSON_METHOD_GMRES, POISSON_BACKEND_SCALAR, &p, NULL, &created);
    if (created) {
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, gmres,
            "GMRES has no multigrid preconditioner");
    }
}

/** A preconditioner's inner cycle must stay symmetric for CG. */
void test_mg_preconditioner_symmetry_is_protected(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
    p.multigrid.pre_smooth = 3;
    p.multigrid.post_smooth = 1;  /* asymmetric: M is then not symmetric */

    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &p, NULL, &created),
        "unequal pre/post smoothing breaks the preconditioner's symmetry");

    /* The cycle, smoother and boundary mode are the preconditioner's too. A W
     * or F cycle is not a V applied harder; whether it stays symmetric depends
     * on how the recursion composes, so it is refused rather than assumed. */
    const char* names[] = {"cycle", "smoother", "bc"};
    for (int f = 0; f < 3; f++) {
        poisson_solver_params_t q = poisson_solver_params_default();
        q.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
        if (f == 0) { q.multigrid.cycle = MG_CYCLE_W; }
        if (f == 1) { q.multigrid.smoother = MG_SMOOTHER_JACOBI; }
        if (f == 2) { q.multigrid.bc = MG_BC_DIRICHLET; }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
            init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &q, NULL, &created),
            names[f]);
    }

    /* The rest of the group is the caller's and is accepted. */
    poisson_solver_params_t ok = poisson_solver_params_default();
    ok.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
    ok.multigrid.pre_smooth = 3;
    ok.multigrid.post_smooth = 3;
    ok.multigrid.coarse_max_iter = 25;
    ok.multigrid.max_levels = 3;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &ok, NULL, &created),
        "a symmetric inner cycle with the caller's sweep counts must be accepted");
}

/** iter % check_interval is undefined behaviour at zero. */
void test_zero_check_interval_is_refused(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.check_interval = 0;

    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &p, NULL, &created));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR, &p, NULL, &created));
}

/* ============================================================================
 * A CALLER'S apply_bc
 * ============================================================================ */

/**
 * Multigrid applies its walls inside the cycle and never calls the hook. It used
 * to install its own routine into this very slot, so assigning to it replaced
 * multigrid's and then did nothing.
 */
void test_multigrid_refuses_a_caller_hook(void) {
    int created;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED,
        init_with(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR, NULL, noop_hook, &created),
        "multigrid must refuse a hook it would ignore");
}

/** Multigrid without a hook still works: the refusal is about the caller's. */
void test_multigrid_without_a_hook_still_works(void) {
    int created;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR, NULL, NULL, &created));
}

/** Every GPU solver applies its walls on the device and never calls the hook. */
void test_gpu_refuses_a_caller_hook(void) {
    int created;
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_JACOBI, POISSON_METHOD_SOR, POISSON_METHOD_REDBLACK_SOR,
        POISSON_METHOD_CG, POISSON_METHOD_BICGSTAB
    };

    int checked = 0;
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        cfd_status_t status =
            init_with(methods[m], POISSON_BACKEND_GPU, NULL, noop_hook, &created);
        if (!created) {
            continue;  /* no CUDA in this build */
        }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
            "a GPU solver must refuse a hook it would ignore");
        checked++;
    }
    printf("GPU hook rejection covered %d solvers\n", checked);
}

/** The CPU Krylov and stationary solvers do honour a hook, and still accept one. */
void test_cpu_solvers_accept_a_hook(void) {
    int created;
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_JACOBI, POISSON_METHOD_SOR, POISSON_METHOD_REDBLACK_SOR,
        POISSON_METHOD_CG, POISSON_METHOD_BICGSTAB, POISSON_METHOD_GMRES
    };
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        cfd_status_t status =
            init_with(methods[m], POISSON_BACKEND_SCALAR, NULL, noop_hook, &created);
        if (!created) {
            continue;
        }
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status,
            "a solver that honours the hook must accept one");
    }
}

/** Default parameters are accepted by every method and backend, as ever. */
void test_defaults_are_always_accepted(void) {
    int created;
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_JACOBI, POISSON_METHOD_GAUSS_SEIDEL, POISSON_METHOD_SOR,
        POISSON_METHOD_REDBLACK_SOR, POISSON_METHOD_CG, POISSON_METHOD_BICGSTAB,
        POISSON_METHOD_GMRES, POISSON_METHOD_MULTIGRID
    };
    const poisson_solver_backend_t backends[] = {
        POISSON_BACKEND_SCALAR, POISSON_BACKEND_OMP,
        POISSON_BACKEND_SIMD, POISSON_BACKEND_GPU
    };

    const int n_methods = (int)(sizeof(methods) / sizeof(methods[0]));

    int checked = 0;
    int scalar_checked = 0;
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        for (size_t b = 0; b < sizeof(backends) / sizeof(backends[0]); b++) {
            cfd_status_t status = init_with(methods[m], backends[b], NULL, NULL, &created);
            if (!created) {
                continue;
            }
            TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status,
                "default parameters must never be refused");
            checked++;
            if (backends[b] == POISSON_BACKEND_SCALAR) {
                scalar_checked++;
            }
        }
    }
    printf("defaults accepted on %d (method, backend) pairs (%d scalar)\n",
           checked, scalar_checked);

    /* Guard against the loop skipping everything and passing vacuously. Counted
     * over the scalar backend only: it is the reference implementation, so every
     * method has one in every build, while OMP, SIMD and GPU are all optional.
     * A total across all four would be a number from whichever machine happened
     * to run it -- which is how this assertion first failed in CI, on a
     * scalar-only build that was behaving perfectly. */
    TEST_ASSERT_EQUAL_INT_MESSAGE(n_methods, scalar_checked,
        "every method must have a scalar implementation that accepts defaults");
}

/**
 * Gauss-Seidel overriding params.sor.omega is deliberately NOT refused.
 *
 * It is documented on the enum member and pinned by
 * test_gauss_seidel_is_sor_at_omega_one. The validator exists to catch silent
 * ignores, not to turn documented behaviour into an error, and the first version
 * of it got this wrong.
 */
void test_gauss_seidel_omega_is_still_accepted(void) {
    int created;
    poisson_solver_params_t p = poisson_solver_params_default();
    p.sor.omega = 1.8;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_GAUSS_SEIDEL, POISSON_BACKEND_SCALAR, &p, NULL, &created));
}

/* ============================================================================
 * AN ACCEPTED GROUP MUST ACTUALLY BE READ
 * ============================================================================ */

/**
 * Run PCG-MG with a V(sweeps, sweeps) inner cycle and return the CG iteration
 * count.
 *
 * More smoothing per level is a better preconditioner, so CG needs fewer
 * iterations. The relation is monotone at every grid size measured (17 through
 * 129), which is what makes it a usable signal that the field was read at all.
 * max_levels is the more obvious knob but a poor test: capping the hierarchy
 * changes the count by one or two either way on small grids, because the first
 * coarsening already handles most of the spectrum there.
 */
static int pcg_mg_iterations(int sweeps) {
    const size_t n = CFG_N * CFG_N;
    double* x = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    /* Something with content on every scale, so the coarse levels matter. */
    for (size_t j = 1; j < CFG_N - 1; j++) {
        for (size_t i = 1; i < CFG_N - 1; i++) {
            double xf = (double)i / (double)(CFG_N - 1);
            double yf = (double)j / (double)(CFG_N - 1);
            rhs[j * CFG_N + i] = sin(3.0 * 3.14159265358979323846 * xf)
                               * sin(3.0 * 3.14159265358979323846 * yf)
                               + 0.5 * (xf - 0.5);
        }
    }
    /* Zero-gradient walls make the operator singular; CG refuses a rhs with a
     * nonzero interior mean, and rightly -- that system has no solution. */
    poisson_make_rhs_compatible(rhs, CFG_N, CFG_N, 1);

    poisson_solver_params_t p = poisson_solver_params_default();
    p.krylov.preconditioner = POISSON_PRECOND_MULTIGRID;
    p.tolerance = 1e-10;
    p.max_iterations = 500;
    p.multigrid.pre_smooth = sweeps;
    p.multigrid.post_smooth = sweeps;  /* equal, or the preconditioner is not symmetric */

    poisson_solver_t* solver =
        poisson_solver_create(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        poisson_solver_init(solver, CFG_N, CFG_N, 1, CFG_H, CFG_H, 0.0, &p));

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, x_temp, rhs, &stats);
    poisson_solver_destroy(solver);

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
    return stats.iterations;
}

/**
 * The multigrid preconditioner used to hardcode its whole configuration and
 * drop the caller's group on the floor. The validator now says CG owns that
 * group when it is multigrid-preconditioned, so accepting it and ignoring it
 * would be the very thing this file exists to prevent.
 */
void test_pcg_mg_reads_the_multigrid_group(void) {
    int light = pcg_mg_iterations(1);  /* V(1,1): a weak preconditioner */
    int heavy = pcg_mg_iterations(4);  /* V(4,4): a strong one */

    printf("PCG-MG iterations: V(1,1) %d, V(4,4) %d\n", light, heavy);
    TEST_ASSERT_GREATER_THAN_INT_MESSAGE(0, heavy, "the solve must run");
    TEST_ASSERT_GREATER_THAN_INT_MESSAGE(heavy, light,
        "more smoothing per level must need fewer CG iterations; equal counts mean "
        "params.multigrid is being accepted and then dropped, which is the exact "
        "failure this file exists to catch");
}

/* ============================================================================
 * EVERY PRESET WORKS AT THE BACKEND IT SHIPS WITH
 * ============================================================================ */

/**
 * A preset that refuses its own default backend is worse than no preset.
 *
 * POISSON_PRESET_MULTIGRID_PCG did exactly that: it left backend at AUTO,
 * which resolves to SIMD wherever AVX2 or NEON is present, and no SIMD CG
 * implements the multigrid preconditioner -- so the one preset whose whole
 * purpose is "give me the fast preconditioned solve" was refused at init on
 * precisely the machines it was for. POISSON_METHOD_MULTIGRID had a matching
 * special case inside poisson_solver_create; the preconditioned form had none,
 * and every existing test passed POISSON_BACKEND_SCALAR explicitly, so nothing
 * exercised the default.
 */
void test_every_preset_runs_at_its_own_backend(void) {
    const size_t N = 33;  /* 2^5+1, so the multigrid presets can build a hierarchy */
    const double h = 1.0 / (double)(N - 1);
    const size_t n = N * N;

    double* x = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    for (size_t j = 1; j < N - 1; j++) {
        for (size_t i = 1; i < N - 1; i++) {
            double xf = (double)i / (double)(N - 1);
            double yf = (double)j / (double)(N - 1);
            rhs[j * N + i] = sin(2.0 * 3.14159265358979323846 * xf)
                           * sin(2.0 * 3.14159265358979323846 * yf);
        }
    }
    poisson_make_rhs_compatible(rhs, N, N, 1);

    const poisson_preset_t presets[] = {
        POISSON_PRESET_DEFAULT, POISSON_PRESET_ACCURATE, POISSON_PRESET_NONSYMMETRIC,
        POISSON_PRESET_SMOOTHER, POISSON_PRESET_MULTIGRID, POISSON_PRESET_MULTIGRID_PCG
    };
    const char* names[] = {
        "DEFAULT", "ACCURATE", "NONSYMMETRIC", "SMOOTHER", "MULTIGRID", "MULTIGRID_PCG"
    };

    for (size_t p = 0; p < sizeof(presets) / sizeof(presets[0]); p++) {
        poisson_solver_config_t cfg = poisson_solver_config_preset(presets[p]);
        poisson_solver_stats_t stats = poisson_solver_stats_default();
        memset(x, 0, n * sizeof(double));
        cfd_clear_error();

        cfd_status_t status =
            poisson_solve(x, x_temp, rhs, N, N, 1, h, h, 0.0, &cfg, &stats);

        char msg[160];
        snprintf(msg, sizeof(msg),
                 "preset %s must solve at the backend it ships with; got %d (%s)",
                 names[p], (int)status,
                 status == CFD_SUCCESS ? "ok" : cfd_get_last_error());
        TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status, msg);
    }

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * A FACE THAT CANNOT TAKE EFFECT
 * ============================================================================ */

/**
 * A 2D grid has no z-faces. poisson_apply_walls skips them and
 * poisson_walls_are_singular ignores them, so prescribing one used to be
 * accepted and then do nothing -- the silent ignore the validator exists to
 * prevent, reached through the validator's own per-face branch.
 */
void test_z_face_on_a_2d_grid_is_refused(void) {
    int created;

    poisson_solver_params_t front = poisson_solver_params_default();
    front.walls.front = POISSON_WALL_DIRICHLET;
    front.walls.values.front = 5.0;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &front, NULL, &created),
        "a prescribed front face cannot take effect on a 2D grid");

    poisson_solver_params_t back = poisson_solver_params_default();
    back.walls.back = POISSON_WALL_DIRICHLET;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_INVALID,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &back, NULL, &created),
        "a prescribed back face cannot take effect on a 2D grid");

    /* The x and y faces of the same 2D grid are honoured and stay accepted. */
    poisson_solver_params_t left = poisson_solver_params_default();
    left.walls.left = POISSON_WALL_DIRICHLET;
    left.walls.values.left = 5.0;
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &left, NULL, &created),
        "an x-face is honoured in 2D and must still be accepted");
}

/** In 3D the same z-face is honoured, so it must be accepted there. */
void test_z_face_on_a_3d_grid_is_accepted(void) {
    poisson_solver_t* solver =
        poisson_solver_create(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);

    poisson_solver_params_t p = poisson_solver_params_default();
    p.walls.front = POISSON_WALL_DIRICHLET;
    p.walls.values.front = 5.0;

    cfd_status_t status = poisson_solver_init(solver, 9, 9, 9,
                                              CFG_H, CFG_H, CFG_H, &p);
    TEST_ASSERT_EQUAL_MESSAGE(CFD_SUCCESS, status,
        "a z-face is a real face in 3D");
    poisson_solver_destroy(solver);
}

/* ============================================================================
 * A REFUSED SOLVE LEAVES NO STALE NUMBERS
 * ============================================================================ */

/**
 * The refusal used to set status and iterations and leave the other four
 * fields alone, so a caller reusing one stats struct read the previous solve's
 * residual and timing next to POISSON_INCOMPATIBLE_RHS -- which looks like a
 * converged result.
 */
void test_refused_solve_resets_the_stats(void) {
    const size_t n = CFG_N * CFG_N;
    double* x = (double*)cfd_calloc(n, sizeof(double));
    double* x_temp = (double*)cfd_calloc(n, sizeof(double));
    double* rhs = (double*)cfd_calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL(x);
    TEST_ASSERT_NOT_NULL(x_temp);
    TEST_ASSERT_NOT_NULL(rhs);

    poisson_solver_t* solver =
        poisson_solver_create(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    poisson_solver_params_t p = poisson_solver_params_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        poisson_solver_init(solver, CFG_N, CFG_N, 1, CFG_H, CFG_H, 0.0, &p));

    /* A solvable problem first, so the stats struct carries real numbers. */
    for (size_t j = 1; j < CFG_N - 1; j++) {
        for (size_t i = 1; i < CFG_N - 1; i++) {
            rhs[j * CFG_N + i] = (double)((i + j) % 3) - 1.0;
        }
    }
    poisson_make_rhs_compatible(rhs, CFG_N, CFG_N, 1);

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        poisson_solver_solve(solver, x, x_temp, rhs, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    double converged_residual = stats.final_residual;

    /* Now an incompatible one, into the same struct. */
    for (size_t k = 0; k < n; k++) {
        rhs[k] = 1.0;  /* strictly positive: nonzero interior mean */
    }
    memset(x, 0, n * sizeof(double));
    cfd_clear_error();
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, &stats);

    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, status);
    TEST_ASSERT_EQUAL(POISSON_INCOMPATIBLE_RHS, stats.status);
    TEST_ASSERT_EQUAL_INT(0, stats.iterations);
    TEST_ASSERT_FALSE_MESSAGE(stats.final_residual == converged_residual,
        "final_residual must not still hold the previous solve's value");
    TEST_ASSERT_EQUAL_DOUBLE_MESSAGE(0.0, stats.elapsed_time_ms,
        "elapsed_time_ms must not hold the previous solve's timing");

    poisson_solver_destroy(solver);
    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

/* ============================================================================
 * NO GPU SOLVER IMPLEMENTS A PRECONDITIONER
 * ============================================================================ */

/**
 * There is no M^-1 apply anywhere under lib/src/solvers/linear/gpu/. The
 * multigrid rule already refused POISSON_PRECOND_MULTIGRID there, but
 * POISSON_PRECOND_JACOBI was accepted and then never used -- the same silent
 * ignore BiCGSTAB had on every backend, one enum value over.
 */
void test_gpu_refuses_any_preconditioner(void) {
    int created;
    const poisson_precond_type_t precond[] = {
        POISSON_PRECOND_JACOBI, POISSON_PRECOND_MULTIGRID
    };
    const poisson_solver_method_t methods[] = {
        POISSON_METHOD_CG, POISSON_METHOD_BICGSTAB
    };

    int checked = 0;
    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        for (size_t c = 0; c < sizeof(precond) / sizeof(precond[0]); c++) {
            poisson_solver_params_t p = poisson_solver_params_default();
            p.krylov.preconditioner = precond[c];
            cfd_status_t status =
                init_with(methods[m], POISSON_BACKEND_GPU, &p, NULL, &created);
            if (!created) {
                continue;  /* no CUDA in this build */
            }
            TEST_ASSERT_EQUAL_MESSAGE(CFD_ERROR_UNSUPPORTED, status,
                "a GPU solver must refuse a preconditioner it does not implement");
            checked++;
        }
    }
    printf("GPU preconditioner rejection covered %d combinations\n", checked);

    /* The same preconditioner on a CPU CG is implemented and stays accepted. */
    poisson_solver_params_t cpu = poisson_solver_params_default();
    cpu.krylov.preconditioner = POISSON_PRECOND_JACOBI;
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
        init_with(POISSON_METHOD_CG, POISSON_BACKEND_SCALAR, &cpu, NULL, &created));
}

/* ============================================================================
 * A PUBLIC HELPER TAKING RAW DIMENSIONS
 * ============================================================================ */

/**
 * poisson_make_rhs_compatible forwards nx/ny/nz into loops that count from 1 to
 * n - 1 in size_t. A degenerate dimension wrapped that to SIZE_MAX and ran off
 * the buffer; there is no interior to take a mean over anyway.
 */
void test_make_rhs_compatible_survives_degenerate_dims(void) {
    double guard[16];
    for (int i = 0; i < 16; i++) {
        guard[i] = 7.0;
    }

    poisson_make_rhs_compatible(guard, 0, 4, 1);
    poisson_make_rhs_compatible(guard, 4, 0, 1);
    poisson_make_rhs_compatible(guard, 1, 1, 1);
    poisson_make_rhs_compatible(guard, 2, 2, 1);
    poisson_make_rhs_compatible(NULL, 4, 4, 1);

    for (int i = 0; i < 16; i++) {
        TEST_ASSERT_EQUAL_DOUBLE_MESSAGE(7.0, guard[i],
            "a grid with no interior must leave the buffer alone");
    }

    /* A real grid still has its interior mean removed. */
    double f[25];
    for (int i = 0; i < 25; i++) {
        f[i] = 1.0;
    }
    poisson_make_rhs_compatible(f, 5, 5, 1);
    double sum = 0.0;
    for (int j = 1; j < 4; j++) {
        for (int i = 1; i < 4; i++) {
            sum += f[j * 5 + i];
        }
    }
    TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(1e-14, 0.0, sum,
        "the interior mean must actually be removed on a real grid");
}

/* ============================================================================
 * STATUS NAMES
 * ============================================================================ */

/**
 * POISSON_INCOMPATIBLE_RHS had no name, so both examples printed it as a
 * generic "error" from a hand-rolled ternary chain that predated it.
 */
void test_every_status_has_a_distinct_name(void) {
    const poisson_solver_status_t all[] = {
        POISSON_CONVERGED, POISSON_MAX_ITER, POISSON_DIVERGED,
        POISSON_STAGNATED, POISSON_INCOMPATIBLE_RHS, POISSON_ERROR
    };
    const size_t n = sizeof(all) / sizeof(all[0]);

    for (size_t i = 0; i < n; i++) {
        const char* name = poisson_solver_status_string(all[i]);
        TEST_ASSERT_NOT_NULL(name);
        TEST_ASSERT_TRUE_MESSAGE(name[0] != '\0', "a status name must not be empty");
        for (size_t j = i + 1; j < n; j++) {
            TEST_ASSERT_FALSE_MESSAGE(
                strcmp(name, poisson_solver_status_string(all[j])) == 0,
                "two statuses must not share a name");
        }
    }

    /* Specifically: the one that used to read as a generic failure. */
    TEST_ASSERT_EQUAL_STRING("incompatible rhs",
        poisson_solver_status_string(POISSON_INCOMPATIBLE_RHS));

    /* A value outside the enum is named, not a NULL deref. */
    TEST_ASSERT_EQUAL_STRING("unknown",
        poisson_solver_status_string((poisson_solver_status_t)123));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_group_not_owned_is_refused);
    RUN_TEST(test_group_owned_is_accepted);
    RUN_TEST(test_bicgstab_refuses_a_preconditioner);
    RUN_TEST(test_restart_outside_gmres_is_refused);
    RUN_TEST(test_mg_preconditioner_backend_limits);
    RUN_TEST(test_mg_preconditioner_symmetry_is_protected);
    RUN_TEST(test_zero_check_interval_is_refused);
    RUN_TEST(test_multigrid_refuses_a_caller_hook);
    RUN_TEST(test_multigrid_without_a_hook_still_works);
    RUN_TEST(test_gpu_refuses_a_caller_hook);
    RUN_TEST(test_cpu_solvers_accept_a_hook);
    RUN_TEST(test_defaults_are_always_accepted);
    RUN_TEST(test_gauss_seidel_omega_is_still_accepted);
    RUN_TEST(test_pcg_mg_reads_the_multigrid_group);
    RUN_TEST(test_every_status_has_a_distinct_name);
    RUN_TEST(test_every_preset_runs_at_its_own_backend);
    RUN_TEST(test_z_face_on_a_2d_grid_is_refused);
    RUN_TEST(test_z_face_on_a_3d_grid_is_accepted);
    RUN_TEST(test_refused_solve_resets_the_stats);
    RUN_TEST(test_gpu_refuses_any_preconditioner);
    RUN_TEST(test_make_rhs_compatible_survives_degenerate_dims);
    return UNITY_END();
}
