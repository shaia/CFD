/**
 * @file test_multigrid_convergence.c
 * @brief Geometric multigrid solver convergence and API tests
 *
 * Tests cover:
 *   - Creation / initialization / backend rules (AUTO -> scalar, others NULL)
 *   - Grid dimension validation (2^k+1 per active dimension)
 *   - V(2,2) per-cycle convergence factor < 0.15 (Dirichlet, acceptance)
 *   - Grid-size-independent cycle counts, 9x9 .. 129x129 (acceptance)
 *   - Neumann mode agrees with RB-SOR after mean subtraction (nullspace)
 *   - W-cycle, F-cycle (FMG discretization accuracy), Jacobi smoother
 *   - 3D grids, mixed dimensions, inhomogeneous Dirichlet data
 *   - mg_max_levels cap (two-grid method)
 *   - poisson_solve_3d convenience preset incl. invalid-dims caching
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include "cfd/core/indexing.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * HELPERS
 * ============================================================================ */

static double* create_field(size_t total) {
    double* field = (double*)cfd_calloc(total, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "field allocation failed");
    return field;
}

/** RHS for -analytic- p = sin(pi x)sin(pi y): f = -2 pi^2 p (p = 0 boundary) */
static void init_dirichlet_rhs_2d(double* rhs, size_t nx, size_t ny,
                                  double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = (double)j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = (double)i * dx;
            rhs[IDX_2D(i, j, nx)] =
                -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/** Neumann-compatible RHS f = cos(2 pi x)cos(2 pi y), interior mean removed */
static void init_neumann_rhs_2d(double* rhs, size_t nx, size_t ny,
                                double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = (double)j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = (double)i * dx;
            rhs[IDX_2D(i, j, nx)] =
                cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }

    double mean = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            mean += rhs[IDX_2D(i, j, nx)];
        }
    }
    mean /= (double)((nx - 2) * (ny - 2));
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            rhs[IDX_2D(i, j, nx)] -= mean;
        }
    }
}

static void subtract_interior_mean_2d(double* f, size_t nx, size_t ny) {
    double mean = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            mean += f[IDX_2D(i, j, nx)];
        }
    }
    mean /= (double)((nx - 2) * (ny - 2));
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            f[IDX_2D(i, j, nx)] -= mean;
        }
    }
}

static void subtract_interior_mean_3d(double* f, size_t nx, size_t ny,
                                      size_t nz) {
    double mean = 0.0;
    for (size_t k = 1; k < nz - 1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                mean += f[IDX_3D(i, j, k, nx, ny)];
            }
        }
    }
    mean /= (double)((nx - 2) * (ny - 2) * (nz - 2));
    for (size_t k = 1; k < nz - 1; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                f[IDX_3D(i, j, k, nx, ny)] -= mean;
            }
        }
    }
}

/**
 * Create + init a multigrid solver, solve, destroy. Asserts creation and
 * init succeed; returns solve status via *stats.
 */
static cfd_status_t solve_mg_2d(double* x, const double* rhs,
                                size_t nx, size_t ny, double dx, double dy,
                                const poisson_solver_params_t* params,
                                poisson_solver_stats_t* stats) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL_MESSAGE(solver, "multigrid create failed");
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        solver, nx, ny, 1, dx, dy, 0.0, params));

    double* x_temp = create_field(nx * ny);
    cfd_status_t status = poisson_solver_solve(solver, x, x_temp, rhs, stats);
    cfd_free(x_temp);
    poisson_solver_destroy(solver);
    return status;
}

/* ============================================================================
 * CREATION / VALIDATION
 * ============================================================================ */

void test_mg_create_init_metadata(void) {
    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL_STRING("multigrid_scalar", solver->name);
    TEST_ASSERT_EQUAL(POISSON_METHOD_MULTIGRID, solver->method);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_SCALAR, solver->backend);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        solver, 33, 33, 1, 1.0 / 32.0, 1.0 / 32.0, 0.0, NULL));
    poisson_solver_destroy(solver);

    /* AUTO resolves to the only (hence best) available backend: scalar */
    solver = poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                   POISSON_BACKEND_AUTO);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(POISSON_BACKEND_SCALAR, solver->backend);
    poisson_solver_destroy(solver);

    /* Explicit non-scalar backends: no silent fallbacks */
    TEST_ASSERT_NULL(poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                           POISSON_BACKEND_SIMD));
    TEST_ASSERT_NULL(poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                           POISSON_BACKEND_OMP));
    TEST_ASSERT_NULL(poisson_solver_create(POISSON_METHOD_MULTIGRID,
                                           POISSON_BACKEND_GPU));
}

void test_mg_invalid_dims_rejected(void) {
    static const size_t bad_dims[][3] = {
        {10, 10, 1}, {33, 34, 1}, {32, 32, 1}, {17, 17, 8},
    };
    static const size_t good_dims[][3] = {
        {17, 33, 1}, {17, 17, 9},
    };

    for (size_t n = 0; n < sizeof(bad_dims) / sizeof(bad_dims[0]); n++) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, poisson_solver_init(
            solver, bad_dims[n][0], bad_dims[n][1], bad_dims[n][2],
            0.1, 0.1, (bad_dims[n][2] > 1) ? 0.1 : 0.0, NULL));
        poisson_solver_destroy(solver);
    }

    for (size_t n = 0; n < sizeof(good_dims) / sizeof(good_dims[0]); n++) {
        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL(solver);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
            solver, good_dims[n][0], good_dims[n][1], good_dims[n][2],
            0.1, 0.1, (good_dims[n][2] > 1) ? 0.1 : 0.0, NULL));
        poisson_solver_destroy(solver);
    }
}

/* ============================================================================
 * CONVERGENCE FACTOR AND GRID INDEPENDENCE (acceptance criteria)
 * ============================================================================ */

void test_mg_vcycle_convergence_factor_dirichlet(void) {
    const size_t NX = 65, NY = 65;
    const double DX = 1.0 / (double)(NX - 1);
    const double DY = 1.0 / (double)(NY - 1);
    const int NUM_CYCLES = 8;

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        solver, NX, NY, 1, DX, DY, 0.0, &params));

    double* x = create_field(NX * NY);
    double* rhs = create_field(NX * NY);
    init_dirichlet_rhs_2d(rhs, NX, NY, DX, DY);

    double res[8];
    for (int n = 0; n < NUM_CYCLES; n++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_iterate(
            solver, x, NULL, rhs, &res[n]));
    }

    /* Every per-cycle residual reduction after the first cycle must beat the
     * V(2,2) Red-Black GS acceptance factor of 0.15 (grid-size-independent) */
    for (int n = 1; n < NUM_CYCLES - 1; n++) {
        TEST_ASSERT_TRUE_MESSAGE(res[n] > 0.0, "residual vanished early");
        double factor = res[n + 1] / res[n];
        TEST_ASSERT_TRUE_MESSAGE(factor < 0.15,
                                 "V(2,2) convergence factor >= 0.15");
    }

    cfd_free(x);
    cfd_free(rhs);
    poisson_solver_destroy(solver);
}

void test_mg_grid_independence_dirichlet(void) {
    static const size_t sizes[] = {9, 17, 33, 65, 129};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int min_iters = 1000, max_iters = 0;

    for (size_t n = 0; n < num_sizes; n++) {
        size_t nx = sizes[n];
        double dx = 1.0 / (double)(nx - 1);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.mg_bc = MG_BC_DIRICHLET;
        params.tolerance = 1e-6;

        double* x = create_field(nx * nx);
        double* rhs = create_field(nx * nx);
        init_dirichlet_rhs_2d(rhs, nx, nx, dx, dx);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
            x, rhs, nx, nx, dx, dx, &params, &stats));
        TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
        TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 15,
                                 "V-cycle count grew beyond 15");

        if (stats.iterations < min_iters) min_iters = stats.iterations;
        if (stats.iterations > max_iters) max_iters = stats.iterations;

        cfd_free(x);
        cfd_free(rhs);
    }

    /* The key multigrid property: cycle count does not grow with grid size */
    TEST_ASSERT_TRUE_MESSAGE(max_iters - min_iters <= 3,
                             "cycle count varies with grid size");
}

void test_mg_neumann_grid_independence(void) {
    static const size_t sizes[] = {17, 33, 65};

    for (size_t n = 0; n < sizeof(sizes) / sizeof(sizes[0]); n++) {
        size_t nx = sizes[n];
        double dx = 1.0 / (double)(nx - 1);

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6; /* default Neumann mode */

        double* x = create_field(nx * nx);
        double* rhs = create_field(nx * nx);
        init_neumann_rhs_2d(rhs, nx, nx, dx, dx);

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
            x, rhs, nx, nx, dx, dx, &params, &stats));
        TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
        TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 15,
                                 "Neumann V-cycle count grew beyond 15");

        cfd_free(x);
        cfd_free(rhs);
    }
}

/* ============================================================================
 * NEUMANN MODE VS CG (nullspace-aware comparison)
 * ============================================================================ */

void test_mg_neumann_matches_rbsor(void) {
    /* Reference is RB-SOR, not CG: RB-SOR applies the zero-gradient BC after
     * every sweep and so converges to the same mirror-BC discrete system as
     * MG. CG freezes boundary values during its Krylov iterations (applying
     * the BC only at start/end), which is a different discrete problem. */
    const size_t NX = 33, NY = 33;
    const double DX = 1.0 / (double)(NX - 1);
    const double DY = 1.0 / (double)(NY - 1);

    double* rhs = create_field(NX * NY);
    init_neumann_rhs_2d(rhs, NX, NY, DX, DY);

    poisson_solver_params_t sor_params = poisson_solver_params_default();
    sor_params.tolerance = 1e-10;
    sor_params.absolute_tolerance = 1e-12;
    sor_params.max_iterations = 20000;

    poisson_solver_t* sor = poisson_solver_create(
        POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(sor);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        sor, NX, NY, 1, DX, DY, 0.0, &sor_params));

    double* x_sor = create_field(NX * NY);
    double* x_temp = create_field(NX * NY);
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_solve(
        sor, x_sor, x_temp, rhs, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    poisson_solver_destroy(sor);

    /* Multigrid, default Neumann mode */
    poisson_solver_params_t mg_params = poisson_solver_params_default();
    mg_params.tolerance = 1e-10;
    mg_params.absolute_tolerance = 1e-12;

    double* x_mg = create_field(NX * NY);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x_mg, rhs, NX, NY, DX, DY, &mg_params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    /* Both solve the same singular system: solutions differ by a constant,
     * so compare after removing the interior mean */
    subtract_interior_mean_2d(x_sor, NX, NY);
    subtract_interior_mean_2d(x_mg, NX, NY);

    double max_diff = 0.0;
    for (size_t j = 1; j < NY - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            double diff = fabs(x_sor[IDX_2D(i, j, NX)]
                             - x_mg[IDX_2D(i, j, NX)]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(max_diff < 1e-6,
                             "MG and RB-SOR disagree beyond 1e-6 after mean removal");

    cfd_free(rhs);
    cfd_free(x_sor);
    cfd_free(x_mg);
    cfd_free(x_temp);
}

/* ============================================================================
 * CYCLE VARIANTS AND SMOOTHERS
 * ============================================================================ */

void test_mg_wcycle_converges(void) {
    const size_t NX = 65;
    const double DX = 1.0 / (double)(NX - 1);

    double* rhs = create_field(NX * NX);
    init_dirichlet_rhs_2d(rhs, NX, NX, DX, DX);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;
    params.tolerance = 1e-6;

    /* V-cycle baseline */
    double* x = create_field(NX * NX);
    poisson_solver_stats_t v_stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &v_stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, v_stats.status);
    cfd_free(x);

    /* W-cycle: at least as strong per cycle */
    params.mg_cycle = MG_CYCLE_W;
    x = create_field(NX * NX);
    poisson_solver_stats_t w_stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &w_stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, w_stats.status);
    TEST_ASSERT_TRUE_MESSAGE(w_stats.iterations <= v_stats.iterations,
                             "W-cycle needed more cycles than V-cycle");
    cfd_free(x);

    cfd_free(rhs);
}

void test_mg_fmg_discretization_accuracy(void) {
    const size_t NX = 65;
    const double DX = 1.0 / (double)(NX - 1);

    double* rhs = create_field(NX * NX);
    init_dirichlet_rhs_2d(rhs, NX, NX, DX, DX);

    /* Reference: fully converged V-cycle solve -> discretization error */
    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;
    params.tolerance = 1e-10;
    params.absolute_tolerance = 1e-12;

    double* x_ref = create_field(NX * NX);
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x_ref, rhs, NX, NX, DX, DX, &params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    double err_ref = 0.0;
    for (size_t j = 1; j < NX - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            double exact = sin(M_PI * (double)i * DX)
                         * sin(M_PI * (double)j * DX);
            double err = fabs(x_ref[IDX_2D(i, j, NX)] - exact);
            if (err > err_ref) err_ref = err;
        }
    }
    cfd_free(x_ref);

    /* One FMG pass (a single iterate on a fresh F-cycle solver) must land
     * within a small factor of the discretization error */
    params.mg_cycle = MG_CYCLE_F;

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        solver, NX, NX, 1, DX, DX, 0.0, &params));

    double* x_fmg = create_field(NX * NX);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_iterate(
        solver, x_fmg, NULL, rhs, NULL));

    double err_fmg = 0.0;
    for (size_t j = 1; j < NX - 1; j++) {
        for (size_t i = 1; i < NX - 1; i++) {
            double exact = sin(M_PI * (double)i * DX)
                         * sin(M_PI * (double)j * DX);
            double err = fabs(x_fmg[IDX_2D(i, j, NX)] - exact);
            if (err > err_fmg) err_fmg = err;
        }
    }
    TEST_ASSERT_TRUE_MESSAGE(err_fmg <= 2.0 * err_ref,
                             "one FMG pass missed discretization accuracy");
    cfd_free(x_fmg);
    poisson_solver_destroy(solver);

    /* A full F-cycle solve converges in very few cycles */
    params.tolerance = 1e-6;
    params.absolute_tolerance = 1e-10;
    double* x = create_field(NX * NX);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 5,
                             "F-cycle solve took more than 5 cycles");
    cfd_free(x);

    cfd_free(rhs);
}

void test_mg_jacobi_smoother_converges(void) {
    const size_t NX = 33;
    const double DX = 1.0 / (double)(NX - 1);

    double* x = create_field(NX * NX);
    double* rhs = create_field(NX * NX);
    init_dirichlet_rhs_2d(rhs, NX, NX, DX, DX);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;
    params.mg_smoother = MG_SMOOTHER_JACOBI;
    params.tolerance = 1e-6;

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    /* Weighted Jacobi smooths less per sweep than RB-GS: looser bound */
    TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 25,
                             "Jacobi-smoothed V-cycle count grew beyond 25");

    cfd_free(x);
    cfd_free(rhs);
}

/* ============================================================================
 * 3D AND MIXED DIMENSIONS
 * ============================================================================ */

static void run_mg_3d_case(size_t n, mg_bc_type_t bc, int compare_rbsor) {
    double d = 1.0 / (double)(n - 1);
    size_t total = n * n * n;

    double* rhs = create_field(total);
    for (size_t k = 0; k < n; k++) {
        double z = (double)k * d;
        for (size_t j = 0; j < n; j++) {
            double y = (double)j * d;
            for (size_t i = 0; i < n; i++) {
                double x = (double)i * d;
                if (bc == MG_BC_DIRICHLET) {
                    rhs[IDX_3D(i, j, k, n, n)] = -3.0 * M_PI * M_PI
                        * sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
                } else {
                    rhs[IDX_3D(i, j, k, n, n)] = cos(2.0 * M_PI * x)
                        * cos(2.0 * M_PI * y) * cos(2.0 * M_PI * z);
                }
            }
        }
    }
    if (bc == MG_BC_NEUMANN) {
        subtract_interior_mean_3d(rhs, n, n, n);
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = bc;
    params.tolerance = 1e-6;

    poisson_solver_t* solver = poisson_solver_create(
        POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    TEST_ASSERT_NOT_NULL(solver);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
        solver, n, n, n, d, d, d, &params));

    double* x = create_field(total);
    double* x_temp = create_field(total);
    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_solve(
        solver, x, x_temp, rhs, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 15,
                             "3D V-cycle count grew beyond 15");
    poisson_solver_destroy(solver);

    if (compare_rbsor) {
        /* RB-SOR solves the same mirror-BC system as MG (see 2D test) */
        poisson_solver_params_t sor_params = poisson_solver_params_default();
        sor_params.tolerance = 1e-10;
        sor_params.absolute_tolerance = 1e-12;
        sor_params.max_iterations = 20000;

        poisson_solver_t* sor = poisson_solver_create(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL(sor);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_init(
            sor, n, n, n, d, d, d, &sor_params));

        double* x_sor = create_field(total);
        TEST_ASSERT_EQUAL(CFD_SUCCESS, poisson_solver_solve(
            sor, x_sor, x_temp, rhs, &stats));
        TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
        poisson_solver_destroy(sor);

        subtract_interior_mean_3d(x, n, n, n);
        subtract_interior_mean_3d(x_sor, n, n, n);

        double max_diff = 0.0;
        for (size_t k = 1; k < n - 1; k++) {
            for (size_t j = 1; j < n - 1; j++) {
                for (size_t i = 1; i < n - 1; i++) {
                    double diff = fabs(x[IDX_3D(i, j, k, n, n)]
                                     - x_sor[IDX_3D(i, j, k, n, n)]);
                    if (diff > max_diff) max_diff = diff;
                }
            }
        }
        TEST_ASSERT_TRUE_MESSAGE(max_diff < 1e-5,
                                 "3D MG and RB-SOR disagree after mean removal");
        cfd_free(x_sor);
    }

    cfd_free(x);
    cfd_free(x_temp);
    cfd_free(rhs);
}

void test_mg_3d_convergence(void) {
    run_mg_3d_case(9, MG_BC_DIRICHLET, 0);
    run_mg_3d_case(17, MG_BC_DIRICHLET, 0);
    run_mg_3d_case(9, MG_BC_NEUMANN, 0);
    run_mg_3d_case(17, MG_BC_NEUMANN, 1);
}

void test_mg_mixed_dims(void) {
    static const size_t dims[][2] = {{17, 33}, {33, 17}};

    for (size_t n = 0; n < sizeof(dims) / sizeof(dims[0]); n++) {
        size_t nx = dims[n][0];
        size_t ny = dims[n][1];
        double dx = 1.0 / (double)(nx - 1);
        double dy = 1.0 / (double)(ny - 1);

        for (int bc = 0; bc < 2; bc++) {
            poisson_solver_params_t params = poisson_solver_params_default();
            params.mg_bc = (bc == 0) ? MG_BC_NEUMANN : MG_BC_DIRICHLET;
            params.tolerance = 1e-6;

            double* x = create_field(nx * ny);
            double* rhs = create_field(nx * ny);
            if (params.mg_bc == MG_BC_DIRICHLET) {
                init_dirichlet_rhs_2d(rhs, nx, ny, dx, dy);
            } else {
                init_neumann_rhs_2d(rhs, nx, ny, dx, dy);
            }

            poisson_solver_stats_t stats = poisson_solver_stats_default();
            TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
                x, rhs, nx, ny, dx, dy, &params, &stats));
            TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
            TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 15,
                                     "mixed-dims cycle count grew beyond 15");

            cfd_free(x);
            cfd_free(rhs);
        }
    }
}

/* ============================================================================
 * INHOMOGENEOUS DIRICHLET DATA
 * ============================================================================ */

void test_mg_dirichlet_inhomogeneous(void) {
    /* p = x^2 + y^2 has Laplacian(p) = 4 represented EXACTLY by the 5-point
     * stencil, so the converged discrete solution equals the analytic one.
     * Verifies caller-supplied boundary values are held fixed and enter the
     * interior stencil correctly. */
    const size_t NX = 17;
    const double DX = 1.0 / (double)(NX - 1);

    double* x = create_field(NX * NX);
    double* rhs = create_field(NX * NX);
    for (size_t j = 0; j < NX; j++) {
        double yv = (double)j * DX;
        for (size_t i = 0; i < NX; i++) {
            double xv = (double)i * DX;
            rhs[IDX_2D(i, j, NX)] = 4.0;
            if (i == 0 || j == 0 || i == NX - 1 || j == NX - 1) {
                x[IDX_2D(i, j, NX)] = xv * xv + yv * yv;
            }
        }
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;
    params.tolerance = 1e-12;
    params.absolute_tolerance = 1e-12;

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);

    for (size_t j = 1; j < NX - 1; j++) {
        double yv = (double)j * DX;
        for (size_t i = 1; i < NX - 1; i++) {
            double xv = (double)i * DX;
            TEST_ASSERT_DOUBLE_WITHIN(1e-8, xv * xv + yv * yv,
                                      x[IDX_2D(i, j, NX)]);
        }
    }

    cfd_free(x);
    cfd_free(rhs);
}

/* ============================================================================
 * LEVEL CAP AND CONVENIENCE API
 * ============================================================================ */

void test_mg_max_levels_cap(void) {
    const size_t NX = 65;
    const double DX = 1.0 / (double)(NX - 1);

    double* x = create_field(NX * NX);
    double* rhs = create_field(NX * NX);
    init_dirichlet_rhs_2d(rhs, NX, NX, DX, DX);

    poisson_solver_params_t params = poisson_solver_params_default();
    params.mg_bc = MG_BC_DIRICHLET;
    params.mg_max_levels = 2;
    /* Two-grid: the 33x33 "coarsest" level needs a real solve, not 50 sweeps */
    params.mg_coarse_max_iter = 500;
    params.tolerance = 1e-6;

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, solve_mg_2d(
        x, rhs, NX, NX, DX, DX, &params, &stats));
    TEST_ASSERT_EQUAL(POISSON_CONVERGED, stats.status);
    TEST_ASSERT_TRUE_MESSAGE(stats.iterations <= 60,
                             "two-grid method failed to converge in 60 cycles");

    cfd_free(x);
    cfd_free(rhs);
}

void test_mg_convenience_api(void) {
    const size_t NX = 33;
    const double DX = 1.0 / (double)(NX - 1);

    double* p = create_field(NX * NX);
    double* p_temp = create_field(NX * NX);
    double* rhs = create_field(NX * NX);
    init_neumann_rhs_2d(rhs, NX, NX, DX, DX);

    int iters = poisson_solve_3d(p, p_temp, rhs, NX, NX, 1, DX, DX, 0.0,
                                 POISSON_SOLVER_MG_SCALAR);
    TEST_ASSERT_TRUE_MESSAGE(iters > 0, "convenience MG solve failed");

    /* Second call with same dims exercises the cached-solver path */
    for (size_t n = 0; n < NX * NX; n++) {
        p[n] = 0.0;
    }
    iters = poisson_solve_3d(p, p_temp, rhs, NX, NX, 1, DX, DX, 0.0,
                             POISSON_SOLVER_MG_SCALAR);
    TEST_ASSERT_TRUE_MESSAGE(iters > 0, "cached convenience MG solve failed");

    /* Invalid dims: init fails, cache slot must not keep a broken solver */
    double* p32 = create_field(32 * 32);
    double* rhs32 = create_field(32 * 32);
    iters = poisson_solve_3d(p32, NULL, rhs32, 32, 32, 1,
                             1.0 / 31.0, 1.0 / 31.0, 0.0,
                             POISSON_SOLVER_MG_SCALAR);
    TEST_ASSERT_EQUAL(-1, iters);
    /* And a valid solve afterwards still works (cache not poisoned) */
    for (size_t n = 0; n < NX * NX; n++) {
        p[n] = 0.0;
    }
    iters = poisson_solve_3d(p, p_temp, rhs, NX, NX, 1, DX, DX, 0.0,
                             POISSON_SOLVER_MG_SCALAR);
    TEST_ASSERT_TRUE_MESSAGE(iters > 0, "MG solve after failed init broke");

    cfd_free(p);
    cfd_free(p_temp);
    cfd_free(rhs);
    cfd_free(p32);
    cfd_free(rhs32);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_mg_create_init_metadata);
    RUN_TEST(test_mg_invalid_dims_rejected);

    RUN_TEST(test_mg_vcycle_convergence_factor_dirichlet);
    RUN_TEST(test_mg_grid_independence_dirichlet);
    RUN_TEST(test_mg_neumann_grid_independence);
    RUN_TEST(test_mg_neumann_matches_rbsor);

    RUN_TEST(test_mg_wcycle_converges);
    RUN_TEST(test_mg_fmg_discretization_accuracy);
    RUN_TEST(test_mg_jacobi_smoother_converges);

    RUN_TEST(test_mg_3d_convergence);
    RUN_TEST(test_mg_mixed_dims);
    RUN_TEST(test_mg_dirichlet_inhomogeneous);

    RUN_TEST(test_mg_max_levels_cap);
    RUN_TEST(test_mg_convenience_api);

    return UNITY_END();
}
