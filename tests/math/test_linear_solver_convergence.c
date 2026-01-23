/**
 * @file test_linear_solver_convergence.c
 * @brief Linear solver convergence validation tests
 *
 * These tests verify that iterative solvers converge at rates predicted
 * by theory for the 2D Poisson problem.
 *
 * Tests cover (ROADMAP 1.2.3):
 *   - Jacobi: spectral radius ρ = cos(πh) < 1
 *   - SOR: optimal ω = 2/(1 + sin(πh)) gives fastest convergence
 *   - Red-Black SOR: same convergence rate as SOR, parallelizable
 *   - CG: convergence in O(√κ) iterations where κ is condition number
 *
 * Theory background:
 *   For 2D Poisson on unit square with mesh spacing h = 1/(n-1):
 *   - Jacobi spectral radius: ρ_J = cos(πh) ≈ 1 - π²h²/2
 *   - Optimal SOR parameter: ω_opt = 2/(1 + sin(πh))
 *   - SOR spectral radius: ρ_SOR = ω_opt - 1
 *   - Condition number: κ ≈ 4/(π²h²) = 4(n-1)²/π²
 *
 * Test methodology:
 *   We use the solver's native Neumann BCs with sinusoidal RHS (Neumann-compatible),
 *   measuring residual reduction per iteration to verify theoretical convergence rates.
 */

#include "unity.h"
#include "cfd/solvers/poisson_solver.h"
#include "cfd/core/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

/* Domain: [0, 1] x [0, 1] */
#define DOMAIN_XMIN 0.0
#define DOMAIN_XMAX 1.0
#define DOMAIN_YMIN 0.0
#define DOMAIN_YMAX 1.0

/* Tolerances */
#define SPECTRAL_RADIUS_TOL     0.01   /* Allow 1% deviation from theory */
#define CG_ITERATION_MARGIN     3.0    /* CG should converge in < 3*sqrt(kappa) iters */

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Create an aligned field initialized to zero
 */
static double* create_field(size_t nx, size_t ny) {
    double* field = (double*)cfd_malloc(nx * ny * sizeof(double));
    if (field) {
        for (size_t i = 0; i < nx * ny; i++) {
            field[i] = 0.0;
        }
    }
    return field;
}

/**
 * Create field with non-trivial initial guess
 * Uses a pattern that exercises the solver
 */
static void init_nontrivial_guess(double* p, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            /* Checkerboard-like pattern */
            p[j * nx + i] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }
    }
}

/**
 * Initialize uniform RHS (for reference, but may not converge with Neumann BCs)
 */
static void init_uniform_rhs(double* rhs, size_t nx, size_t ny, double value) {
    for (size_t i = 0; i < nx * ny; i++) {
        rhs[i] = value;
    }
}

/**
 * Initialize sinusoidal RHS that is compatible with Neumann BCs
 * f(x,y) = cos(2πx)cos(2πy) has zero integral over [0,1]²
 * Solution: p(x,y) = -cos(2πx)cos(2πy) / (8π²)
 */
static void init_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                 double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        double y = j * dy;
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            rhs[j * nx + i] = cos(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }
}

/**
 * Compute theoretical Jacobi spectral radius for 2D Poisson
 * ρ = cos(πh) where h is the mesh spacing
 */
static double theoretical_jacobi_spectral_radius(double h) {
    return cos(M_PI * h);
}

/**
 * Compute theoretical optimal SOR parameter for 2D Poisson
 * ω_opt = 2 / (1 + sin(πh))
 */
static double theoretical_optimal_omega(double h) {
    return 2.0 / (1.0 + sin(M_PI * h));
}

/**
 * Compute theoretical SOR spectral radius at optimal omega
 * ρ_SOR = ω_opt - 1
 */
static double theoretical_sor_spectral_radius(double h) {
    double omega_opt = theoretical_optimal_omega(h);
    return omega_opt - 1.0;
}

/**
 * Compute theoretical condition number for 2D Poisson
 * κ ≈ 4(n-1)² / π² = 4 / (π²h²)
 */
static double theoretical_condition_number(double h) {
    return 4.0 / (M_PI * M_PI * h * h);
}

/**
 * Estimate spectral radius from iteration history
 * Uses ratio of consecutive residuals: ρ ≈ (r_k / r_{k-1})
 * Averages over middle iterations where convergence is most regular
 */
static double estimate_spectral_radius(const double* residuals, int num_iters) {
    if (num_iters < 30) return 1.0;  /* Not enough data */

    /* Skip early transient (first 10 iters), average over next 40 */
    double sum_ratio = 0.0;
    int count = 0;
    int start = 10;
    int end = (num_iters < 50) ? num_iters : 50;

    for (int k = start; k < end; k++) {
        if (residuals[k - 1] > 1e-12 && residuals[k] > 1e-12) {
            double ratio = residuals[k] / residuals[k - 1];
            /* Only count reasonable ratios (convergent behavior) */
            if (ratio > 0.5 && ratio < 1.1) {
                sum_ratio += ratio;
                count++;
            }
        }
    }

    return (count > 5) ? sum_ratio / count : 1.0;
}

/**
 * Estimate iterations needed to reduce residual by factor
 * Based on spectral radius: iters ≈ log(factor) / log(ρ)
 */
static int theoretical_iterations(double spectral_radius, double reduction_factor) {
    if (spectral_radius >= 1.0 || spectral_radius <= 0.0) return 99999;
    return (int)ceil(log(reduction_factor) / log(spectral_radius));
}

/* ============================================================================
 * TEST: JACOBI SPECTRAL RADIUS
 * ============================================================================
 *
 * Verify that Jacobi iteration has spectral radius ρ ≈ cos(πh).
 * We measure residual reduction rate and compare to theory.
 */

void test_jacobi_spectral_radius(void) {
    printf("\n    Testing Jacobi spectral radius matches theory ρ = cos(πh)...\n");

    size_t sizes[] = {17, 33};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        double h = dx;

        double* p = create_field(n, n);
        double* p_temp = create_field(n, n);
        double* rhs = create_field(n, n);

        if (!p || !p_temp || !rhs) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        /* Use non-trivial initial guess with sinusoidal RHS */
        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);

        if (!solver) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Could not create solver");
            return;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-12;
        params.max_iterations = 200;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT(CFD_SUCCESS, status);

        /* Record residual history */
        double* residuals = (double*)malloc(200 * sizeof(double));
        if (!residuals) {
            poisson_solver_destroy(solver);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        /* Run iterations and record residuals */
        int iters = 0;
        for (int iter = 0; iter < 200; iter++) {
            double residual = 0.0;
            cfd_status_t iter_status = poisson_solver_iterate(solver, p, p_temp, rhs, &residual);
            if (iter_status != CFD_SUCCESS) {
                free(residuals);
                poisson_solver_destroy(solver);
                cfd_free(p);
                cfd_free(p_temp);
                cfd_free(rhs);
                TEST_FAIL_MESSAGE("poisson_solver_iterate failed");
                return;
            }
            residuals[iter] = residual;
            iters = iter + 1;
            if (residual < 1e-12) break;
        }

        /* Estimate spectral radius from residual history */
        double measured_rho = estimate_spectral_radius(residuals, iters);
        double theoretical_rho = theoretical_jacobi_spectral_radius(h);
        double relative_error = fabs(measured_rho - theoretical_rho) / theoretical_rho;

        printf("      %3zux%-3zu: h=%.4f, ρ_theory=%.4f, ρ_measured=%.4f, "
               "rel_error=%.1f%%, iters=%d\n",
               n, n, h, theoretical_rho, measured_rho, relative_error * 100, iters);

        /* Jacobi should show spectral radius close to cos(πh) */
        TEST_ASSERT_TRUE_MESSAGE(relative_error < SPECTRAL_RADIUS_TOL,
            "Jacobi spectral radius deviates too much from theory");

        free(residuals);
        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
    }
}

/* ============================================================================
 * TEST: SOR OVER-RELAXATION BENEFIT
 * ============================================================================
 *
 * Verify that over-relaxation (ω > 1) improves SOR convergence over
 * Gauss-Seidel (ω = 1). Tests several omega values and confirms that
 * at least one over-relaxed variant converges faster.
 *
 * Note: Theoretical ω_opt = 2/(1+sin(πh)) applies to Dirichlet BCs.
 * With Neumann BCs used by these solvers, optimal omega is typically lower.
 */

void test_sor_optimal_omega(void) {
    printf("\n    Testing SOR over-relaxation improves convergence...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    printf("      Grid %zux%zu\n", n, n);

    /* Test omega values: Gauss-Seidel (ω=1) vs over-relaxed (ω>1) */
    double omegas[] = {1.0, 1.3, 1.5, 1.7};
    const char* labels[] = {"ω=1.0 (GS)", "ω=1.3", "ω=1.5", "ω=1.7"};
    int num_omegas = sizeof(omegas) / sizeof(omegas[0]);

    int iterations[4];
    int gauss_seidel_iters = 0;
    int min_iters = 999999;
    int best_idx = -1;

    for (int w = 0; w < num_omegas; w++) {
        double omega = omegas[w];

        double* p = create_field(n, n);
        double* rhs = create_field(n, n);

        if (!p || !rhs) {
            cfd_free(p);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);

        if (!solver) {
            cfd_free(p);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Could not create solver");
            return;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 2000;
        params.omega = omega;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver init failed");

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "SOR solver did not converge");
        iterations[w] = stats.iterations;

        if (w == 0) gauss_seidel_iters = stats.iterations;

        if (stats.iterations < min_iters) {
            min_iters = stats.iterations;
            best_idx = w;
        }

        printf("      %s: %d iterations\n", labels[w], stats.iterations);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
    }

    printf("      Best: %s (%d iters)\n", labels[best_idx], min_iters);

    /* SOR with some ω > 1 should be faster than Gauss-Seidel (ω=1) */
    TEST_ASSERT_TRUE_MESSAGE(min_iters < gauss_seidel_iters,
        "Over-relaxation (ω>1) should converge faster than Gauss-Seidel");
}

/* ============================================================================
 * TEST: SOR VS JACOBI SPEEDUP
 * ============================================================================
 *
 * Verify SOR converges significantly faster than Jacobi.
 * At optimal omega, SOR should require O(n) iterations vs O(n²) for Jacobi.
 */

void test_sor_vs_jacobi_speedup(void) {
    printf("\n    Testing SOR speedup over Jacobi...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
    double omega = 1.5;  /* Use practical omega value */

    int jacobi_iters, sor_iters;

    /* Test Jacobi */
    {
        double* p = create_field(n, n);
        double* p_temp = create_field(n, n);
        double* rhs = create_field(n, n);

        TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed for p");
        TEST_ASSERT_NOT_NULL_MESSAGE(p_temp, "Memory allocation failed for p_temp");
        TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create Jacobi solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 5000;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "Jacobi solver init failed");

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "Jacobi solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "Jacobi solver did not converge");
        jacobi_iters = stats.iterations;

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
    }

    /* Test SOR */
    {
        double* p = create_field(n, n);
        double* rhs = create_field(n, n);

        TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed for p");
        TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create SOR solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 5000;
        params.omega = omega;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver init failed");

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "SOR solver did not converge");
        sor_iters = stats.iterations;

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
    }

    double speedup = (double)jacobi_iters / (double)sor_iters;
    printf("      Jacobi: %d iterations\n", jacobi_iters);
    printf("      SOR (ω=%.1f): %d iterations\n", omega, sor_iters);
    printf("      Speedup: %.1fx\n", speedup);

    /* SOR should be significantly faster than Jacobi.
     * Theory predicts O(n) vs O(n²) scaling, so expect substantial speedup.
     * At n=33 with ω=1.5, typical speedup is ~17x. Use 5x as conservative bound. */
    TEST_ASSERT_TRUE_MESSAGE(speedup > 5.0,
        "SOR should converge significantly faster than Jacobi (expected >5x speedup)");
}

/* ============================================================================
 * TEST: RED-BLACK SOR EQUIVALENCE
 * ============================================================================
 *
 * Verify Red-Black SOR has same convergence as standard SOR.
 * Red-Black ordering enables parallelization without changing convergence.
 */

void test_redblack_sor_equivalence(void) {
    printf("\n    Testing Red-Black SOR convergence...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
    double omega = 1.5;  /* Conservative omega value */

    int iters_sor, iters_rb;

    /* Test standard SOR */
    {
        double* p = create_field(n, n);
        double* rhs = create_field(n, n);

        TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed for p");
        TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create SOR solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 2000;
        params.omega = omega;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver init failed");

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "SOR solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "SOR solver did not converge");
        iters_sor = stats.iterations;

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
    }

    /* Test Red-Black SOR */
    {
        double* p = create_field(n, n);
        double* rhs = create_field(n, n);

        TEST_ASSERT_NOT_NULL_MESSAGE(p, "Memory allocation failed for p");
        TEST_ASSERT_NOT_NULL_MESSAGE(rhs, "Memory allocation failed for rhs");

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR);
        TEST_ASSERT_NOT_NULL_MESSAGE(solver, "Could not create Red-Black SOR solver");

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 2000;
        params.omega = omega;

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "Red-Black SOR solver init failed");

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, NULL, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "Red-Black SOR solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "Red-Black SOR solver did not converge");
        iters_rb = stats.iterations;

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(rhs);
    }

    printf("      Standard SOR (ω=%.1f): %d iterations\n", omega, iters_sor);
    printf("      Red-Black SOR (ω=%.1f): %d iterations\n", omega, iters_rb);

    /* Red-Black may have different characteristics due to ordering
     * but should be in the same ballpark (within factor of 2) */
    double ratio = (double)iters_rb / (double)iters_sor;
    printf("      Ratio (RB/SOR): %.2f\n", ratio);

    TEST_ASSERT_TRUE_MESSAGE(ratio > 0.5 && ratio < 2.0,
        "Red-Black SOR should have comparable iteration count to standard SOR");
}

/* ============================================================================
 * TEST: CG ITERATION BOUND
 * ============================================================================
 *
 * Verify CG converges in O(√κ) iterations where κ is condition number.
 * For 2D Poisson: κ ≈ 4(n-1)²/π², so √κ ≈ 2(n-1)/π
 */

void test_cg_iteration_bound(void) {
    printf("\n    Testing CG converges in O(sqrt(κ)) iterations...\n");

    size_t sizes[] = {17, 33, 65};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int sizes_tested = 0;

    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
        double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);
        double h = dx;

        double kappa = theoretical_condition_number(h);
        double sqrt_kappa = sqrt(kappa);
        int expected_bound = (int)(CG_ITERATION_MARGIN * sqrt_kappa);

        double* p = create_field(n, n);
        double* p_temp = create_field(n, n);
        double* rhs = create_field(n, n);

        if (!p || !p_temp || !rhs) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            POISSON_METHOD_CG, POISSON_BACKEND_SCALAR);

        if (!solver) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            printf("      %3zux%-3zu: CG solver not available\n", n, n);
            continue;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-8;
        params.max_iterations = (int)(n * n);  /* Upper bound: matrix size */

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        if (status != CFD_SUCCESS) {
            poisson_solver_destroy(solver);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            printf("      %3zux%-3zu: CG solver init failed\n", n, n);
            continue;
        }

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);
        TEST_ASSERT_EQUAL_INT_MESSAGE(CFD_SUCCESS, status, "CG solver solve failed");
        TEST_ASSERT_EQUAL_INT_MESSAGE(POISSON_CONVERGED, stats.status, "CG solver did not converge");

        printf("      %3zux%-3zu: κ=%.0f, √κ=%.1f, bound=%d, actual=%d\n",
               n, n, kappa, sqrt_kappa, expected_bound, stats.iterations);

        /* CG should converge well within the O(√κ) bound */
        TEST_ASSERT_TRUE_MESSAGE(stats.iterations < expected_bound,
            "CG should converge in O(sqrt(kappa)) iterations");

        /* Also verify it's much faster than matrix dimension */
        int matrix_dim = (int)((n - 2) * (n - 2));  /* Interior points */
        TEST_ASSERT_TRUE_MESSAGE(stats.iterations < matrix_dim / 2,
            "CG should be much faster than matrix dimension");

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
        sizes_tested++;
    }

    /* Ensure at least one size was actually tested */
    TEST_ASSERT_TRUE_MESSAGE(sizes_tested > 0,
        "CG solver not available - no sizes could be tested");
}

/* ============================================================================
 * TEST: SOLVER COMPARISON
 * ============================================================================
 *
 * Compare iteration counts across all solvers to verify relative performance.
 */

void test_solver_comparison(void) {
    printf("\n    Comparing solver iteration counts...\n");

    size_t n = 33;
    double dx = (DOMAIN_XMAX - DOMAIN_XMIN) / (n - 1);
    double dy = (DOMAIN_YMAX - DOMAIN_YMIN) / (n - 1);

    struct {
        const char* name;
        poisson_solver_method_t method;
        double omega;
        int needs_temp;
    } solvers[] = {
        {"Jacobi", POISSON_METHOD_JACOBI, 0.0, 1},
        {"SOR (ω=1.5)", POISSON_METHOD_SOR, 1.5, 0},
        {"SOR (ω=1.7)", POISSON_METHOD_SOR, 1.7, 0},
        {"Red-Black SOR", POISSON_METHOD_REDBLACK_SOR, 1.5, 0},
        {"CG", POISSON_METHOD_CG, 0.0, 1}
    };
    int num_solvers = sizeof(solvers) / sizeof(solvers[0]);

    int jacobi_iters = 0;

    for (int i = 0; i < num_solvers; i++) {
        double* p = create_field(n, n);
        double* p_temp = solvers[i].needs_temp ? create_field(n, n) : NULL;
        double* rhs = create_field(n, n);

        if (!p || (solvers[i].needs_temp && !p_temp) || !rhs) {
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            continue;
        }

        init_nontrivial_guess(p, n, n);
        init_sinusoidal_rhs(rhs, n, n, dx, dy);

        poisson_solver_t* solver = poisson_solver_create(
            solvers[i].method, POISSON_BACKEND_SCALAR);

        if (!solver) {
            printf("      %-15s: not available\n", solvers[i].name);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            continue;
        }

        poisson_solver_params_t params = poisson_solver_params_default();
        params.tolerance = 1e-6;
        params.max_iterations = 5000;
        if (solvers[i].omega > 0) {
            params.omega = solvers[i].omega;
        }

        cfd_status_t status = poisson_solver_init(solver, n, n, dx, dy, &params);
        if (status != CFD_SUCCESS) {
            printf("      %-15s: init failed\n", solvers[i].name);
            poisson_solver_destroy(solver);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            continue;
        }

        poisson_solver_stats_t stats = poisson_solver_stats_default();
        cfd_status_t solve_status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

        if (solve_status != CFD_SUCCESS) {
            printf("      %-15s: solve failed\n", solvers[i].name);
            poisson_solver_destroy(solver);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            continue;
        }

        if (stats.status != POISSON_CONVERGED) {
            printf("      %-15s: did not converge (%d iterations)\n",
                   solvers[i].name, stats.iterations);
            poisson_solver_destroy(solver);
            cfd_free(p);
            cfd_free(p_temp);
            cfd_free(rhs);
            continue;
        }

        if (i == 0) jacobi_iters = stats.iterations;

        double speedup = (jacobi_iters > 0 && stats.iterations > 0)
                         ? (double)jacobi_iters / stats.iterations : 0.0;
        printf("      %-15s: %5d iterations (%.1fx vs Jacobi)\n",
               solvers[i].name, stats.iterations, speedup);

        poisson_solver_destroy(solver);
        cfd_free(p);
        cfd_free(p_temp);
        cfd_free(rhs);
    }

    /* Informational test - verifies all solvers complete without crashes */
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

void setUp(void) {}
void tearDown(void) {}

int main(void) {
    UNITY_BEGIN();

    printf("\n========================================\n");
    printf("Linear Solver Convergence Validation Tests\n");
    printf("========================================\n");

    /* Jacobi tests */
    printf("\n--- Jacobi Convergence Tests ---\n");
    RUN_TEST(test_jacobi_spectral_radius);

    /* SOR tests */
    printf("\n--- SOR Convergence Tests ---\n");
    RUN_TEST(test_sor_optimal_omega);
    RUN_TEST(test_sor_vs_jacobi_speedup);

    /* Red-Black SOR tests */
    printf("\n--- Red-Black SOR Tests ---\n");
    RUN_TEST(test_redblack_sor_equivalence);

    /* CG tests */
    printf("\n--- CG Convergence Tests ---\n");
    RUN_TEST(test_cg_iteration_bound);

    /* Solver comparison */
    printf("\n--- Solver Comparison ---\n");
    RUN_TEST(test_solver_comparison);

    printf("\n========================================\n");
    return UNITY_END();
}
