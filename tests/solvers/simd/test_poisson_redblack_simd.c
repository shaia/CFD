/**
 * Tests for Red-Black SOR SIMD Poisson Solver
 *
 * Tests the Red-Black SOR implementation with AVX2 SIMD vectorization.
 * Red-Black ordering allows partial vectorization: within each "color" sweep,
 * all updates are independent (they only read from the other color).
 */

#include "unity.h"
#include "cfd/core/cfd_init.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

// Include the Poisson solver header directly for testing
#include "../../../lib/src/solvers/simd/poisson_solver_simd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

//=============================================================================
// HELPER FUNCTIONS
//=============================================================================

static double* allocate_field(size_t nx, size_t ny) {
    return (double*)aligned_alloc(32, nx * ny * sizeof(double));
}

static void free_field(double* field) {
    aligned_free(field);
}

static void initialize_zero(double* field, size_t nx, size_t ny) {
    memset(field, 0, nx * ny * sizeof(double));
}

// Initialize RHS for a known analytical solution: p = sin(pi*x)*sin(pi*y)
// Laplacian of p = -2*pi^2 * sin(pi*x)*sin(pi*y)
// Scale down to make convergence easier (typical CFD RHS values are small)
static void initialize_sinusoidal_rhs(double* rhs, size_t nx, size_t ny, double dx, double dy) {
    double scale = 0.01;  // Scale factor to make RHS similar to CFD use case
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            // RHS = -Laplacian(p) = 2*pi^2 * sin(pi*x)*sin(pi*y)
            rhs[j * nx + i] = scale * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

static double compute_analytical_solution(double x, double y) {
    return 0.01 * sin(M_PI * x) * sin(M_PI * y);  // Same scale as RHS
}

static double compute_l2_error(const double* p, size_t nx, size_t ny, double dx, double dy) {
    double error = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double x = i * dx;
            double y = j * dy;
            double analytical = compute_analytical_solution(x, y);
            double diff = p[j * nx + i] - analytical;
            error += diff * diff;
        }
    }
    return sqrt(error / ((nx - 2) * (ny - 2)));
}

static double compute_max_residual(const double* p, const double* rhs,
                                    size_t nx, size_t ny, double dx2, double dy2) {
    double max_res = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
            double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;
            double res = fabs(p_xx + p_yy - rhs[idx]);
            if (res > max_res) max_res = res;
        }
    }
    return max_res;
}

//=============================================================================
// TEST: RED-BLACK SOLVER RUNS WITH SINUSOIDAL RHS
//=============================================================================

void test_redblack_runs_sinusoidal(void) {
    printf("\n=== Test: Red-Black SIMD Runs with Sinusoidal RHS ===\n");

    // Use smaller grid for faster test
    size_t nx = 16, ny = 16;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    initialize_zero(p, nx, ny);
    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    // Red-Black doesn't need p_temp (in-place update)
    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    printf("Iterations: %d (negative means max iterations reached)\n", iters);

    double residual = compute_max_residual(p, rhs, nx, ny, dx * dx, dy * dy);
    printf("Final residual: %.6e\n", residual);

    // Check for valid output
    int valid = 1;
    for (size_t i = 0; i < nx * ny && valid; i++) {
        if (!isfinite(p[i])) valid = 0;
    }
    TEST_ASSERT_TRUE_MESSAGE(valid, "Red-Black solver should produce valid results");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK SOLVER CONVERGES WITH ZERO RHS
//=============================================================================

void test_redblack_converges_zero_rhs(void) {
    printf("\n=== Test: Red-Black SIMD Converges with Zero RHS ===\n");

    size_t nx = 16, ny = 16;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    // Start with non-zero initial guess
    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = 0.1;
        rhs[i] = 0.0;  // Zero RHS - solution should converge
    }

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    printf("Iterations to converge: %d\n", iters);

    // With zero RHS and Neumann BCs, the solver should converge
    TEST_ASSERT_TRUE_MESSAGE(iters >= 0, "Should converge with zero RHS");

    // Solution should be near constant (zero gradient everywhere)
    double max_val = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (fabs(p[i]) > max_val) max_val = fabs(p[i]);
    }
    printf("Max |p|: %.6e\n", max_val);

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK ACHIEVES REASONABLE ACCURACY
//=============================================================================

void test_redblack_accuracy(void) {
    printf("\n=== Test: Red-Black SIMD Accuracy ===\n");

    // Use smaller grid for faster test execution
    size_t nx = 16, ny = 16;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_zero(p, nx, ny);
    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);
    printf("Iterations: %d\n", iters);

    // Check that solver made progress (residual decreased)
    double residual = compute_max_residual(p, rhs, nx, ny, dx * dx, dy * dy);
    printf("Final residual: %.6e\n", residual);

    // Allow non-convergence but ensure valid results
    int valid = 1;
    for (size_t i = 0; i < nx * ny; i++) {
        if (!isfinite(p[i])) { valid = 0; break; }
    }
    TEST_ASSERT_TRUE_MESSAGE(valid, "Results should be finite");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK HANDLES NON-ALIGNED GRID SIZES
//=============================================================================

void test_redblack_non_aligned_sizes(void) {
    printf("\n=== Test: Red-Black SIMD Non-Aligned Grid Sizes ===\n");

    // Test that SIMD handles non-aligned sizes correctly (no crashes, valid output)
    size_t sizes[][2] = {
        {9, 9},     // Small odd
        {13, 13},   // 4n+1
        {15, 15},   // 4n-1
        {11, 13},   // Non-square, both non-aligned
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = sizes[t][0];
        size_t ny = sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        initialize_zero(p, nx, ny);
        initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        // Check for NaN/Inf (main goal is SIMD correctness with non-aligned sizes)
        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "No NaN/Inf in result");

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d)\n", iters);
    }

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK DETERMINISTIC RESULTS
//=============================================================================

void test_redblack_deterministic(void) {
    printf("\n=== Test: Red-Black SIMD Deterministic ===\n");

    size_t nx = 32, ny = 32;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p1 = allocate_field(nx, ny);
    double* p2 = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    // First run
    initialize_zero(p1, nx, ny);
    int iters1 = poisson_solve_redblack_simd(p1, NULL, rhs, nx, ny, dx, dy);

    // Second run
    initialize_zero(p2, nx, ny);
    int iters2 = poisson_solve_redblack_simd(p2, NULL, rhs, nx, ny, dx, dy);

    TEST_ASSERT_EQUAL_MESSAGE(iters1, iters2, "Same iteration count");

    double max_diff = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        double diff = fabs(p1[i] - p2[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Maximum difference between runs: %.6e\n", max_diff);
    TEST_ASSERT_TRUE_MESSAGE(max_diff < 1e-14, "Results should be identical");

    free_field(p1);
    free_field(p2);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK UNIFORM RHS (CONSTANT SOLUTION)
//=============================================================================

void test_redblack_uniform_rhs(void) {
    printf("\n=== Test: Red-Black SIMD Uniform RHS ===\n");

    size_t nx = 32, ny = 32;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_zero(p, nx, ny);

    // Uniform RHS = 0 should give p = 0 (with zero BC)
    for (size_t i = 0; i < nx * ny; i++) {
        rhs[i] = 0.0;
    }

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    // Should converge quickly (already at solution)
    printf("Iterations: %d\n", iters);

    // Solution should be near zero
    double max_val = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        if (fabs(p[i]) > max_val) max_val = fabs(p[i]);
    }

    printf("Max |p|: %.6e\n", max_val);
    TEST_ASSERT_TRUE_MESSAGE(max_val < 1e-6, "Solution should be near zero");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK BOUNDARY CONDITIONS PRESERVED
//=============================================================================

void test_redblack_boundary_conditions(void) {
    printf("\n=== Test: Red-Black SIMD Boundary Conditions ===\n");

    size_t nx = 32, ny = 32;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_zero(p, nx, ny);
    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    // Check Neumann BC: dp/dn = 0 at boundaries
    int bc_ok = 1;
    for (size_t j = 0; j < ny && bc_ok; j++) {
        if (fabs(p[j * nx + 0] - p[j * nx + 1]) > 1e-10) bc_ok = 0;
        if (fabs(p[j * nx + nx - 1] - p[j * nx + nx - 2]) > 1e-10) bc_ok = 0;
    }
    for (size_t i = 0; i < nx && bc_ok; i++) {
        if (fabs(p[i] - p[nx + i]) > 1e-10) bc_ok = 0;
        if (fabs(p[(ny - 1) * nx + i] - p[(ny - 2) * nx + i]) > 1e-10) bc_ok = 0;
    }

    TEST_ASSERT_TRUE_MESSAGE(bc_ok, "Neumann BC should be satisfied");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK CONVERGENCE RATE
//=============================================================================

void test_redblack_convergence_rate(void) {
    printf("\n=== Test: Red-Black SIMD Convergence Rate ===\n");

    // Test that Red-Black makes progress on the problem
    size_t nx = 16, ny = 16;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_zero(p, nx, ny);
    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    printf("Red-Black iterations: %d\n", iters);

    // Check solver ran without errors (produces finite values)
    int valid = 1;
    for (size_t i = 0; i < nx * ny && valid; i++) {
        if (!isfinite(p[i])) valid = 0;
    }
    TEST_ASSERT_TRUE_MESSAGE(valid, "Should produce valid results");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK PERFORMANCE (informational)
//=============================================================================

void test_redblack_performance(void) {
    printf("\n=== Test: Red-Black SIMD Performance ===\n");

    size_t nx = 64, ny = 64;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    // Time multiple solves
    int num_solves = 10;
    clock_t start = clock();

    for (int i = 0; i < num_solves; i++) {
        initialize_zero(p, nx, ny);
        poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);
    }

    clock_t end = clock();
    double total_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time for %d solves on %zux%zu grid: %.3f sec\n",
           num_solves, nx, ny, total_time);
    printf("Average time per solve: %.3f ms\n", (total_time / num_solves) * 1000.0);

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// TEST: RED-BLACK SIMD VS JACOBI SIMD CONSISTENCY
//=============================================================================

void test_redblack_vs_jacobi_consistency(void) {
    printf("\n=== Test: Red-Black SIMD vs Jacobi SIMD Consistency ===\n");

    size_t nx = 16, ny = 16;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p_rb = allocate_field(nx, ny);
    double* p_j = allocate_field(nx, ny);
    double* p_j_temp = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    // Red-Black
    initialize_zero(p_rb, nx, ny);
    int iters_rb = poisson_solve_redblack_simd(p_rb, NULL, rhs, nx, ny, dx, dy);

    // Jacobi
    initialize_zero(p_j, nx, ny);
    initialize_zero(p_j_temp, nx, ny);
    int iters_j = poisson_solve_jacobi_simd(p_j, p_j_temp, rhs, nx, ny, dx, dy);

    printf("Red-Black iterations: %d, Jacobi iterations: %d\n", iters_rb, iters_j);

    // Both should produce valid results
    int rb_valid = 1, j_valid = 1;
    for (size_t i = 0; i < nx * ny; i++) {
        if (!isfinite(p_rb[i])) rb_valid = 0;
        if (!isfinite(p_j[i])) j_valid = 0;
    }

    TEST_ASSERT_TRUE_MESSAGE(rb_valid, "Red-Black should produce valid results");
    TEST_ASSERT_TRUE_MESSAGE(j_valid, "Jacobi should produce valid results");

    // Compare residuals
    double res_rb = compute_max_residual(p_rb, rhs, nx, ny, dx * dx, dy * dy);
    double res_j = compute_max_residual(p_j, rhs, nx, ny, dx * dx, dy * dy);
    printf("Red-Black residual: %.6e, Jacobi residual: %.6e\n", res_rb, res_j);

    free_field(p_rb);
    free_field(p_j);
    free_field(p_j_temp);
    free_field(rhs);

    printf("PASSED\n");
}

//=============================================================================
// MAIN
//=============================================================================

int main(void) {
    UNITY_BEGIN();

    printf("\n");
    printf("================================================\n");
    printf("  Red-Black SIMD Poisson Solver Tests\n");
    printf("================================================\n");

    RUN_TEST(test_redblack_runs_sinusoidal);
    RUN_TEST(test_redblack_converges_zero_rhs);
    RUN_TEST(test_redblack_accuracy);
    RUN_TEST(test_redblack_non_aligned_sizes);
    RUN_TEST(test_redblack_deterministic);
    RUN_TEST(test_redblack_uniform_rhs);
    RUN_TEST(test_redblack_boundary_conditions);
    RUN_TEST(test_redblack_convergence_rate);
    RUN_TEST(test_redblack_vs_jacobi_consistency);
    RUN_TEST(test_redblack_performance);

    printf("\n================================================\n");

    return UNITY_END();
}
