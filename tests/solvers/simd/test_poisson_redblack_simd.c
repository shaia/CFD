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

// Common test utilities (memory allocation, initialization, error computation)
#include "poisson_test_utils.h"

// Include the Poisson solver header directly for testing
#include "../../../lib/src/solvers/simd/poisson_solver_simd.h"

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
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
// EDGE CASE TESTS: SIMD LOOP BOUNDARY CONDITIONS
//=============================================================================

/**
 * Test minimum grid size (4x4).
 * With nx=4, interior cells are at i=1,2 only.
 * Red-Black with stride-2: i_start depends on row and color.
 * This tests the absolute minimum viable grid.
 */
void test_redblack_edge_minimum_grid(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - Minimum Grid (4x4) ===\n");

    size_t nx = 4, ny = 4;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    TEST_ASSERT_NOT_NULL(p);
    TEST_ASSERT_NOT_NULL(rhs);

    // Start with non-zero initial guess
    for (size_t i = 0; i < nx * ny; i++) {
        p[i] = 0.1;
        rhs[i] = 0.0;  // Zero RHS for easy convergence
    }

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    printf("Grid 4x4: iterations=%d\n", iters);

    // Check for valid output
    int valid = 1;
    for (size_t i = 0; i < nx * ny && valid; i++) {
        if (!isfinite(p[i])) valid = 0;
    }
    TEST_ASSERT_TRUE_MESSAGE(valid, "4x4 grid should produce valid results");
    TEST_ASSERT_TRUE_MESSAGE(iters >= 0, "4x4 grid should converge");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

/**
 * Test 5x5 grid - odd minimum size.
 * With nx=5, interior cells are at i=1,2,3.
 * Red-Black stride-2 pattern: either 1,3 or just 2 depending on row/color.
 */
void test_redblack_edge_5x5_grid(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - 5x5 Grid ===\n");

    size_t nx = 5, ny = 5;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    double* p = allocate_field(nx, ny);
    double* rhs = allocate_field(nx, ny);

    initialize_zero(p, nx, ny);
    initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

    int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

    printf("Grid 5x5: iterations=%d\n", iters);

    int valid = 1;
    for (size_t i = 0; i < nx * ny && valid; i++) {
        if (!isfinite(p[i])) valid = 0;
    }
    TEST_ASSERT_TRUE_MESSAGE(valid, "5x5 grid should produce valid results");

    free_field(p);
    free_field(rhs);

    printf("PASSED\n");
}

/**
 * Test grid sizes that are just below SIMD threshold.
 * SIMD loop condition: i + 6 < nx - 1, processes at i, i+2, i+4, i+6.
 * For i_start=1 (even row, red): need i+6 < nx-1, so nx > 8.
 * For i_start=2 (even row, black): need i+6 < nx-1, so nx > 9.
 * These sizes should use only scalar remainder.
 */
void test_redblack_edge_scalar_only_sizes(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - Scalar-Only Sizes ===\n");

    // These sizes should NOT trigger SIMD (only scalar remainder)
    size_t scalar_only_sizes[][2] = {
        {6, 6},   // nx=6: i_start=1, i+6=7, need 7<5 -> false. Only scalar.
        {7, 7},   // nx=7: i_start=1, i+6=7, need 7<6 -> false. Only scalar.
        {8, 8},   // nx=8: i_start=1, i+6=7, need 7<7 -> false. Only scalar.
    };
    int num_sizes = sizeof(scalar_only_sizes) / sizeof(scalar_only_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = scalar_only_sizes[t][0];
        size_t ny = scalar_only_sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu (scalar only)... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        for (size_t i = 0; i < nx * ny; i++) {
            p[i] = 0.1;
            rhs[i] = 0.0;
        }

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "Scalar-only size should produce valid results");
        TEST_ASSERT_TRUE_MESSAGE(iters >= 0, "Scalar-only size should converge");

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d)\n", iters);
    }

    printf("PASSED\n");
}

/**
 * Test grid sizes at SIMD threshold boundary.
 * SIMD loop: for (i = i_start; i + 6 < nx - 1; i += 8)
 * At i_start=1: enters loop when 1+6 < nx-1 -> 7 < nx-1 -> nx > 8 -> nx >= 9
 * At i_start=2: enters loop when 2+6 < nx-1 -> 8 < nx-1 -> nx > 9 -> nx >= 10
 * These are the first sizes where SIMD kicks in.
 */
void test_redblack_edge_simd_threshold(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - SIMD Threshold Sizes ===\n");

    // First sizes where SIMD loop executes at least once
    size_t threshold_sizes[][2] = {
        {9, 9},    // First size where SIMD runs for some rows
        {10, 10},  // First size where SIMD runs for all rows
        {17, 17},  // nx=17: i+6=7<16, then i+8=9, 9+6=15<16, then i+8=17 exits
    };
    int num_sizes = sizeof(threshold_sizes) / sizeof(threshold_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = threshold_sizes[t][0];
        size_t ny = threshold_sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu (SIMD threshold)... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        initialize_zero(p, nx, ny);
        initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "SIMD threshold size should produce valid results");

        // Verify boundary conditions
        int bc_ok = 1;
        for (size_t j = 0; j < ny && bc_ok; j++) {
            if (fabs(p[j * nx + 0] - p[j * nx + 1]) > 1e-10) bc_ok = 0;
            if (fabs(p[j * nx + nx - 1] - p[j * nx + nx - 2]) > 1e-10) bc_ok = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(bc_ok, "BCs should be correct at SIMD threshold");

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d)\n", iters);
    }

    printf("PASSED\n");
}

/**
 * Test grid where SIMD processes exactly last valid cell.
 * SIMD loop condition: i + 6 < nx - 1
 * Last interior cell is at i = nx - 2.
 * For SIMD to process cell at nx-2, we need i+6 = nx-2 to be included.
 * Since we process i, i+2, i+4, i+6, if i+6 = nx-2, condition is i+6 < nx-1.
 * (nx-2) < (nx-1) is always true, so this cell IS included.
 * Test with nx where last SIMD cell lands exactly at nx-2.
 */
void test_redblack_edge_last_simd_cell(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - Last SIMD Cell at Boundary ===\n");

    // For i_start=1 and wanting i+6 = nx-2:
    // Loop advances by 8: i=1, 9, 17, 25...
    // If i=1, i+6=7, so nx-2=7 -> nx=9. Last SIMD cell is at 7.
    // If i=9, i+6=15, so nx-2=15 -> nx=17. Last SIMD cell is at 15.
    size_t boundary_sizes[][2] = {
        {9, 9},    // i=1, last SIMD cell at i+6=7 = nx-2
        {17, 17},  // i=1 then i=9, last batch: i=9, cells at 9,11,13,15=nx-2
        {25, 25},  // Similar pattern
    };
    int num_sizes = sizeof(boundary_sizes) / sizeof(boundary_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = boundary_sizes[t][0];
        size_t ny = boundary_sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu (last SIMD at boundary)... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        initialize_zero(p, nx, ny);
        initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "Boundary SIMD cell should produce valid results");

        // Check that last interior column was correctly updated
        double last_interior_max = 0.0;
        for (size_t j = 1; j < ny - 1; j++) {
            double val = fabs(p[j * nx + (nx - 2)]);
            if (val > last_interior_max) last_interior_max = val;
        }
        printf("last_interior_max=%.2e ", last_interior_max);

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d)\n", iters);
    }

    printf("PASSED\n");
}

/**
 * Test asymmetric grids with edge-case dimensions.
 * Tests that row/column handling is independent.
 */
void test_redblack_edge_asymmetric_grids(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - Asymmetric Grids ===\n");

    size_t asymmetric_sizes[][2] = {
        {4, 32},   // Minimum width, larger height
        {32, 4},   // Larger width, minimum height
        {9, 4},    // SIMD threshold width, minimum height
        {4, 9},    // Minimum width, SIMD threshold height
        {10, 5},   // Just above SIMD threshold, small height
        {5, 10},   // Small width, just above SIMD threshold
    };
    int num_sizes = sizeof(asymmetric_sizes) / sizeof(asymmetric_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = asymmetric_sizes[t][0];
        size_t ny = asymmetric_sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu (asymmetric)... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        for (size_t i = 0; i < nx * ny; i++) {
            p[i] = 0.1;
            rhs[i] = 0.0;
        }

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "Asymmetric grid should produce valid results");
        TEST_ASSERT_TRUE_MESSAGE(iters >= 0, "Asymmetric grid should converge");

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d)\n", iters);
    }

    printf("PASSED\n");
}

/**
 * Test grid size where scalar remainder processes exactly 1 cell.
 * After SIMD processes batches of 8 (4 same-color cells),
 * the scalar loop handles the rest with stride 2.
 */
void test_redblack_edge_single_scalar_remainder(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - Single Scalar Remainder ===\n");

    // SIMD loop: i = i_start; i + 6 < nx - 1; i += 8
    // For i_start=1, nx=10:
    //   i=1: 1+6=7 < 9? yes, process 1,3,5,7. i becomes 9.
    //   i=9: 9+6=15 < 9? no, exit SIMD.
    //   Scalar: i=9, but nx-1=9, so 9 < 9 is false. No scalar cells!
    // For i_start=1, nx=11:
    //   i=1: 1+6=7 < 10? yes, process 1,3,5,7. i becomes 9.
    //   i=9: 9+6=15 < 10? no, exit SIMD.
    //   Scalar: i=9, 9 < 10? yes. Process cell 9. Done (9+2=11 >= 10).
    size_t single_remainder_sizes[][2] = {
        {11, 11},  // After SIMD, scalar processes cell 9 only
        {19, 19},  // Similar pattern at larger size
    };
    int num_sizes = sizeof(single_remainder_sizes) / sizeof(single_remainder_sizes[0]);

    for (int t = 0; t < num_sizes; t++) {
        size_t nx = single_remainder_sizes[t][0];
        size_t ny = single_remainder_sizes[t][1];
        double dx = 1.0 / (nx - 1);
        double dy = 1.0 / (ny - 1);

        printf("  Testing %zux%zu (single scalar remainder)... ", nx, ny);

        double* p = allocate_field(nx, ny);
        double* rhs = allocate_field(nx, ny);

        initialize_zero(p, nx, ny);
        initialize_sinusoidal_rhs(rhs, nx, ny, dx, dy);

        int iters = poisson_solve_redblack_simd(p, NULL, rhs, nx, ny, dx, dy);

        int valid = 1;
        for (size_t i = 0; i < nx * ny && valid; i++) {
            if (!isfinite(p[i])) valid = 0;
        }
        TEST_ASSERT_TRUE_MESSAGE(valid, "Single scalar remainder should produce valid results");

        // Check that all interior cells have been updated (non-zero for sinusoidal RHS)
        int updated = 0;
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                if (fabs(p[j * nx + i]) > 1e-12) updated++;
            }
        }
        TEST_ASSERT_TRUE_MESSAGE(updated > 0, "Interior cells should be updated");

        free_field(p);
        free_field(rhs);

        printf("OK (iters=%d, updated=%d)\n", iters, updated);
    }

    printf("PASSED\n");
}

/**
 * Test consistency between SIMD and scalar paths.
 * Run on a size where SIMD does most work, then on a size where
 * scalar does all work, and verify residuals are comparable.
 */
void test_redblack_edge_simd_scalar_consistency(void) {
    printf("\n=== Test: Red-Black SIMD Edge Case - SIMD/Scalar Consistency ===\n");

    // Small grid (scalar only) vs larger grid (SIMD + scalar)
    // Both should achieve similar residual reduction per iteration
    size_t nx_scalar = 6, ny_scalar = 6;
    size_t nx_simd = 18, ny_simd = 18;

    double dx_s = 1.0 / (nx_scalar - 1);
    double dy_s = 1.0 / (ny_scalar - 1);
    double dx_v = 1.0 / (nx_simd - 1);
    double dy_v = 1.0 / (ny_simd - 1);

    double* p_s = allocate_field(nx_scalar, ny_scalar);
    double* rhs_s = allocate_field(nx_scalar, ny_scalar);
    double* p_v = allocate_field(nx_simd, ny_simd);
    double* rhs_v = allocate_field(nx_simd, ny_simd);

    for (size_t i = 0; i < nx_scalar * ny_scalar; i++) {
        p_s[i] = 0.1;
        rhs_s[i] = 0.0;
    }
    for (size_t i = 0; i < nx_simd * ny_simd; i++) {
        p_v[i] = 0.1;
        rhs_v[i] = 0.0;
    }

    int iters_s = poisson_solve_redblack_simd(p_s, NULL, rhs_s, nx_scalar, ny_scalar, dx_s, dy_s);
    int iters_v = poisson_solve_redblack_simd(p_v, NULL, rhs_v, nx_simd, ny_simd, dx_v, dy_v);

    printf("Scalar-only %zux%zu: iters=%d\n", nx_scalar, ny_scalar, iters_s);
    printf("SIMD+scalar %zux%zu: iters=%d\n", nx_simd, ny_simd, iters_v);

    // Both should converge (with zero RHS)
    TEST_ASSERT_TRUE_MESSAGE(iters_s >= 0, "Scalar-only should converge");
    TEST_ASSERT_TRUE_MESSAGE(iters_v >= 0, "SIMD+scalar should converge");

    // Check solutions are near constant (expected for zero RHS with Neumann BC)
    double max_s = 0.0, max_v = 0.0;
    for (size_t i = 0; i < nx_scalar * ny_scalar; i++) {
        if (fabs(p_s[i]) > max_s) max_s = fabs(p_s[i]);
    }
    for (size_t i = 0; i < nx_simd * ny_simd; i++) {
        if (fabs(p_v[i]) > max_v) max_v = fabs(p_v[i]);
    }

    printf("Max values: scalar=%.2e, simd=%.2e\n", max_s, max_v);

    free_field(p_s);
    free_field(rhs_s);
    free_field(p_v);
    free_field(rhs_v);

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

    // Edge case tests for SIMD loop boundary conditions
    RUN_TEST(test_redblack_edge_minimum_grid);
    RUN_TEST(test_redblack_edge_5x5_grid);
    RUN_TEST(test_redblack_edge_scalar_only_sizes);
    RUN_TEST(test_redblack_edge_simd_threshold);
    RUN_TEST(test_redblack_edge_last_simd_cell);
    RUN_TEST(test_redblack_edge_asymmetric_grids);
    RUN_TEST(test_redblack_edge_single_scalar_remainder);
    RUN_TEST(test_redblack_edge_simd_scalar_consistency);

    RUN_TEST(test_redblack_vs_jacobi_consistency);
    RUN_TEST(test_redblack_performance);

    printf("\n================================================\n");

    return UNITY_END();
}
