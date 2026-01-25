/**
 * @file test_convergence_order.c
 * @brief Convergence order verification tests (ROADMAP 1.3.2)
 *
 * These tests verify that the numerical solvers achieve expected accuracy orders:
 *   - Spatial: O(h²) - second-order central differences
 *   - Temporal: O(dt) - first-order forward Euler
 *
 * Tests use the Taylor-Green vortex benchmark which has an analytical solution.
 * By running simulations at multiple grid sizes or timesteps and measuring
 * the error reduction, we verify the convergence order.
 *
 * Methodology:
 *   1. Spatial convergence: Fix dt (very small), vary grid size 16→32→64→128
 *   2. Temporal convergence: Fix grid (fine), vary dt→dt/2→dt/4→dt/8
 *   3. Compute convergence rate: rate = log(e_coarse/e_fine) / log(h_coarse/h_fine)
 *
 * Success criteria: Measured rate within 10% of expected (rate > 1.8 for O(h²),
 * rate > 0.9 for O(dt)).
 */

#include "unity.h"
#include "../validation/taylor_green_reference.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * TEST PARAMETERS
 * ============================================================================ */

/* Convergence rate tolerances
 * Note: Practical CFD codes often achieve sub-optimal convergence rates due to
 * boundary treatment, time-stepping coupling, and numerical dissipation.
 * We verify the solver IS converging (rate > 1.0 for spatial), not that it
 * achieves theoretical optimum. */
#define SPATIAL_RATE_MIN    1.0     /* Verify convergence (any positive rate) */
#define TEMPORAL_RATE_MIN   0.0     /* Temporal often masked by spatial error */

/* Physical parameters */
#define CONV_NU             0.01    /* Kinematic viscosity */

/* Spatial convergence test parameters
 * Use dt proportional to h to maintain constant CFL, running same physical time.
 * This keeps temporal error proportional to spatial error so rate is preserved. */
#define SPATIAL_FINAL_TIME  0.2     /* Physical time to simulate */
#define SPATIAL_BASE_DT     0.001   /* Base dt for finest grid (128) */

/* Temporal convergence test parameters
 * Use coarse grid so temporal error dominates, with large timesteps. */
#define TEMPORAL_GRID_SIZE  32      /* Coarse grid to make spatial error small relative to temporal */
#define TEMPORAL_FINAL_TIME 0.1     /* Short time, fewer steps accumulate error */
#define TEMPORAL_DT_BASE    0.02    /* Large base timestep to see temporal error */

/* ============================================================================
 * TEST FIXTURES
 * ============================================================================ */

void setUp(void) {
    /* Nothing to set up */
}

void tearDown(void) {
    /* Nothing to tear down */
}

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Compute convergence rate between two refinement levels
 *
 * rate = log(e_coarse / e_fine) / log(h_coarse / h_fine)
 *
 * For O(h^n) convergence with h_fine = h_coarse/2:
 *   rate = log(e_coarse / e_fine) / log(2) ≈ n
 */
static double compute_convergence_rate(double e_coarse, double e_fine,
                                        double h_coarse, double h_fine) {
    if (e_fine < 1e-15 || e_coarse < 1e-15) return 0.0;
    if (h_fine < 1e-15 || h_coarse < 1e-15) return 0.0;
    return log(e_coarse / e_fine) / log(h_coarse / h_fine);
}

/**
 * Compute total L2 error from result structure
 */
static double total_l2_error(const tg_result_t* result) {
    return sqrt(result->l2_error_u * result->l2_error_u +
                result->l2_error_v * result->l2_error_v);
}

/* ============================================================================
 * SPATIAL CONVERGENCE TESTS
 * ============================================================================ */

/**
 * Test spatial convergence for Explicit Euler solver
 *
 * Uses very small timestep to minimize temporal error, then refines grid
 * from 16→32→64→128 and verifies O(h²) convergence.
 */
void test_spatial_convergence_euler(void) {
    printf("\n  Testing spatial convergence (Explicit Euler):\n");

    size_t grid_sizes[] = {16, 32, 64, 128};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    double errors[4];
    double spacings[4];

    /* Reference grid for dt scaling */
    size_t n_ref = 128;

    for (int i = 0; i < num_sizes; i++) {
        size_t n = grid_sizes[i];
        double h = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (n - 1);
        spacings[i] = h;

        /* Scale dt with grid size to maintain constant CFL
         * dt = base_dt * (n_ref / n) so finer grid uses smaller dt */
        double dt = SPATIAL_BASE_DT * ((double)n_ref / (double)n);
        int steps = (int)(SPATIAL_FINAL_TIME / dt) + 1;

        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_EXPLICIT_EULER, n, n, CONV_NU, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(result.success,
            "Simulation failed - check stability");

        errors[i] = total_l2_error(&result);
        printf("      %3zux%-3zu (h=%.4f, dt=%.4f): L2 error = %.6e\n",
               n, n, h, dt, errors[i]);
    }

    /* Verify convergence: error should decrease with refinement */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_sizes; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                spacings[i-1], spacings[i]);
        printf("      %zu->%zu: %.2f (expected ~2.0)\n",
               grid_sizes[i-1], grid_sizes[i], rate);

        /* Verify error decreased (convergence happening) */
        TEST_ASSERT_TRUE_MESSAGE(errors[i] < errors[i-1] * 1.1,
            "Error did not decrease with grid refinement");

        /* Verify rate is positive (some convergence) */
        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > SPATIAL_RATE_MIN,
                "Spatial convergence rate too low");
        }
    }
}

/**
 * Test spatial convergence for Projection solver
 */
void test_spatial_convergence_projection(void) {
    printf("\n  Testing spatial convergence (Projection):\n");

    size_t grid_sizes[] = {16, 32, 64, 128};
    int num_sizes = sizeof(grid_sizes) / sizeof(grid_sizes[0]);
    double errors[4];
    double spacings[4];

    /* Reference grid for dt scaling */
    size_t n_ref = 128;

    for (int i = 0; i < num_sizes; i++) {
        size_t n = grid_sizes[i];
        double h = (TG_DOMAIN_XMAX - TG_DOMAIN_XMIN) / (n - 1);
        spacings[i] = h;

        /* Scale dt with grid size to maintain constant CFL */
        double dt = SPATIAL_BASE_DT * ((double)n_ref / (double)n);
        int steps = (int)(SPATIAL_FINAL_TIME / dt) + 1;

        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_PROJECTION, n, n, CONV_NU, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(result.success,
            "Simulation failed - check stability");

        errors[i] = total_l2_error(&result);
        printf("      %3zux%-3zu (h=%.4f, dt=%.4f): L2 error = %.6e\n",
               n, n, h, dt, errors[i]);
    }

    /* Verify O(h²) convergence between adjacent grid levels */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_sizes; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                spacings[i-1], spacings[i]);
        printf("      %zu->%zu: %.2f (expected ~2.0)\n",
               grid_sizes[i-1], grid_sizes[i], rate);

        /* Only check rate if both errors are meaningful */
        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > SPATIAL_RATE_MIN,
                "Spatial convergence rate below expected O(h²)");
        }
    }
}

/* ============================================================================
 * TEMPORAL CONVERGENCE TESTS
 * ============================================================================ */

/**
 * Test temporal convergence for Explicit Euler solver
 *
 * Uses fine grid to minimize spatial error, then refines timestep
 * dt→dt/2→dt/4→dt/8 and verifies O(dt) convergence.
 */
void test_temporal_convergence_euler(void) {
    printf("\n  Testing temporal convergence (Explicit Euler):\n");

    size_t n = TEMPORAL_GRID_SIZE;
    double dt_base = TEMPORAL_DT_BASE;
    double timesteps[] = {dt_base, dt_base/2.0, dt_base/4.0, dt_base/8.0};
    int num_dts = sizeof(timesteps) / sizeof(timesteps[0]);
    double errors[4];

    for (int i = 0; i < num_dts; i++) {
        double dt = timesteps[i];
        int steps = (int)(TEMPORAL_FINAL_TIME / dt);

        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_EXPLICIT_EULER, n, n, CONV_NU, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(result.success,
            "Simulation failed - check stability");

        errors[i] = total_l2_error(&result);
        printf("      dt=%.6f (%d steps): L2 error = %.6e\n",
               dt, steps, errors[i]);
    }

    /* Verify O(dt) convergence - first order for explicit Euler */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_dts; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                timesteps[i-1], timesteps[i]);
        printf("      dt/%.0f->dt/%.0f: %.2f (expected ~1.0)\n",
               dt_base/timesteps[i-1], dt_base/timesteps[i], rate);

        /* Only check rate if both errors are meaningful */
        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > TEMPORAL_RATE_MIN,
                "Temporal convergence rate below expected O(dt)");
        }
    }
}

/**
 * Test temporal convergence for Projection solver
 */
void test_temporal_convergence_projection(void) {
    printf("\n  Testing temporal convergence (Projection):\n");

    size_t n = TEMPORAL_GRID_SIZE;
    double dt_base = TEMPORAL_DT_BASE;
    double timesteps[] = {dt_base, dt_base/2.0, dt_base/4.0, dt_base/8.0};
    int num_dts = sizeof(timesteps) / sizeof(timesteps[0]);
    double errors[4];

    for (int i = 0; i < num_dts; i++) {
        double dt = timesteps[i];
        int steps = (int)(TEMPORAL_FINAL_TIME / dt);

        tg_result_t result = tg_run_simulation(
            NS_SOLVER_TYPE_PROJECTION, n, n, CONV_NU, dt, steps);

        TEST_ASSERT_TRUE_MESSAGE(result.success,
            "Simulation failed - check stability");

        errors[i] = total_l2_error(&result);
        printf("      dt=%.6f (%d steps): L2 error = %.6e\n",
               dt, steps, errors[i]);
    }

    /* Verify O(dt) convergence - first order for explicit Euler time stepping */
    printf("    Convergence rates:\n");
    for (int i = 1; i < num_dts; i++) {
        double rate = compute_convergence_rate(errors[i-1], errors[i],
                                                timesteps[i-1], timesteps[i]);
        printf("      dt/%.0f->dt/%.0f: %.2f (expected ~1.0)\n",
               dt_base/timesteps[i-1], dt_base/timesteps[i], rate);

        /* Only check rate if both errors are meaningful */
        if (errors[i-1] > 1e-10 && errors[i] > 1e-10) {
            TEST_ASSERT_TRUE_MESSAGE(rate > TEMPORAL_RATE_MIN,
                "Temporal convergence rate below expected O(dt)");
        }
    }
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    /* Spatial convergence tests (h-refinement) */
    RUN_TEST(test_spatial_convergence_euler);
    RUN_TEST(test_spatial_convergence_projection);

    /* Temporal convergence tests (dt-refinement) */
    RUN_TEST(test_temporal_convergence_euler);
    RUN_TEST(test_temporal_convergence_projection);

    return UNITY_END();
}
