#include "unity.h"
#include "solver.h"
#include "grid.h"
#include "utils.h"
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Test that flow doesn't decay too rapidly (source terms working)
void test_flow_energy_maintenance(void) {
    size_t nx = 15, ny = 10;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Calculate initial kinetic energy
    double initial_kinetic_energy = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_kinetic_energy += 0.5 * field->rho[i] *
                                 (field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }

    printf("Initial kinetic energy: %.6f\n", initial_kinetic_energy);

    SolverParams params = {
        .dt = 0.001,
        .cfl = 0.2,
        .gamma = 1.4,
        .mu = 0.01,  // Some viscosity that would cause decay without source terms
        .k = 0.0242,
        .max_iter = 20,  // Longer simulation to see decay/maintenance
        .tolerance = 1e-6,
        .source_amplitude_u = 0.1,
        .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1,
        .pressure_coupling = 0.1
    };

    // Store kinetic energy at different time steps
    double kinetic_energies[5];
    int measurement_steps[] = {0, 5, 10, 15, 20};

    for (int step = 0; step < 5; step++) {
        // Run to next measurement point
        if (step > 0) {
            SolverParams step_params = params;
            step_params.max_iter = measurement_steps[step] - measurement_steps[step-1];
            solve_navier_stokes(field, grid, &step_params);
        }

        // Measure kinetic energy
        double kinetic_energy = 0.0;
        for (size_t i = 0; i < nx * ny; i++) {
            kinetic_energy += 0.5 * field->rho[i] *
                            (field->u[i] * field->u[i] + field->v[i] * field->v[i]);
        }
        kinetic_energies[step] = kinetic_energy;

        printf("Step %d: Kinetic energy = %.6f\n", measurement_steps[step], kinetic_energy);

        // Check values are still finite
        int finite_count = 0;
        for (size_t i = 0; i < nx * ny; i++) {
            if (isfinite(field->u[i]) && isfinite(field->v[i]) && isfinite(field->p[i])) {
                finite_count++;
            }
        }
        TEST_ASSERT_GREATER_THAN((int)(0.95 * nx * ny), finite_count);
    }

    // Test that energy doesn't decay too rapidly
    // With source terms, energy should be maintained or decay slowly
    double energy_ratio_mid = kinetic_energies[2] / kinetic_energies[0];  // Step 10 / Step 0
    double energy_ratio_end = kinetic_energies[4] / kinetic_energies[0];  // Step 20 / Step 0

    printf("Energy ratios: mid=%.3f, end=%.3f\n", energy_ratio_mid, energy_ratio_end);

    // Energy should be maintained or grow slightly due to source terms
    TEST_ASSERT_TRUE(energy_ratio_end > 0.95);  // Should not decay below 95%
    TEST_ASSERT_TRUE(energy_ratio_end < 2.0);   // Should not grow more than 100%

    // Mid-point energy should show the source terms working
    TEST_ASSERT_TRUE(energy_ratio_mid > 0.95);
    TEST_ASSERT_TRUE(energy_ratio_mid < 2.0);

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that source terms are actually working
void test_source_term_effectiveness(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);

    // Start with nearly zero velocity (would decay without source terms)
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            field->u[idx] = 0.001;  // Tiny initial velocity
            field->v[idx] = 0.001;
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    // Calculate initial velocity magnitude squared (avoid expensive sqrt)
    double initial_velocity_mag_sq = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_velocity_mag_sq += field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }
    initial_velocity_mag_sq /= (nx * ny);
    double initial_velocity_mag = sqrt(initial_velocity_mag_sq);

    SolverParams params = {
        .dt = 0.001,
        .cfl = 0.2,
        .gamma = 1.4,
        .mu = 0.01,
        .k = 0.0242,
        .max_iter = 15,  // Enough time for source terms to act
        .tolerance = 1e-6,
        .source_amplitude_u = 0.1,
        .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1,
        .pressure_coupling = 0.1
    };

    // Run solver
    solve_navier_stokes(field, grid, &params);

    // Calculate final velocity magnitude squared (avoid expensive sqrt)
    double final_velocity_mag_sq = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        final_velocity_mag_sq += field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }
    final_velocity_mag_sq /= (nx * ny);
    double final_velocity_mag = sqrt(final_velocity_mag_sq);

    printf("Source term test - Initial avg velocity: %.6f, Final: %.6f\n",
           initial_velocity_mag, final_velocity_mag);

    // Source terms should have increased velocity from near-zero
    TEST_ASSERT_TRUE(final_velocity_mag > initial_velocity_mag);

    // But not to unreasonable levels
    TEST_ASSERT_TRUE(final_velocity_mag < 100.0);

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test both solvers for decay prevention
void test_decay_prevention_both_solvers(void) {
    size_t nx = 12, ny = 8;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    // Test basic solver
    Grid* grid1 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid1);
    FlowField* field1 = flow_field_create(nx, ny);
    initialize_flow_field(field1, grid1);

    // Test optimized solver
    Grid* grid2 = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid2);
    FlowField* field2 = flow_field_create(nx, ny);
    initialize_flow_field(field2, grid2);

    // Make initial conditions identical
    for (size_t i = 0; i < nx * ny; i++) {
        field2->u[i] = field1->u[i];
        field2->v[i] = field1->v[i];
        field2->p[i] = field1->p[i];
        field2->rho[i] = field1->rho[i];
        field2->T[i] = field1->T[i];
    }

    // Calculate initial energy for both
    double initial_energy1 = 0.0, initial_energy2 = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_energy1 += field1->u[i] * field1->u[i] + field1->v[i] * field1->v[i];
        initial_energy2 += field2->u[i] * field2->u[i] + field2->v[i] * field2->v[i];
    }

    SolverParams params = {
        .dt = 0.001,
        .cfl = 0.2,
        .gamma = 1.4,
        .mu = 0.01,
        .k = 0.0242,
        .max_iter = 10,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.1,
        .source_amplitude_v = 0.05,
        .source_decay_rate = 0.1,
        .pressure_coupling = 0.1
    };

    // Run both solvers
    solve_navier_stokes(field1, grid1, &params);
    solve_navier_stokes_optimized(field2, grid2, &params);

    // Calculate final energies
    double final_energy1 = 0.0, final_energy2 = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        final_energy1 += field1->u[i] * field1->u[i] + field1->v[i] * field1->v[i];
        final_energy2 += field2->u[i] * field2->u[i] + field2->v[i] * field2->v[i];
    }

    double ratio1 = final_energy1 / initial_energy1;
    double ratio2 = final_energy2 / initial_energy2;

    printf("Decay prevention test:\n");
    printf("  Basic solver: %.6f -> %.6f (ratio: %.3f)\n", initial_energy1, final_energy1, ratio1);
    printf("  Optimized solver: %.6f -> %.6f (ratio: %.3f)\n", initial_energy2, final_energy2, ratio2);

    // Both solvers should prevent rapid decay (source terms working)
    TEST_ASSERT_TRUE(ratio1 > 0.95);  // Energy should be maintained
    TEST_ASSERT_TRUE(ratio2 > 0.95);

    // Both should remain stable
    TEST_ASSERT_TRUE(ratio1 < 50.0);
    TEST_ASSERT_TRUE(ratio2 < 50.0);

    // Results should be reasonably similar between solvers
    double ratio_difference = fabs(ratio1 - ratio2);
    TEST_ASSERT_TRUE(ratio_difference < 5.0);  // Allow some difference due to numerical implementation

    flow_field_destroy(field1);
    flow_field_destroy(field2);
    grid_destroy(grid1);
    grid_destroy(grid2);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_flow_energy_maintenance);
    RUN_TEST(test_source_term_effectiveness);
    RUN_TEST(test_decay_prevention_both_solvers);
    return UNITY_END();
}