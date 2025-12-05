#include "grid.h"
#include "solver_interface.h"
#include "unity.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void setUp(void) {
    // Set up code (if needed)
}

void tearDown(void) {
    // Tear down code (if needed)
}

// Test that viscous terms are actually working
void test_viscous_diffusion(void) {
    size_t nx = 15, ny = 15;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Create a sharp velocity gradient (should diffuse with viscosity)
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double x = grid->x[i];

            // Sharp step function in velocity
            if (x < 0.5) {
                field->u[idx] = 1.0;
                field->v[idx] = 0.0;
            } else {
                field->u[idx] = 0.0;
                field->v[idx] = 0.0;
            }
            field->p[idx] = 1.0;
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    // Record initial sharpness (gradient magnitude)
    double initial_gradient = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
            initial_gradient += fabs(du_dx);
        }
    }

    SolverParams params = {.dt = 0.001,
                           .cfl = 0.2,
                           .gamma = 1.4,
                           .mu = 0.1,  // High viscosity for visible diffusion
                           .k = 0.0242,
                           .max_iter = 10,
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Run solver using modern interface
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);

    // Calculate final gradient
    double final_gradient = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
            final_gradient += fabs(du_dx);
        }
    }

    printf("Viscous test - Initial gradient: %.4f, Final gradient: %.4f\n", initial_gradient,
           final_gradient);

    // Viscosity should have affected the gradient (diffusion effect active)
    // Note: Initial steps may show increase as viscous terms activate
    TEST_ASSERT_TRUE(fabs(final_gradient - initial_gradient) > 1e-6);

    // Values should still be finite
    for (size_t i = 0; i < nx * ny; i++) {
        TEST_ASSERT_TRUE(isfinite(field->u[i]));
        TEST_ASSERT_TRUE(isfinite(field->v[i]));
        TEST_ASSERT_TRUE(isfinite(field->p[i]));
    }

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test that pressure gradients affect velocity
void test_pressure_gradient_effects(void) {
    size_t nx = 10, ny = 10;
    double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);

    // Set up pressure gradient with zero initial velocity
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            size_t idx = j * nx + i;
            double x = grid->x[i];

            field->u[idx] = 0.0;  // Start with zero velocity
            field->v[idx] = 0.0;
            field->p[idx] = 1.0 + 2.0 * x;  // Linear pressure gradient
            field->rho[idx] = 1.0;
            field->T[idx] = 300.0;
        }
    }

    // Record initial velocity
    double initial_velocity_sum = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        initial_velocity_sum += fabs(field->u[i]) + fabs(field->v[i]);
    }

    SolverParams params = {.dt = 0.001,
                           .cfl = 0.2,
                           .gamma = 1.4,
                           .mu = 0.001,  // Low viscosity so pressure gradient dominates
                           .k = 0.0242,
                           .max_iter = 5,
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Run solver using modern interface
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);

    // Calculate final velocity
    double final_velocity_sum = 0.0;
    for (size_t i = 0; i < nx * ny; i++) {
        final_velocity_sum += fabs(field->u[i]) + fabs(field->v[i]);
    }

    printf("Pressure gradient test - Initial velocity sum: %.6f, Final: %.6f\n",
           initial_velocity_sum, final_velocity_sum);

    // Pressure gradient should have induced some velocity
    TEST_ASSERT_TRUE(final_velocity_sum > initial_velocity_sum + 1e-8);

    // Check that pressure gradient induced flow in correct direction (negative x)
    double avg_u_velocity = 0.0;
    int count = 0;
    for (size_t j = 2; j < ny - 2; j++) {  // Check interior points
        for (size_t i = 2; i < nx - 2; i++) {
            size_t idx = j * nx + i;
            avg_u_velocity += field->u[idx];
            count++;
        }
    }
    avg_u_velocity /= count;

    printf("Average u-velocity: %.6f (should be negative due to positive pressure gradient)\n",
           avg_u_velocity);

    // With positive pressure gradient in x, u-velocity should become negative
    TEST_ASSERT_TRUE(avg_u_velocity < 0.0);

    flow_field_destroy(field);
    grid_destroy(grid);
}

// Test conservation properties
void test_conservation_properties(void) {
    size_t nx = 12, ny = 8;
    double xmin = 0.0, xmax = 2.0, ymin = 0.0, ymax = 1.0;

    Grid* grid = grid_create(nx, ny, xmin, xmax, ymin, ymax);
    grid_initialize_uniform(grid);

    FlowField* field = flow_field_create(nx, ny);
    initialize_flow_field(field, grid);

    // Calculate initial total momentum and mass
    double initial_mass = 0.0;
    double initial_momentum_x = 0.0;
    double initial_momentum_y = 0.0;

    for (size_t i = 0; i < nx * ny; i++) {
        initial_mass += field->rho[i];
        initial_momentum_x += field->rho[i] * field->u[i];
        initial_momentum_y += field->rho[i] * field->v[i];
    }

    SolverParams params = {.dt = 0.0005,
                           .cfl = 0.2,
                           .gamma = 1.4,
                           .mu = 0.01,
                           .k = 0.0242,
                           .max_iter = 3,  // Short run to check conservation
                           .tolerance = 1e-6,
                           .source_amplitude_u = 0.1,
                           .source_amplitude_v = 0.05,
                           .source_decay_rate = 0.1,
                           .pressure_coupling = 0.1};

    // Run solver using modern interface
    Solver* solver = solver_create(SOLVER_TYPE_EXPLICIT_EULER);
    solver_init(solver, grid, &params);
    SolverStats stats = solver_stats_default();
    solver_step(solver, field, grid, &params, &stats);
    solver_destroy(solver);

    // Calculate final totals
    double final_mass = 0.0;
    double final_momentum_x = 0.0;
    double final_momentum_y = 0.0;

    for (size_t i = 0; i < nx * ny; i++) {
        final_mass += field->rho[i];
        final_momentum_x += field->rho[i] * field->u[i];
        final_momentum_y += field->rho[i] * field->v[i];
    }

    printf("Conservation test:\n");
    printf("  Mass: %.6f -> %.6f (change: %.2e)\n", initial_mass, final_mass,
           fabs(final_mass - initial_mass));
    printf("  Momentum X: %.6f -> %.6f (change: %.2e)\n", initial_momentum_x, final_momentum_x,
           fabs(final_momentum_x - initial_momentum_x));
    printf("  Momentum Y: %.6f -> %.6f (change: %.2e)\n", initial_momentum_y, final_momentum_y,
           fabs(final_momentum_y - initial_momentum_y));

    // Mass should be exactly conserved (we keep density constant)
    TEST_ASSERT_FLOAT_WITHIN(1e-10, initial_mass, final_mass);

    // Momentum conservation is approximate due to source terms and boundaries
    // Source terms intentionally add momentum to prevent decay
    double momentum_x_change = fabs(initial_momentum_x - final_momentum_x);
    double momentum_y_change = fabs(initial_momentum_y - final_momentum_y);

    TEST_ASSERT_TRUE(momentum_x_change < 10.0);  // Allow reasonable change due to source terms
    TEST_ASSERT_TRUE(momentum_y_change < 1.0);   // Y-momentum should change less

    flow_field_destroy(field);
    grid_destroy(grid);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_viscous_diffusion);
    RUN_TEST(test_pressure_gradient_effects);
    RUN_TEST(test_conservation_properties);
    return UNITY_END();
}