/**
 * Projection Method Solver (Chorin's Method)
 *
 * The projection method solves the incompressible Navier-Stokes equations
 * by splitting the solution into two steps:
 *
 * 1. Predictor Step: Compute intermediate velocity u* ignoring pressure
 *    u* = u^n + dt * (-u·∇u + ν∇²u + f)
 *
 * 2. Pressure Projection: Solve Poisson equation for pressure
 *    ∇²p = (ρ/dt) * ∇·u*
 *
 * 3. Corrector Step: Project velocity to be divergence-free
 *    u^(n+1) = u* - (dt/ρ) * ∇p
 *
 * This ensures ∇·u^(n+1) = 0 (incompressibility constraint)
 */

#include "cfd/core/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/solver_interface.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Poisson solver parameters
#define POISSON_MAX_ITER  1000
#define POISSON_TOLERANCE 1e-6
#define POISSON_OMEGA     1.5  // SOR relaxation parameter (1 < omega < 2)

// Physical limits
#define MAX_VELOCITY 100.0
#define MAX_PRESSURE 1000.0

/**
 * Solve the Poisson equation for pressure using Successive Over-Relaxation (SOR)
 *
 * ∇²p = rhs
 *
 * With Neumann boundary conditions (dp/dn = 0)
 */
static int solve_poisson_sor(double* p, const double* rhs, size_t nx, size_t ny, double dx,
                             double dy, int max_iter, double tolerance) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    if (factor < 1e-10) {
        return -1;
    }

    double inv_factor = 1.0 / factor;

    int converged = 0;
    int iter;

    for (iter = 0; iter < max_iter; iter++) {
        double max_residual = 0.0;

        // Red-black Gauss-Seidel with SOR for better convergence
        for (int color = 0; color < 2; color++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    // Red-black ordering
                    if ((int)((i + j) % 2) != color) {
                        continue;
                    }

                    size_t idx = (j * nx) + i;

                    // Compute Laplacian stencil
                    double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
                    double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;

                    // Residual: ∇²p - rhs
                    double residual = p_xx + p_yy - rhs[idx];

                    // Track maximum residual
                    if (fabs(residual) > max_residual) {
                        max_residual = fabs(residual);
                    }

                    // SOR update
                    double p_new = (rhs[idx] - (p[idx + 1] + p[idx - 1]) / dx2 -
                                    (p[idx + nx] + p[idx - nx]) / dy2) *
                                   (-inv_factor);

                    p[idx] = p[idx] + (POISSON_OMEGA * (p_new - p[idx]));
                }
            }
        }

        // Apply Neumann boundary conditions (using scalar CPU backend)
        bc_apply_scalar_cpu(p, nx, ny, BC_TYPE_NEUMANN);

        // Check convergence
        if (max_residual < tolerance) {
            converged = 1;
            break;
        }
    }

    return converged ? iter : -1;
}

/**
 * Projection Method Solver
 */
cfd_status_t solve_projection_method(flow_field* field, const grid* grid,
                                     const solver_params* params) {
    if (!field || !grid || !params) {
        return CFD_ERROR_INVALID;
    }
    if (field->nx < 3 || field->ny < 3) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;

    // Get grid spacing (assume uniform grid)
    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;  // Viscosity (treated as kinematic for ρ=1)

    // Allocate temporary arrays
    double* u_star = (double*)cfd_calloc(size, sizeof(double));
    double* v_star = (double*)cfd_calloc(size, sizeof(double));
    double* p_new = (double*)cfd_calloc(size, sizeof(double));
    double* rhs = (double*)cfd_calloc(size, sizeof(double));

    if (!u_star || !v_star || !p_new || !rhs) {
        cfd_free(u_star);
        cfd_free(v_star);
        cfd_free(p_new);
        cfd_free(rhs);
        return CFD_ERROR_NOMEM;
    }

    // Copy current values
    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(p_new, field->p, size * sizeof(double));

    // Main iteration loop
    for (int iter = 0; iter < params->max_iter; iter++) {
        // ============================================================
        // STEP 1: Predictor - Compute intermediate velocity u*
        // ============================================================
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;

                double u = field->u[idx];
                double v = field->v[idx];

                // Convective terms: -u·∇u
                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

                double conv_u = (u * du_dx) + (v * du_dy);
                double conv_v = (u * dv_dx) + (v * dv_dy);

                // Viscous terms: ν∇²u
                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);

                double visc_u = nu * (d2u_dx2 + d2u_dy2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2);

                // Source terms (optional, for maintaining flow)
                double source_u = 0.0;
                double source_v = 0.0;
                if (params->source_amplitude_u > 0) {
                    double x = grid->x[i];
                    double y = grid->y[j];
                    source_u = params->source_amplitude_u * sin(M_PI * y) *
                               exp(-params->source_decay_rate * iter * dt);
                    source_v = params->source_amplitude_v * sin(2.0 * M_PI * x) *
                               exp(-params->source_decay_rate * iter * dt);
                }

                // Intermediate velocity (without pressure gradient)
                u_star[idx] = u + (dt * (-conv_u + visc_u + source_u));
                v_star[idx] = v + (dt * (-conv_v + visc_v + source_v));

                // Limit velocities
                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
            }
        }

        // Apply Neumann BCs (zero gradient) to intermediate velocity for proper
        // divergence computation. Final velocity gets periodic BCs later.
        // Using scalar CPU backend explicitly for baseline CPU solver.
        bc_apply_velocity_cpu(u_star, v_star, nx, ny, BC_TYPE_NEUMANN);

        // ============================================================
        // STEP 2: Solve Poisson equation for pressure
        // ∇²p = (ρ/dt) * ∇·u*
        // ============================================================

        // Compute RHS: divergence of intermediate velocity
        double rho = field->rho[0];  // Assume constant density
        if (rho < 1e-10) {
            rho = 1.0;
        }

        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;

                double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);

                double divergence = du_star_dx + dv_star_dy;
                rhs[idx] = (rho / dt) * divergence;
            }
        }

        // Solve Poisson equation
        int poisson_iters =
            solve_poisson_sor(p_new, rhs, nx, ny, dx, dy, POISSON_MAX_ITER, POISSON_TOLERANCE);

        if (poisson_iters < 0) {
            // Poisson solver didn't converge - use simple pressure update as fallback
            for (size_t idx = 0; idx < size; idx++) {
                p_new[idx] = field->p[idx] - (0.1 * dt * rhs[idx]);
            }
        }

        // ============================================================
        // STEP 3: Corrector - Project velocity to be divergence-free
        // u^(n+1) = u* - (dt/ρ) * ∇p
        // ============================================================
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                size_t idx = (j * nx) + i;

                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);

                field->u[idx] = u_star[idx] - ((dt / rho) * dp_dx);
                field->v[idx] = v_star[idx] - ((dt / rho) * dp_dy);

                // Limit velocities
                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
            }
        }

        // Update pressure
        memcpy(field->p, p_new, size * sizeof(double));

        // Apply boundary conditions
        apply_boundary_conditions(field, grid);

        // Check for NaN
        for (size_t k = 0; k < size; k++) {
            if (!isfinite(field->u[k]) || !isfinite(field->v[k]) || !isfinite(field->p[k])) {
                cfd_free(u_star);
                cfd_free(v_star);
                cfd_free(p_new);
                cfd_free(rhs);
                return CFD_ERROR_DIVERGED;
            }
        }
    }

    // Free temporary arrays
    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(p_new);
    cfd_free(rhs);

    return CFD_SUCCESS;
}
