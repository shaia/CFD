#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Poisson solver parameters
#define POISSON_MAX_ITER  1000
#define POISSON_TOLERANCE 1e-6
#define POISSON_OMEGA     1.5

// Physical limits
#define MAX_VELOCITY 100.0

static int solve_poisson_sor_omp(double* p, const double* rhs, size_t nx, size_t ny, double dx,
                             double dy, int max_iter, double tolerance) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double factor = 2.0 * (1.0 / dx2 + 1.0 / dy2);

    if (factor < 1e-10) return -1;

    double inv_factor = 1.0 / factor;
    int iter;
    int converged = 0;

    for (iter = 0; iter < max_iter; iter++) {
        double max_residual = 0.0;
        
        // Red-Black Gauss-Seidel
        for (int color = 0; color < 2; color++) {
            double local_max_res = 0.0;
            int i, j;
            
            #pragma omp parallel private(i, j) shared(max_residual)
            {
                double thread_max_res = 0.0;
                
                #pragma omp for schedule(static)
                for (j = 1; j < (int)ny - 1; j++) {
                    for (i = 1; i < (int)nx - 1; i++) {
                        if ((i + j) % 2 != color) continue;

                        size_t idx = j * nx + i;

                        double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
                        double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;
                        double residual = p_xx + p_yy - rhs[idx];

                        if (fabs(residual) > thread_max_res) {
                            thread_max_res = fabs(residual);
                        }

                        double p_new = (rhs[idx] - (p[idx + 1] + p[idx - 1]) / dx2 -
                                        (p[idx + nx] + p[idx - nx]) / dy2) * (-inv_factor);
                        
                        p[idx] = p[idx] + POISSON_OMEGA * (p_new - p[idx]);
                    }
                }
                
                #pragma omp critical
                {
                    if (thread_max_res > local_max_res) {
                        local_max_res = thread_max_res;
                    }
                }
            } // end parallel
            
            if (local_max_res > max_residual) max_residual = local_max_res;
        }

        // Apply BCs
        int i, j;
        #pragma omp parallel for private(j) schedule(static)
        for (j = 0; j < (int)ny; j++) {
            p[j * nx + 0] = p[j * nx + 1];
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        }
        #pragma omp parallel for private(i) schedule(static)
        for (i = 0; i < (int)nx; i++) {
            p[i] = p[nx + i];
            p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
        }

        if (max_residual < tolerance) {
            converged = 1;
            break;
        }
    }

    return converged ? iter : -1;
}

void solve_projection_method_omp(FlowField* field, const Grid* grid, const SolverParams* params) {
    if (!field || !grid || !params) return;
    if (field->nx < 3 || field->ny < 3) return;

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t size = nx * ny;

    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double dt = params->dt;
    double nu = params->mu;

    double* u_star = (double*)cfd_calloc(size, sizeof(double));
    double* v_star = (double*)cfd_calloc(size, sizeof(double));
    double* p_new = (double*)cfd_calloc(size, sizeof(double));
    double* rhs = (double*)cfd_calloc(size, sizeof(double));

    if (!u_star || !v_star || !p_new || !rhs) {
        cfd_free(u_star); cfd_free(v_star); cfd_free(p_new); cfd_free(rhs);
        return;
    }

    memcpy(u_star, field->u, size * sizeof(double));
    memcpy(v_star, field->v, size * sizeof(double));
    memcpy(p_new, field->p, size * sizeof(double));

    for (int iter = 0; iter < params->max_iter; iter++) {
        // STEP 1: Predictor
        int i, j;
        // Use static scheduling for uniform grid load balancing
        #pragma omp parallel for private(i) schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = j * nx + i;

                double u = field->u[idx];
                double v = field->v[idx];

                double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * dx);
                double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * dy);
                double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * dx);
                double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * dy);

                double conv_u = u * du_dx + v * du_dy;
                double conv_v = u * dv_dx + v * dv_dy;

                double d2u_dx2 = (field->u[idx + 1] - 2.0 * u + field->u[idx - 1]) / (dx * dx);
                double d2u_dy2 = (field->u[idx + nx] - 2.0 * u + field->u[idx - nx]) / (dy * dy);
                double d2v_dx2 = (field->v[idx + 1] - 2.0 * v + field->v[idx - 1]) / (dx * dx);
                double d2v_dy2 = (field->v[idx + nx] - 2.0 * v + field->v[idx - nx]) / (dy * dy);

                double visc_u = nu * (d2u_dx2 + d2u_dy2);
                double visc_v = nu * (d2v_dx2 + d2v_dy2);

                double source_u = 0.0;
                double source_v = 0.0;
                if (params->source_amplitude_u > 0) {
                    source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]) *
                               exp(-params->source_decay_rate * iter * dt);
                    source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]) *
                               exp(-params->source_decay_rate * iter * dt);
                }

                u_star[idx] = u + dt * (-conv_u + visc_u + source_u);
                v_star[idx] = v + dt * (-conv_v + visc_v + source_v);

                u_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_star[idx]));
                v_star[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_star[idx]));
            }
        }

        // BCs for intermediate velocity
        #pragma omp parallel for private(j) schedule(static)
        for (j = 0; j < (int)ny; j++) {
            u_star[j * nx + 0] = u_star[j * nx + 1];
            u_star[j * nx + nx - 1] = u_star[j * nx + nx - 2];
            v_star[j * nx + 0] = v_star[j * nx + 1];
            v_star[j * nx + nx - 1] = v_star[j * nx + nx - 2];
        }
        #pragma omp parallel for private(i) schedule(static)
        for (i = 0; i < (int)nx; i++) {
            u_star[i] = u_star[nx + i];
            u_star[(ny - 1) * nx + i] = u_star[(ny - 2) * nx + i];
            v_star[i] = v_star[nx + i];
            v_star[(ny - 1) * nx + i] = v_star[(ny - 2) * nx + i];
        }

        // STEP 2: Pressure
        double rho = field->rho[0] < 1e-10 ? 1.0 : field->rho[0];
        
        #pragma omp parallel for private(i) schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = j * nx + i;
                double du_star_dx = (u_star[idx + 1] - u_star[idx - 1]) / (2.0 * dx);
                double dv_star_dy = (v_star[idx + nx] - v_star[idx - nx]) / (2.0 * dy);
                rhs[idx] = (rho / dt) * (du_star_dx + dv_star_dy);
            }
        }

        int poisson_iters = solve_poisson_sor_omp(p_new, rhs, nx, ny, dx, dy, POISSON_MAX_ITER, POISSON_TOLERANCE);
        
        if (poisson_iters < 0) {
            // Fallback
             long long idx;
             #pragma omp parallel for private(idx) schedule(static)
             for (idx = 0; idx < (long long)size; idx++) {
                p_new[idx] = field->p[idx] - 0.1 * dt * rhs[idx];
            }
        }

        // STEP 3: Corrector
        #pragma omp parallel for private(i) schedule(static)
        for (j = 1; j < (int)ny - 1; j++) {
            for (i = 1; i < (int)nx - 1; i++) {
                size_t idx = j * nx + i;
                double dp_dx = (p_new[idx + 1] - p_new[idx - 1]) / (2.0 * dx);
                double dp_dy = (p_new[idx + nx] - p_new[idx - nx]) / (2.0 * dy);

                field->u[idx] = u_star[idx] - (dt / rho) * dp_dx;
                field->v[idx] = v_star[idx] - (dt / rho) * dp_dy;
                
                field->u[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->u[idx]));
                field->v[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, field->v[idx]));
            }
        }

        memcpy(field->p, p_new, size * sizeof(double));
        apply_boundary_conditions(field, grid);
    }

    cfd_free(u_star);
    cfd_free(v_star);
    cfd_free(p_new);
    cfd_free(rhs);
}
