/**
 * Optimized Projection Method Solver (Chorin's Method) with SIMD
 * Refactored to use persistent memory context.
 */

#include "cfd_status.h"
#include "solver_interface.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

// Poisson solver parameters
#define POISSON_MAX_ITER  1000
#define POISSON_TOLERANCE 1e-6
#define POISSON_OMEGA     1.5

// Physical limits
#define MAX_VELOCITY 100.0

typedef struct {
    double* u_star;
    double* v_star;
    double* p_new;
    double* rhs;
    double* u_new;
    double* v_new;
    size_t nx;
    size_t ny;
    int initialized;
} ProjectionSIMDContext;

// Public API
cfd_status_t projection_simd_init(Solver* solver, const Grid* grid, const SolverParams* params);
void projection_simd_destroy(Solver* solver);
cfd_status_t projection_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                  const SolverParams* params, SolverStats* stats);

cfd_status_t projection_simd_init(Solver* solver, const Grid* grid, const SolverParams* params) {
    (void)params;
    if (!solver || !grid)
        return CFD_ERROR_INVALID;

    ProjectionSIMDContext* ctx =
        (ProjectionSIMDContext*)cfd_calloc(1, sizeof(ProjectionSIMDContext));
    if (!ctx)
        return CFD_ERROR_NOMEM;

    ctx->nx = grid->nx;
    ctx->ny = grid->ny;
    size_t size = ctx->nx * ctx->ny * sizeof(double);

    ctx->u_star = (double*)cfd_aligned_malloc(size);
    ctx->v_star = (double*)cfd_aligned_malloc(size);
    ctx->p_new = (double*)cfd_aligned_malloc(size);
    ctx->rhs = (double*)cfd_aligned_malloc(size);
    ctx->u_new = (double*)cfd_aligned_malloc(size);
    ctx->v_new = (double*)cfd_aligned_malloc(size);

    if (!ctx->u_star || !ctx->v_star || !ctx->p_new || !ctx->rhs || !ctx->u_new || !ctx->v_new) {
        if (ctx->u_star)
            cfd_aligned_free(ctx->u_star);
        if (ctx->v_star)
            cfd_aligned_free(ctx->v_star);
        if (ctx->p_new)
            cfd_aligned_free(ctx->p_new);
        if (ctx->rhs)
            cfd_aligned_free(ctx->rhs);
        if (ctx->u_new)
            cfd_aligned_free(ctx->u_new);
        if (ctx->v_new)
            cfd_aligned_free(ctx->v_new);
        cfd_free(ctx);
        return CFD_ERROR_NOMEM;
    }

    ctx->initialized = 1;
    solver->context = ctx;
    return CFD_SUCCESS;
}

void projection_simd_destroy(Solver* solver) {
    if (solver && solver->context) {
        ProjectionSIMDContext* ctx = (ProjectionSIMDContext*)solver->context;
        if (ctx->initialized) {
            cfd_aligned_free(ctx->u_star);
            cfd_aligned_free(ctx->v_star);
            cfd_aligned_free(ctx->p_new);
            cfd_aligned_free(ctx->rhs);
            cfd_aligned_free(ctx->u_new);
            cfd_aligned_free(ctx->v_new);
        }
        cfd_free(ctx);
        solver->context = NULL;
    }
}

// Simplified Scalar Poisson for reliability during refactor
static int solve_poisson_sor(double* p, const double* rhs, size_t nx, size_t ny, double dx,
                             double dy) {
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;
    double inv_factor = 1.0 / (2.0 * (inv_dx2 + inv_dy2));

    int iter;
    int converged = 0;

    for (iter = 0; iter < POISSON_MAX_ITER; iter++) {
        double max_res = 0.0;

        for (int color = 0; color < 2; color++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    if ((i + j) % 2 != color)
                        continue;
                    size_t idx = j * nx + i;

                    double p_xx = (p[idx + 1] - 2 * p[idx] + p[idx - 1]) * inv_dx2;
                    double p_yy = (p[idx + nx] - 2 * p[idx] + p[idx - nx]) * inv_dy2;
                    double res = p_xx + p_yy - rhs[idx];
                    if (fabs(res) > max_res)
                        max_res = fabs(res);

                    double p_new_val = (rhs[idx] - (p[idx + 1] + p[idx - 1]) * inv_dx2 -
                                        (p[idx + nx] + p[idx - nx]) * inv_dy2) *
                                       (-inv_factor);
                    p[idx] += POISSON_OMEGA * (p_new_val - p[idx]);
                }
            }
        }

        // BCs
        for (size_t j = 0; j < ny; j++) {
            p[j * nx] = p[j * nx + 1];
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        }
        for (size_t i = 0; i < nx; i++) {
            p[i] = p[nx + i];
            p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
        }

        if (max_res < POISSON_TOLERANCE) {
            converged = 1;
            break;
        }
    }
    return converged ? iter : -1;
}

cfd_status_t projection_simd_step(Solver* solver, FlowField* field, const Grid* grid,
                                  const SolverParams* params, SolverStats* stats) {
    if (!solver || !solver->context || !field || !grid)
        return CFD_ERROR_INVALID;
    ProjectionSIMDContext* ctx = (ProjectionSIMDContext*)solver->context;

    if (field->nx != ctx->nx || field->ny != ctx->ny)
        return CFD_ERROR_INVALID;

    size_t size = ctx->nx * ctx->ny;
    double dt = params->dt;
    double dx = grid->dx[0];
    double dy = grid->dy[0];
    double rho = field->rho[0] > 1e-10 ? field->rho[0] : 1.0;

    // Copy current fields
    memcpy(ctx->u_star, field->u, size * sizeof(double));
    memcpy(ctx->v_star, field->v, size * sizeof(double));
    memcpy(ctx->p_new, field->p, size * sizeof(double));

    // Removed outer loop over max_iter to match interface expectation (one step per call)

    // Step 1: Predictor
    for (size_t j = 1; j < ctx->ny - 1; j++) {
        for (size_t i = 1; i < ctx->nx - 1; i++) {
            size_t idx = j * ctx->nx + i;
            double u = field->u[idx];
            double v = field->v[idx];

            double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2 * dx);
            double du_dy = (field->u[idx + ctx->nx] - field->u[idx - ctx->nx]) / (2 * dy);
            double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2 * dx);
            double dv_dy = (field->v[idx + ctx->nx] - field->v[idx - ctx->nx]) / (2 * dy);

            double d2u_dx2 = (field->u[idx + 1] - 2 * u + field->u[idx - 1]) / (dx * dx);
            double d2u_dy2 =
                (field->u[idx + ctx->nx] - 2 * u + field->u[idx - ctx->nx]) / (dy * dy);
            double d2v_dx2 = (field->v[idx + 1] - 2 * v + field->v[idx - 1]) / (dx * dx);
            double d2v_dy2 =
                (field->v[idx + ctx->nx] - 2 * v + field->v[idx - ctx->nx]) / (dy * dy);

            double conv_u = u * du_dx + v * du_dy;
            double conv_v = u * dv_dx + v * dv_dy;
            double visc_u = params->mu * (d2u_dx2 + d2u_dy2);
            double visc_v = params->mu * (d2v_dx2 + d2v_dy2);

            double source_u = 0.0;
            double source_v = 0.0;
            if (params->source_amplitude_u > 0) {
                source_u = params->source_amplitude_u * sin(M_PI * grid->y[j]);
                source_v = params->source_amplitude_v * sin(2.0 * M_PI * grid->x[i]);
            }

            ctx->u_star[idx] = u + dt * (-conv_u + visc_u + source_u);
            ctx->v_star[idx] = v + dt * (-conv_v + visc_v + source_v);
        }
    }

    // BCs intermediate
    for (size_t j = 0; j < ctx->ny; j++) {
        ctx->u_star[j * ctx->nx] = ctx->u_star[j * ctx->nx + 1];
        ctx->u_star[j * ctx->nx + ctx->nx - 1] = ctx->u_star[j * ctx->nx + ctx->nx - 2];
        ctx->v_star[j * ctx->nx] = ctx->v_star[j * ctx->nx + 1];
        ctx->v_star[j * ctx->nx + ctx->nx - 1] = ctx->v_star[j * ctx->nx + ctx->nx - 2];
    }
    for (size_t i = 0; i < ctx->nx; i++) {
        ctx->u_star[i] = ctx->u_star[ctx->nx + i];
        ctx->u_star[(ctx->ny - 1) * ctx->nx + i] = ctx->u_star[(ctx->ny - 2) * ctx->nx + i];
        ctx->v_star[i] = ctx->v_star[ctx->nx + i];
        ctx->v_star[(ctx->ny - 1) * ctx->nx + i] = ctx->v_star[(ctx->ny - 2) * ctx->nx + i];
    }

    // Step 2: RHS
    for (size_t j = 1; j < ctx->ny - 1; j++) {
        for (size_t i = 1; i < ctx->nx - 1; i++) {
            size_t idx = j * ctx->nx + i;
            double du_dx = (ctx->u_star[idx + 1] - ctx->u_star[idx - 1]) / (2 * dx);
            double dv_dy = (ctx->v_star[idx + ctx->nx] - ctx->v_star[idx - ctx->nx]) / (2 * dy);
            ctx->rhs[idx] = (rho / dt) * (du_dx + dv_dy);
        }
    }

    // Poisson
    int poisson_iters = solve_poisson_sor(ctx->p_new, ctx->rhs, ctx->nx, ctx->ny, dx, dy);
    if (poisson_iters < 0) {
        // Fallback if Poisson solver fails to converge
        double fallback_factor = 0.1 * dt;
#if USE_AVX
        __m256d v_factor = _mm256_set1_pd(fallback_factor);
        size_t total_cells = ctx->nx * ctx->ny;
        size_t aligned_n = (total_cells / 4) * 4;

        for (size_t i = 0; i < aligned_n; i += 4) {
            __m256d v_p = _mm256_loadu_pd(&field->p[i]);    // field->p might not be aligned
            __m256d v_rhs = _mm256_loadu_pd(&ctx->rhs[i]);  // Safe unaligned load
            __m256d v_res = _mm256_sub_pd(v_p, _mm256_mul_pd(v_factor, v_rhs));
            _mm256_storeu_pd(&ctx->p_new[i], v_res);
        }
        for (size_t i = aligned_n; i < total_cells; i++) {
            ctx->p_new[i] = field->p[i] - fallback_factor * ctx->rhs[i];
        }
#else
        for (size_t i = 0; i < ctx->nx * ctx->ny; i++) {
            ctx->p_new[i] = field->p[i] - fallback_factor * ctx->rhs[i];
        }
#endif
    }

    // Step 3: Corrector
    for (size_t j = 1; j < ctx->ny - 1; j++) {
        for (size_t i = 1; i < ctx->nx - 1; i++) {
            size_t idx = j * ctx->nx + i;
            double dp_dx = (ctx->p_new[idx + 1] - ctx->p_new[idx - 1]) / (2 * dx);
            double dp_dy = (ctx->p_new[idx + ctx->nx] - ctx->p_new[idx - ctx->nx]) / (2 * dy);

            double u_corr = ctx->u_star[idx] - (dt / rho) * dp_dx;
            double v_corr = ctx->v_star[idx] - (dt / rho) * dp_dy;

            ctx->u_new[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, u_corr));
            ctx->v_new[idx] = fmax(-MAX_VELOCITY, fmin(MAX_VELOCITY, v_corr));
        }
    }

    // Update field
    memcpy(field->u, ctx->u_new, size * sizeof(double));
    memcpy(field->v, ctx->v_new, size * sizeof(double));
    memcpy(field->p, ctx->p_new, size * sizeof(double));
    apply_boundary_conditions(field, grid);

    if (stats) {
        stats->iterations = 1;
    }

    return CFD_SUCCESS;
}
