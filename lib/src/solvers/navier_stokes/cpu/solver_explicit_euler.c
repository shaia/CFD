#include "cfd/core/cfd_status.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/grid.h"
#include "cfd/core/indexing.h"
#include "cfd/core/math_utils.h"
#include "cfd/core/memory.h"


#include "cfd/solvers/energy_solver.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include "../boundary_copy_utils.h"

#include <math.h>
#include <stdio.h>
#include <string.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical stability limits for numerical computation
#define MAX_DERIVATIVE_LIMIT        100.0   // Maximum allowed first derivative magnitude (1/s)
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0  // Maximum allowed second derivative magnitude (1/s²)
#define MAX_VELOCITY_LIMIT          100.0   // Maximum allowed velocity magnitude (m/s)
#define MAX_DIVERGENCE_LIMIT        10.0    // Maximum allowed velocity divergence (1/s)

// Initial condition constants
#define INIT_U_BASE   1.0
#define INIT_U_VAR    0.1
#define INIT_V_VAR    0.05
#define INIT_PRESSURE 1.0
#define INIT_DENSITY  1.0
#define INIT_TEMP     300.0

// Perturbation constants
#define PERTURB_CENTER_X    1.0
#define PERTURB_CENTER_Y    0.5
#define PERTURB_RADIUS      0.2
#define PERTURB_WIDTH_SQ    0.02
#define PERTURB_MAG         0.1
#define PERTURB_GRAD_FACTOR 2.0

// Time stepping and stability constants
#define VELOCITY_EPSILON      1e-20
#define SPEED_EPSILON         1e-10
#define DT_MAX_LIMIT          0.01
#define DT_MIN_LIMIT          1e-6
#define DT_CONSERVATIVE_LIMIT 0.0001

// Update limits
#define UPDATE_LIMIT           1.0
#define PRESSURE_UPDATE_FACTOR 0.1

// Helper function to initialize ns_solver_params_t with default values
ns_solver_params_t ns_solver_params_default(void) {
    ns_solver_params_t params = {.dt = DEFAULT_TIME_STEP,
                            .cfl = DEFAULT_CFL_NUMBER,
                            .gamma = DEFAULT_GAMMA,
                            .mu = DEFAULT_VISCOSITY,
                            .k = DEFAULT_THERMAL_CONDUCTIVITY,
                            .max_iter = DEFAULT_MAX_ITERATIONS,
                            .tolerance = DEFAULT_TOLERANCE,
                            .source_amplitude_u = DEFAULT_SOURCE_AMPLITUDE_U,
                            .source_amplitude_v = DEFAULT_SOURCE_AMPLITUDE_V,
                            .source_decay_rate = DEFAULT_SOURCE_DECAY_RATE,
                            .pressure_coupling = DEFAULT_PRESSURE_COUPLING,
                            .alpha = 0.0,
                            .beta = 0.0,
                            .T_ref = 0.0,
                            .gravity = {0.0, 0.0, 0.0},
                            .heat_source_func = NULL,
                            .heat_source_context = NULL};
    return params;
}
flow_field* flow_field_create(size_t nx, size_t ny, size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) {
        cfd_set_error(CFD_ERROR_INVALID, "Flow field dimensions must be positive");
        return NULL;
    }

    flow_field* field = (flow_field*)cfd_calloc(1, sizeof(flow_field));
    if (field == NULL) {
        return NULL;
    }

    field->nx = nx;
    field->ny = ny;
    field->nz = nz;

    size_t total = nx * ny * nz;

    // Allocate 32-byte aligned memory for flow variables (optimized for SIMD operations)
    field->u = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->v = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->w = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->p = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->rho = (double*)cfd_aligned_calloc(total, sizeof(double));
    field->T = (double*)cfd_aligned_calloc(total, sizeof(double));

    if (!field->u || !field->v || !field->w || !field->p || !field->rho || !field->T) {
        flow_field_destroy(field);
        return NULL;
    }

    return field;
}

void flow_field_destroy(flow_field* field) {
    if (field != NULL) {
        cfd_aligned_free(field->u);
        cfd_aligned_free(field->v);
        cfd_aligned_free(field->w);
        cfd_aligned_free(field->p);
        cfd_aligned_free(field->rho);
        cfd_aligned_free(field->T);
        cfd_free(field);
    }
}

void initialize_flow_field(flow_field* field, const grid* grid) {
    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;
    size_t plane = nx * ny;

    for (size_t k = 0; k < nz; k++) {
        size_t base = k * plane;
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t idx = base + IDX_2D(i, j, nx);
                double x = grid->x[i];
                double y = grid->y[j];

                field->u[idx] =
                    INIT_U_BASE + (INIT_U_VAR * sin(M_PI * y));
                field->v[idx] = INIT_V_VAR * sin(2.0 * M_PI * x);
                field->w[idx] = 0.0;
                field->p[idx] = INIT_PRESSURE;
                field->rho[idx] = INIT_DENSITY;
                field->T[idx] = INIT_TEMP;

                double cx = PERTURB_CENTER_X, cy = PERTURB_CENTER_Y;
                double r = sqrt(((x - cx) * (x - cx)) + ((y - cy) * (y - cy)));
                if (r < PERTURB_RADIUS) {
                    field->p[idx] += PERTURB_MAG * exp(-r * r / PERTURB_WIDTH_SQ);
                    double dp_dx = -PERTURB_MAG * PERTURB_GRAD_FACTOR * (x - cx) / PERTURB_WIDTH_SQ *
                                   exp(-r * r / PERTURB_WIDTH_SQ);
                    double dp_dy = -PERTURB_MAG * PERTURB_GRAD_FACTOR * (y - cy) / PERTURB_WIDTH_SQ *
                                   exp(-r * r / PERTURB_WIDTH_SQ);
                    field->u[idx] += -PERTURB_MAG * dp_dx;
                    field->v[idx] += -PERTURB_MAG * dp_dy;
                }
            }
        }
    }
}

void compute_time_step(flow_field* field, const grid* grid, ns_solver_params_t* params) {
    double max_speed = 0.0;
    double dx_min = grid->dx[0];
    double dy_min = grid->dy[0];

    // Find minimum grid spacing
    for (size_t i = 0; i < grid->nx - 1; i++) {
        dx_min = min_double(dx_min, grid->dx[i]);
    }
    for (size_t j = 0; j < grid->ny - 1; j++) {
        dy_min = min_double(dy_min, grid->dy[j]);
    }

    // Find maximum wave speed
    int has_w = (grid->nz > 1 && field->w);
    for (size_t j = 0; j < field->ny; j++) {
        for (size_t i = 0; i < field->nx; i++) {
            size_t idx = IDX_2D(i, j, field->nx);
            double u_speed = fabs(field->u[idx]);
            double v_speed = fabs(field->v[idx]);
            double sound_speed = sqrt(params->gamma * field->p[idx] / field->rho[idx]);

            // Optimized velocity magnitude calculation - avoid sqrt when possible
            double vel_mag_sq = (u_speed * u_speed) + (v_speed * v_speed);
            if (has_w) {
                double w_speed = fabs(field->w[idx]);
                vel_mag_sq += w_speed * w_speed;
            }
            double vel_mag = (vel_mag_sq > VELOCITY_EPSILON) ? sqrt(vel_mag_sq) : 0.0;
            double local_speed = vel_mag + sound_speed;
            max_speed = max_double(max_speed, local_speed);
        }
    }

    // Prevent division by zero and ensure reasonable time step
    if (max_speed < SPEED_EPSILON) {
        max_speed = 1.0;  // Use default if speeds are too small
    }

    // Include z-direction in CFL if 3D
    double dmin = min_double(dx_min, dy_min);
    if (grid->nz > 1 && grid->dz) {
        double dz_min = grid->dz[0];
        for (size_t k = 0; k < grid->nz - 1; k++) {
            dz_min = min_double(dz_min, grid->dz[k]);
        }
        dmin = min_double(dmin, dz_min);
    }

    // Compute time step based on CFL condition with safety factor
    double dt_cfl = params->cfl * dmin / max_speed;

    // Thermal diffusion stability constraint: dt < dx^2 / (2 * alpha * ndim)
    double dt_thermal = dt_cfl;
    if (params->alpha > 0.0) {
        int ndim = (grid->nz > 1) ? 3 : 2;
        dt_thermal = (dmin * dmin) / (2.0 * params->alpha * ndim);
        dt_thermal *= params->cfl;  // Apply safety factor
    }

    double dt_stable = min_double(dt_cfl, dt_thermal);

    // Limit time step to reasonable bounds
    double dt_max = DT_MAX_LIMIT;  // Maximum allowed time step
    double dt_min = DT_MIN_LIMIT;  // Minimum allowed time step

    params->dt = max_double(dt_min, min_double(dt_max, dt_stable));
}

void apply_boundary_conditions(flow_field* field, const grid* grid) {
    (void)grid;
    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;
    size_t plane = nx * ny;

    /* Apply periodic BCs in x and y for each z-plane */
    for (size_t k = 0; k < nz; k++) {
        size_t base = k * plane;

        /* x-direction periodic */
        for (size_t j = 0; j < ny; j++) {
            size_t left = base + IDX_2D(0, j, nx);
            size_t right = base + IDX_2D(nx - 1, j, nx);
            size_t src_left = base + IDX_2D(nx - 2, j, nx);
            size_t src_right = base + IDX_2D(1, j, nx);

            field->u[left] = field->u[src_left];
            field->v[left] = field->v[src_left];
            field->w[left] = field->w[src_left];
            field->p[left] = field->p[src_left];
            field->rho[left] = field->rho[src_left];
            field->T[left] = field->T[src_left];

            field->u[right] = field->u[src_right];
            field->v[right] = field->v[src_right];
            field->w[right] = field->w[src_right];
            field->p[right] = field->p[src_right];
            field->rho[right] = field->rho[src_right];
            field->T[right] = field->T[src_right];
        }

        /* y-direction periodic */
        for (size_t i = 0; i < nx; i++) {
            size_t bot = base + i;
            size_t top = base + IDX_2D(i, ny - 1, nx);
            size_t src_bot = base + IDX_2D(i, ny - 2, nx);
            size_t src_top = base + IDX_2D(i, 1, nx);

            field->u[bot] = field->u[src_bot];
            field->v[bot] = field->v[src_bot];
            field->w[bot] = field->w[src_bot];
            field->p[bot] = field->p[src_bot];
            field->rho[bot] = field->rho[src_bot];
            field->T[bot] = field->T[src_bot];

            field->u[top] = field->u[src_top];
            field->v[top] = field->v[src_top];
            field->w[top] = field->w[src_top];
            field->p[top] = field->p[src_top];
            field->rho[top] = field->rho[src_top];
            field->T[top] = field->T[src_top];
        }
    }

    /* z-direction periodic (only when nz > 1) */
    if (nz > 1) {
        size_t front_base = 0;
        size_t back_base = (nz - 1) * plane;
        size_t src_front = (nz - 2) * plane;
        size_t src_back = 1 * plane;

        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                size_t off = IDX_2D(i, j, nx);

                field->u[front_base + off] = field->u[src_front + off];
                field->v[front_base + off] = field->v[src_front + off];
                field->w[front_base + off] = field->w[src_front + off];
                field->p[front_base + off] = field->p[src_front + off];
                field->rho[front_base + off] = field->rho[src_front + off];
                field->T[front_base + off] = field->T[src_front + off];

                field->u[back_base + off] = field->u[src_back + off];
                field->v[back_base + off] = field->v[src_back + off];
                field->w[back_base + off] = field->w[src_back + off];
                field->p[back_base + off] = field->p[src_back + off];
                field->rho[back_base + off] = field->rho[src_back + off];
                field->T[back_base + off] = field->T[src_back + off];
            }
        }
    }
}

// Helper function to compute source terms consistently across all solvers
void compute_source_terms(double x, double y, double z, int iter, double dt,
                          const ns_solver_params_t* params,
                          double* source_u, double* source_v, double* source_w) {
    // Use custom source function if provided
    if (params->source_func) {
        double t = iter * dt;
        params->source_func(x, y, z, t, params->source_context, source_u, source_v, source_w);
        return;
    }

    // Default source term implementation (no default w-source)
    *source_u =
        params->source_amplitude_u * sin(M_PI * y) * exp(-params->source_decay_rate * iter * dt);
    *source_v = params->source_amplitude_v * sin(2.0 * M_PI * x) *
                exp(-params->source_decay_rate * iter * dt);
    *source_w = 0.0;
}

// Internal explicit Euler implementation
// This is called by the solver registry - not part of public API
cfd_status_t explicit_euler_impl(flow_field* field, const grid* grid, const ns_solver_params_t* params) {
    if (field->nx < 3 || field->ny < 3 || (field->nz > 1 && field->nz < 3)) {
        return CFD_ERROR_INVALID;
    }

    size_t nx = field->nx;
    size_t ny = field->ny;
    size_t nz = field->nz;

    /* Reject non-uniform z-spacing (solver uses constant inv_2dz/inv_dz2) */
    if (nz > 1 && grid->dz) {
        for (size_t k = 1; k < nz - 1; k++) {
            if (fabs(grid->dz[k] - grid->dz[0]) > 1e-14) {
                return CFD_ERROR_INVALID;
            }
        }
    }

    size_t plane = nx * ny;
    size_t total = plane * nz;
    size_t bytes = total * sizeof(double);

    /* Branch-free 3D constants: when nz==1, stride_z=0, inv_2dz=0, inv_dz2=0
     * so all z-terms vanish, producing identical results to 2D. */
    size_t stride_z = (nz > 1) ? plane : 0;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end   = (nz > 1) ? (nz - 1) : 1;
    double inv_2dz = (nz > 1 && grid->dz) ? 1.0 / (2.0 * grid->dz[0]) : 0.0;
    double inv_dz2 = (nz > 1 && grid->dz) ? 1.0 / (grid->dz[0] * grid->dz[0]) : 0.0;

    double* u_new = (double*)cfd_calloc(total, sizeof(double));
    double* v_new = (double*)cfd_calloc(total, sizeof(double));
    double* w_new = (double*)cfd_calloc(total, sizeof(double));
    double* p_new = (double*)cfd_calloc(total, sizeof(double));
    double* rho_new = (double*)cfd_calloc(total, sizeof(double));

    if (!u_new || !v_new || !w_new || !p_new || !rho_new) {
        cfd_free(u_new); cfd_free(v_new); cfd_free(w_new);
        cfd_free(p_new); cfd_free(rho_new);
        return CFD_ERROR_NOMEM;
    }

    memcpy(u_new, field->u, bytes);
    memcpy(v_new, field->v, bytes);
    memcpy(w_new, field->w, bytes);
    memcpy(p_new, field->p, bytes);
    memcpy(rho_new, field->rho, bytes);

    double conservative_dt = fmin(params->dt, DT_CONSERVATIVE_LIMIT);

    for (int iter = 0; iter < params->max_iter; iter++) {
        for (size_t k = k_start; k < k_end; k++) {
            for (size_t j = 1; j < ny - 1; j++) {
                for (size_t i = 1; i < nx - 1; i++) {
                    size_t idx = k * stride_z + IDX_2D(i, j, nx);

                    if (field->rho[idx] <= 1e-10) {
                        continue;
                    }
                    if (fabs(grid->dx[i]) < 1e-10 || fabs(grid->dy[j]) < 1e-10) {
                        continue;
                    }

                    double u_c = field->u[idx];
                    double v_c = field->v[idx];
                    double w_c = field->w[idx];

                    /* First derivatives (central differences) */
                    double du_dx = (field->u[idx + 1] - field->u[idx - 1]) / (2.0 * grid->dx[i]);
                    double du_dy = (field->u[idx + nx] - field->u[idx - nx]) / (2.0 * grid->dy[j]);
                    double du_dz = (field->u[idx + stride_z] - field->u[idx - stride_z]) * inv_2dz;

                    double dv_dx = (field->v[idx + 1] - field->v[idx - 1]) / (2.0 * grid->dx[i]);
                    double dv_dy = (field->v[idx + nx] - field->v[idx - nx]) / (2.0 * grid->dy[j]);
                    double dv_dz = (field->v[idx + stride_z] - field->v[idx - stride_z]) * inv_2dz;

                    double dw_dx = (field->w[idx + 1] - field->w[idx - 1]) / (2.0 * grid->dx[i]);
                    double dw_dy = (field->w[idx + nx] - field->w[idx - nx]) / (2.0 * grid->dy[j]);
                    double dw_dz = (field->w[idx + stride_z] - field->w[idx - stride_z]) * inv_2dz;

                    /* Pressure gradients */
                    double dp_dx = (field->p[idx + 1] - field->p[idx - 1]) / (2.0 * grid->dx[i]);
                    double dp_dy = (field->p[idx + nx] - field->p[idx - nx]) / (2.0 * grid->dy[j]);
                    double dp_dz = (field->p[idx + stride_z] - field->p[idx - stride_z]) * inv_2dz;

                    /* Second derivatives (viscous terms) */
                    double d2u_dx2 = (field->u[idx + 1] - 2.0 * u_c + field->u[idx - 1]) /
                                     (grid->dx[i] * grid->dx[i]);
                    double d2u_dy2 = (field->u[idx + nx] - 2.0 * u_c + field->u[idx - nx]) /
                                     (grid->dy[j] * grid->dy[j]);
                    double d2u_dz2 = (field->u[idx + stride_z] - 2.0 * u_c +
                                      field->u[idx - stride_z]) * inv_dz2;

                    double d2v_dx2 = (field->v[idx + 1] - 2.0 * v_c + field->v[idx - 1]) /
                                     (grid->dx[i] * grid->dx[i]);
                    double d2v_dy2 = (field->v[idx + nx] - 2.0 * v_c + field->v[idx - nx]) /
                                     (grid->dy[j] * grid->dy[j]);
                    double d2v_dz2 = (field->v[idx + stride_z] - 2.0 * v_c +
                                      field->v[idx - stride_z]) * inv_dz2;

                    double d2w_dx2 = (field->w[idx + 1] - 2.0 * w_c + field->w[idx - 1]) /
                                     (grid->dx[i] * grid->dx[i]);
                    double d2w_dy2 = (field->w[idx + nx] - 2.0 * w_c + field->w[idx - nx]) /
                                     (grid->dy[j] * grid->dy[j]);
                    double d2w_dz2 = (field->w[idx + stride_z] - 2.0 * w_c +
                                      field->w[idx - stride_z]) * inv_dz2;

                    double nu = params->mu / fmax(field->rho[idx], 1e-10);
                    nu = fmin(nu, 1.0);

                    /* Clamp derivatives */
                    du_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dx));
                    du_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dy));
                    du_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, du_dz));
                    dv_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dx));
                    dv_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dy));
                    dv_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dv_dz));
                    dw_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dx));
                    dw_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dy));
                    dw_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dw_dz));
                    dp_dx = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dx));
                    dp_dy = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dy));
                    dp_dz = fmax(-MAX_DERIVATIVE_LIMIT, fmin(MAX_DERIVATIVE_LIMIT, dp_dz));
                    d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
                    d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
                    d2u_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dz2));
                    d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
                    d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));
                    d2v_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dz2));
                    d2w_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dx2));
                    d2w_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dy2));
                    d2w_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dz2));

                    /* Source terms */
                    double x = grid->x[i];
                    double y = grid->y[j];
                    double z = (nz > 1 && grid->z) ? grid->z[k] : 0.0;
                    double source_u, source_v, source_w;
                    compute_source_terms(x, y, z, iter, conservative_dt, params,
                                         &source_u, &source_v, &source_w);

                    /* Boussinesq buoyancy source */
                    energy_compute_buoyancy(field->T[idx], params,
                                            &source_u, &source_v, &source_w);

                    /* u-momentum */
                    double du = conservative_dt *
                        (-u_c * du_dx - v_c * du_dy - w_c * du_dz
                         - dp_dx / field->rho[idx]
                         + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2)
                         + source_u);

                    /* v-momentum */
                    double dv = conservative_dt *
                        (-u_c * dv_dx - v_c * dv_dy - w_c * dv_dz
                         - dp_dy / field->rho[idx]
                         + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2)
                         + source_v);

                    /* w-momentum */
                    double dw = conservative_dt *
                        (-u_c * dw_dx - v_c * dw_dy - w_c * dw_dz
                         - dp_dz / field->rho[idx]
                         + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2)
                         + source_w);

                    du = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, du));
                    dv = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dv));
                    dw = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dw));

                    u_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, u_c + du));
                    v_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, v_c + dv));
                    w_new[idx] = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, w_c + dw));

                    /* Pressure update from divergence */
                    double divergence = du_dx + dv_dy + dw_dz;
                    divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
                    double dp = -PRESSURE_UPDATE_FACTOR * conservative_dt * field->rho[idx] * divergence;
                    dp = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dp));
                    p_new[idx] = field->p[idx] + dp;

                    rho_new[idx] = field->rho[idx];
                }
            }
        }

        /* Copy new solution */
        memcpy(field->u, u_new, bytes);
        memcpy(field->v, v_new, bytes);
        memcpy(field->w, w_new, bytes);
        memcpy(field->p, p_new, bytes);
        memcpy(field->rho, rho_new, bytes);

        /* Energy equation: advance temperature using updated velocity */
        {
            cfd_status_t energy_status = energy_step_explicit(field, grid, params,
                                                               conservative_dt,
                                                               iter * conservative_dt);
            if (energy_status != CFD_SUCCESS) {
                cfd_free(u_new); cfd_free(v_new); cfd_free(w_new);
                cfd_free(p_new); cfd_free(rho_new);
                return energy_status;
            }
        }

        /* Save caller BCs, apply periodic BCs, restore caller BCs.
         * Use _3d helper for 6-face copy when nz > 1. */
        copy_boundary_velocities_3d(u_new, v_new, w_new,
                                    field->u, field->v, field->w, nx, ny, nz);
        apply_boundary_conditions(field, grid);
        copy_boundary_velocities_3d(field->u, field->v, field->w,
                                    u_new, v_new, w_new, nx, ny, nz);

        /* NaN/Inf check */
        int has_nan = 0;
        for (size_t n = 0; n < total; n++) {
            if (!isfinite(field->u[n]) || !isfinite(field->v[n]) ||
                !isfinite(field->w[n]) || !isfinite(field->p[n])) {
                has_nan = 1;
                break;
            }
        }
        if (has_nan) {
            cfd_free(u_new); cfd_free(v_new); cfd_free(w_new);
            cfd_free(p_new); cfd_free(rho_new);
            cfd_set_error(CFD_ERROR_DIVERGED,
                          "NaN/Inf detected in explicit_euler step");
            return CFD_ERROR_DIVERGED;
        }
    }

    cfd_free(u_new); cfd_free(v_new); cfd_free(w_new);
    cfd_free(p_new); cfd_free(rho_new);

    return CFD_SUCCESS;
}
