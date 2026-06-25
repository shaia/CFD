/**
 * @file solver_rk_gpu.cu
 * @brief GPU-accelerated explicit Runge-Kutta Navier-Stokes solvers (RK2/RK4).
 *
 * CUDA port of the scalar RK2 (Heun) and RK4 (classical) integrators
 * (lib/src/solvers/navier_stokes/cpu/solver_rk{2,4}.c). The device RHS kernel
 * mirrors the shared scalar kernel ns_momentum_rhs_scalar.h exactly: periodic
 * stencil indexing (no ghost-cell reliance), the same derivative/divergence
 * clamps, the default sinusoidal momentum source, and Boussinesq buoyancy. As
 * in the CPU path, boundary conditions are NOT applied between RK stages — only
 * after the full step — which is required to preserve RK temporal order.
 *
 * One driver, solve_rk_gpu(..., order), serves both RK2 (order=2, stages k1,k2)
 * and RK4 (order=4, stages k1..k4). The energy equation, thermal BCs, and
 * caller-set velocity-BC restoration reuse the shared device kernels in
 * gpu_shared_kernels.cuh (the same ones used by the projection GPU backend).
 *
 * GPU-specific limitations vs the CPU reference (rejected, not silently wrong):
 *   - host source_func / heat_source_func callbacks cannot run on the device;
 *   - only uniform grid spacing is supported (the CPU path allows non-uniform x/y).
 * Like the projection GPU backend, caller-set velocity boundaries are restored
 * from their initial (upload-time) values rather than re-running the full host
 * apply_boundary_conditions() each step; GPU vs CPU is validated within tolerance.
 *
 * Branch-free 3D: when nz==1, stride_z=0 and inv_2dz=inv_dz2=0 collapse all
 * z-terms, matching the 2D code path.
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/gpu_device.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include "gpu_shared_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physical stability limits — identical to the scalar RHS kernel
// (ns_momentum_rhs_scalar.h) and the velocity update (solver_rk4.c).
#define MAX_VELOCITY_LIMIT          100.0
#define MAX_DERIVATIVE_LIMIT        100.0
#define MAX_SECOND_DERIVATIVE_LIMIT 1000.0
#define MAX_DIVERGENCE_LIMIT        10.0
#define PRESSURE_UPDATE_FACTOR      0.1

// Explicit-Euler-only limits (order==1 path), mirroring solver_explicit_euler.c:
// the per-step increment is clamped to +/-UPDATE_LIMIT before being added, and dt
// is capped at DT_CONSERVATIVE_LIMIT. These do NOT apply to the RK2/RK4 paths.
#define UPDATE_LIMIT          1.0
#define DT_CONSERVATIVE_LIMIT 0.0001

// ============================================================================
// CUDA kernels
// ============================================================================

// Momentum RHS — device port of compute_rhs (ns_momentum_rhs_scalar.h).
// Writes the semi-discrete RHS of (u,v,w,p) for every interior point using
// periodic stencil indices. x_coord/y_coord are the grid->x / grid->y arrays
// (needed by the default sinusoidal source term).
__global__ void kernel_rk_rhs(const double* __restrict__ u, const double* __restrict__ v,
                              const double* __restrict__ w, const double* __restrict__ p,
                              const double* __restrict__ rho, const double* __restrict__ T,
                              double* __restrict__ rhs_u, double* __restrict__ rhs_v,
                              double* __restrict__ rhs_w, double* __restrict__ rhs_p,
                              const double* __restrict__ x_coord,
                              const double* __restrict__ y_coord,
                              size_t nx, size_t ny, size_t nz,
                              size_t stride_z, int k_start, int k_end,
                              double dx, double dy, double inv_2dz, double inv_dz2,
                              double mu, double beta, double T_ref,
                              double gx, double gy, double gz,
                              double amp_u, double amp_v, double decay,
                              int iter, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = (size_t)k * stride_z + IDX_2D(i, j, nx);

            if (rho[idx] <= 1e-10) {
                rhs_u[idx] = 0.0; rhs_v[idx] = 0.0; rhs_w[idx] = 0.0; rhs_p[idx] = 0.0;
                continue;
            }

            // Periodic stencil indices in x/y/z — avoids relying on ghost cells.
            // When nz==1: stride_z=0 so kd=ku=idx and the z-terms vanish.
            size_t il = (i > 1)      ? idx - 1       : (size_t)k * stride_z + IDX_2D(nx - 2, j, nx);
            size_t ir = (i < nx - 2) ? idx + 1       : (size_t)k * stride_z + IDX_2D(1, j, nx);
            size_t jd = (j > 1)      ? idx - nx      : (size_t)k * stride_z + IDX_2D(i, ny - 2, nx);
            size_t ju = (j < ny - 2) ? idx + nx      : (size_t)k * stride_z + IDX_2D(i, 1, nx);
            size_t kd = (k > 1)      ? idx - stride_z : (size_t)(nz - 2) * stride_z + IDX_2D(i, j, nx);
            size_t ku = (k < nz - 2) ? idx + stride_z : (size_t)1 * stride_z + IDX_2D(i, j, nx);

            double inv_2dx = 1.0 / (2.0 * dx);
            double inv_2dy = 1.0 / (2.0 * dy);
            double inv_dx2 = 1.0 / (dx * dx);
            double inv_dy2 = 1.0 / (dy * dy);

            // First derivatives (central differences)
            double du_dx = (u[ir] - u[il]) * inv_2dx;
            double du_dy = (u[ju] - u[jd]) * inv_2dy;
            double du_dz = (u[ku] - u[kd]) * inv_2dz;
            double dv_dx = (v[ir] - v[il]) * inv_2dx;
            double dv_dy = (v[ju] - v[jd]) * inv_2dy;
            double dv_dz = (v[ku] - v[kd]) * inv_2dz;
            double dw_dx = (w[ir] - w[il]) * inv_2dx;
            double dw_dy = (w[ju] - w[jd]) * inv_2dy;
            double dw_dz = (w[ku] - w[kd]) * inv_2dz;

            double dp_dx = (p[ir] - p[il]) * inv_2dx;
            double dp_dy = (p[ju] - p[jd]) * inv_2dy;
            double dp_dz = (p[ku] - p[kd]) * inv_2dz;

            // Second derivatives (viscous terms)
            double d2u_dx2 = (u[ir] - 2.0 * u[idx] + u[il]) * inv_dx2;
            double d2u_dy2 = (u[ju] - 2.0 * u[idx] + u[jd]) * inv_dy2;
            double d2u_dz2 = (u[ku] - 2.0 * u[idx] + u[kd]) * inv_dz2;
            double d2v_dx2 = (v[ir] - 2.0 * v[idx] + v[il]) * inv_dx2;
            double d2v_dy2 = (v[ju] - 2.0 * v[idx] + v[jd]) * inv_dy2;
            double d2v_dz2 = (v[ku] - 2.0 * v[idx] + v[kd]) * inv_dz2;
            double d2w_dx2 = (w[ir] - 2.0 * w[idx] + w[il]) * inv_dx2;
            double d2w_dy2 = (w[ju] - 2.0 * w[idx] + w[jd]) * inv_dy2;
            double d2w_dz2 = (w[ku] - 2.0 * w[idx] + w[kd]) * inv_dz2;

            double nu = mu / fmax(rho[idx], 1e-10);
            nu = fmin(nu, 1.0);

            // Clamp first derivatives
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

            // Clamp second derivatives
            d2u_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dx2));
            d2u_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dy2));
            d2u_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2u_dz2));
            d2v_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dx2));
            d2v_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dy2));
            d2v_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2v_dz2));
            d2w_dx2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dx2));
            d2w_dy2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dy2));
            d2w_dz2 = fmax(-MAX_SECOND_DERIVATIVE_LIMIT, fmin(MAX_SECOND_DERIVATIVE_LIMIT, d2w_dz2));

            // Default sinusoidal momentum source (compute_source_terms with no
            // source_func; host callbacks are rejected before launch).
            double t = iter * dt;
            double decay_factor = exp(-decay * t);
            double source_u = amp_u * sin(M_PI * y_coord[j]) * decay_factor;
            double source_v = amp_v * sin(2.0 * M_PI * x_coord[i]) * decay_factor;
            double source_w = 0.0;

            // Boussinesq buoyancy source (no-op when beta==0).
            double dT = T[idx] - T_ref;
            source_u += -beta * dT * gx;
            source_v += -beta * dT * gy;
            source_w += -beta * dT * gz;

            rhs_u[idx] = -u[idx] * du_dx - v[idx] * du_dy - w[idx] * du_dz
                         - dp_dx / rho[idx]
                         + nu * (d2u_dx2 + d2u_dy2 + d2u_dz2) + source_u;
            rhs_v[idx] = -u[idx] * dv_dx - v[idx] * dv_dy - w[idx] * dv_dz
                         - dp_dy / rho[idx]
                         + nu * (d2v_dx2 + d2v_dy2 + d2v_dz2) + source_v;
            rhs_w[idx] = -u[idx] * dw_dx - v[idx] * dw_dy - w[idx] * dw_dz
                         - dp_dz / rho[idx]
                         + nu * (d2w_dx2 + d2w_dy2 + d2w_dz2) + source_w;

            double divergence = du_dx + dv_dy + dw_dz;
            divergence = fmax(-MAX_DIVERGENCE_LIMIT, fmin(MAX_DIVERGENCE_LIMIT, divergence));
            rhs_p[idx] = -PRESSURE_UPDATE_FACTOR * rho[idx] * divergence;
        }
    }
}

// Stage update: out = base + factor*k over the full field. clamp_velocity != 0
// clamps to [-MAX_VELOCITY_LIMIT, MAX_VELOCITY_LIMIT] (used for u/v/w; pressure
// is updated unclamped). Mirrors apply_stage_update in solver_rk4.c.
__global__ void kernel_axpy1(double* __restrict__ out, const double* __restrict__ base,
                             const double* __restrict__ k, double factor, int clamp_velocity,
                             size_t total) {
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total) {
        double val = base[n] + factor * k[n];
        if (clamp_velocity)
            val = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, val));
        out[n] = val;
    }
}

// Forward-Euler update (order==1): out = base + clamp(dt*k, +/-UPDATE_LIMIT),
// then clamp velocity to +/-MAX_VELOCITY_LIMIT when clamp_velocity != 0 (pressure
// passes clamp_velocity == 0). Mirrors the increment-clamp + velocity-clamp of
// solver_explicit_euler.c exactly (du = clamp(conservative_dt*rhs); u += du).
__global__ void kernel_euler_update(double* __restrict__ out, const double* __restrict__ base,
                                    const double* __restrict__ k, double dt, int clamp_velocity,
                                    size_t total) {
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total) {
        double inc = fmax(-UPDATE_LIMIT, fmin(UPDATE_LIMIT, dt * k[n]));
        double val = base[n] + inc;
        if (clamp_velocity)
            val = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, val));
        out[n] = val;
    }
}

// Final RK combine: out = base + coef*(w1*k1 + w2*k2 + w3*k3 + w4*k4).
// RK2 passes (w1,w2,w3,w4)=(1,1,0,0) with coef=dt/2; RK4 (1,2,2,1) with dt/6.
// Unused stage pointers may alias a live buffer as long as their weight is 0.
__global__ void kernel_rk_combine(double* __restrict__ out, const double* __restrict__ base,
                                  const double* __restrict__ k1, const double* __restrict__ k2,
                                  const double* __restrict__ k3, const double* __restrict__ k4,
                                  double w1, double w2, double w3, double w4,
                                  double coef, int clamp_velocity, size_t total) {
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n < total) {
        double val = base[n] + coef * (w1 * k1[n] + w2 * k2[n] + w3 * k3[n] + w4 * k4[n]);
        if (clamp_velocity)
            val = fmax(-MAX_VELOCITY_LIMIT, fmin(MAX_VELOCITY_LIMIT, val));
        out[n] = val;
    }
}

// ============================================================================
// Host driver
// ============================================================================

extern "C" {

// Reject non-uniform spacing in one axis: GPU RK kernels use a constant dx/dy/dz.
// spacing has `count` entries; returns 1 if all equal spacing[0] within tol.
static int spacing_is_uniform(const double* spacing, size_t count) {
    if (!spacing || count == 0)
        return 1;
    for (size_t i = 1; i < count; i++) {
        if (fabs(spacing[i] - spacing[0]) > 1e-12)
            return 0;
    }
    return 1;
}

// Shared RK2/RK4 GPU driver. order must be 2 or 4.
static cfd_status_t solve_rk_gpu(flow_field* field, const grid* g,
                                 const ns_solver_params_t* params,
                                 const gpu_config_t* config, int order) {
    if (!field || !g || !params)
        return CFD_ERROR_INVALID;
    if (order != 1 && order != 2 && order != 4)
        return CFD_ERROR_INVALID;

    gpu_config_t cfg = config ? *config : gpu_config_default();
    size_t nx = field->nx, ny = field->ny, nz = field->nz;
    if (!gpu_should_use(&cfg, nx, ny, nz, params->max_iter))
        return CFD_ERROR;

    // Host callbacks cannot run on the device.
    if (params->source_func != NULL) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "GPU RK solver does not support host source_func callbacks; "
                      "use a CPU, OMP, or AVX2 solver");
        return CFD_ERROR_UNSUPPORTED;
    }
    // Energy-equation support (heat_source_func + thermal BC types/grid).
    {
        cfd_status_t e = gpu_check_energy_support(params, nx, ny, nz);
        if (e != CFD_SUCCESS)
            return e;
    }
    // Grid spacing must be present and non-degenerate: the kernels read a single
    // g->dx[0]/g->dy[0]/g->dz[0] and divide by it, so a NULL array or a zero
    // first entry would be a NULL dereference / division by zero (the scalar RHS
    // kernel guards each cell against near-zero dx/dy — mirror that here). This
    // must run before spacing_is_uniform() and the dx/dy reads below.
    if (!g->dx || !g->dy || (nz > 1 && !g->dz) ||
        fabs(g->dx[0]) < 1e-10 || fabs(g->dy[0]) < 1e-10 ||
        (nz > 1 && fabs(g->dz[0]) < 1e-10)) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "GPU RK solver requires non-zero grid spacing (dx/dy/dz)");
        return CFD_ERROR_INVALID;
    }
    // Uniform spacing requirement (CPU RK allows non-uniform x/y; GPU does not).
    if (!spacing_is_uniform(g->dx, nx > 0 ? nx - 1 : 0) ||
        !spacing_is_uniform(g->dy, ny > 0 ? ny - 1 : 0) ||
        (nz > 1 && !spacing_is_uniform(g->dz, nz - 1))) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "GPU RK solver requires uniform grid spacing");
        return CFD_ERROR_UNSUPPORTED;
    }

    size_t total = nx * ny * nz;
    size_t bytes = total * sizeof(double);
    size_t stride_z = (nz > 1) ? nx * ny : 0;
    int k_start = (nz > 1) ? 1 : 0;
    int k_end = (nz > 1) ? (int)(nz - 2) : 0;
    double dx = g->dx[0], dy = g->dy[0];
    double inv_2dz = (nz > 1) ? 0.5 / g->dz[0] : 0.0;
    double inv_dz2 = (nz > 1) ? 1.0 / (g->dz[0] * g->dz[0]) : 0.0;
    // Energy-kernel coefficients (uniform spacing).
    double inv_2dx = 0.5 / dx, inv_2dy = 0.5 / dy;
    double inv_dx2 = 1.0 / (dx * dx), inv_dy2 = 1.0 / (dy * dy);

    cfd_status_t status = CFD_SUCCESS;
    cudaStream_t stream = 0;

    double* d_u = 0; double* d_v = 0; double* d_w = 0; double* d_p = 0;
    double* d_rho = 0; double* d_T = 0; double* d_T_new = 0;
    double* d_u0 = 0; double* d_v0 = 0; double* d_w0 = 0; double* d_p0 = 0;
    double* d_x = 0; double* d_y = 0;
    double* d_u_bc = 0; double* d_v_bc = 0; double* d_w_bc = 0;
    double* d_k[4][4];
    for (int s = 0; s < 4; s++)
        for (int f = 0; f < 4; f++)
            d_k[s][f] = 0;

#define RK_ALLOC(ptr, nbytes)                              \
    do {                                                   \
        if (cudaMalloc(&(ptr), (nbytes)) != cudaSuccess) { \
            status = CFD_ERROR_NOMEM;                      \
            goto cleanup;                                  \
        }                                                  \
    } while (0)

    if (cudaStreamCreate(&stream) != cudaSuccess)
        return CFD_ERROR;

    RK_ALLOC(d_u, bytes);   RK_ALLOC(d_v, bytes);   RK_ALLOC(d_w, bytes);   RK_ALLOC(d_p, bytes);
    RK_ALLOC(d_rho, bytes); RK_ALLOC(d_T, bytes);   RK_ALLOC(d_T_new, bytes);
    RK_ALLOC(d_u0, bytes);  RK_ALLOC(d_v0, bytes);  RK_ALLOC(d_w0, bytes);  RK_ALLOC(d_p0, bytes);
    RK_ALLOC(d_x, nx * sizeof(double));
    RK_ALLOC(d_y, ny * sizeof(double));
    RK_ALLOC(d_u_bc, bytes); RK_ALLOC(d_v_bc, bytes); RK_ALLOC(d_w_bc, bytes);
    for (int s = 0; s < order; s++)
        for (int f = 0; f < 4; f++)
            RK_ALLOC(d_k[s][f], bytes);

    // Upload state + coordinates + initial boundary values.
    cudaMemcpyAsync(d_u, field->u, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, field->v, bytes, cudaMemcpyHostToDevice, stream);
    if (field->w) cudaMemcpyAsync(d_w, field->w, bytes, cudaMemcpyHostToDevice, stream);
    else          cudaMemsetAsync(d_w, 0, bytes, stream);
    cudaMemcpyAsync(d_p, field->p, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_rho, field->rho, bytes, cudaMemcpyHostToDevice, stream);
    if (field->T) cudaMemcpyAsync(d_T, field->T, bytes, cudaMemcpyHostToDevice, stream);
    else          cudaMemsetAsync(d_T, 0, bytes, stream);
    cudaMemcpyAsync(d_x, g->x, nx * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y, g->y, ny * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_u_bc, field->u, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v_bc, field->v, bytes, cudaMemcpyHostToDevice, stream);
    if (field->w) cudaMemcpyAsync(d_w_bc, field->w, bytes, cudaMemcpyHostToDevice, stream);
    else          cudaMemsetAsync(d_w_bc, 0, bytes, stream);
    {
        cudaError_t serr = cudaStreamSynchronize(stream);
        if (serr != cudaSuccess) {
            fprintf(stderr, "CUDA error in RK GPU solve (upload): %s\n",
                    cudaGetErrorString(serr));
            status = CFD_ERROR;
            goto cleanup;
        }
    }

    {
        int energy_on = (params->alpha > 0.0);
        double dt = params->dt;
        // Explicit Euler (order==1) caps dt at DT_CONSERVATIVE_LIMIT, matching
        // solver_explicit_euler.c. For RK2/RK4 dt_eff == dt (no behavior change).
        double dt_eff = (order == 1) ? fmin(dt, DT_CONSERVATIVE_LIMIT) : dt;
        double mu = params->mu;
        double half_dt = 0.5 * dt;

        dim3 block(cfg.block_size_x, cfg.block_size_y);
        dim3 grid_dim((unsigned)((nx - 2 + block.x - 1) / block.x),
                      (unsigned)((ny - 2 + block.y - 1) / block.y));
        int n1d = 256;
        int g1d = (int)((total + n1d - 1) / n1d);

        // 1D launch extent for the boundary-restore kernel.
        size_t max_bc_dim;
        if (nz == 1) {
            max_bc_dim = (nx > ny) ? nx : ny;
        } else {
            max_bc_dim = nx * nz;
            if (ny * nz > max_bc_dim) max_bc_dim = ny * nz;
            if (nx * ny > max_bc_dim) max_bc_dim = nx * ny;
        }
        int bc_block = 256;
        int bc_grid = (int)((max_bc_dim + bc_block - 1) / bc_block);

        for (int iter = 0; iter < params->max_iter; iter++) {
            // Save Q^n.
            cudaMemcpyAsync(d_u0, d_u, bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_v0, d_v, bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_w0, d_w, bytes, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_p0, d_p, bytes, cudaMemcpyDeviceToDevice, stream);

            for (int s = 0; s < order; s++) {
                if (s > 0) {
                    // Form Q^n + factor*k_{s-1} into the working field.
                    double factor = (order == 4) ? ((s == 3) ? dt : half_dt) : dt;
                    kernel_axpy1<<<g1d, n1d, 0, stream>>>(d_u, d_u0, d_k[s - 1][0], factor, 1, total);
                    kernel_axpy1<<<g1d, n1d, 0, stream>>>(d_v, d_v0, d_k[s - 1][1], factor, 1, total);
                    kernel_axpy1<<<g1d, n1d, 0, stream>>>(d_w, d_w0, d_k[s - 1][2], factor, 1, total);
                    kernel_axpy1<<<g1d, n1d, 0, stream>>>(d_p, d_p0, d_k[s - 1][3], factor, 0, total);
                }
                // Zero stage buffers so boundary cells (never written by the RHS
                // kernel) contribute 0 to the stage-update / combine.
                for (int f = 0; f < 4; f++)
                    cudaMemsetAsync(d_k[s][f], 0, bytes, stream);
                kernel_rk_rhs<<<grid_dim, block, 0, stream>>>(
                    d_u, d_v, d_w, d_p, d_rho, d_T,
                    d_k[s][0], d_k[s][1], d_k[s][2], d_k[s][3],
                    d_x, d_y, nx, ny, nz, stride_z, k_start, k_end,
                    dx, dy, inv_2dz, inv_dz2, mu,
                    params->beta, params->T_ref,
                    params->gravity[0], params->gravity[1], params->gravity[2],
                    params->source_amplitude_u, params->source_amplitude_v,
                    params->source_decay_rate, iter, dt_eff);
            }

            // Final update Q^{n+1}.
            if (order == 1) {
                // Forward Euler with increment clamp: Q += clamp(dt_eff*k, +/-1).
                // Velocity clamped to +/-MAX_VELOCITY (flag 1); pressure not (flag 0).
                kernel_euler_update<<<g1d, n1d, 0, stream>>>(
                    d_u, d_u0, d_k[0][0], dt_eff, 1, total);
                kernel_euler_update<<<g1d, n1d, 0, stream>>>(
                    d_v, d_v0, d_k[0][1], dt_eff, 1, total);
                kernel_euler_update<<<g1d, n1d, 0, stream>>>(
                    d_w, d_w0, d_k[0][2], dt_eff, 1, total);
                kernel_euler_update<<<g1d, n1d, 0, stream>>>(
                    d_p, d_p0, d_k[0][3], dt_eff, 0, total);
            } else {
                // RK2: dt/2*(k1+k2); RK4: dt/6*(k1+2k2+2k3+k4).
                // For RK2, stages 2,3 alias d_k[0][*] with zero weight.
                double coef = (order == 4) ? (dt / 6.0) : (dt / 2.0);
                double w1 = 1.0, w2 = (order == 4) ? 2.0 : 1.0;
                double w3 = (order == 4) ? 2.0 : 0.0, w4 = (order == 4) ? 1.0 : 0.0;
                int i3 = (order == 4) ? 2 : 0;
                int i4 = (order == 4) ? 3 : 0;
                kernel_rk_combine<<<g1d, n1d, 0, stream>>>(
                    d_u, d_u0, d_k[0][0], d_k[1][0], d_k[i3][0], d_k[i4][0],
                    w1, w2, w3, w4, coef, 1, total);
                kernel_rk_combine<<<g1d, n1d, 0, stream>>>(
                    d_v, d_v0, d_k[0][1], d_k[1][1], d_k[i3][1], d_k[i4][1],
                    w1, w2, w3, w4, coef, 1, total);
                kernel_rk_combine<<<g1d, n1d, 0, stream>>>(
                    d_w, d_w0, d_k[0][2], d_k[1][2], d_k[i3][2], d_k[i4][2],
                    w1, w2, w3, w4, coef, 1, total);
                kernel_rk_combine<<<g1d, n1d, 0, stream>>>(
                    d_p, d_p0, d_k[0][3], d_k[1][3], d_k[i3][3], d_k[i4][3],
                    w1, w2, w3, w4, coef, 0, total);
            }

            // Energy equation — advance T with the corrected velocity, then
            // enforce thermal BCs (shared kernels). Swap instead of DtoD copy.
            if (energy_on) {
                kernel_energy_step<<<grid_dim, block, 0, stream>>>(
                    d_T, d_u, d_v, d_w, d_T_new,
                    nx, ny, stride_z, k_start, k_end, params->alpha,
                    inv_2dx, inv_2dy, inv_2dz, inv_dx2, inv_dy2, inv_dz2, dt_eff);
                double* tmp_T = d_T;
                d_T = d_T_new;
                d_T_new = tmp_T;
                apply_thermal_bcs_gpu(d_T, nx, ny, nz, &params->thermal_bc, stream);
            }

            // Restore caller-set velocity boundaries (after the full step only).
            kernel_copy_velocity_boundaries<<<bc_grid, bc_block, 0, stream>>>(
                d_u, d_v, d_w, d_u_bc, d_v_bc, d_w_bc, nx, ny, nz);
        }

        // Sync the time-stepping loop before downloading; a failure here means
        // an async memcpy/memset or kernel launch failed, so skip the download
        // and report the error rather than returning partial/undefined results.
        cudaError_t serr = cudaStreamSynchronize(stream);
        if (serr != cudaSuccess) {
            fprintf(stderr, "CUDA error in RK GPU solve (time-stepping): %s\n",
                    cudaGetErrorString(serr));
            status = CFD_ERROR;
            goto cleanup;
        }

        // Download results.
        cudaMemcpyAsync(field->u, d_u, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(field->v, d_v, bytes, cudaMemcpyDeviceToHost, stream);
        if (field->w) cudaMemcpyAsync(field->w, d_w, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(field->p, d_p, bytes, cudaMemcpyDeviceToHost, stream);
        if (field->T) cudaMemcpyAsync(field->T, d_T, bytes, cudaMemcpyDeviceToHost, stream);
        serr = cudaStreamSynchronize(stream);
        if (serr != cudaSuccess) {
            fprintf(stderr, "CUDA error in RK GPU solve (download): %s\n",
                    cudaGetErrorString(serr));
            status = CFD_ERROR;
        }

        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "CUDA error in RK GPU solve: %s\n", cudaGetErrorString(kerr));
            status = CFD_ERROR;
        }
    }

cleanup:
    cudaFree(d_u);   cudaFree(d_v);   cudaFree(d_w);   cudaFree(d_p);
    cudaFree(d_rho); cudaFree(d_T);   cudaFree(d_T_new);
    cudaFree(d_u0);  cudaFree(d_v0);  cudaFree(d_w0);  cudaFree(d_p0);
    cudaFree(d_x);   cudaFree(d_y);
    cudaFree(d_u_bc); cudaFree(d_v_bc); cudaFree(d_w_bc);
    for (int s = 0; s < 4; s++)
        for (int f = 0; f < 4; f++)
            cudaFree(d_k[s][f]);
    if (stream)
        cudaStreamDestroy(stream);
    return status;

#undef RK_ALLOC
}

cfd_status_t solve_explicit_euler_method_gpu(flow_field* field, const grid* grid,
                                             const ns_solver_params_t* params,
                                             const gpu_config_t* config) {
    return solve_rk_gpu(field, grid, params, config, 1);
}

cfd_status_t solve_rk2_method_gpu(flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params, const gpu_config_t* config) {
    return solve_rk_gpu(field, grid, params, config, 2);
}

cfd_status_t solve_rk4_method_gpu(flow_field* field, const grid* grid,
                                  const ns_solver_params_t* params, const gpu_config_t* config) {
    return solve_rk_gpu(field, grid, params, config, 4);
}

}  // extern "C"
