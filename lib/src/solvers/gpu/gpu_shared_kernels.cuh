/**
 * @file gpu_shared_kernels.cuh
 * @brief Shared CUDA device/host helpers used by more than one GPU NS solver.
 *
 * Single source of truth for the GPU energy step, per-face thermal boundary
 * conditions, caller-set velocity-BC restoration, the CUDA error-check macros,
 * and the energy-parameter support check. Both the projection
 * (`solver_projection_jacobi_gpu.cu`) and Runge-Kutta (`solver_rk_gpu.cu`)
 * backends `#include` this header so the energy/thermal numerics cannot drift.
 *
 * Like the SIMD template headers (lib/src/solvers/linear/simd_template/), the
 * kernels and helpers are declared `static` so each including translation unit
 * gets its own internal-linkage copy — no duplicate-symbol link errors. This
 * header is #include-only and is never compiled as a standalone TU.
 */
#ifndef CFD_GPU_SHARED_KERNELS_CUH
#define CFD_GPU_SHARED_KERNELS_CUH

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/cfd_status.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            return CFD_ERROR;                                                \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            return;                                                          \
        }                                                                    \
    } while (0)

// ============================================================================
// Shared CUDA kernels
// ============================================================================

// Energy equation: explicit-Euler advection-diffusion of temperature.
// dT/dt + u*grad(T) = alpha*lap(T). Same central-difference numerics as the
// scalar/AVX2 reference (energy_solver_avx2.c). Interior only; boundary values
// are written afterwards by the thermal-BC kernels. Branch-free 3D: stride_z=0
// and inv_2dz=inv_dz2=0 collapse the z-terms when nz==1.
static __global__ void kernel_energy_step(const double* __restrict__ T,
                                          const double* __restrict__ u,
                                          const double* __restrict__ v,
                                          const double* __restrict__ w,
                                          double* __restrict__ T_new,
                                          size_t nx, size_t ny,
                                          size_t stride_z, int k_start, int k_end,
                                          double alpha,
                                          double inv_2dx, double inv_2dy, double inv_2dz,
                                          double inv_dx2, double inv_dy2, double inv_dz2,
                                          double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        for (int k = k_start; k <= k_end; k++) {
            size_t idx = k * stride_z + IDX_2D(i, j, nx);
            double T_c = T[idx];

            double dT_dx = (T[idx + 1] - T[idx - 1]) * inv_2dx;
            double dT_dy = (T[idx + nx] - T[idx - nx]) * inv_2dy;
            double dT_dz = (T[idx + stride_z] - T[idx - stride_z]) * inv_2dz;
            double adv = u[idx] * dT_dx + v[idx] * dT_dy + w[idx] * dT_dz;

            double d2x = (T[idx + 1] - 2.0 * T_c + T[idx - 1]) * inv_dx2;
            double d2y = (T[idx + nx] - 2.0 * T_c + T[idx - nx]) * inv_dy2;
            double d2z = (T[idx + stride_z] - 2.0 * T_c + T[idx - stride_z]) * inv_dz2;
            double diff = alpha * (d2x + d2y + d2z);

            T_new[idx] = T_c + dt * (diff - adv);
        }
    }
}

// Per-face thermal boundary conditions, mirroring energy_apply_thermal_bcs
// (energy/cpu/energy_solver.c). Each kernel writes one boundary face; the host
// launches them sequentially in the order left, right, bottom, top, back, front
// so shared corners/edges resolve with last-applied-wins precedence, matching
// the scalar reference. type uses bc_type_t (PERIODIC/NEUMANN/DIRICHLET).

// x-faces: one thread per (j,k). is_right selects i=nx-1 vs i=0.
static __global__ void kernel_thermal_xface(double* __restrict__ T, size_t nx, size_t ny, size_t nz,
                                            size_t plane, int is_right, int type, double value) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ny * nz) return;
    size_t k = tid / ny;
    size_t j = tid % ny;
    size_t base = k * plane;
    if (!is_right) {
        size_t idx = base + j * nx;  // i=0
        if (type == BC_TYPE_DIRICHLET) T[idx] = value;
        else if (type == BC_TYPE_NEUMANN) T[idx] = T[idx + 1];
        else if (type == BC_TYPE_PERIODIC) T[idx] = T[base + j * nx + (nx - 2)];
    } else {
        size_t idx = base + j * nx + (nx - 1);  // i=nx-1
        if (type == BC_TYPE_DIRICHLET) T[idx] = value;
        else if (type == BC_TYPE_NEUMANN) T[idx] = T[idx - 1];
        else if (type == BC_TYPE_PERIODIC) T[idx] = T[base + j * nx + 1];
    }
}

// y-faces: one thread per (i,k). is_top selects j=ny-1 vs j=0.
static __global__ void kernel_thermal_yface(double* __restrict__ T, size_t nx, size_t ny, size_t nz,
                                            size_t plane, int is_top, int type, double value) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * nz) return;
    size_t k = tid / nx;
    size_t i = tid % nx;
    size_t base = k * plane;
    if (!is_top) {
        size_t idx = base + i;  // j=0
        if (type == BC_TYPE_DIRICHLET) T[idx] = value;
        else if (type == BC_TYPE_NEUMANN) T[idx] = T[idx + nx];
        else if (type == BC_TYPE_PERIODIC) T[idx] = T[base + (ny - 2) * nx + i];
    } else {
        size_t idx = base + (ny - 1) * nx + i;  // j=ny-1
        if (type == BC_TYPE_DIRICHLET) T[idx] = value;
        else if (type == BC_TYPE_NEUMANN) T[idx] = T[idx - nx];
        else if (type == BC_TYPE_PERIODIC) T[idx] = T[base + nx + i];
    }
}

// z-faces (3D only): one thread per (i,j). is_front selects k=nz-1 vs k=0.
static __global__ void kernel_thermal_zface(double* __restrict__ T, size_t nx, size_t ny, size_t nz,
                                            size_t plane, int is_front, int type, double value) {
    size_t off = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (off >= plane) return;
    if (!is_front) {
        size_t idx = off;  // k=0 (back)
        if (type == BC_TYPE_DIRICHLET) T[idx] = value;
        else if (type == BC_TYPE_NEUMANN) T[idx] = T[plane + off];
        else if (type == BC_TYPE_PERIODIC) T[idx] = T[(nz - 2) * plane + off];
    } else {
        size_t fb = (nz - 1) * plane;  // k=nz-1 (front)
        if (type == BC_TYPE_DIRICHLET) T[fb + off] = value;
        else if (type == BC_TYPE_NEUMANN) T[fb + off] = T[(nz - 2) * plane + off];
        else if (type == BC_TYPE_PERIODIC) T[fb + off] = T[plane + off];
    }
}

// Copy boundary values from stored BC arrays to velocity arrays
// This preserves caller-set boundary conditions (e.g., Dirichlet for lid-driven cavity)
// 2D (nz==1): top/bottom and left/right boundaries in k=0 plane
// 3D (nz>1): face-partitioned to avoid corner/edge write races (z > y > x priority)
static __global__ void kernel_copy_velocity_boundaries(double* __restrict__ u,
                                                       double* __restrict__ v,
                                                       double* __restrict__ w,
                                                       const double* __restrict__ u_bc,
                                                       const double* __restrict__ v_bc,
                                                       const double* __restrict__ w_bc,
                                                       size_t nx, size_t ny, size_t nz) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;

    if (nz == 1) {
        // 2D: top/bottom (j=0, j=ny-1) — one thread per column
        if (tid < nx) {
            size_t i = tid;
            size_t bot = i;
            size_t top = IDX_2D(i, ny - 1, nx);
            u[bot] = u_bc[bot]; u[top] = u_bc[top];
            v[bot] = v_bc[bot]; v[top] = v_bc[top];
        }
        // 2D: left/right (i=0, i=nx-1) — one thread per row
        if (tid < ny) {
            size_t j = tid;
            size_t left = IDX_2D(0, j, nx);
            size_t right = IDX_2D(nx - 1, j, nx);
            u[left] = u_bc[left]; u[right] = u_bc[right];
            v[left] = v_bc[left]; v[right] = v_bc[right];
        }
    } else {
        size_t ny_int = (ny > 2) ? ny - 2 : 0;
        size_t nz_int = (nz > 2) ? nz - 2 : 0;

        // y-faces (top/bottom): all i [0..nx-1], interior k [1..nz-2]
        if (nz_int > 0 && tid < nx * nz_int) {
            size_t ki = tid / nx;
            size_t i = tid % nx;
            size_t k = ki + 1;
            size_t base = k * plane;
            size_t bot = base + i;
            size_t top = base + IDX_2D(i, ny - 1, nx);
            u[bot] = u_bc[bot]; u[top] = u_bc[top];
            v[bot] = v_bc[bot]; v[top] = v_bc[top];
            if (w && w_bc) { w[bot] = w_bc[bot]; w[top] = w_bc[top]; }
        }

        // x-faces (left/right): interior j [1..ny-2], interior k [1..nz-2]
        if (nz_int > 0 && ny_int > 0 && tid < ny_int * nz_int) {
            size_t ki = tid / ny_int;
            size_t ji = tid % ny_int;
            size_t k = ki + 1;
            size_t j = ji + 1;
            size_t base = k * plane;
            size_t left = base + IDX_2D(0, j, nx);
            size_t right = base + IDX_2D(nx - 1, j, nx);
            u[left] = u_bc[left]; u[right] = u_bc[right];
            v[left] = v_bc[left]; v[right] = v_bc[right];
            if (w && w_bc) { w[left] = w_bc[left]; w[right] = w_bc[right]; }
        }

        // z-faces (k=0, k=nz-1): all (i, j) — owns edges/corners
        if (tid < plane) {
            size_t zbot = tid;
            size_t ztop = (nz - 1) * plane + tid;
            u[zbot] = u_bc[zbot]; u[ztop] = u_bc[ztop];
            v[zbot] = v_bc[zbot]; v[ztop] = v_bc[ztop];
            if (w && w_bc) { w[zbot] = w_bc[zbot]; w[ztop] = w_bc[ztop]; }
        }
    }
}

// ============================================================================
// Shared host helpers
// ============================================================================

// Only PERIODIC/NEUMANN/DIRICHLET are valid thermal BC types (mirrors the
// scalar is_supported_thermal_bc guard in energy/cpu/energy_solver.c).
static int thermal_bc_type_ok(bc_type_t t) {
    return t == BC_TYPE_PERIODIC || t == BC_TYPE_NEUMANN || t == BC_TYPE_DIRICHLET;
}

// Apply per-face thermal BCs to d_T, replicating energy_apply_thermal_bcs:
// faces are launched in order left, right, bottom, top, back, front on a single
// stream so shared corners/edges resolve with last-applied-wins precedence.
static void apply_thermal_bcs_gpu(double* d_T, size_t nx, size_t ny, size_t nz,
                                  const ns_thermal_bc_config_t* tbc, cudaStream_t stream) {
    size_t plane = nx * ny;
    int blk = 256;
    int g_x = (int)((ny * nz + blk - 1) / blk);
    int g_y = (int)((nx * nz + blk - 1) / blk);
    int g_z = (int)((plane + blk - 1) / blk);
    const bc_dirichlet_values_t* dv = &tbc->dirichlet_values;
    kernel_thermal_xface<<<g_x, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 0, (int)tbc->left,   dv->left);
    kernel_thermal_xface<<<g_x, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 1, (int)tbc->right,  dv->right);
    kernel_thermal_yface<<<g_y, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 0, (int)tbc->bottom, dv->bottom);
    kernel_thermal_yface<<<g_y, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 1, (int)tbc->top,    dv->top);
    if (nz > 1) {
        kernel_thermal_zface<<<g_z, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 0, (int)tbc->back,  dv->back);
        kernel_thermal_zface<<<g_z, blk, 0, stream>>>(d_T, nx, ny, nz, plane, 1, (int)tbc->front, dv->front);
    }
}

// Validate that the energy-equation parameters are runnable on the GPU. Host
// heat-source callbacks cannot execute on the device, only PERIODIC/NEUMANN/
// DIRICHLET thermal BCs are implemented, and a requested BC must fit the grid
// (Neumann reads one adjacent interior cell; Periodic wraps via nx-2/ny-2/nz-2).
// Mirrors the guards in solve_projection_method_gpu and the scalar energy path.
// Returns CFD_SUCCESS when energy is disabled (alpha<=0) or fully supported.
static cfd_status_t gpu_check_energy_support(const ns_solver_params_t* params,
                                             size_t nx, size_t ny, size_t nz) {
    if (params->alpha <= 0.0) {
        return CFD_SUCCESS;
    }
    if (params->heat_source_func != NULL) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
                      "GPU energy solver does not support host heat_source_func callbacks; "
                      "use a CPU, OMP, or AVX2 solver");
        return CFD_ERROR_UNSUPPORTED;
    }
    const ns_thermal_bc_config_t* tbc = &params->thermal_bc;
    int ok = thermal_bc_type_ok(tbc->left) && thermal_bc_type_ok(tbc->right) &&
             thermal_bc_type_ok(tbc->bottom) && thermal_bc_type_ok(tbc->top);
    if (nz > 1)
        ok = ok && thermal_bc_type_ok(tbc->front) && thermal_bc_type_ok(tbc->back);
    if (!ok) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "GPU energy: unsupported thermal BC type on a face");
        return CFD_ERROR_INVALID;
    }
    if (((tbc->left == BC_TYPE_NEUMANN || tbc->right == BC_TYPE_NEUMANN) && nx < 2) ||
        ((tbc->bottom == BC_TYPE_NEUMANN || tbc->top == BC_TYPE_NEUMANN) && ny < 2) ||
        ((tbc->left == BC_TYPE_PERIODIC || tbc->right == BC_TYPE_PERIODIC) && nx < 3) ||
        ((tbc->bottom == BC_TYPE_PERIODIC || tbc->top == BC_TYPE_PERIODIC) && ny < 3) ||
        (nz > 1 && (tbc->back == BC_TYPE_NEUMANN || tbc->front == BC_TYPE_NEUMANN) && nz < 2) ||
        (nz > 1 && (tbc->back == BC_TYPE_PERIODIC || tbc->front == BC_TYPE_PERIODIC) && nz < 3)) {
        cfd_set_error(CFD_ERROR_INVALID,
                      "GPU energy: grid too small for the requested thermal BC type");
        return CFD_ERROR_INVALID;
    }
    return CFD_SUCCESS;
}

#endif  /* CFD_GPU_SHARED_KERNELS_CUH */
