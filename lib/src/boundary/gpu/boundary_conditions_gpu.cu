/**
 * GPU Boundary Conditions Implementation (CUDA)
 *
 * CUDA kernels and host wrapper functions for applying boundary conditions
 * to fields stored in device memory. Mirrors the host-side API for consistency.
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include "cfd/core/indexing.h"

// Block size for 1D boundary kernels
#define BC_BLOCK_SIZE 256

// ============================================================================
// CUDA Kernels - Neumann (Zero Gradient)
// ============================================================================

/**
 * Apply Neumann BC to a scalar field
 * Each thread handles one boundary point (either row or column)
 */
__global__ void kernel_bc_neumann_scalar(double* field, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries (one thread per row)
    if (idx < (int)ny) {
        field[IDX_2D(0, idx, nx)] = field[IDX_2D(1, idx, nx)];                  // Left: copy from interior
        field[IDX_2D(nx - 1, idx, nx)] = field[IDX_2D(nx - 2, idx, nx)];        // Right: copy from interior
    }

    // Top and bottom boundaries (one thread per column)
    if (idx < (int)nx) {
        field[idx] = field[IDX_2D(idx, 1, nx)];                           // Bottom: copy from interior
        field[IDX_2D(idx, ny - 1, nx)] = field[IDX_2D(idx, ny - 2, nx)];  // Top: copy from interior
    }
}

/**
 * Apply Neumann BC to velocity components (u and v)
 * Processes both components in a single kernel for efficiency
 */
__global__ void kernel_bc_neumann_velocity(double* u, double* v, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries
    if (idx < (int)ny) {
        // u component
        u[IDX_2D(0, idx, nx)] = u[IDX_2D(1, idx, nx)];
        u[IDX_2D(nx - 1, idx, nx)] = u[IDX_2D(nx - 2, idx, nx)];
        // v component
        v[IDX_2D(0, idx, nx)] = v[IDX_2D(1, idx, nx)];
        v[IDX_2D(nx - 1, idx, nx)] = v[IDX_2D(nx - 2, idx, nx)];
    }

    // Top and bottom boundaries
    if (idx < (int)nx) {
        // u component
        u[idx] = u[IDX_2D(idx, 1, nx)];
        u[IDX_2D(idx, ny - 1, nx)] = u[IDX_2D(idx, ny - 2, nx)];
        // v component
        v[idx] = v[IDX_2D(idx, 1, nx)];
        v[IDX_2D(idx, ny - 1, nx)] = v[IDX_2D(idx, ny - 2, nx)];
    }
}

// ============================================================================
// CUDA Kernels - Periodic
// ============================================================================

/**
 * Apply Periodic BC to a scalar field
 */
__global__ void kernel_bc_periodic_scalar(double* field, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries (periodic in x)
    if (idx < (int)ny) {
        field[IDX_2D(0, idx, nx)] = field[IDX_2D(nx - 2, idx, nx)];             // Left: copy from right interior
        field[IDX_2D(nx - 1, idx, nx)] = field[IDX_2D(1, idx, nx)];             // Right: copy from left interior
    }

    // Top and bottom boundaries (periodic in y)
    if (idx < (int)nx) {
        field[idx] = field[IDX_2D(idx, ny - 2, nx)];                // Bottom: copy from top interior
        field[IDX_2D(idx, ny - 1, nx)] = field[IDX_2D(idx, 1, nx)]; // Top: copy from bottom interior
    }
}

/**
 * Apply Periodic BC to velocity components
 */
__global__ void kernel_bc_periodic_velocity(double* u, double* v, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries (periodic in x)
    if (idx < (int)ny) {
        u[IDX_2D(0, idx, nx)] = u[IDX_2D(nx - 2, idx, nx)];
        u[IDX_2D(nx - 1, idx, nx)] = u[IDX_2D(1, idx, nx)];
        v[IDX_2D(0, idx, nx)] = v[IDX_2D(nx - 2, idx, nx)];
        v[IDX_2D(nx - 1, idx, nx)] = v[IDX_2D(1, idx, nx)];
    }

    // Top and bottom boundaries (periodic in y)
    if (idx < (int)nx) {
        u[idx] = u[IDX_2D(idx, ny - 2, nx)];
        u[IDX_2D(idx, ny - 1, nx)] = u[IDX_2D(idx, 1, nx)];
        v[idx] = v[IDX_2D(idx, ny - 2, nx)];
        v[IDX_2D(idx, ny - 1, nx)] = v[IDX_2D(idx, 1, nx)];
    }
}

// ============================================================================
// CUDA Kernels - Dirichlet (Fixed Value)
// ============================================================================

/**
 * Apply Dirichlet BC to a scalar field
 * Sets boundary values to specified fixed values
 */
__global__ void kernel_bc_dirichlet_scalar(double* field, size_t nx, size_t ny,
                                            double val_left, double val_right,
                                            double val_top, double val_bottom) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries (one thread per row)
    if (idx < (int)ny) {
        field[IDX_2D(0, idx, nx)] = val_left;
        field[IDX_2D(nx - 1, idx, nx)] = val_right;
    }

    // Top and bottom boundaries (one thread per column)
    if (idx < (int)nx) {
        field[idx] = val_bottom;
        field[IDX_2D(idx, ny - 1, nx)] = val_top;
    }
}

/**
 * Apply Dirichlet BC to velocity components (u and v)
 */
__global__ void kernel_bc_dirichlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                              double u_left, double u_right,
                                              double u_top, double u_bottom,
                                              double v_left, double v_right,
                                              double v_top, double v_bottom) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries
    if (idx < (int)ny) {
        u[IDX_2D(0, idx, nx)] = u_left;
        u[IDX_2D(nx - 1, idx, nx)] = u_right;
        v[IDX_2D(0, idx, nx)] = v_left;
        v[IDX_2D(nx - 1, idx, nx)] = v_right;
    }

    // Top and bottom boundaries
    if (idx < (int)nx) {
        u[idx] = u_bottom;
        u[IDX_2D(idx, ny - 1, nx)] = u_top;
        v[idx] = v_bottom;
        v[IDX_2D(idx, ny - 1, nx)] = v_top;
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

extern "C" void bc_apply_neumann_gpu(double* d_field, size_t nx, size_t ny, cudaStream_t stream) {
    if (!d_field || nx < 3 || ny < 3) {
        return;
    }

    // Launch enough threads to cover max(nx, ny)
    size_t max_dim = (nx > ny) ? nx : ny;
    int num_blocks = (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);

    kernel_bc_neumann_scalar<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_field, nx, ny);
}

extern "C" void bc_apply_scalar_gpu(double* d_field, size_t nx, size_t ny, bc_type_t type,
                                    cudaStream_t stream) {
    if (!d_field || nx < 3 || ny < 3) {
        return;
    }

    size_t max_dim = (nx > ny) ? nx : ny;
    int num_blocks = (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);

    switch (type) {
        case BC_TYPE_NEUMANN:
            kernel_bc_neumann_scalar<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_field, nx, ny);
            break;

        case BC_TYPE_PERIODIC:
            kernel_bc_periodic_scalar<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_field, nx, ny);
            break;

        case BC_TYPE_DIRICHLET:
        case BC_TYPE_NOSLIP:
        case BC_TYPE_INLET:
        case BC_TYPE_OUTLET:
        default:
            // Default to Neumann for safety (same as host implementation)
            kernel_bc_neumann_scalar<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_field, nx, ny);
            break;
    }
}

extern "C" void bc_apply_velocity_gpu(double* d_u, double* d_v, size_t nx, size_t ny, bc_type_t type,
                                      cudaStream_t stream) {
    if (!d_u || !d_v || nx < 3 || ny < 3) {
        return;
    }

    size_t max_dim = (nx > ny) ? nx : ny;
    int num_blocks = (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);

    switch (type) {
        case BC_TYPE_NEUMANN:
            kernel_bc_neumann_velocity<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_u, d_v, nx, ny);
            break;

        case BC_TYPE_PERIODIC:
            kernel_bc_periodic_velocity<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_u, d_v, nx, ny);
            break;

        case BC_TYPE_DIRICHLET:
        case BC_TYPE_NOSLIP:
        case BC_TYPE_INLET:
        case BC_TYPE_OUTLET:
        default:
            kernel_bc_neumann_velocity<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(d_u, d_v, nx, ny);
            break;
    }
}

// ============================================================================
// Dirichlet BC Wrapper Functions
// ============================================================================

extern "C" void bc_apply_dirichlet_scalar_gpu(double* d_field, size_t nx, size_t ny,
                                               const bc_dirichlet_values_t* values,
                                               cudaStream_t stream) {
    if (!d_field || !values || nx < 3 || ny < 3) {
        return;
    }

    size_t max_dim = (nx > ny) ? nx : ny;
    int num_blocks = (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);

    kernel_bc_dirichlet_scalar<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
        d_field, nx, ny,
        values->left, values->right, values->top, values->bottom);
}

extern "C" void bc_apply_dirichlet_velocity_gpu(double* d_u, double* d_v, size_t nx, size_t ny,
                                                 const bc_dirichlet_values_t* u_values,
                                                 const bc_dirichlet_values_t* v_values,
                                                 cudaStream_t stream) {
    if (!d_u || !d_v || !u_values || !v_values || nx < 3 || ny < 3) {
        return;
    }

    size_t max_dim = (nx > ny) ? nx : ny;
    int num_blocks = (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);

    kernel_bc_dirichlet_velocity<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
        d_u, d_v, nx, ny,
        u_values->left, u_values->right, u_values->top, u_values->bottom,
        v_values->left, v_values->right, v_values->top, v_values->bottom);
}

// ============================================================================
// 3D CUDA Kernels - Neumann (Zero Gradient)
// ============================================================================

/**
 * Apply Neumann BC to a 3D scalar field.
 * Boundary cells are partitioned across face groups to avoid write races:
 *   z-faces: all (i,j) pairs — owns edges/corners on z-boundary planes
 *   y-faces: all i, interior k only [1..nz-2] — no overlap with z-faces
 *   x-faces: interior j [1..ny-2] and interior k [1..nz-2] — no overlap
 * Launch with max((ny-2)*(nz-2), nx*(nz-2), nx*ny) threads.
 */
__global__ void kernel_bc_neumann_scalar_3d(double* field, size_t nx, size_t ny, size_t nz) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;
    size_t ny_int = (ny > 2) ? ny - 2 : 0;
    size_t nz_int = (nz > 2) ? nz - 2 : 0;

    // Left/right boundaries: interior j [1..ny-2], interior k [1..nz-2]
    if (nz_int > 0 && ny_int > 0 && tid < ny_int * nz_int) {
        size_t ki = tid / ny_int;
        size_t ji = tid % ny_int;
        size_t k = ki + 1;
        size_t j = ji + 1;
        size_t base = k * plane;
        field[base + IDX_2D(0, j, nx)] = field[base + IDX_2D(1, j, nx)];
        field[base + IDX_2D(nx - 1, j, nx)] = field[base + IDX_2D(nx - 2, j, nx)];
    }

    // Top/bottom boundaries: all i [0..nx-1], interior k [1..nz-2]
    if (nz_int > 0 && tid < nx * nz_int) {
        size_t ki = tid / nx;
        size_t i = tid % nx;
        size_t k = ki + 1;
        size_t base = k * plane;
        field[base + i] = field[base + IDX_2D(i, 1, nx)];
        field[base + IDX_2D(i, ny - 1, nx)] = field[base + IDX_2D(i, ny - 2, nx)];
    }

    // Z-face boundaries: all (i, j) pairs — handles edges/corners on z-planes
    if (nz > 1 && tid < plane) {
        field[tid] = field[plane + tid];                             // k=0 from k=1
        field[(nz - 1) * plane + tid] = field[(nz - 2) * plane + tid];  // k=nz-1 from k=nz-2
    }
}

/**
 * Apply Neumann BC to 3D velocity components (u, v, w).
 * Same face-partitioned threading as the scalar kernel to avoid write races.
 */
__global__ void kernel_bc_neumann_velocity_3d(double* u, double* v, double* w,
                                               size_t nx, size_t ny, size_t nz) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;
    size_t ny_int = (ny > 2) ? ny - 2 : 0;
    size_t nz_int = (nz > 2) ? nz - 2 : 0;

    // Left/right boundaries: interior j [1..ny-2], interior k [1..nz-2]
    if (nz_int > 0 && ny_int > 0 && tid < ny_int * nz_int) {
        size_t ki = tid / ny_int;
        size_t ji = tid % ny_int;
        size_t k = ki + 1;
        size_t j = ji + 1;
        size_t base = k * plane;
        size_t left = base + IDX_2D(0, j, nx);
        size_t left_int = base + IDX_2D(1, j, nx);
        size_t right = base + IDX_2D(nx - 1, j, nx);
        size_t right_int = base + IDX_2D(nx - 2, j, nx);
        u[left] = u[left_int];      u[right] = u[right_int];
        v[left] = v[left_int];      v[right] = v[right_int];
        if (w) { w[left] = w[left_int]; w[right] = w[right_int]; }
    }

    // Top/bottom boundaries: all i [0..nx-1], interior k [1..nz-2]
    if (nz_int > 0 && tid < nx * nz_int) {
        size_t ki = tid / nx;
        size_t i = tid % nx;
        size_t k = ki + 1;
        size_t base = k * plane;
        size_t bot = base + i;
        size_t bot_int = base + IDX_2D(i, 1, nx);
        size_t top = base + IDX_2D(i, ny - 1, nx);
        size_t top_int = base + IDX_2D(i, ny - 2, nx);
        u[bot] = u[bot_int];        u[top] = u[top_int];
        v[bot] = v[bot_int];        v[top] = v[top_int];
        if (w) { w[bot] = w[bot_int]; w[top] = w[top_int]; }
    }

    // Z-face boundaries: all (i, j) — handles edges/corners on z-planes
    if (nz > 1 && tid < plane) {
        size_t zbot = tid;
        size_t zbot_int = plane + tid;
        size_t ztop = (nz - 1) * plane + tid;
        size_t ztop_int = (nz - 2) * plane + tid;
        u[zbot] = u[zbot_int];      u[ztop] = u[ztop_int];
        v[zbot] = v[zbot_int];      v[ztop] = v[ztop_int];
        if (w) { w[zbot] = w[zbot_int]; w[ztop] = w[ztop_int]; }
    }
}

// ============================================================================
// 3D CUDA Kernels - Periodic
// ============================================================================

__global__ void kernel_bc_periodic_scalar_3d(double* field, size_t nx, size_t ny, size_t nz) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;
    size_t ny_int = (ny > 2) ? ny - 2 : 0;
    size_t nz_int = (nz > 2) ? nz - 2 : 0;

    // Left/right (periodic in x): interior j [1..ny-2], interior k [1..nz-2]
    if (nz_int > 0 && ny_int > 0 && tid < ny_int * nz_int) {
        size_t ki = tid / ny_int;
        size_t ji = tid % ny_int;
        size_t k = ki + 1;
        size_t j = ji + 1;
        size_t base = k * plane;
        field[base + IDX_2D(0, j, nx)] = field[base + IDX_2D(nx - 2, j, nx)];
        field[base + IDX_2D(nx - 1, j, nx)] = field[base + IDX_2D(1, j, nx)];
    }

    // Top/bottom (periodic in y): all i [0..nx-1], interior k [1..nz-2]
    if (nz_int > 0 && tid < nx * nz_int) {
        size_t ki = tid / nx;
        size_t i = tid % nx;
        size_t k = ki + 1;
        size_t base = k * plane;
        field[base + i] = field[base + IDX_2D(i, ny - 2, nx)];
        field[base + IDX_2D(i, ny - 1, nx)] = field[base + IDX_2D(i, 1, nx)];
    }

    // Z-faces (periodic in z): all (i, j) — handles edges/corners on z-planes
    if (nz > 1 && tid < plane) {
        field[tid] = field[(nz - 2) * plane + tid];
        field[(nz - 1) * plane + tid] = field[plane + tid];
    }
}

__global__ void kernel_bc_periodic_velocity_3d(double* u, double* v, double* w,
                                                size_t nx, size_t ny, size_t nz) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t plane = nx * ny;
    size_t ny_int = (ny > 2) ? ny - 2 : 0;
    size_t nz_int = (nz > 2) ? nz - 2 : 0;

    // Left/right (periodic in x): interior j [1..ny-2], interior k [1..nz-2]
    if (nz_int > 0 && ny_int > 0 && tid < ny_int * nz_int) {
        size_t ki = tid / ny_int;
        size_t ji = tid % ny_int;
        size_t k = ki + 1;
        size_t j = ji + 1;
        size_t base = k * plane;
        size_t left = base + IDX_2D(0, j, nx);
        size_t left_src = base + IDX_2D(nx - 2, j, nx);
        size_t right = base + IDX_2D(nx - 1, j, nx);
        size_t right_src = base + IDX_2D(1, j, nx);
        u[left] = u[left_src];      u[right] = u[right_src];
        v[left] = v[left_src];      v[right] = v[right_src];
        if (w) { w[left] = w[left_src]; w[right] = w[right_src]; }
    }

    // Top/bottom (periodic in y): all i [0..nx-1], interior k [1..nz-2]
    if (nz_int > 0 && tid < nx * nz_int) {
        size_t ki = tid / nx;
        size_t i = tid % nx;
        size_t k = ki + 1;
        size_t base = k * plane;
        size_t bot = base + i;
        size_t bot_src = base + IDX_2D(i, ny - 2, nx);
        size_t top = base + IDX_2D(i, ny - 1, nx);
        size_t top_src = base + IDX_2D(i, 1, nx);
        u[bot] = u[bot_src];        u[top] = u[top_src];
        v[bot] = v[bot_src];        v[top] = v[top_src];
        if (w) { w[bot] = w[bot_src]; w[top] = w[top_src]; }
    }

    // Z-faces (periodic in z): all (i, j) — handles edges/corners on z-planes
    if (nz > 1 && tid < plane) {
        size_t zbot = tid;
        size_t zbot_src = (nz - 2) * plane + tid;
        size_t ztop = (nz - 1) * plane + tid;
        size_t ztop_src = plane + tid;
        u[zbot] = u[zbot_src];      u[ztop] = u[ztop_src];
        v[zbot] = v[zbot_src];      v[ztop] = v[ztop_src];
        if (w) { w[zbot] = w[zbot_src]; w[ztop] = w[ztop_src]; }
    }
}

// ============================================================================
// 3D Host Wrapper Functions
// ============================================================================

static int bc_3d_num_blocks(size_t nx, size_t ny, size_t nz) {
    // Thread count matches the face-partitioned kernels:
    //   x-faces: (ny-2)*(nz-2), y-faces: nx*(nz-2), z-faces: nx*ny
    size_t nz_int = (nz > 2) ? nz - 2 : 0;
    size_t ny_int = (ny > 2) ? ny - 2 : 0;
    size_t max_dim = ny_int * nz_int;
    if (nx * nz_int > max_dim) max_dim = nx * nz_int;
    if (nx * ny > max_dim) max_dim = nx * ny;
    return (int)((max_dim + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE);
}

extern "C" void bc_apply_scalar_3d_gpu(double* d_field, size_t nx, size_t ny, size_t nz,
                                        bc_type_t type, cudaStream_t stream) {
    if (!d_field || nx < 3 || ny < 3) return;
    if (nz == 1) {
        bc_apply_scalar_gpu(d_field, nx, ny, type, stream);
        return;
    }
    if (nz < 3) return;

    int num_blocks = bc_3d_num_blocks(nx, ny, nz);
    switch (type) {
        case BC_TYPE_NEUMANN:
            kernel_bc_neumann_scalar_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_field, nx, ny, nz);
            break;
        case BC_TYPE_PERIODIC:
            kernel_bc_periodic_scalar_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_field, nx, ny, nz);
            break;
        default:
            kernel_bc_neumann_scalar_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_field, nx, ny, nz);
            break;
    }
}

extern "C" void bc_apply_velocity_3d_gpu(double* d_u, double* d_v, double* d_w,
                                          size_t nx, size_t ny, size_t nz,
                                          bc_type_t type, cudaStream_t stream) {
    if (!d_u || !d_v || nx < 3 || ny < 3) return;
    if (nz == 1) {
        bc_apply_velocity_gpu(d_u, d_v, nx, ny, type, stream);
        return;
    }
    if (nz < 3) return;

    int num_blocks = bc_3d_num_blocks(nx, ny, nz);
    switch (type) {
        case BC_TYPE_NEUMANN:
            kernel_bc_neumann_velocity_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, d_w, nx, ny, nz);
            break;
        case BC_TYPE_PERIODIC:
            kernel_bc_periodic_velocity_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, d_w, nx, ny, nz);
            break;
        default:
            kernel_bc_neumann_velocity_3d<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, d_w, nx, ny, nz);
            break;
    }
}
