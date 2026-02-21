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
