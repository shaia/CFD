/**
 * GPU Boundary Conditions Implementation (CUDA)
 *
 * CUDA kernels and host wrapper functions for applying boundary conditions
 * to fields stored in device memory. Mirrors the host-side API for consistency.
 */

#include "cfd/core/boundary_conditions_gpu.cuh"

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
        field[idx * nx] = field[idx * nx + 1];                  // Left: copy from interior
        field[idx * nx + nx - 1] = field[idx * nx + nx - 2];    // Right: copy from interior
    }

    // Top and bottom boundaries (one thread per column)
    if (idx < (int)nx) {
        field[idx] = field[nx + idx];                           // Bottom: copy from interior
        field[(ny - 1) * nx + idx] = field[(ny - 2) * nx + idx];  // Top: copy from interior
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
        u[idx * nx] = u[idx * nx + 1];
        u[idx * nx + nx - 1] = u[idx * nx + nx - 2];
        // v component
        v[idx * nx] = v[idx * nx + 1];
        v[idx * nx + nx - 1] = v[idx * nx + nx - 2];
    }

    // Top and bottom boundaries
    if (idx < (int)nx) {
        // u component
        u[idx] = u[nx + idx];
        u[(ny - 1) * nx + idx] = u[(ny - 2) * nx + idx];
        // v component
        v[idx] = v[nx + idx];
        v[(ny - 1) * nx + idx] = v[(ny - 2) * nx + idx];
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
        field[idx * nx] = field[idx * nx + nx - 2];             // Left: copy from right interior
        field[idx * nx + nx - 1] = field[idx * nx + 1];         // Right: copy from left interior
    }

    // Top and bottom boundaries (periodic in y)
    if (idx < (int)nx) {
        field[idx] = field[(ny - 2) * nx + idx];                // Bottom: copy from top interior
        field[(ny - 1) * nx + idx] = field[nx + idx];           // Top: copy from bottom interior
    }
}

/**
 * Apply Periodic BC to velocity components
 */
__global__ void kernel_bc_periodic_velocity(double* u, double* v, size_t nx, size_t ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Left and right boundaries (periodic in x)
    if (idx < (int)ny) {
        u[idx * nx] = u[idx * nx + nx - 2];
        u[idx * nx + nx - 1] = u[idx * nx + 1];
        v[idx * nx] = v[idx * nx + nx - 2];
        v[idx * nx + nx - 1] = v[idx * nx + 1];
    }

    // Top and bottom boundaries (periodic in y)
    if (idx < (int)nx) {
        u[idx] = u[(ny - 2) * nx + idx];
        u[(ny - 1) * nx + idx] = u[nx + idx];
        v[idx] = v[(ny - 2) * nx + idx];
        v[(ny - 1) * nx + idx] = v[nx + idx];
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
