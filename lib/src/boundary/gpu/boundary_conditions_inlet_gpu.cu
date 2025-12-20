/**
 * Inlet Boundary Conditions - GPU (CUDA) Implementation
 *
 * CUDA kernels for applying inlet velocity boundary conditions to fields
 * stored in device memory.
 *
 * Supports:
 * - Uniform velocity profile
 * - Parabolic profile (fully-developed flow)
 * - Velocity specification by components or magnitude+direction
 *
 * Note: Custom profile callbacks are not supported on GPU as they would
 * require device function pointers. Use bc_apply_inlet() on host for
 * custom profiles.
 */

#include "cfd/boundary/boundary_conditions_gpu.cuh"
#include <math.h>

#define BC_BLOCK_SIZE 256

// ============================================================================
// CUDA Device Functions - Inlet Velocity Computation
// ============================================================================

/**
 * Compute velocity from inlet configuration at given normalized position.
 * Device function callable from kernels.
 */
__device__ void inlet_compute_velocity_gpu(bc_edge_t edge, bc_inlet_profile_t profile,
                                            bc_inlet_spec_type_t spec_type,
                                            double u_spec, double v_spec,
                                            double magnitude, double direction,
                                            double position,
                                            double* u_out, double* v_out) {
    double u_base = 0.0, v_base = 0.0;

    /* Get base velocity components from specification type */
    switch (spec_type) {
        case BC_INLET_SPEC_VELOCITY:
            u_base = u_spec;
            v_base = v_spec;
            break;

        case BC_INLET_SPEC_MAGNITUDE_DIR:
            u_base = magnitude * cos(direction);
            v_base = magnitude * sin(direction);
            break;

        case BC_INLET_SPEC_MASS_FLOW:
            /* For GPU, mass flow is pre-computed to avg_velocity on host
             * and passed via u_spec/v_spec based on edge */
            u_base = u_spec;
            v_base = v_spec;
            break;

        default:
            u_base = 0.0;
            v_base = 0.0;
            break;
    }

    /* Apply profile shape */
    switch (profile) {
        case BC_INLET_PROFILE_UNIFORM:
            *u_out = u_base;
            *v_out = v_base;
            break;

        case BC_INLET_PROFILE_PARABOLIC: {
            double profile_factor = 4.0 * position * (1.0 - position);
            *u_out = u_base * profile_factor;
            *v_out = v_base * profile_factor;
            break;
        }

        case BC_INLET_PROFILE_CUSTOM:
            /* Custom profiles not supported on GPU - use uniform fallback */
            *u_out = u_base;
            *v_out = v_base;
            break;

        default:
            *u_out = u_base;
            *v_out = v_base;
            break;
    }
}

// ============================================================================
// CUDA Kernels - Inlet Boundary Conditions
// ============================================================================

/**
 * Apply inlet BC to left boundary (column 0)
 * One thread per row
 */
__global__ void kernel_bc_inlet_left(double* u, double* v, size_t nx, size_t ny,
                                      bc_inlet_profile_t profile,
                                      bc_inlet_spec_type_t spec_type,
                                      double u_spec, double v_spec,
                                      double magnitude, double direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (int)ny) {
        double position = (ny > 1) ? (double)idx / (double)(ny - 1) : 0.5;
        double u_val, v_val;
        inlet_compute_velocity_gpu(BC_EDGE_LEFT, profile, spec_type,
                                    u_spec, v_spec, magnitude, direction,
                                    position, &u_val, &v_val);
        u[idx * nx] = u_val;
        v[idx * nx] = v_val;
    }
}

/**
 * Apply inlet BC to right boundary (column nx-1)
 * One thread per row
 */
__global__ void kernel_bc_inlet_right(double* u, double* v, size_t nx, size_t ny,
                                       bc_inlet_profile_t profile,
                                       bc_inlet_spec_type_t spec_type,
                                       double u_spec, double v_spec,
                                       double magnitude, double direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (int)ny) {
        double position = (ny > 1) ? (double)idx / (double)(ny - 1) : 0.5;
        double u_val, v_val;
        inlet_compute_velocity_gpu(BC_EDGE_RIGHT, profile, spec_type,
                                    u_spec, v_spec, magnitude, direction,
                                    position, &u_val, &v_val);
        u[idx * nx + (nx - 1)] = u_val;
        v[idx * nx + (nx - 1)] = v_val;
    }
}

/**
 * Apply inlet BC to bottom boundary (row 0)
 * One thread per column
 */
__global__ void kernel_bc_inlet_bottom(double* u, double* v, size_t nx, size_t ny,
                                        bc_inlet_profile_t profile,
                                        bc_inlet_spec_type_t spec_type,
                                        double u_spec, double v_spec,
                                        double magnitude, double direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (int)nx) {
        double position = (nx > 1) ? (double)idx / (double)(nx - 1) : 0.5;
        double u_val, v_val;
        inlet_compute_velocity_gpu(BC_EDGE_BOTTOM, profile, spec_type,
                                    u_spec, v_spec, magnitude, direction,
                                    position, &u_val, &v_val);
        u[idx] = u_val;
        v[idx] = v_val;
    }
}

/**
 * Apply inlet BC to top boundary (row ny-1)
 * One thread per column
 */
__global__ void kernel_bc_inlet_top(double* u, double* v, size_t nx, size_t ny,
                                     bc_inlet_profile_t profile,
                                     bc_inlet_spec_type_t spec_type,
                                     double u_spec, double v_spec,
                                     double magnitude, double direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (int)nx) {
        double position = (nx > 1) ? (double)idx / (double)(nx - 1) : 0.5;
        double u_val, v_val;
        inlet_compute_velocity_gpu(BC_EDGE_TOP, profile, spec_type,
                                    u_spec, v_spec, magnitude, direction,
                                    position, &u_val, &v_val);
        u[(ny - 1) * nx + idx] = u_val;
        v[(ny - 1) * nx + idx] = v_val;
    }
}

// ============================================================================
// Host Wrapper Function
// ============================================================================

extern "C" cfd_status_t bc_apply_inlet_gpu(double* d_u, double* d_v, size_t nx, size_t ny,
                                            const bc_inlet_config_t* config,
                                            cudaStream_t stream) {
    if (!d_u || !d_v || !config || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    /* Custom profiles not supported on GPU */
    if (config->profile == BC_INLET_PROFILE_CUSTOM) {
        return CFD_ERROR_UNSUPPORTED;
    }

    /* Extract configuration for kernel */
    double u_spec = 0.0, v_spec = 0.0;
    double magnitude = 0.0, direction = 0.0;

    switch (config->spec_type) {
        case BC_INLET_SPEC_VELOCITY:
            u_spec = config->spec.velocity.u;
            v_spec = config->spec.velocity.v;
            break;

        case BC_INLET_SPEC_MAGNITUDE_DIR:
            magnitude = config->spec.magnitude_dir.magnitude;
            direction = config->spec.magnitude_dir.direction;
            break;

        case BC_INLET_SPEC_MASS_FLOW: {
            /* Pre-compute velocity from mass flow rate */
            double avg_velocity = config->spec.mass_flow.mass_flow_rate /
                                   (config->spec.mass_flow.density * config->spec.mass_flow.inlet_length);
            switch (config->edge) {
                case BC_EDGE_LEFT:
                    u_spec = avg_velocity;
                    v_spec = 0.0;
                    break;
                case BC_EDGE_RIGHT:
                    u_spec = -avg_velocity;
                    v_spec = 0.0;
                    break;
                case BC_EDGE_BOTTOM:
                    u_spec = 0.0;
                    v_spec = avg_velocity;
                    break;
                case BC_EDGE_TOP:
                    u_spec = 0.0;
                    v_spec = -avg_velocity;
                    break;
                default:
                    u_spec = avg_velocity;
                    v_spec = 0.0;
                    break;
            }
            break;
        }

        default:
            return CFD_ERROR_INVALID;
    }

    /* Launch appropriate kernel based on edge */
    int num_threads;
    int num_blocks;

    switch (config->edge) {
        case BC_EDGE_LEFT:
            num_threads = (int)ny;
            num_blocks = (num_threads + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE;
            kernel_bc_inlet_left<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, nx, ny, config->profile, config->spec_type,
                u_spec, v_spec, magnitude, direction);
            break;

        case BC_EDGE_RIGHT:
            num_threads = (int)ny;
            num_blocks = (num_threads + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE;
            kernel_bc_inlet_right<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, nx, ny, config->profile, config->spec_type,
                u_spec, v_spec, magnitude, direction);
            break;

        case BC_EDGE_BOTTOM:
            num_threads = (int)nx;
            num_blocks = (num_threads + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE;
            kernel_bc_inlet_bottom<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, nx, ny, config->profile, config->spec_type,
                u_spec, v_spec, magnitude, direction);
            break;

        case BC_EDGE_TOP:
            num_threads = (int)nx;
            num_blocks = (num_threads + BC_BLOCK_SIZE - 1) / BC_BLOCK_SIZE;
            kernel_bc_inlet_top<<<num_blocks, BC_BLOCK_SIZE, 0, stream>>>(
                d_u, d_v, nx, ny, config->profile, config->spec_type,
                u_spec, v_spec, magnitude, direction);
            break;

        default:
            return CFD_ERROR_INVALID;
    }

    /* Check for kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return CFD_ERROR;
    }

    return CFD_SUCCESS;
}
