/**
 * Boundary Conditions - Scalar (CPU) Implementation
 *
 * Baseline scalar implementations of boundary conditions.
 * No SIMD, no OpenMP - pure C loops.
 *
 * Neumann, Periodic, and Dirichlet are generated from the shared template.
 * Inlet is in boundary_conditions_inlet_scalar.c.
 * Outlet is in boundary_conditions_outlet_scalar.c.
 */

#include "cfd/core/indexing.h"

#define BC_CORE_FUNC_PREFIX scalar
#define BC_CORE_USE_OMP 0
#include "../boundary_conditions_core_impl.h"

cfd_status_t bc_apply_symmetry_scalar_impl(double* u, double* v, double* w,
                                            size_t nx, size_t ny,
                                            size_t nz, size_t stride_z,
                                            const bc_symmetry_config_t* config) {
    if (!u || !v || !config) {
        return CFD_ERROR_INVALID;
    }
    if (nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    size_t j, i, k;
    bc_edge_t edges = config->edges;

    /* Left boundary (X-symmetry plane at x=0):
     * - u = 0 (normal velocity is zero)
     * - dv/dx = 0 (tangential gradient is zero, copy from interior)
     * - dw/dx = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_LEFT) {
        for (k = 0; k < nz; k++) {
            size_t base = k * stride_z;
            for (j = 0; j < ny; j++) {
                u[base + IDX_2D(0, j, nx)] = 0.0;
                v[base + IDX_2D(0, j, nx)] = v[base + IDX_2D(1, j, nx)];
            }
            if (w) {
                for (j = 0; j < ny; j++) {
                    w[base + IDX_2D(0, j, nx)] = w[base + IDX_2D(1, j, nx)];
                }
            }
        }
    }

    /* Right boundary (X-symmetry plane at x=Lx):
     * - u = 0 (normal velocity is zero)
     * - dv/dx = 0 (tangential gradient is zero, copy from interior)
     * - dw/dx = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_RIGHT) {
        for (k = 0; k < nz; k++) {
            size_t base = k * stride_z;
            for (j = 0; j < ny; j++) {
                u[base + IDX_2D(nx - 1, j, nx)] = 0.0;
                v[base + IDX_2D(nx - 1, j, nx)] = v[base + IDX_2D(nx - 2, j, nx)];
            }
            if (w) {
                for (j = 0; j < ny; j++) {
                    w[base + IDX_2D(nx - 1, j, nx)] = w[base + IDX_2D(nx - 2, j, nx)];
                }
            }
        }
    }

    /* Bottom boundary (Y-symmetry plane at y=0):
     * - v = 0 (normal velocity is zero)
     * - du/dy = 0 (tangential gradient is zero, copy from interior)
     * - dw/dy = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_BOTTOM) {
        for (k = 0; k < nz; k++) {
            size_t base = k * stride_z;
            for (i = 0; i < nx; i++) {
                v[base + i] = 0.0;
                u[base + i] = u[base + nx + i];
            }
            if (w) {
                for (i = 0; i < nx; i++) {
                    w[base + i] = w[base + nx + i];
                }
            }
        }
    }

    /* Top boundary (Y-symmetry plane at y=Ly):
     * - v = 0 (normal velocity is zero)
     * - du/dy = 0 (tangential gradient is zero, copy from interior)
     * - dw/dy = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_TOP) {
        for (k = 0; k < nz; k++) {
            size_t base = k * stride_z;
            double* u_top = u + base + ((ny - 1) * nx);
            double* v_top = v + base + ((ny - 1) * nx);
            double* u_interior = u + base + ((ny - 2) * nx);
            for (i = 0; i < nx; i++) {
                v_top[i] = 0.0;
                u_top[i] = u_interior[i];
            }
            if (w) {
                double* w_top = w + base + ((ny - 1) * nx);
                double* w_interior = w + base + ((ny - 2) * nx);
                for (i = 0; i < nx; i++) {
                    w_top[i] = w_interior[i];
                }
            }
        }
    }

    /* Back boundary (Z-symmetry plane at z=0):
     * - w = 0 (normal velocity is zero, only when w is provided)
     * - du/dz = 0 (tangential gradient is zero, copy from interior)
     * - dv/dz = 0 (tangential gradient is zero, copy from interior) */
    if ((edges & BC_EDGE_BACK) && nz > 1) {
        size_t plane_size = nx * ny;
        for (i = 0; i < plane_size; i++) {
            if (w) {
                w[i] = 0.0;
            }
            u[i] = u[stride_z + i];
            v[i] = v[stride_z + i];
        }
    }

    /* Front boundary (Z-symmetry plane at z=Lz):
     * - w = 0 (normal velocity is zero, only when w is provided)
     * - du/dz = 0 (tangential gradient is zero, copy from interior)
     * - dv/dz = 0 (tangential gradient is zero, copy from interior) */
    if ((edges & BC_EDGE_FRONT) && nz > 1) {
        size_t plane_size = nx * ny;
        size_t front = (nz - 1) * stride_z;
        size_t interior = (nz - 2) * stride_z;
        for (i = 0; i < plane_size; i++) {
            if (w) {
                w[front + i] = 0.0;
            }
            u[front + i] = u[interior + i];
            v[front + i] = v[interior + i];
        }
    }

    return CFD_SUCCESS;
}

/* Scalar backend implementation table
 * Note: bc_apply_inlet_scalar_impl is defined in boundary_conditions_inlet_scalar.c
 * Note: bc_apply_outlet_scalar_impl is defined in boundary_conditions_outlet_scalar.c */
const bc_backend_impl_t bc_impl_scalar = {
    .apply_neumann = bc_apply_neumann_scalar_impl,
    .apply_periodic = bc_apply_periodic_scalar_impl,
    .apply_dirichlet = bc_apply_dirichlet_scalar_impl,
    .apply_inlet = bc_apply_inlet_scalar_impl,
    .apply_outlet = bc_apply_outlet_scalar_impl,
    .apply_symmetry = bc_apply_symmetry_scalar_impl
};
