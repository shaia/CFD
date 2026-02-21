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

cfd_status_t bc_apply_symmetry_scalar_impl(double* u, double* v, size_t nx, size_t ny,
                                            const bc_symmetry_config_t* config) {
    if (!u || !v || !config) {
        return CFD_ERROR_INVALID;
    }
    if (nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    size_t j, i;
    bc_edge_t edges = config->edges;

    /* Left boundary (X-symmetry plane at x=0):
     * - u = 0 (normal velocity is zero)
     * - dv/dx = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_LEFT) {
        for (j = 0; j < ny; j++) {
            u[IDX_2D(0, j, nx)] = 0.0;
            v[IDX_2D(0, j, nx)] = v[IDX_2D(1, j, nx)];
        }
    }

    /* Right boundary (X-symmetry plane at x=Lx):
     * - u = 0 (normal velocity is zero)
     * - dv/dx = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_RIGHT) {
        for (j = 0; j < ny; j++) {
            u[IDX_2D(nx - 1, j, nx)] = 0.0;
            v[IDX_2D(nx - 1, j, nx)] = v[IDX_2D(nx - 2, j, nx)];
        }
    }

    /* Bottom boundary (Y-symmetry plane at y=0):
     * - v = 0 (normal velocity is zero)
     * - du/dy = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_BOTTOM) {
        for (i = 0; i < nx; i++) {
            v[i] = 0.0;
            u[i] = u[nx + i];
        }
    }

    /* Top boundary (Y-symmetry plane at y=Ly):
     * - v = 0 (normal velocity is zero)
     * - du/dy = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_TOP) {
        double* u_top = u + (ny - 1) * nx;
        double* v_top = v + (ny - 1) * nx;
        double* u_interior = u + (ny - 2) * nx;
        for (i = 0; i < nx; i++) {
            v_top[i] = 0.0;
            u_top[i] = u_interior[i];
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
