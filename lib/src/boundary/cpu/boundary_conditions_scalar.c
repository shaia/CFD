/**
 * Boundary Conditions - Scalar (CPU) Implementation
 *
 * Baseline scalar implementations of boundary conditions.
 * No SIMD, no OpenMP - pure C loops.
 *
 * Inlet boundary conditions are implemented in boundary_conditions_inlet_scalar.c
 */

#include "../boundary_conditions_internal.h"

/* ============================================================================
 * Scalar Implementations
 * ============================================================================ */

void bc_apply_neumann_scalar_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + 1];
        field[(j * nx) + nx - 1] = field[(j * nx) + nx - 2];
    }

    /* Top and bottom boundaries */
    double* bottom_dst = field;
    double* bottom_src = field + nx;
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + ((ny - 2) * nx);

    for (i = 0; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

void bc_apply_periodic_scalar_impl(double* field, size_t nx, size_t ny) {
    size_t j, i;

    /* Left and right boundaries (periodic in x) */
    for (j = 0; j < ny; j++) {
        field[(j * nx) + 0] = field[(j * nx) + nx - 2];
        field[(j * nx) + nx - 1] = field[(j * nx) + 1];
    }

    /* Top and bottom boundaries (periodic in y) */
    double* bottom_dst = field;
    double* bottom_src = field + ((ny - 2) * nx);
    double* top_dst = field + ((ny - 1) * nx);
    double* top_src = field + nx;

    for (i = 0; i < nx; i++) {
        bottom_dst[i] = bottom_src[i];
        top_dst[i] = top_src[i];
    }
}

void bc_apply_dirichlet_scalar_impl(double* field, size_t nx, size_t ny,
                                     const bc_dirichlet_values_t* values) {
    size_t j, i;

    /* Left boundary (column 0) */
    for (j = 0; j < ny; j++) {
        field[j * nx] = values->left;
    }

    /* Right boundary (column nx-1) */
    for (j = 0; j < ny; j++) {
        field[j * nx + (nx - 1)] = values->right;
    }

    /* Bottom boundary (row 0) */
    for (i = 0; i < nx; i++) {
        field[i] = values->bottom;
    }

    /* Top boundary (row ny-1) */
    double* top_row = field + ((ny - 1) * nx);
    for (i = 0; i < nx; i++) {
        top_row[i] = values->top;
    }
}

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
            u[j * nx] = 0.0;
            v[j * nx] = v[j * nx + 1];
        }
    }

    /* Right boundary (X-symmetry plane at x=Lx):
     * - u = 0 (normal velocity is zero)
     * - dv/dx = 0 (tangential gradient is zero, copy from interior) */
    if (edges & BC_EDGE_RIGHT) {
        for (j = 0; j < ny; j++) {
            u[j * nx + (nx - 1)] = 0.0;
            v[j * nx + (nx - 1)] = v[j * nx + (nx - 2)];
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
