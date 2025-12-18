/**
 * Boundary Conditions Implementation
 *
 * Unified boundary condition functions for CFD simulations.
 * Eliminates code duplication across CPU, SIMD, and OMP solvers.
 *
 * This file provides:
 * - Core scalar implementations (baseline)
 * - Runtime backend selection and dispatch
 * - Explicit backend-specific API functions
 *
 * Optimized implementations are in separate files:
 * - boundary_conditions_simd.c (AVX2/SSE2)
 * - boundary_conditions_omp.c (OpenMP)
 * - boundary_conditions_gpu.cu (CUDA)
 */

#include "../boundary_conditions_internal.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/logging.h"
#include <stdbool.h>
#include <stdio.h>

/* ============================================================================
 * Global Backend State
 * ============================================================================ */

static bc_backend_t g_current_backend = BC_BACKEND_AUTO;

/* ============================================================================
 * Scalar Implementations (Baseline)
 * ============================================================================ */

/**
 * Apply Neumann boundary conditions (zero gradient) to a scalar field.
 */
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

/**
 * Apply periodic boundary conditions to a scalar field.
 */
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

/**
 * Apply Dirichlet (fixed value) boundary conditions to a scalar field.
 */
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

/* ============================================================================
 * Backend Availability Detection
 * ============================================================================ */

bool bc_backend_available(bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_AUTO:
        case BC_BACKEND_SCALAR:
            return true;

        case BC_BACKEND_SIMD:
#ifdef BC_HAS_SIMD
            return true;
#else
            return false;
#endif

        case BC_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            return true;
#else
            return false;
#endif

        default:
            return false;
    }
}

/* ============================================================================
 * Backend Selection API
 * ============================================================================ */

bc_backend_t bc_get_backend(void) {
    return g_current_backend;
}

const char* bc_get_backend_name(void) {
    switch (g_current_backend) {
        case BC_BACKEND_AUTO:
            /* Return the actual backend being used */
#ifdef CFD_ENABLE_OPENMP
            return "auto (omp)";
#elif defined(BC_HAS_SIMD)
            return "auto (simd)";
#else
            return "auto (scalar)";
#endif

        case BC_BACKEND_SCALAR:
            return "scalar";

        case BC_BACKEND_SIMD:
#ifdef BC_HAS_SIMD
            return "simd";
#else
            return "simd (unavailable, using scalar)";
#endif

        case BC_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            return "omp";
#else
            return "omp (unavailable, using scalar)";
#endif

        default:
            return "unknown";
    }
}

bool bc_set_backend(bc_backend_t backend) {
    if (!bc_backend_available(backend) && backend != BC_BACKEND_AUTO) {
        /* Backend not available - set anyway but return false to indicate failure */
        g_current_backend = backend;
        return false;
    }
    g_current_backend = backend;
    return true;
}

/* ============================================================================
 * Internal Dispatch Helpers
 * ============================================================================ */

static cfd_status_t apply_neumann_with_backend(double* field, size_t nx, size_t ny, bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            bc_apply_neumann_scalar_impl(field, nx, ny);
            return CFD_SUCCESS;

#ifdef BC_HAS_SIMD
        case BC_BACKEND_SIMD:
            bc_apply_neumann_simd_impl(field, nx, ny);
            return CFD_SUCCESS;
#endif

#ifdef CFD_ENABLE_OPENMP
        case BC_BACKEND_OMP:
            bc_apply_neumann_omp_impl(field, nx, ny);
            return CFD_SUCCESS;
#endif

        case BC_BACKEND_AUTO:
        default:
            /* Auto: priority OMP > SIMD > Scalar */
#ifdef CFD_ENABLE_OPENMP
            bc_apply_neumann_omp_impl(field, nx, ny);
#elif defined(BC_HAS_SIMD)
            bc_apply_neumann_simd_impl(field, nx, ny);
#else
            bc_apply_neumann_scalar_impl(field, nx, ny);
#endif
            return CFD_SUCCESS;
    }
}

static cfd_status_t apply_periodic_with_backend(double* field, size_t nx, size_t ny, bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            bc_apply_periodic_scalar_impl(field, nx, ny);
            return CFD_SUCCESS;

#ifdef BC_HAS_SIMD
        case BC_BACKEND_SIMD:
            bc_apply_periodic_simd_impl(field, nx, ny);
            return CFD_SUCCESS;
#endif

#ifdef CFD_ENABLE_OPENMP
        case BC_BACKEND_OMP:
            bc_apply_periodic_omp_impl(field, nx, ny);
            return CFD_SUCCESS;
#endif

        case BC_BACKEND_AUTO:
        default:
#ifdef CFD_ENABLE_OPENMP
            bc_apply_periodic_omp_impl(field, nx, ny);
#elif defined(BC_HAS_SIMD)
            bc_apply_periodic_simd_impl(field, nx, ny);
#else
            bc_apply_periodic_scalar_impl(field, nx, ny);
#endif
            return CFD_SUCCESS;
    }
}

static cfd_status_t apply_dirichlet_with_backend(double* field, size_t nx, size_t ny,
                                                  const bc_dirichlet_values_t* values,
                                                  bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            bc_apply_dirichlet_scalar_impl(field, nx, ny, values);
            return CFD_SUCCESS;

#ifdef BC_HAS_SIMD
        case BC_BACKEND_SIMD:
            bc_apply_dirichlet_simd_impl(field, nx, ny, values);
            return CFD_SUCCESS;
#endif

#ifdef CFD_ENABLE_OPENMP
        case BC_BACKEND_OMP:
            bc_apply_dirichlet_omp_impl(field, nx, ny, values);
            return CFD_SUCCESS;
#endif

        case BC_BACKEND_AUTO:
        default:
#ifdef CFD_ENABLE_OPENMP
            bc_apply_dirichlet_omp_impl(field, nx, ny, values);
#elif defined(BC_HAS_SIMD)
            bc_apply_dirichlet_simd_impl(field, nx, ny, values);
#else
            bc_apply_dirichlet_scalar_impl(field, nx, ny, values);
#endif
            return CFD_SUCCESS;
    }
}

static cfd_status_t apply_scalar_field_bc(double* field, size_t nx, size_t ny, bc_type_t type, bc_backend_t backend) {
    if (!field || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }

    switch (type) {
        case BC_TYPE_NEUMANN:
            return apply_neumann_with_backend(field, nx, ny, backend);

        case BC_TYPE_PERIODIC:
            return apply_periodic_with_backend(field, nx, ny, backend);

        case BC_TYPE_DIRICHLET:
            cfd_warning("BC_TYPE_DIRICHLET requires bc_apply_dirichlet_*() functions with values");
            return CFD_ERROR_INVALID;

        case BC_TYPE_NOSLIP:
            cfd_warning("BC_TYPE_NOSLIP not implemented");
            return CFD_ERROR_UNSUPPORTED;

        case BC_TYPE_INLET:
            cfd_warning("BC_TYPE_INLET not implemented");
            return CFD_ERROR_UNSUPPORTED;

        case BC_TYPE_OUTLET:
            cfd_warning("BC_TYPE_OUTLET not implemented");
            return CFD_ERROR_UNSUPPORTED;

        default:
            cfd_warning("Unknown BC type requested");
            return CFD_ERROR_INVALID;
    }
}

/* ============================================================================
 * Public API - Global Backend
 * ============================================================================ */

cfd_status_t bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type) {
    return apply_scalar_field_bc(field, nx, ny, type, g_current_backend);
}

cfd_status_t bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, g_current_backend);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, g_current_backend);
}

/* ============================================================================
 * Public API - Explicit Backend Selection
 * ============================================================================ */

cfd_status_t bc_apply_scalar_cpu(double* field, size_t nx, size_t ny, bc_type_t type) {
    return apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_SCALAR);
}

cfd_status_t bc_apply_scalar_simd(double* field, size_t nx, size_t ny, bc_type_t type) {
#ifdef BC_HAS_SIMD
    return apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_SIMD);
#else
    (void)field; (void)nx; (void)ny; (void)type;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_scalar_omp(double* field, size_t nx, size_t ny, bc_type_t type) {
#ifdef CFD_ENABLE_OPENMP
    return apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_OMP);
#else
    (void)field; (void)nx; (void)ny; (void)type;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_velocity_cpu(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_SCALAR);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_SCALAR);
}

cfd_status_t bc_apply_velocity_simd(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
#ifdef BC_HAS_SIMD
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_SIMD);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_SIMD);
#else
    (void)u; (void)v; (void)nx; (void)ny; (void)type;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_velocity_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
#ifdef CFD_ENABLE_OPENMP
    if (!u || !v || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_OMP);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_OMP);
#else
    (void)u; (void)v; (void)nx; (void)ny; (void)type;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

/* ============================================================================
 * Public API - Dirichlet Boundary Conditions
 * ============================================================================ */

cfd_status_t bc_apply_dirichlet_scalar(double* field, size_t nx, size_t ny,
                                        const bc_dirichlet_values_t* values) {
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, g_current_backend);
}

cfd_status_t bc_apply_dirichlet_velocity(double* u, double* v, size_t nx, size_t ny,
                                          const bc_dirichlet_values_t* u_values,
                                          const bc_dirichlet_values_t* v_values) {
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, g_current_backend);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, g_current_backend);
}

/* Backend-specific Dirichlet implementations */

cfd_status_t bc_apply_dirichlet_scalar_cpu(double* field, size_t nx, size_t ny,
                                            const bc_dirichlet_values_t* values) {
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, BC_BACKEND_SCALAR);
}

cfd_status_t bc_apply_dirichlet_scalar_simd(double* field, size_t nx, size_t ny,
                                             const bc_dirichlet_values_t* values) {
#ifdef BC_HAS_SIMD
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, BC_BACKEND_SIMD);
#else
    (void)field; (void)nx; (void)ny; (void)values;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_dirichlet_scalar_omp(double* field, size_t nx, size_t ny,
                                            const bc_dirichlet_values_t* values) {
#ifdef CFD_ENABLE_OPENMP
    if (!field || !values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    return apply_dirichlet_with_backend(field, nx, ny, values, BC_BACKEND_OMP);
#else
    (void)field; (void)nx; (void)ny; (void)values;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_dirichlet_velocity_cpu(double* u, double* v, size_t nx, size_t ny,
                                              const bc_dirichlet_values_t* u_values,
                                              const bc_dirichlet_values_t* v_values) {
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, BC_BACKEND_SCALAR);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, BC_BACKEND_SCALAR);
}

cfd_status_t bc_apply_dirichlet_velocity_simd(double* u, double* v, size_t nx, size_t ny,
                                               const bc_dirichlet_values_t* u_values,
                                               const bc_dirichlet_values_t* v_values) {
#ifdef BC_HAS_SIMD
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, BC_BACKEND_SIMD);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, BC_BACKEND_SIMD);
#else
    (void)u; (void)v; (void)nx; (void)ny; (void)u_values; (void)v_values;
    return CFD_ERROR_UNSUPPORTED;
#endif
}

cfd_status_t bc_apply_dirichlet_velocity_omp(double* u, double* v, size_t nx, size_t ny,
                                              const bc_dirichlet_values_t* u_values,
                                              const bc_dirichlet_values_t* v_values) {
#ifdef CFD_ENABLE_OPENMP
    if (!u || !v || !u_values || !v_values || nx < 3 || ny < 3) {
        return CFD_ERROR_INVALID;
    }
    cfd_status_t status = apply_dirichlet_with_backend(u, nx, ny, u_values, BC_BACKEND_OMP);
    if (status != CFD_SUCCESS) {
        return status;
    }
    return apply_dirichlet_with_backend(v, nx, ny, v_values, BC_BACKEND_OMP);
#else
    (void)u; (void)v; (void)nx; (void)ny; (void)u_values; (void)v_values;
    return CFD_ERROR_UNSUPPORTED;
#endif
}
