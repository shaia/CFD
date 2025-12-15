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

#include "boundary_conditions_internal.h"
#include "cfd/core/logging.h"
#include <stdbool.h>

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
        /* Still set it - will fall back to scalar */
        g_current_backend = backend;
        return false;
    }
    g_current_backend = backend;
    return true;
}

/* ============================================================================
 * Internal Dispatch Helpers
 * ============================================================================ */

static void apply_neumann_with_backend(double* field, size_t nx, size_t ny, bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            bc_apply_neumann_scalar_impl(field, nx, ny);
            break;

        case BC_BACKEND_SIMD:
#ifdef BC_HAS_SIMD
            bc_apply_neumann_simd_impl(field, nx, ny);
#else
            bc_apply_neumann_scalar_impl(field, nx, ny);
#endif
            break;

        case BC_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            bc_apply_neumann_omp_impl(field, nx, ny);
#else
            bc_apply_neumann_scalar_impl(field, nx, ny);
#endif
            break;

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
            break;
    }
}

static void apply_periodic_with_backend(double* field, size_t nx, size_t ny, bc_backend_t backend) {
    switch (backend) {
        case BC_BACKEND_SCALAR:
            bc_apply_periodic_scalar_impl(field, nx, ny);
            break;

        case BC_BACKEND_SIMD:
#ifdef BC_HAS_SIMD
            bc_apply_periodic_simd_impl(field, nx, ny);
#else
            bc_apply_periodic_scalar_impl(field, nx, ny);
#endif
            break;

        case BC_BACKEND_OMP:
#ifdef CFD_ENABLE_OPENMP
            bc_apply_periodic_omp_impl(field, nx, ny);
#else
            bc_apply_periodic_scalar_impl(field, nx, ny);
#endif
            break;

        case BC_BACKEND_AUTO:
        default:
#ifdef CFD_ENABLE_OPENMP
            bc_apply_periodic_omp_impl(field, nx, ny);
#elif defined(BC_HAS_SIMD)
            bc_apply_periodic_simd_impl(field, nx, ny);
#else
            bc_apply_periodic_scalar_impl(field, nx, ny);
#endif
            break;
    }
}

static void apply_scalar_field_bc(double* field, size_t nx, size_t ny, bc_type_t type, bc_backend_t backend) {
    if (!field || nx < 3 || ny < 3) {
        return;
    }

    switch (type) {
        case BC_TYPE_NEUMANN:
            apply_neumann_with_backend(field, nx, ny, backend);
            break;

        case BC_TYPE_PERIODIC:
            apply_periodic_with_backend(field, nx, ny, backend);
            break;

        case BC_TYPE_DIRICHLET:
            cfd_warning("BC_TYPE_DIRICHLET not implemented, falling back to Neumann");
            apply_neumann_with_backend(field, nx, ny, backend);
            break;

        case BC_TYPE_NOSLIP:
            cfd_warning("BC_TYPE_NOSLIP not implemented, falling back to Neumann");
            apply_neumann_with_backend(field, nx, ny, backend);
            break;

        case BC_TYPE_INLET:
            cfd_warning("BC_TYPE_INLET not implemented, falling back to Neumann");
            apply_neumann_with_backend(field, nx, ny, backend);
            break;

        case BC_TYPE_OUTLET:
            cfd_warning("BC_TYPE_OUTLET not implemented, falling back to Neumann");
            apply_neumann_with_backend(field, nx, ny, backend);
            break;

        default:
            cfd_warning("Unknown BC type requested, falling back to Neumann");
            apply_neumann_with_backend(field, nx, ny, backend);
            break;
    }
}

/* ============================================================================
 * Public API - Global Backend
 * ============================================================================ */

void bc_apply_scalar(double* field, size_t nx, size_t ny, bc_type_t type) {
    apply_scalar_field_bc(field, nx, ny, type, g_current_backend);
}

void bc_apply_velocity(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return;
    }
    apply_scalar_field_bc(u, nx, ny, type, g_current_backend);
    apply_scalar_field_bc(v, nx, ny, type, g_current_backend);
}

/* ============================================================================
 * Public API - Explicit Backend Selection
 * ============================================================================ */

void bc_apply_scalar_cpu(double* field, size_t nx, size_t ny, bc_type_t type) {
    apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_SCALAR);
}

void bc_apply_scalar_simd(double* field, size_t nx, size_t ny, bc_type_t type) {
    apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_SIMD);
}

void bc_apply_scalar_omp(double* field, size_t nx, size_t ny, bc_type_t type) {
    apply_scalar_field_bc(field, nx, ny, type, BC_BACKEND_OMP);
}

void bc_apply_velocity_cpu(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return;
    }
    apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_SCALAR);
    apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_SCALAR);
}

void bc_apply_velocity_simd(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return;
    }
    apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_SIMD);
    apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_SIMD);
}

void bc_apply_velocity_omp(double* u, double* v, size_t nx, size_t ny, bc_type_t type) {
    if (!u || !v || nx < 3 || ny < 3) {
        return;
    }
    apply_scalar_field_bc(u, nx, ny, type, BC_BACKEND_OMP);
    apply_scalar_field_bc(v, nx, ny, type, BC_BACKEND_OMP);
}
