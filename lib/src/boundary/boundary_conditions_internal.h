/**
 * Boundary Conditions Internal Header
 *
 * Internal declarations for boundary condition implementations.
 * Not for public use - include boundary_conditions.h instead.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_INTERNAL_H
#define CFD_BOUNDARY_CONDITIONS_INTERNAL_H

#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/cfd_status.h"
#include <stddef.h>

/* ============================================================================
 * Backend Implementation Function Types
 *
 * Each backend provides these functions. If a backend is not available,
 * it provides NULL pointers instead.
 * ============================================================================ */

/** Function type for applying Neumann/Periodic BCs to a scalar field */
typedef void (*bc_apply_scalar_fn)(double* field, size_t nx, size_t ny);

/** Function type for applying Dirichlet BCs to a scalar field */
typedef void (*bc_apply_dirichlet_fn)(double* field, size_t nx, size_t ny,
                                       const bc_dirichlet_values_t* values);

/**
 * Backend implementation table.
 * Each backend fills in its function pointers.
 * NULL means the function is not available for that backend.
 */
typedef struct {
    bc_apply_scalar_fn apply_neumann;
    bc_apply_scalar_fn apply_periodic;
    bc_apply_dirichlet_fn apply_dirichlet;
} bc_backend_impl_t;

/* ============================================================================
 * Backend Implementation Tables
 *
 * Each backend provides a table of function pointers.
 * Declared here, defined in each backend's source file.
 * ============================================================================ */

/** Scalar (baseline) implementation - always available */
extern const bc_backend_impl_t bc_impl_scalar;

/** SIMD (AVX2/SSE2) implementation - NULL if not compiled with SIMD */
extern const bc_backend_impl_t bc_impl_simd;

/** OpenMP implementation - NULL if not compiled with OpenMP */
extern const bc_backend_impl_t bc_impl_omp;

/* ============================================================================
 * Internal Scalar Implementations
 *
 * These are the actual baseline implementations.
 * ============================================================================ */

void bc_apply_neumann_scalar_impl(double* field, size_t nx, size_t ny);
void bc_apply_periodic_scalar_impl(double* field, size_t nx, size_t ny);
void bc_apply_dirichlet_scalar_impl(double* field, size_t nx, size_t ny,
                                     const bc_dirichlet_values_t* values);

#endif /* CFD_BOUNDARY_CONDITIONS_INTERNAL_H */
