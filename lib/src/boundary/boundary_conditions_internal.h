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

/** OpenMP implementation - NULL if not compiled with OpenMP */
extern const bc_backend_impl_t bc_impl_omp;

/**
 * Architecture-specific SIMD + OpenMP implementations.
 * Each file provides its own table; only one will be non-NULL at compile time.
 */
extern const bc_backend_impl_t bc_impl_avx2_omp;  /* AVX2 + OMP (x86-64) */
extern const bc_backend_impl_t bc_impl_neon_omp;  /* NEON + OMP (ARM64) */

/**
 * SIMD + OpenMP unified interface with runtime architecture detection.
 * Dispatches to AVX2 or NEON based on detected CPU at runtime.
 * Defined in simd_omp/boundary_conditions_simd_omp_dispatch.c
 */
extern const bc_backend_impl_t bc_impl_simd_omp;

/**
 * Runtime check for SIMD+OMP availability.
 * Since bc_impl_simd_omp always has non-NULL function pointers (for dispatch),
 * this function checks if the underlying SIMD backend is actually available.
 */
bool bc_simd_omp_backend_available(void);

/**
 * Get the name of the detected SIMD architecture at runtime.
 * Returns "avx2", "neon", or "none".
 */
const char* bc_simd_omp_get_arch_name(void);

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
