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

/** Function type for applying inlet BCs to velocity fields */
typedef cfd_status_t (*bc_apply_inlet_fn)(double* u, double* v, size_t nx, size_t ny,
                                           const bc_inlet_config_t* config);

/** Function type for applying outlet BCs to a scalar field */
typedef cfd_status_t (*bc_apply_outlet_fn)(double* field, size_t nx, size_t ny,
                                            const bc_outlet_config_t* config);

/**
 * Backend implementation table.
 * Each backend fills in its function pointers.
 * NULL means the function is not available for that backend.
 */
typedef struct {
    bc_apply_scalar_fn apply_neumann;
    bc_apply_scalar_fn apply_periodic;
    bc_apply_dirichlet_fn apply_dirichlet;
    bc_apply_inlet_fn apply_inlet;
    bc_apply_outlet_fn apply_outlet;
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
extern const bc_backend_impl_t bc_impl_avx2;  /* AVX2 + OMP (x86-64) */
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
 * Error Handling (Internal)
 *
 * Used by dispatcher to report errors through the user-configurable handler.
 * ============================================================================ */

/**
 * Report an error through the configured error handler.
 * Called by internal code when an error occurs that cannot be returned normally.
 *
 * @param error_code  The error code
 * @param function    Name of the function where the error occurred
 * @param message     Human-readable error message
 */
void bc_report_error(bc_error_code_t error_code, const char* function, const char* message);

/* ============================================================================
 * Internal Scalar Implementations
 *
 * These are the actual baseline implementations.
 * ============================================================================ */

void bc_apply_neumann_scalar_impl(double* field, size_t nx, size_t ny);
void bc_apply_periodic_scalar_impl(double* field, size_t nx, size_t ny);
void bc_apply_dirichlet_scalar_impl(double* field, size_t nx, size_t ny,
                                     const bc_dirichlet_values_t* values);
cfd_status_t bc_apply_inlet_scalar_impl(double* u, double* v, size_t nx, size_t ny,
                                         const bc_inlet_config_t* config);

/* OpenMP inlet implementation - defined in omp/boundary_conditions_inlet_omp.c */
cfd_status_t bc_apply_inlet_omp_impl(double* u, double* v, size_t nx, size_t ny,
                                      const bc_inlet_config_t* config);

/* AVX2+OMP inlet implementation - defined in avx2/boundary_conditions_inlet_avx2.c */
cfd_status_t bc_apply_inlet_avx2_impl(double* u, double* v, size_t nx, size_t ny,
                                       const bc_inlet_config_t* config);

/* NEON+OMP inlet implementation - defined in neon/boundary_conditions_inlet_neon_omp.c */
cfd_status_t bc_apply_inlet_neon_omp_impl(double* u, double* v, size_t nx, size_t ny,
                                           const bc_inlet_config_t* config);

/* Outlet implementations */
cfd_status_t bc_apply_outlet_scalar_impl(double* field, size_t nx, size_t ny,
                                          const bc_outlet_config_t* config);

/* OpenMP outlet implementation - defined in omp/boundary_conditions_outlet_omp.c */
cfd_status_t bc_apply_outlet_omp_impl(double* field, size_t nx, size_t ny,
                                       const bc_outlet_config_t* config);

/* AVX2+OMP outlet implementation - defined in avx2/boundary_conditions_outlet_avx2.c */
cfd_status_t bc_apply_outlet_avx2_impl(double* field, size_t nx, size_t ny,
                                        const bc_outlet_config_t* config);

/* NEON+OMP outlet implementation - defined in neon/boundary_conditions_outlet_neon_omp.c */
cfd_status_t bc_apply_outlet_neon_omp_impl(double* field, size_t nx, size_t ny,
                                            const bc_outlet_config_t* config);

#endif /* CFD_BOUNDARY_CONDITIONS_INTERNAL_H */
