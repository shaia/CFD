/**
 * Boundary Conditions Internal Header
 *
 * Internal declarations for boundary condition implementations.
 * Not for public use - include boundary_conditions.h instead.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_INTERNAL_H
#define CFD_BOUNDARY_CONDITIONS_INTERNAL_H

#include "cfd/boundary/boundary_conditions.h"
#include <stddef.h>

/**
 * Apply Neumann (zero gradient) boundary conditions using scalar operations.
 * This is the baseline implementation used when no optimizations are available.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_neumann_scalar_impl(double* field, size_t nx, size_t ny);

/**
 * Apply periodic boundary conditions using scalar operations.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_periodic_scalar_impl(double* field, size_t nx, size_t ny);

#ifdef CFD_ENABLE_OPENMP
/**
 * Apply Neumann boundary conditions with OpenMP parallelization.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_neumann_omp_impl(double* field, size_t nx, size_t ny);

/**
 * Apply periodic boundary conditions with OpenMP parallelization.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_periodic_omp_impl(double* field, size_t nx, size_t ny);
#endif

/* SIMD implementation availability detection */
#if defined(__AVX2__) || defined(__AVX__) || (defined(_MSC_VER) && defined(__AVX2__))
#define BC_HAS_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#define BC_HAS_SSE2 1
#endif

#if defined(BC_HAS_AVX2) || defined(BC_HAS_SSE2)
#define BC_HAS_SIMD 1

/**
 * Apply Neumann boundary conditions with SIMD optimization.
 * Uses AVX2 or SSE2 for contiguous memory operations.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_neumann_simd_impl(double* field, size_t nx, size_t ny);

/**
 * Apply periodic boundary conditions with SIMD optimization.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 */
void bc_apply_periodic_simd_impl(double* field, size_t nx, size_t ny);

/**
 * Apply Dirichlet boundary conditions with SIMD optimization.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 * @param values Pointer to struct containing boundary values
 */
void bc_apply_dirichlet_simd_impl(double* field, size_t nx, size_t ny,
                                   const bc_dirichlet_values_t* values);
#endif

/* ============================================================================
 * Dirichlet Boundary Condition Implementations
 * ============================================================================ */

/**
 * Apply Dirichlet (fixed value) boundary conditions using scalar operations.
 * This is the baseline implementation used when no optimizations are available.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 * @param values Pointer to struct containing boundary values
 */
void bc_apply_dirichlet_scalar_impl(double* field, size_t nx, size_t ny,
                                     const bc_dirichlet_values_t* values);

#ifdef CFD_ENABLE_OPENMP
/**
 * Apply Dirichlet boundary conditions with OpenMP parallelization.
 *
 * @param field  Pointer to field array (nx * ny elements)
 * @param nx     Number of grid points in x-direction
 * @param ny     Number of grid points in y-direction
 * @param values Pointer to struct containing boundary values
 */
void bc_apply_dirichlet_omp_impl(double* field, size_t nx, size_t ny,
                                  const bc_dirichlet_values_t* values);
#endif

#endif /* CFD_BOUNDARY_CONDITIONS_INTERNAL_H */
