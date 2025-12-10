#ifndef FIELD_OPS_H
#define FIELD_OPS_H

#include "solver_interface.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Field Operations API - Utility functions for flow field calculations
 *
 * This module provides common operations on flow fields that can be used
 * for visualization, analysis, or post-processing.
 */

/**
 * Calculate velocity magnitude field
 *
 * Computes sqrt(u^2 + v^2) for each grid point.
 *
 * @param field The flow field containing u and v velocity components
 * @param nx Number of grid points in x direction
 * @param ny Number of grid points in y direction
 * @return Allocated array of velocity magnitudes (caller must free), or NULL on error
 */
double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny);

/**
 * Calculate velocity magnitude squared field
 *
 * Computes u^2 + v^2 for each grid point.
 * Performance optimized version avoiding sqrt call.
 *
 * @param field The flow field containing u and v velocity components
 * @param nx Number of grid points in x direction
 * @param ny Number of grid points in y direction
 * @return Allocated array of squared velocity magnitudes (caller must free), or NULL on error
 */
double* calculate_velocity_magnitude_squared(const FlowField* field, size_t nx, size_t ny);

#ifdef __cplusplus
}
#endif

#endif  // FIELD_OPS_H
