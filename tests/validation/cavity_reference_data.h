/**
 * @file cavity_reference_data.h
 * @brief Reference data for lid-driven cavity validation tests
 *
 * This file contains pre-computed "golden" values that serve as the source
 * of truth for regression testing. These values were computed with known-good
 * solver configurations and validated against Ghia et al. (1982) reference data.
 *
 * To regenerate reference data, run tests with CAVITY_GENERATE_REFERENCE=1
 */

#ifndef CAVITY_REFERENCE_DATA_H
#define CAVITY_REFERENCE_DATA_H

/* ============================================================================
 * GHIA ET AL. (1982) PUBLISHED REFERENCE DATA
 * ============================================================================
 * These are the authoritative benchmark values from:
 *   "High-Re solutions for incompressible flow using the Navier-Stokes
 *    equations and a multigrid method"
 *   Journal of Computational Physics, Vol. 48, pp. 387-411
 */

/* y-coordinates for vertical centerline (x=0.5) */
static const double GHIA_Y_COORDS[] = {
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
};

/* u-velocity along vertical centerline at Re=100 */
static const double GHIA_U_RE100[] = {
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
    0.78871, 0.84123, 1.00000
};

/* u-velocity along vertical centerline at Re=400 */
static const double GHIA_U_RE400[] = {
    0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
    -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756,
    0.68439, 0.75837, 1.00000
};

/* u-velocity along vertical centerline at Re=1000 */
static const double GHIA_U_RE1000[] = {
    0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805,
    -0.10648, -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117,
    0.57492, 0.65928, 1.00000
};

/* x-coordinates for horizontal centerline (y=0.5) */
static const double GHIA_X_COORDS[] = {
    0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
    0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
    0.9609, 0.9688, 1.0000
};

/* v-velocity along horizontal centerline at Re=100 */
static const double GHIA_V_RE100[] = {
    0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
    0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864,
    -0.07391, -0.05906, 0.00000
};

/* v-velocity along horizontal centerline at Re=400 */
static const double GHIA_V_RE400[] = {
    0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203,
    0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254,
    -0.15663, -0.12146, 0.00000
};

/* v-velocity along horizontal centerline at Re=1000 */
static const double GHIA_V_RE1000[] = {
    0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075,
    0.32235, 0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714,
    -0.27669, -0.21388, 0.00000
};

#define GHIA_NUM_POINTS 17

/* Named indices for key reference points */
#define GHIA_IDX_CENTER     8   /* Index for y=0.5 (u_center) and x=0.5 (v_center) */
#define GHIA_IDX_U_MIN_RE100 7  /* Index of minimum u-velocity for Re=100 (y≈0.4531) */

/* Convenience macros for common reference values */
#define GHIA_U_CENTER_RE100  GHIA_U_RE100[GHIA_IDX_CENTER]      /* -0.20581 */
#define GHIA_U_MIN_RE100     GHIA_U_RE100[GHIA_IDX_U_MIN_RE100] /* -0.21090 */
#define GHIA_V_CENTER_RE100  GHIA_V_RE100[GHIA_IDX_CENTER]      /* 0.05454 */

/* ============================================================================
 * SOLVER REFERENCE VALUES
 * ============================================================================
 * Pre-computed values from our solver at specific configurations.
 * Used for regression testing to detect unintended solver changes.
 *
 * Configuration for these values:
 *   - Grid: 33x33
 *   - Solver: Projection method
 *   - Time step: 0.001 (fast mode) / 0.0005 (full mode)
 *   - Steps: 1000 (fast mode) / 5000 (full mode)
 */

typedef struct {
    double max_velocity;
    double kinetic_energy;
    double u_at_center;      /* u at (0.5, 0.5) */
    double v_at_center;      /* v at (0.5, 0.5) */
    double u_min_centerline; /* min u along x=0.5 */
} cavity_reference_t;

/* Reference values for Re=100, 33x33 grid, 1000 steps, dt=0.001 (fast mode)
 * Note: In fast mode, flow hasn't fully developed, so values differ from Ghia */
static const cavity_reference_t REF_RE100_33x33_FAST = {
    .max_velocity = 0.45,         /* Interior max velocity magnitude */
    .kinetic_energy = 22.6,       /* KE after partial development */
    .u_at_center = -0.09,         /* Ghia: -0.20581 (not converged) */
    .v_at_center = 0.14,          /* Ghia: 0.05454 */
    .u_min_centerline = -0.12     /* Ghia: -0.21090 */
};

/* Reference values for Re=100, 33x33 grid, 5000 steps, dt=0.0005 (full validation) */
static const cavity_reference_t REF_RE100_33x33_FULL = {
    .max_velocity = 0.80,         /* Better developed flow */
    .kinetic_energy = 35.0,       /* Higher KE after more development */
    .u_at_center = -0.18,         /* Closer to Ghia: -0.20581 */
    .v_at_center = 0.08,          /* Closer to Ghia: 0.05454 */
    .u_min_centerline = -0.19     /* Closer to Ghia: -0.21090 */
};

/* ============================================================================
 * TOLERANCE DEFINITIONS
 * ============================================================================
 * Tolerances depend on both grid resolution and iteration count.
 * Fast mode (fewer iterations) needs looser tolerances.
 */

#ifndef CAVITY_FULL_VALIDATION
#define CAVITY_FULL_VALIDATION 0
#endif

/* Ghia validation tolerances - these are HONEST scientific targets.
 * If tests fail, the solver needs improvement, not looser tolerances.
 *
 * Industry standard: RMS < 0.05 is excellent, < 0.10 is acceptable
 */
#define GHIA_TOLERANCE_COARSE  0.15  /* For grids < 25x25 */
#define GHIA_TOLERANCE_MEDIUM  0.10  /* For 25-33x grids */
#define GHIA_TOLERANCE_FINE    0.05  /* For grids > 64x64 */

/* All backends now achieve the scientific target (RMS < 0.10).
 * Scalar CPU uses CG Poisson solver for reliable convergence.
 * SIMD backends achieve RMS < 0.05.
 * Using 0.10 to match the scientific target (GHIA_TOLERANCE_MEDIUM). */
#define GHIA_TOLERANCE_CURRENT 0.10  /* Matches scientific target */

/* Tolerance for regression testing (should be very tight) */
#define REGRESSION_TOLERANCE   0.01  /* 1% relative difference */

#endif /* CAVITY_REFERENCE_DATA_H */
