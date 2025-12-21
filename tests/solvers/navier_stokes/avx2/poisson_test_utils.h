/**
 * Common Test Utilities for SIMD Poisson Solver Tests
 *
 * Shared helper functions for memory allocation, initialization, and
 * error computation used by both Jacobi and Red-Black SIMD Poisson tests.
 */

#ifndef POISSON_TEST_UTILS_H
#define POISSON_TEST_UTILS_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Allocate a field with 32-byte alignment for AVX2 SIMD operations.
 * Size is rounded up to multiple of 32 as required by aligned_alloc on some platforms.
 */
static inline double* allocate_field(size_t nx, size_t ny) {
    size_t size = nx * ny * sizeof(double);
    // Round up to multiple of 32 (required by aligned_alloc on some platforms)
    size_t aligned_size = (size + 31) & ~((size_t)31);
    return (double*)aligned_alloc(32, aligned_size);
}

/**
 * Free a field allocated with allocate_field.
 */
static inline void free_field(double* field) {
    aligned_free(field);
}

/**
 * Initialize a field to zero.
 */
static inline void initialize_zero(double* field, size_t nx, size_t ny) {
    memset(field, 0, nx * ny * sizeof(double));
}

/**
 * Initialize RHS for a known analytical solution: p = sin(pi*x)*sin(pi*y)
 * Laplacian of p = -2*pi^2 * sin(pi*x)*sin(pi*y)
 * Scale down to make convergence easier (typical CFD RHS values are small).
 */
static inline void initialize_sinusoidal_rhs(double* rhs, size_t nx, size_t ny,
                                              double dx, double dy) {
    double scale = 0.01;  // Scale factor to make RHS similar to CFD use case
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = i * dx;
            double y = j * dy;
            // RHS = -Laplacian(p) = 2*pi^2 * sin(pi*x)*sin(pi*y)
            rhs[j * nx + i] = scale * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

/**
 * Compute the analytical solution at a given point.
 * Solution: p = 0.01 * sin(pi*x) * sin(pi*y)
 */
static inline double compute_analytical_solution(double x, double y) {
    return 0.01 * sin(M_PI * x) * sin(M_PI * y);  // Same scale as RHS
}

/**
 * Compute the L2 error between computed and analytical solutions.
 * Only considers interior points (excludes boundaries).
 */
static inline double compute_l2_error(const double* p, size_t nx, size_t ny,
                                       double dx, double dy) {
    double error = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double x = i * dx;
            double y = j * dy;
            double analytical = compute_analytical_solution(x, y);
            double diff = p[j * nx + i] - analytical;
            error += diff * diff;
        }
    }
    return sqrt(error / ((nx - 2) * (ny - 2)));
}

/**
 * Compute the maximum residual of the Poisson equation.
 * Residual = |Laplacian(p) - rhs| at each interior point.
 */
static inline double compute_max_residual(const double* p, const double* rhs,
                                           size_t nx, size_t ny,
                                           double dx2, double dy2) {
    double max_res = 0.0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            size_t idx = j * nx + i;
            double p_xx = (p[idx + 1] - 2.0 * p[idx] + p[idx - 1]) / dx2;
            double p_yy = (p[idx + nx] - 2.0 * p[idx] + p[idx - nx]) / dy2;
            double res = fabs(p_xx + p_yy - rhs[idx]);
            if (res > max_res) max_res = res;
        }
    }
    return max_res;
}

#endif // POISSON_TEST_UTILS_H
