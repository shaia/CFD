/**
 * @file stencils.h
 * @brief Finite difference stencil operations
 *
 * Shared stencil implementations used by both solvers and tests.
 * All stencils are O(h²) accurate central differences.
 *
 * This header provides inline functions for:
 *   - First derivatives (∂f/∂x, ∂f/∂y)
 *   - Second derivatives (∂²f/∂x², ∂²f/∂y²)
 *   - Laplacian (∇²f = ∂²f/∂x² + ∂²f/∂y²)
 *   - Divergence (∇·F = ∂u/∂x + ∂v/∂y)
 *   - Gradient (∇f = (∂f/∂x, ∂f/∂y))
 */

#ifndef CFD_MATH_STENCILS_H
#define CFD_MATH_STENCILS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FIRST DERIVATIVE STENCILS (O(h²) central difference)
 * ============================================================================
 *
 * Formula: df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
 * Error: O(dx²)
 */

/**
 * Central difference first derivative in x-direction
 *
 * @param f_ip1  Function value at x+dx (f[i+1])
 * @param f_im1  Function value at x-dx (f[i-1])
 * @param dx     Grid spacing in x
 * @return Approximation of df/dx
 */
static inline double stencil_first_deriv_x(double f_ip1, double f_im1, double dx) {
    return (f_ip1 - f_im1) / (2.0 * dx);
}

/**
 * Central difference first derivative in y-direction
 *
 * @param f_jp1  Function value at y+dy (f[j+1])
 * @param f_jm1  Function value at y-dy (f[j-1])
 * @param dy     Grid spacing in y
 * @return Approximation of df/dy
 */
static inline double stencil_first_deriv_y(double f_jp1, double f_jm1, double dy) {
    return (f_jp1 - f_jm1) / (2.0 * dy);
}

/* ============================================================================
 * SECOND DERIVATIVE STENCILS (O(h²) central difference)
 * ============================================================================
 *
 * Formula: d²f/dx² ≈ (f[i+1] - 2*f[i] + f[i-1]) / dx²
 * Error: O(dx²)
 */

/**
 * Central difference second derivative in x-direction
 *
 * @param f_ip1  Function value at x+dx (f[i+1])
 * @param f_i    Function value at x (f[i])
 * @param f_im1  Function value at x-dx (f[i-1])
 * @param dx     Grid spacing in x
 * @return Approximation of d²f/dx²
 */
static inline double stencil_second_deriv_x(double f_ip1, double f_i, double f_im1, double dx) {
    return (f_ip1 - 2.0 * f_i + f_im1) / (dx * dx);
}

/**
 * Central difference second derivative in y-direction
 *
 * @param f_jp1  Function value at y+dy (f[j+1])
 * @param f_j    Function value at y (f[j])
 * @param f_jm1  Function value at y-dy (f[j-1])
 * @param dy     Grid spacing in y
 * @return Approximation of d²f/dy²
 */
static inline double stencil_second_deriv_y(double f_jp1, double f_j, double f_jm1, double dy) {
    return (f_jp1 - 2.0 * f_j + f_jm1) / (dy * dy);
}

/* ============================================================================
 * LAPLACIAN STENCIL (5-point stencil, O(h²))
 * ============================================================================
 *
 * Formula: ∇²f = ∂²f/∂x² + ∂²f/∂y²
 *             ≈ (f[i+1] - 2*f[i] + f[i-1])/dx² + (f[j+1] - 2*f[j] + f[j-1])/dy²
 * Error: O(h²) where h = max(dx, dy)
 */

/**
 * 2D Laplacian using 5-point stencil
 *
 * @param f_ip1  Function value at x+dx (f[i+1,j])
 * @param f_im1  Function value at x-dx (f[i-1,j])
 * @param f_jp1  Function value at y+dy (f[i,j+1])
 * @param f_jm1  Function value at y-dy (f[i,j-1])
 * @param f_ij   Function value at center (f[i,j])
 * @param dx     Grid spacing in x
 * @param dy     Grid spacing in y
 * @return Approximation of ∇²f
 */
static inline double stencil_laplacian_2d(double f_ip1, double f_im1,
                                          double f_jp1, double f_jm1,
                                          double f_ij, double dx, double dy) {
    return (f_ip1 - 2.0 * f_ij + f_im1) / (dx * dx) +
           (f_jp1 - 2.0 * f_ij + f_jm1) / (dy * dy);
}

/* ============================================================================
 * DIVERGENCE STENCIL (O(h²))
 * ============================================================================
 *
 * Formula: ∇·F = ∂u/∂x + ∂v/∂y
 *             ≈ (u[i+1] - u[i-1])/(2*dx) + (v[j+1] - v[j-1])/(2*dy)
 * Error: O(h²) where h = max(dx, dy)
 */

/**
 * 2D divergence of vector field (u, v)
 *
 * @param u_ip1  u-component at x+dx
 * @param u_im1  u-component at x-dx
 * @param v_jp1  v-component at y+dy
 * @param v_jm1  v-component at y-dy
 * @param dx     Grid spacing in x
 * @param dy     Grid spacing in y
 * @return Approximation of ∇·(u,v) = ∂u/∂x + ∂v/∂y
 */
static inline double stencil_divergence_2d(double u_ip1, double u_im1,
                                           double v_jp1, double v_jm1,
                                           double dx, double dy) {
    return (u_ip1 - u_im1) / (2.0 * dx) + (v_jp1 - v_jm1) / (2.0 * dy);
}

/* ============================================================================
 * GRADIENT STENCIL (O(h²))
 * ============================================================================
 *
 * Formula: ∇f = (∂f/∂x, ∂f/∂y)
 * Uses central differences for each component.
 */

/**
 * 2D gradient x-component (same as first derivative in x)
 */
#define stencil_gradient_x stencil_first_deriv_x

/**
 * 2D gradient y-component (same as first derivative in y)
 */
#define stencil_gradient_y stencil_first_deriv_y

#ifdef __cplusplus
}
#endif

#endif /* CFD_MATH_STENCILS_H */
