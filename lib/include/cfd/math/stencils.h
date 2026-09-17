/**
 * @file stencils.h
 * @brief Finite difference stencil operations
 *
 * Shared stencil implementations used by both solvers and tests.
 * All stencils are O(h²) accurate central differences, except the
 * first-order upwind derivatives used for convective terms.
 *
 * This header provides inline functions for:
 *   - First derivatives (∂f/∂x, ∂f/∂y, ∂f/∂z)
 *   - First-order upwind first derivatives (convective terms)
 *   - Second derivatives (∂²f/∂x², ∂²f/∂y², ∂²f/∂z²)
 *   - Laplacian 2D (∇²f = ∂²f/∂x² + ∂²f/∂y²) and 3D (+ ∂²f/∂z²)
 *   - Divergence 2D (∇·F = ∂u/∂x + ∂v/∂y) and 3D (+ ∂w/∂z)
 *   - Gradient (∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z))
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

/**
 * Central difference first derivative in z-direction
 *
 * @param f_kp1  Function value at z+dz (f[k+1])
 * @param f_km1  Function value at z-dz (f[k-1])
 * @param dz     Grid spacing in z
 * @return Approximation of df/dz
 */
static inline double stencil_first_deriv_z(double f_kp1, double f_km1, double dz) {
    return (f_kp1 - f_km1) / (2.0 * dz);
}

/* ============================================================================
 * FIRST-ORDER UPWIND FIRST DERIVATIVE STENCILS (O(h))
 * ============================================================================
 *
 * The advecting velocity selects the one-sided difference:
 *   vel >= 0: df/dx ≈ (f[i] - f[i-1]) / dx   (information flows from the left)
 *   vel <  0: df/dx ≈ (f[i+1] - f[i]) / dx   (information flows from the right)
 * Error: O(dx); the leading error term is numerical diffusion |vel|*dx/2.
 *
 * `vel` is the velocity that carries f (u_c for df/dx, v_c for df/dy), not the
 * derivative being computed. A NaN velocity selects the forward difference.
 */

/**
 * Undivided first-order upwind difference along one axis
 *
 * Owns the direction convention shared by every upwind stencil. Scale the
 * result by 1/h; for branch-free 2D/3D solver loops, multiply by a precomputed
 * inverse spacing that is 0.0 on inactive axes instead of dividing by h.
 *
 * @param f_p  Function value at the + neighbor (f[i+1])
 * @param f_c  Function value at the center (f[i])
 * @param f_m  Function value at the - neighbor (f[i-1])
 * @param vel  Advecting velocity component along this axis
 * @return f_c - f_m when vel >= 0, otherwise f_p - f_c
 */
static inline double stencil_upwind_diff(double f_p, double f_c, double f_m, double vel) {
    return (vel >= 0.0) ? (f_c - f_m) : (f_p - f_c);
}

/**
 * First-order upwind first derivative in x-direction
 *
 * @param f_ip1  Function value at x+dx (f[i+1])
 * @param f_i    Function value at x (f[i])
 * @param f_im1  Function value at x-dx (f[i-1])
 * @param dx     Grid spacing in x
 * @param vel    Advecting velocity in x (u)
 * @return Approximation of df/dx
 */
static inline double stencil_upwind_deriv_x(double f_ip1, double f_i, double f_im1,
                                            double dx, double vel) {
    return stencil_upwind_diff(f_ip1, f_i, f_im1, vel) / dx;
}

/**
 * First-order upwind first derivative in y-direction
 *
 * @param f_jp1  Function value at y+dy (f[j+1])
 * @param f_j    Function value at y (f[j])
 * @param f_jm1  Function value at y-dy (f[j-1])
 * @param dy     Grid spacing in y
 * @param vel    Advecting velocity in y (v)
 * @return Approximation of df/dy
 */
static inline double stencil_upwind_deriv_y(double f_jp1, double f_j, double f_jm1,
                                            double dy, double vel) {
    return stencil_upwind_diff(f_jp1, f_j, f_jm1, vel) / dy;
}

/**
 * First-order upwind first derivative in z-direction
 *
 * @param f_kp1  Function value at z+dz (f[k+1])
 * @param f_k    Function value at z (f[k])
 * @param f_km1  Function value at z-dz (f[k-1])
 * @param dz     Grid spacing in z (must be > 0)
 * @param vel    Advecting velocity in z (w)
 * @return Approximation of df/dz
 */
static inline double stencil_upwind_deriv_z(double f_kp1, double f_k, double f_km1,
                                            double dz, double vel) {
    return stencil_upwind_diff(f_kp1, f_k, f_km1, vel) / dz;
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

/**
 * Central difference second derivative in z-direction
 *
 * @param f_kp1  Function value at z+dz (f[k+1])
 * @param f_k    Function value at z (f[k])
 * @param f_km1  Function value at z-dz (f[k-1])
 * @param dz     Grid spacing in z
 * @return Approximation of d²f/dz²
 */
static inline double stencil_second_deriv_z(double f_kp1, double f_k, double f_km1, double dz) {
    return (f_kp1 - 2.0 * f_k + f_km1) / (dz * dz);
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
 * 3D LAPLACIAN STENCIL (7-point stencil, O(h²))
 * ============================================================================
 *
 * Formula: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
 * Error: O(h²) where h = max(dx, dy, dz)
 *
 * NOTE: Do NOT call with dz=0. For branch-free 2D/3D solver loops, use
 * precomputed inv_dz2=0.0 with multiplication instead.
 */

/**
 * 3D Laplacian using 7-point stencil
 *
 * @param f_ip1  Function value at x+dx (f[i+1,j,k])
 * @param f_im1  Function value at x-dx (f[i-1,j,k])
 * @param f_jp1  Function value at y+dy (f[i,j+1,k])
 * @param f_jm1  Function value at y-dy (f[i,j-1,k])
 * @param f_kp1  Function value at z+dz (f[i,j,k+1])
 * @param f_km1  Function value at z-dz (f[i,j,k-1])
 * @param f_ijk  Function value at center (f[i,j,k])
 * @param dx     Grid spacing in x
 * @param dy     Grid spacing in y
 * @param dz     Grid spacing in z (must be > 0)
 * @return Approximation of ∇²f
 */
static inline double stencil_laplacian_3d(double f_ip1, double f_im1,
                                          double f_jp1, double f_jm1,
                                          double f_kp1, double f_km1,
                                          double f_ijk,
                                          double dx, double dy, double dz) {
    return (f_ip1 - 2.0 * f_ijk + f_im1) / (dx * dx) +
           (f_jp1 - 2.0 * f_ijk + f_jm1) / (dy * dy) +
           (f_kp1 - 2.0 * f_ijk + f_km1) / (dz * dz);
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

/**
 * 3D divergence of vector field (u, v, w)
 *
 * @param u_ip1  u-component at x+dx
 * @param u_im1  u-component at x-dx
 * @param v_jp1  v-component at y+dy
 * @param v_jm1  v-component at y-dy
 * @param w_kp1  w-component at z+dz
 * @param w_km1  w-component at z-dz
 * @param dx     Grid spacing in x
 * @param dy     Grid spacing in y
 * @param dz     Grid spacing in z (must be > 0)
 * @return Approximation of ∇·(u,v,w) = ∂u/∂x + ∂v/∂y + ∂w/∂z
 */
static inline double stencil_divergence_3d(double u_ip1, double u_im1,
                                           double v_jp1, double v_jm1,
                                           double w_kp1, double w_km1,
                                           double dx, double dy, double dz) {
    return (u_ip1 - u_im1) / (2.0 * dx) +
           (v_jp1 - v_jm1) / (2.0 * dy) +
           (w_kp1 - w_km1) / (2.0 * dz);
}

/* ============================================================================
 * GRADIENT STENCIL (O(h²))
 * ============================================================================
 *
 * Formula: ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z)
 * Uses central differences for each component.
 */

/**
 * Gradient x-component (same as first derivative in x)
 */
#define stencil_gradient_x stencil_first_deriv_x

/**
 * Gradient y-component (same as first derivative in y)
 */
#define stencil_gradient_y stencil_first_deriv_y

/**
 * Gradient z-component (same as first derivative in z)
 */
#define stencil_gradient_z stencil_first_deriv_z

#ifdef __cplusplus
}
#endif

#endif /* CFD_MATH_STENCILS_H */
