/**
 * @file linear_solver_internal.h
 * @brief Internal declarations for linear solver implementations
 *
 * This header is not part of the public API.
 */

#ifndef CFD_LINEAR_SOLVER_INTERNAL_H
#define CFD_LINEAR_SOLVER_INTERNAL_H

#include "cfd/solvers/poisson_solver.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FACTORY FUNCTIONS
 *
 * All SIMD backends use runtime CPU detection (AVX2/NEON) via the SIMD
 * dispatcher. See simd/linear_solver_simd_dispatch.c for details.
 * ============================================================================ */

/* Jacobi solvers */
poisson_solver_t* create_jacobi_scalar_solver(void);
poisson_solver_t* create_jacobi_simd_solver(void);

/* SOR solvers */
poisson_solver_t* create_sor_scalar_solver(void);
poisson_solver_t* create_sor_simd_solver(void);

/* Red-Black SOR solvers */
poisson_solver_t* create_redblack_scalar_solver(void);
poisson_solver_t* create_redblack_simd_solver(void);

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_jacobi_omp_solver(void);
poisson_solver_t* create_redblack_omp_solver(void);
poisson_solver_t* create_cg_omp_solver(void);
poisson_solver_t* create_bicgstab_omp_solver(void);
#endif

/* Conjugate Gradient solvers */
poisson_solver_t* create_cg_scalar_solver(void);
poisson_solver_t* create_cg_simd_solver(void);

#ifdef CFD_HAS_CUDA
/* GPU (CUDA) solvers — defined in linear/gpu/*.cu, linked via cfd_cuda */
poisson_solver_t* create_jacobi_gpu_solver(void);
poisson_solver_t* create_cg_gpu_solver(void);
poisson_solver_t* create_bicgstab_gpu_solver(void);
poisson_solver_t* create_redblack_gpu_solver(void);
poisson_solver_t* create_sor_gpu_solver(void);
#endif

/* BiCGSTAB solvers (for non-symmetric systems) */
poisson_solver_t* create_bicgstab_scalar_solver(void);
poisson_solver_t* create_bicgstab_simd_solver(void);

/* GMRES solvers (restarted GMRES(m) for non-symmetric systems) */
poisson_solver_t* create_gmres_scalar_solver(void);
poisson_solver_t* create_gmres_simd_solver(void);

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_gmres_omp_solver(void);
#endif

/* Geometric multigrid solvers (V/W/F cycles; scalar and OpenMP backends) */
poisson_solver_t* create_multigrid_scalar_solver(void);

#ifdef CFD_ENABLE_OPENMP
poisson_solver_t* create_multigrid_omp_solver(void);
#endif

/**
 * Reject POISSON_PRECOND_MULTIGRID on backends that don't implement it.
 * Only the scalar CG solver supports the MG preconditioner; silently
 * ignoring it would be a forbidden silent fallback.
 */
static inline cfd_status_t poisson_solver_reject_mg_precond(
    const poisson_solver_params_t* params) {
    if (params && params->preconditioner == POISSON_PRECOND_MULTIGRID) {
        cfd_set_error(CFD_ERROR_UNSUPPORTED,
            "POISSON_PRECOND_MULTIGRID is only supported by the scalar CG solver");
        return CFD_ERROR_UNSUPPORTED;
    }
    return CFD_SUCCESS;
}

/* ============================================================================
 * CG ALGORITHM CONSTANTS
 * ============================================================================ */

/**
 * Threshold for detecting CG breakdown (division by near-zero).
 * If (p, Ap) or (r, r) falls below this, the algorithm has stagnated
 * or encountered a singular/near-singular system.
 */
#define CG_BREAKDOWN_THRESHOLD 1e-30

/**
 * Macro for CG breakdown check with early return.
 * Used when a denominator (p_dot_Ap or r_dot_r) becomes too small.
 *
 * @param value The value to check against breakdown threshold
 * @param stats Pointer to stats structure (may be NULL)
 * @param iter Current iteration index
 * @param res_norm Current residual norm
 * @param start_time Start time for elapsed time calculation
 */
#define CG_CHECK_BREAKDOWN(value, stats, iter, res_norm, start_time) \
    do { \
        if (fabs(value) < CG_BREAKDOWN_THRESHOLD) { \
            if (stats) { \
                (stats)->status = POISSON_STAGNATED; \
                (stats)->iterations = (iter) + 1; \
                (stats)->final_residual = (res_norm); \
                (stats)->elapsed_time_ms = poisson_solver_get_time_ms() - (start_time); \
            } \
            return CFD_ERROR_MAX_ITER; \
        } \
    } while (0)

/* ============================================================================
 * BICGSTAB ALGORITHM CONSTANTS
 * ============================================================================ */

/**
 * Threshold for detecting BiCGSTAB breakdown (division by near-zero).
 * If rho, (r_hat,v), or (t,t) falls below this, the algorithm has stagnated.
 */
#define BICGSTAB_BREAKDOWN_THRESHOLD 1e-30

/* ============================================================================
 * GMRES ALGORITHM CONSTANTS
 * ============================================================================ */

/**
 * Default restart length m for GMRES(m) when params.restart <= 0.
 * The Krylov basis holds m+1 grid-sized vectors, so this bounds memory.
 */
#define GMRES_DEFAULT_RESTART 30

/**
 * Threshold for detecting GMRES happy (lucky) breakdown.
 * When the Arnoldi subdiagonal H[j+1,j] (a vector norm) falls below this, the
 * exact solution already lies in the current Krylov subspace: this signals
 * CONVERGENCE, not failure. Larger than CG_BREAKDOWN_THRESHOLD because
 * H[j+1,j] is an sqrt-scaled norm rather than a squared dot product.
 */
#define GMRES_BREAKDOWN_THRESHOLD 1e-12

/* ============================================================================
 * OPENMP LOOP BOUNDS
 * ============================================================================ */

/**
 * Convert size_t to int for OpenMP loop bounds.
 * OpenMP (MSVC 2.0) requires int loop variables, but grid dimensions are size_t.
 * Shared by the OMP primitives and the SIMD solver templates.
 *
 * @param val The size_t value to convert
 * @return int value, or 0 on overflow (CFD_ERROR_LIMIT_EXCEEDED set)
 */
static inline int poisson_solver_size_to_int(size_t val) {
    if (val > (size_t)INT_MAX) {
        cfd_set_error(CFD_ERROR_LIMIT_EXCEEDED, "Grid size exceeds INT_MAX for OpenMP loop");
        return 0;
    }
    return (int)val;
}

/* ============================================================================
 * GRID SIZE LIMITS
 * ============================================================================ */

/**
 * Validate grid dimensions at init and compute the field size n = nx*ny*nz.
 *
 * Rejects nx or ny above INT_MAX, where poisson_solver_size_to_int returns 0 and
 * every int-bounded primitive loop becomes a no-op (a zero residual would then
 * read as convergence), and grids where `vectors` grid-sized double arrays would
 * not fit in size_t bytes. Each multiplication is checked before it is taken, so
 * n cannot wrap and work arrays cannot be allocated undersized.
 * Assumes nx, ny >= 3 (validated by poisson_solver_init).
 *
 * @param vectors number of grid-sized double arrays in the largest single
 *                allocation (>= 1)
 * @param n_out   receives nx*ny*nz on success
 * @return CFD_SUCCESS, or CFD_ERROR_LIMIT_EXCEEDED (error set)
 */
static inline cfd_status_t poisson_solver_validate_grid_size(
    size_t nx, size_t ny, size_t nz, size_t vectors, size_t* n_out)
{
    size_t max_n = SIZE_MAX / sizeof(double) / vectors;
    if (nx > (size_t)INT_MAX || ny > (size_t)INT_MAX ||
        ny > max_n / nx || nz > max_n / (nx * ny)) {
        cfd_set_error(CFD_ERROR_LIMIT_EXCEEDED, "Grid size exceeds indexable limits");
        return CFD_ERROR_LIMIT_EXCEEDED;
    }
    *n_out = nx * ny * nz;
    return CFD_SUCCESS;
}

/* ============================================================================
 * SIMD BACKEND AVAILABILITY (Runtime detection)
 * ============================================================================ */

/**
 * Check if SIMD backend is available at runtime.
 * Uses cfd_detect_simd_arch() from cpu_features.h.
 */
bool poisson_solver_simd_backend_available(void);

/**
 * Get the name of the detected SIMD architecture.
 * Returns "avx2", "neon", or "none".
 */
const char* poisson_solver_simd_get_arch_name(void);

/* ============================================================================
 * 3D LOOP BOUNDS HELPERS
 *
 * Centralizes the nz-dependent logic so each solver's init doesn't repeat it.
 * When nz==1 (2D): stride_z=0, k_start=0, k_end=1 → single k-iteration,
 * z-stencil terms vanish naturally.
 * ============================================================================ */

/**
 * Compute 3D loop bounds from solver dimensions.
 */
static inline void poisson_solver_compute_3d_bounds(
    size_t nz, size_t nx, size_t ny,
    size_t* stride_z, size_t* k_start, size_t* k_end)
{
    *stride_z = (nz > 1) ? (nx * ny) : 0;
    *k_start  = (nz > 1) ? 1 : 0;
    *k_end    = (nz > 1) ? (nz - 1) : 1;
}

/**
 * Compute inv_dz2 safely (0.0 when dz==0, avoiding division by zero).
 */
static inline double poisson_solver_compute_inv_dz2(double dz) {
    return (dz > 0.0) ? (1.0 / (dz * dz)) : 0.0;
}

/* ============================================================================
 * OPTIMAL SOR OMEGA COMPUTATION
 * ============================================================================ */

/**
 * Compute optimal SOR relaxation parameter for a 2D/3D Laplacian.
 *
 * Uses the Jacobi spectral radius for a rectangular grid:
 *   2D: rho_J = [cos(pi/(nx-1))/dx^2 + cos(pi/(ny-1))/dy^2]
 *                / [1/dx^2 + 1/dy^2]
 *   3D: rho_J = [cos(pi/(nx-1))/dx^2 + cos(pi/(ny-1))/dy^2
 *                + cos(pi/(nz-1))/dz^2]
 *                / [1/dx^2 + 1/dy^2 + 1/dz^2]
 *   omega_opt = 2 / (1 + sqrt(1 - rho_J^2))
 *
 * When nz <= 1 or dz <= 0, the z-component is ignored and the 2D formula
 * is used.
 */
static inline double poisson_solver_compute_optimal_omega(
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz)
{
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = poisson_solver_compute_inv_dz2(dz);

    double num = cos(M_PI / (double)(nx - 1)) * inv_dx2
               + cos(M_PI / (double)(ny - 1)) * inv_dy2;
    double denom = inv_dx2 + inv_dy2;

    if (nz > 1 && inv_dz2 > 0.0) {
        num   += cos(M_PI / (double)(nz - 1)) * inv_dz2;
        denom += inv_dz2;
    }

    double rho_j = num / denom;
    return 2.0 / (1.0 + sqrt(1.0 - (rho_j * rho_j)));
}

/**
 * Optimal SOR omega for zero-gradient (Neumann) walls.
 *
 * Relaxed as poisson_solver_wall_omega() describes, the iteration is SOR on the
 * Neumann matrix A itself, which is consistently ordered, so Young's
 * omega_opt = 2 / (1 + sqrt(1 - rho_J^2)) applies. rho_J = 1 - lambda, with lambda
 * the smallest nonzero eigenvalue of A v = lambda D v, D the diagonal of A; the
 * zero eigenvalue is the constant, the null space every Neumann problem has.
 *
 * The slowest mode varies as cos((i - 1/2) pi / N_a) along one axis a, where N_a
 * is its number of interior points, and is constant along the others. Its
 * Rayleigh quotient is
 *   lambda_a = w_a (2 - 2 cos(pi / N_a)) / (factor - deficit_a)
 *   deficit_a = w_a (4 / N_a) cos^2(pi / (2 N_a)) + sum over b != a of w_b (2 / N_b)
 * with w_b = 1/h_b^2, factor = 2 (w_x + w_y + w_z) the diagonal away from the
 * walls, and deficit_a what the walls take off the diagonal as that mode weighs
 * it. lambda is the smallest lambda_a. A Rayleigh quotient can only overestimate
 * the eigenvalue, so the omega this gives never exceeds the optimum, and it
 * approaches it as the grid grows.
 *
 * An axis with fewer than two interior points has no such mode and is skipped;
 * with no axis left there is nothing to relax, and omega is 1.
 */
static inline double poisson_solver_compute_neumann_omega(
    size_t nx, size_t ny, size_t nz,
    double dx, double dy, double dz)
{
    size_t n[3] = { nx, ny, nz };
    double w[3] = { 1.0 / (dx * dx), 1.0 / (dy * dy), 0.0 };
    if (nz > 1) {
        w[2] = poisson_solver_compute_inv_dz2(dz);
    }
    double factor = 2.0 * (w[0] + w[1] + w[2]);

    /* Interior points per active axis; a 2D grid has none along z */
    double interior[3];
    for (int a = 0; a < 3; a++) {
        interior[a] = (w[a] > 0.0) ? (double)(n[a] - 2) : 0.0;
    }

    double lambda = -1.0;
    for (int a = 0; a < 3; a++) {
        if (interior[a] < 2.0) {
            continue;
        }
        double half = cos(M_PI / (2.0 * interior[a]));
        double deficit = w[a] * (4.0 / interior[a]) * half * half;
        for (int b = 0; b < 3; b++) {
            if (b != a && interior[b] >= 1.0) {
                deficit += w[b] * (2.0 / interior[b]);
            }
        }
        double mode = w[a] * (2.0 - 2.0 * cos(M_PI / interior[a])) / (factor - deficit);
        if (lambda < 0.0 || mode < lambda) {
            lambda = mode;
        }
    }
    if (lambda < 0.0) {
        return 1.0;
    }
    double rho_j = 1.0 - lambda;
    return 2.0 / (1.0 + sqrt(1.0 - (rho_j * rho_j)));
}

/* ============================================================================
 * SOR RELAXATION AT ZERO-GRADIENT WALLS
 * ============================================================================ */

/**
 * Stencil weight of the zero-gradient walls a point touches along one axis:
 * inv_h2 for each of its two neighbours on that axis that is a wall. With n
 * points along the axis, walls included, interior index 1 touches the low wall
 * and n - 2 the high one; a lone interior point (n == 3) touches both.
 */
static inline double poisson_solver_wall_weight(size_t index, size_t n, double inv_h2)
{
    int walls = (index == 1) + (index + 2 == n);
    return walls * inv_h2;
}

/**
 * Relaxation factor for a point beside the default zero-gradient walls.
 *
 * poisson_solver_apply_bc() copies each wall from its inner neighbour after the
 * sweep, so while sweeping, a point next to a wall reads its own previous value
 * through the copy. Relaxed with the plain omega, that stale self-coupling leaves
 * the iteration inconsistently ordered: SOR theory no longer predicts its best
 * omega, which lies well above any formula.
 *
 * Scaling the point's step by factor / (factor - wall_weight) removes the stale
 * term. The update becomes relaxation against the point's real neighbours only,
 * which is SOR on the Neumann matrix itself, whose optimum
 * poisson_solver_compute_neumann_omega() gives. factor is the full stencil weight
 * 2 (1/dx^2 + 1/dy^2 + 1/dz^2) and wall_weight the sum of
 * poisson_solver_wall_weight() over the axes: on a square 2D grid an edge point
 * relaxes with 4/3 omega and a corner point with 2 omega. The converged solution
 * is unchanged, because with the walls copied both updates solve the same
 * equation.
 *
 * A point whose every neighbour is a wall has nothing to relax against and keeps
 * its value.
 */
static inline double poisson_solver_wall_omega(double omega, double factor, double wall_weight)
{
    if (wall_weight <= 0.0) {
        return omega;
    }
    double interior = factor - wall_weight;
    return (interior > 0.0) ? omega * factor / interior : 0.0;
}

/**
 * Whether a CPU solver relaxes against the default zero-gradient wall copy.
 * A caller-supplied apply_bc holds wall values that are data rather than copies
 * of the interior, so its points keep the plain omega.
 */
static inline int poisson_solver_uses_default_walls(const poisson_solver_t* solver)
{
    return solver->apply_bc == NULL;
}

/**
 * Relaxation factors for one row of interior points (fixed j and k): w_row for
 * the points away from the x-walls, w_edge for the first and last (i == 1 and
 * i == nx - 2). Both are the plain omega when walls is 0.
 */
static inline void poisson_solver_row_omegas(
    int walls, double omega, double factor,
    size_t nx, size_t ny, size_t nz, size_t j, size_t k,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double* w_row, double* w_edge)
{
    if (!walls) {
        *w_row = omega;
        *w_edge = omega;
        return;
    }
    double w_yz = poisson_solver_wall_weight(j, ny, inv_dy2)
                + poisson_solver_wall_weight(k, nz, inv_dz2);
    *w_row = poisson_solver_wall_omega(omega, factor, w_yz);
    *w_edge = poisson_solver_wall_omega(omega, factor,
                                        w_yz + poisson_solver_wall_weight(1, nx, inv_dx2));
}

/**
 * The omega a SOR or Red-Black SOR solver relaxes with.
 *
 * Gauss-Seidel is SOR at omega = 1, whatever params.omega says. Otherwise an
 * explicit omega > 0 is used as given, and omega <= 0, the default, asks for the
 * optimum: the Neumann formula for the default wall copy, and the Dirichlet
 * formula above when the caller supplies its own apply_bc. poisson_solver_init()
 * fills in the grid before calling the backend init that resolves omega, so a
 * custom apply_bc must be installed before it.
 */
static inline double poisson_solver_resolve_omega(const poisson_solver_t* solver, double omega)
{
    if (solver->method == POISSON_METHOD_GAUSS_SEIDEL) {
        return 1.0;
    }
    if (omega > 0.0) {
        return omega;
    }
    if (poisson_solver_uses_default_walls(solver)) {
        return poisson_solver_compute_neumann_omega(
            solver->nx, solver->ny, solver->nz, solver->dx, solver->dy, solver->dz);
    }
    return poisson_solver_compute_optimal_omega(
        solver->nx, solver->ny, solver->nz, solver->dx, solver->dy, solver->dz);
}

/* ============================================================================
 * INTERNAL HELPER FUNCTIONS
 * ============================================================================ */

/**
 * Common solve loop used by all iterative solvers
 *
 * Implements the iteration control, convergence checking, and statistics.
 *
 * @param solver Initialized solver
 * @param x Solution vector
 * @param x_temp Temporary buffer
 * @param rhs Right-hand side
 * @param stats Output statistics
 * @return CFD_SUCCESS if converged
 */
cfd_status_t poisson_solver_solve_common(
    poisson_solver_t* solver,
    double* x,
    double* x_temp,
    const double* rhs,
    poisson_solver_stats_t* stats);

/**
 * Get current time in milliseconds (platform-independent)
 */
double poisson_solver_get_time_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* CFD_LINEAR_SOLVER_INTERNAL_H */
