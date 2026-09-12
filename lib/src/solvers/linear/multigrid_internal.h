/**
 * @file multigrid_internal.h
 * @brief Internal data structures and grid-transfer operators for geometric multigrid
 *
 * Not part of the public API. The algorithm itself lives in
 * multigrid_template/linear_solver_multigrid_template.h. The scalar transfer
 * operators live in cpu/multigrid_transfer.c with external linkage but are not
 * exported from the shared library (no CFD_LIBRARY_EXPORT, hidden visibility):
 * the operator unit test links them from the static library, or compiles the
 * translation unit directly in shared builds. The restriction weight rule and
 * the interior mean are static inline here so every backend shares one copy.
 */

#ifndef CFD_MULTIGRID_INTERNAL_H
#define CFD_MULTIGRID_INTERNAL_H

#include "cfd/core/indexing.h"
#include "cfd/solvers/poisson_solver.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Defaults resolved at init when the corresponding param is 0 */
#define MG_DEFAULT_PRE_SMOOTH        2
#define MG_DEFAULT_POST_SMOOTH       2
#define MG_DEFAULT_COARSE_MAX_ITER   50

/* Coarsest-grid size floor per BC mode: a 3x3 Neumann grid's single interior
 * point has an identically zero mirror-BC operator (all stencil neighbors
 * mirror the center), so Neumann coarsening stops at 5. A 3x3 Dirichlet grid
 * is solved exactly by one Gauss-Seidel sweep. */
#define MG_MIN_COARSE_DIM_DIRICHLET  3
#define MG_MIN_COARSE_DIM_NEUMANN    5

/* Weighted Jacobi smoother relaxation: optimal high-frequency damping */
#define MG_JACOBI_OMEGA              (2.0 / 3.0)

/* Minimum points one OpenMP multigrid region must cover to use the thread team.
 * Each OMP primitive opens a region per plane, and starting and joining a team
 * costs as much as sweeping tens of thousands of points, so smaller planes (every
 * coarse level, and small grids) run serially (omp/linear_solver_multigrid_omp.c).
 * test_omp_consistency keeps a configuration above it. */
#define MG_OMP_MIN_POINTS            32768

/**
 * Per-level grid data for the multigrid hierarchy.
 *
 * Level 0 (finest) borrows the caller's x/rhs arrays, so its x and rhs
 * pointers stay NULL; coarser levels own their buffers.
 */
typedef struct {
    size_t nx, ny, nz;      /* Grid dimensions at this level (nz==1 for 2D) */
    size_t total;           /* nx * ny * nz */
    size_t stride_z;        /* nx*ny for 3D, 0 for 2D (branch-free stencils) */
    size_t k_start, k_end;  /* z loop bounds ([1,nz-1) for 3D, [0,1) for 2D) */
    double dx2, dy2;        /* Squared grid spacings at this level */
    double inv_dz2;         /* 1/dz^2 at this level (0.0 for 2D) */
    double inv_factor;      /* 1/(2/dx2 + 2/dy2 + 2*inv_dz2): smoother diagonal inverse */
    double* x;              /* Correction vector (levels >= 1; NULL at level 0) */
    double* x_temp;         /* Jacobi smoother buffer (NULL unless Jacobi smoother) */
    double* rhs;            /* Restricted residual / restricted b (levels >= 1; NULL at level 0) */
    double* residual;       /* Residual buffer (all levels except coarsest) */
} mg_level_t;

/**
 * Multigrid solver context (stored in poisson_solver_t.context)
 */
typedef struct {
    int num_levels;         /* >= 1; [0] = finest, [num_levels-1] = coarsest */
    mg_level_t* levels;

    /* Resolved parameters (defaults already applied) */
    mg_cycle_type_t cycle_type;
    mg_smoother_type_t smoother_type;
    mg_bc_type_t bc_mode;
    int nu1, nu2;           /* Pre/post smoothing sweeps */
    int coarse_max_iter;    /* Smoother sweeps on the coarsest grid */

    int fmg_pending;        /* 1: next iterate() performs the FMG nested iteration */
    int initialized;
} mg_context_t;

/** n is of the form 2^k+1 (k>=1): n-1 is a power of two */
static inline bool mg_is_pow2_plus1(size_t n) {
    return (n >= 3) && (((n - 1) & (n - 2)) == 0);
}

/* ============================================================================
 * GRID-TRANSFER OPERATORS (cpu/multigrid_transfer.c)
 *
 * Scalar reference operators. The OpenMP backend keeps row-parallel static
 * counterparts with the same arithmetic in omp/linear_solver_multigrid_omp.c;
 * both use mg_weights_1d and mg_interior_mean below, so the two backends
 * cannot diverge in the fold rule or the mean.
 *
 * External linkage, intentionally NOT CFD_LIBRARY_EXPORT (hidden outside a
 * shared library build).
 * All write coarse/fine INTERIOR points only. Restriction never reads fine
 * boundary values (coarse-interior stencils stay within the fine interior);
 * prolongation reads coarse boundary values, which the caller must have set
 * consistently with the BC mode (zero for Dirichlet corrections, mirrored
 * for Neumann).
 *
 * fold_neumann: with zero-gradient BCs the boundary vertex is slaved to its
 * interior neighbor (mirror plane at the half-cell), so prolongation reads
 * mirrored ghosts. The adjoint of that folded prolongation doubles the
 * toward-boundary weight at boundary-adjacent coarse points. Pass 1 in
 * Neumann mode to keep restriction the exact adjoint (up to the standard
 * 1/4 (2D) or 1/8 (3D) factor); pass 0 for Dirichlet.
 * ============================================================================ */

/**
 * Per-dimension 1D restriction weights at a coarse index: (1,2,1) in the
 * interior. With fold_neumann, the toward-boundary weight doubles at
 * boundary-adjacent coarse points (I==1 and/or I==n_coarse-2) — the exact
 * adjoint of prolongation through mirrored (zero-gradient) ghost values.
 */
static inline void mg_weights_1d(size_t idx, size_t n_coarse, int fold_neumann,
                                 double w[3]) {
    w[0] = (fold_neumann && idx == 1) ? 2.0 : 1.0;
    w[1] = 2.0;
    w[2] = (fold_neumann && idx == n_coarse - 2) ? 2.0 : 1.0;
}

/** Full-weighting restriction, 2D (4-2-1)/16 stencil; coarse interior only */
void mg_restrict_2d(const double* fine, double* coarse,
                    size_t nxf, size_t nyf, size_t nxc, size_t nyc,
                    int fold_neumann);

/** Full-weighting restriction, 3D 27-point (8-4-2-1)/64 stencil */
void mg_restrict_3d(const double* fine, double* coarse,
                    size_t nxf, size_t nyf, size_t nzf,
                    size_t nxc, size_t nyc, size_t nzc,
                    int fold_neumann);

/** Bilinear prolongation, adds into fine interior points only */
void mg_prolongate_add_2d(const double* coarse, double* fine,
                          size_t nxc, size_t nyc, size_t nxf, size_t nyf);

/** Trilinear prolongation, adds into fine interior points only */
void mg_prolongate_add_3d(const double* coarse, double* fine,
                          size_t nxc, size_t nyc, size_t nzc,
                          size_t nxf, size_t nyf, size_t nzf);

/**
 * Mean of f over interior points. Serial by design: every backend must
 * produce the identical mean, so it is never computed with a parallel
 * reduction.
 */
static inline double mg_interior_mean(const double* f, size_t nx, size_t ny,
                                      size_t nz) {
    size_t stride_z = (nz > 1) ? nx * ny : 0;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? nz - 1 : 1;

    double sum = 0.0;
    size_t count = 0;
    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                sum += f[k * stride_z + IDX_2D(i, j, nx)];
                count++;
            }
        }
    }
    return (count > 0) ? sum / (double)count : 0.0;
}

/** Subtract the interior mean from interior points (boundary untouched) */
void mg_subtract_interior_mean(double* f, size_t nx, size_t ny, size_t nz);

#ifdef __cplusplus
}
#endif

#endif /* CFD_MULTIGRID_INTERNAL_H */
