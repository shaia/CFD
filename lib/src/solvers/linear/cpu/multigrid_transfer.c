/**
 * @file multigrid_transfer.c
 * @brief Grid-transfer operators for geometric multigrid (restriction/prolongation)
 *
 * Pure array math with no library runtime dependencies, so the operator unit
 * test can compile this translation unit directly (see multigrid_internal.h).
 *
 * Grid relationship: nxf = 2*nxc - 1 (fine dims are 2^k+1, coarse (2^(k-1))+1).
 * Coarse interior point (I,J,K) sits at fine point (2I,2J,2K); its
 * full-weighting stencil spans fine indices 2I-1..2I+1, which always lie in
 * the fine interior for coarse-interior I. Restriction therefore never reads
 * fine boundary values, and both operators write interior points only.
 */

#include "../multigrid_internal.h"

#include "cfd/core/indexing.h"

/* ============================================================================
 * RESTRICTION (fine -> coarse, full weighting)
 * ============================================================================ */

/**
 * Per-dimension 1D restriction weights at a coarse index: (1,2,1) in the
 * interior. With fold_neumann, the toward-boundary weight doubles at
 * boundary-adjacent coarse points (I==1 and/or I==n_coarse-2) — the exact
 * adjoint of prolongation through mirrored (zero-gradient) ghost values.
 */
static void mg_weights_1d(size_t idx, size_t n_coarse, int fold_neumann,
                          double w[3]) {
    w[0] = (fold_neumann && idx == 1) ? 2.0 : 1.0;
    w[1] = 2.0;
    w[2] = (fold_neumann && idx == n_coarse - 2) ? 2.0 : 1.0;
}

void mg_restrict_2d(const double* fine, double* coarse,
                    size_t nxf, size_t nyf, size_t nxc, size_t nyc,
                    int fold_neumann) {
    (void)nyf;
    /* Separable full weighting: interior stencil (1/16)*(4 center, 2 edge,
     * 1 corner) */
    for (size_t J = 1; J < nyc - 1; J++) {
        double wy[3];
        mg_weights_1d(J, nyc, fold_neumann, wy);
        for (size_t I = 1; I < nxc - 1; I++) {
            double wx[3];
            mg_weights_1d(I, nxc, fold_neumann, wx);

            size_t i = 2 * I;
            size_t j = 2 * J;
            double sum = 0.0;
            for (int oj = -1; oj <= 1; oj++) {
                for (int oi = -1; oi <= 1; oi++) {
                    double w = wx[oi + 1] * wy[oj + 1];
                    sum += w * fine[IDX_2D(i + oi, j + oj, nxf)];
                }
            }
            coarse[IDX_2D(I, J, nxc)] = sum / 16.0;
        }
    }
}

void mg_restrict_3d(const double* fine, double* coarse,
                    size_t nxf, size_t nyf, size_t nzf,
                    size_t nxc, size_t nyc, size_t nzc,
                    int fold_neumann) {
    (void)nzf;
    /* Separable 27-point full weighting: interior stencil (1/64)*(8 center,
     * 4 face, 2 edge, 1 corner) */
    for (size_t K = 1; K < nzc - 1; K++) {
        double wz[3];
        mg_weights_1d(K, nzc, fold_neumann, wz);
        for (size_t J = 1; J < nyc - 1; J++) {
            double wy[3];
            mg_weights_1d(J, nyc, fold_neumann, wy);
            for (size_t I = 1; I < nxc - 1; I++) {
                double wx[3];
                mg_weights_1d(I, nxc, fold_neumann, wx);

                size_t i = 2 * I;
                size_t j = 2 * J;
                size_t k = 2 * K;
                double sum = 0.0;
                for (int ok = -1; ok <= 1; ok++) {
                    for (int oj = -1; oj <= 1; oj++) {
                        for (int oi = -1; oi <= 1; oi++) {
                            double w = wx[oi + 1] * wy[oj + 1] * wz[ok + 1];
                            sum += w * fine[IDX_3D(i + oi, j + oj, k + ok,
                                                   nxf, nyf)];
                        }
                    }
                }
                coarse[IDX_3D(I, J, K, nxc, nyc)] = sum / 64.0;
            }
        }
    }
}

/* ============================================================================
 * PROLONGATION (coarse -> fine, bilinear/trilinear, additive)
 * ============================================================================ */

void mg_prolongate_add_2d(const double* coarse, double* fine,
                          size_t nxc, size_t nyc, size_t nxf, size_t nyf) {
    (void)nyc;
    for (size_t j = 1; j < nyf - 1; j++) {
        size_t J = j >> 1;
        int jodd = (int)(j & 1);
        for (size_t i = 1; i < nxf - 1; i++) {
            size_t I = i >> 1;
            int iodd = (int)(i & 1);
            double e;
            if (!iodd && !jodd) {
                e = coarse[IDX_2D(I, J, nxc)];
            } else if (iodd && !jodd) {
                e = 0.5 * (coarse[IDX_2D(I, J, nxc)]
                         + coarse[IDX_2D(I + 1, J, nxc)]);
            } else if (!iodd && jodd) {
                e = 0.5 * (coarse[IDX_2D(I, J, nxc)]
                         + coarse[IDX_2D(I, J + 1, nxc)]);
            } else {
                e = 0.25 * (coarse[IDX_2D(I, J, nxc)]
                          + coarse[IDX_2D(I + 1, J, nxc)]
                          + coarse[IDX_2D(I, J + 1, nxc)]
                          + coarse[IDX_2D(I + 1, J + 1, nxc)]);
            }
            fine[IDX_2D(i, j, nxf)] += e;
        }
    }
}

void mg_prolongate_add_3d(const double* coarse, double* fine,
                          size_t nxc, size_t nyc, size_t nzc,
                          size_t nxf, size_t nyf, size_t nzf) {
    (void)nzc;
    for (size_t k = 1; k < nzf - 1; k++) {
        size_t K = k >> 1;
        size_t nk = (k & 1) ? 2 : 1;
        double wk = (k & 1) ? 0.5 : 1.0;
        for (size_t j = 1; j < nyf - 1; j++) {
            size_t J = j >> 1;
            size_t nj = (j & 1) ? 2 : 1;
            double wj = (j & 1) ? 0.5 : 1.0;
            for (size_t i = 1; i < nxf - 1; i++) {
                size_t I = i >> 1;
                size_t ni = (i & 1) ? 2 : 1;
                double wi = (i & 1) ? 0.5 : 1.0;
                double e = 0.0;
                for (size_t ck = 0; ck < nk; ck++) {
                    for (size_t cj = 0; cj < nj; cj++) {
                        for (size_t ci = 0; ci < ni; ci++) {
                            e += wi * wj * wk
                               * coarse[IDX_3D(I + ci, J + cj, K + ck,
                                               nxc, nyc)];
                        }
                    }
                }
                fine[IDX_3D(i, j, k, nxf, nyf)] += e;
            }
        }
    }
}

/* ============================================================================
 * INTERIOR MEAN (Neumann nullspace handling)
 * ============================================================================ */

double mg_interior_mean(const double* f, size_t nx, size_t ny, size_t nz) {
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

void mg_subtract_interior_mean(double* f, size_t nx, size_t ny, size_t nz) {
    double mean = mg_interior_mean(f, nx, ny, nz);

    size_t stride_z = (nz > 1) ? nx * ny : 0;
    size_t k_start = (nz > 1) ? 1 : 0;
    size_t k_end = (nz > 1) ? nz - 1 : 1;

    for (size_t k = k_start; k < k_end; k++) {
        for (size_t j = 1; j < ny - 1; j++) {
            for (size_t i = 1; i < nx - 1; i++) {
                f[k * stride_z + IDX_2D(i, j, nx)] -= mean;
            }
        }
    }
}
