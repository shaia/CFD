/**
 * @file test_multigrid_operators.c
 * @brief Unit tests for multigrid grid-transfer operators
 *
 * Compiles lib/src/solvers/linear/cpu/multigrid_transfer.c directly (the
 * operators have internal linkage and are not exported from the library).
 * Deliberately uses no poisson_solver_* API and no cfd runtime — pure math
 * verification of the operators before they are used in the solver.
 *
 * Tests cover:
 *   - Restriction exactness for constant and linear fields (2D/3D)
 *   - Restriction writes coarse interior only (boundary contract)
 *   - Prolongation exactness for constant and linear fields (2D/3D)
 *   - Prolongation adds into fine interior only (boundary contract)
 *   - Adjointness: <Rf, g> = (1/4)<f, Pg> in 2D, (1/8) in 3D
 *   - Conservation: sum(Rf) = (1/4)sum(f) in 2D, (1/8) in 3D
 *   - Interior mean subtraction
 *   - 2^k+1 dimension predicate
 */

#include "unity.h"
#include "../../lib/src/solvers/linear/multigrid_internal.h"
#include "cfd/core/indexing.h"

#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

#define SENTINEL 999.0

/* ============================================================================
 * HELPERS
 * ============================================================================ */

/** Deterministic LCG in [-1, 1] (no rand() — reproducible across platforms) */
static unsigned int g_lcg_state = 12345u;

static double lcg_next(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return (double)(g_lcg_state >> 8) / (double)(1u << 24) * 2.0 - 1.0;
}

static double* alloc_zeroed(size_t n) {
    double* p = (double*)calloc(n, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(p, "allocation failed");
    return p;
}

/** Fill a 2D field with f(x,y) = a + b*x + c*y on the unit square */
static void fill_linear_2d(double* f, size_t nx, size_t ny,
                           double a, double b, double c) {
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            f[IDX_2D(i, j, nx)] = a + b * (double)i * dx + c * (double)j * dy;
        }
    }
}

/** Fill a 3D field with f(x,y,z) = a + b*x + c*y + d*z on the unit cube */
static void fill_linear_3d(double* f, size_t nx, size_t ny, size_t nz,
                           double a, double b, double c, double d) {
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);
    double dz = 1.0 / (double)(nz - 1);
    for (size_t k = 0; k < nz; k++) {
        for (size_t j = 0; j < ny; j++) {
            for (size_t i = 0; i < nx; i++) {
                f[IDX_3D(i, j, k, nx, ny)] = a + b * (double)i * dx
                                           + c * (double)j * dy
                                           + d * (double)k * dz;
            }
        }
    }
}

static int is_boundary_2d(size_t i, size_t j, size_t nx, size_t ny) {
    return i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
}

static int is_boundary_3d(size_t i, size_t j, size_t k,
                          size_t nx, size_t ny, size_t nz) {
    return is_boundary_2d(i, j, nx, ny) || k == 0 || k == nz - 1;
}

/* ============================================================================
 * RESTRICTION TESTS (2D)
 * ============================================================================ */

void test_restrict_2d_constant_exact(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* fine = alloc_zeroed(NXF * NYF);
    double* coarse = alloc_zeroed(NXC * NYC);

    fill_linear_2d(fine, NXF, NYF, 3.7, 0.0, 0.0);
    mg_restrict_2d(fine, coarse, NXF, NYF, NXC, NYC, 0);

    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            TEST_ASSERT_DOUBLE_WITHIN(1e-14, 3.7, coarse[IDX_2D(I, J, NXC)]);
        }
    }

    free(fine);
    free(coarse);
}

void test_restrict_2d_linear_exact(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* fine = alloc_zeroed(NXF * NYF);
    double* coarse = alloc_zeroed(NXC * NYC);

    fill_linear_2d(fine, NXF, NYF, 1.0, 2.0, 3.0);
    mg_restrict_2d(fine, coarse, NXF, NYF, NXC, NYC, 0);

    /* Coarse node (I,J) is coincident with fine node (2I,2J); full weighting
     * of a linear field returns the center value exactly (symmetric stencil) */
    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            double expected = fine[IDX_2D(2 * I, 2 * J, NXF)];
            TEST_ASSERT_DOUBLE_WITHIN(1e-13, expected,
                                      coarse[IDX_2D(I, J, NXC)]);
        }
    }

    free(fine);
    free(coarse);
}

void test_restrict_2d_boundary_contract(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* fine = alloc_zeroed(NXF * NYF);
    double* coarse = alloc_zeroed(NXC * NYC);

    for (size_t i = 0; i < NXF * NYF; i++) {
        fine[i] = lcg_next();
    }
    for (size_t J = 0; J < NYC; J++) {
        for (size_t I = 0; I < NXC; I++) {
            if (is_boundary_2d(I, J, NXC, NYC)) {
                coarse[IDX_2D(I, J, NXC)] = SENTINEL;
            }
        }
    }

    mg_restrict_2d(fine, coarse, NXF, NYF, NXC, NYC, 0);

    for (size_t J = 0; J < NYC; J++) {
        for (size_t I = 0; I < NXC; I++) {
            if (is_boundary_2d(I, J, NXC, NYC)) {
                TEST_ASSERT_EQUAL_DOUBLE(SENTINEL, coarse[IDX_2D(I, J, NXC)]);
            }
        }
    }

    free(fine);
    free(coarse);
}

/* ============================================================================
 * RESTRICTION TESTS (3D)
 * ============================================================================ */

void test_restrict_3d_constant_and_linear_exact(void) {
    const size_t NF = 9, NC = 5;
    double* fine = alloc_zeroed(NF * NF * NF);
    double* coarse = alloc_zeroed(NC * NC * NC);

    /* Constant */
    fill_linear_3d(fine, NF, NF, NF, -2.5, 0.0, 0.0, 0.0);
    mg_restrict_3d(fine, coarse, NF, NF, NF, NC, NC, NC, 0);
    for (size_t K = 1; K < NC - 1; K++) {
        for (size_t J = 1; J < NC - 1; J++) {
            for (size_t I = 1; I < NC - 1; I++) {
                TEST_ASSERT_DOUBLE_WITHIN(1e-14, -2.5,
                                          coarse[IDX_3D(I, J, K, NC, NC)]);
            }
        }
    }

    /* Linear */
    fill_linear_3d(fine, NF, NF, NF, 1.0, 2.0, 3.0, 4.0);
    mg_restrict_3d(fine, coarse, NF, NF, NF, NC, NC, NC, 0);
    for (size_t K = 1; K < NC - 1; K++) {
        for (size_t J = 1; J < NC - 1; J++) {
            for (size_t I = 1; I < NC - 1; I++) {
                double expected = fine[IDX_3D(2 * I, 2 * J, 2 * K, NF, NF)];
                TEST_ASSERT_DOUBLE_WITHIN(1e-13, expected,
                                          coarse[IDX_3D(I, J, K, NC, NC)]);
            }
        }
    }

    free(fine);
    free(coarse);
}

/* ============================================================================
 * PROLONGATION TESTS (2D)
 * ============================================================================ */

void test_prolongate_2d_constant_exact(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* fine = alloc_zeroed(NXF * NYF);
    double* coarse = alloc_zeroed(NXC * NYC);

    fill_linear_2d(coarse, NXC, NYC, 4.2, 0.0, 0.0); /* incl. boundary */
    mg_prolongate_add_2d(coarse, fine, NXC, NYC, NXF, NYF);

    for (size_t j = 0; j < NYF; j++) {
        for (size_t i = 0; i < NXF; i++) {
            if (is_boundary_2d(i, j, NXF, NYF)) {
                /* Additive prolongation must not touch the fine boundary */
                TEST_ASSERT_EQUAL_DOUBLE(0.0, fine[IDX_2D(i, j, NXF)]);
            } else {
                TEST_ASSERT_DOUBLE_WITHIN(1e-14, 4.2, fine[IDX_2D(i, j, NXF)]);
            }
        }
    }

    free(fine);
    free(coarse);
}

void test_prolongate_2d_linear_exact(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* fine = alloc_zeroed(NXF * NYF);
    double* coarse = alloc_zeroed(NXC * NYC);
    double* expected = alloc_zeroed(NXF * NYF);

    fill_linear_2d(coarse, NXC, NYC, 1.0, 2.0, 3.0);
    fill_linear_2d(expected, NXF, NYF, 1.0, 2.0, 3.0);
    mg_prolongate_add_2d(coarse, fine, NXC, NYC, NXF, NYF);

    /* Bilinear interpolation is exact for linear functions */
    for (size_t j = 1; j < NYF - 1; j++) {
        for (size_t i = 1; i < NXF - 1; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(1e-13, expected[IDX_2D(i, j, NXF)],
                                      fine[IDX_2D(i, j, NXF)]);
        }
    }

    free(fine);
    free(coarse);
    free(expected);
}

/* ============================================================================
 * PROLONGATION TESTS (3D)
 * ============================================================================ */

void test_prolongate_3d_linear_exact(void) {
    const size_t NF = 9, NC = 5;
    double* fine = alloc_zeroed(NF * NF * NF);
    double* coarse = alloc_zeroed(NC * NC * NC);
    double* expected = alloc_zeroed(NF * NF * NF);

    fill_linear_3d(coarse, NC, NC, NC, 1.0, 2.0, 3.0, 4.0);
    fill_linear_3d(expected, NF, NF, NF, 1.0, 2.0, 3.0, 4.0);
    mg_prolongate_add_3d(coarse, fine, NC, NC, NC, NF, NF, NF);

    for (size_t k = 0; k < NF; k++) {
        for (size_t j = 0; j < NF; j++) {
            for (size_t i = 0; i < NF; i++) {
                if (is_boundary_3d(i, j, k, NF, NF, NF)) {
                    TEST_ASSERT_EQUAL_DOUBLE(0.0,
                                             fine[IDX_3D(i, j, k, NF, NF)]);
                } else {
                    TEST_ASSERT_DOUBLE_WITHIN(1e-13,
                                              expected[IDX_3D(i, j, k, NF, NF)],
                                              fine[IDX_3D(i, j, k, NF, NF)]);
                }
            }
        }
    }

    free(fine);
    free(coarse);
    free(expected);
}

/* ============================================================================
 * ADJOINTNESS: <Rf, g> = c * <f, Pg>  (c = 1/4 in 2D, 1/8 in 3D)
 *
 * Essential for the coarse-grid operator to inherit positive-definiteness.
 * f, g have zero boundary values so the interior-truncated operators satisfy
 * the identity exactly.
 * ============================================================================ */

void test_adjointness_2d(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* f = alloc_zeroed(NXF * NYF);
    double* g = alloc_zeroed(NXC * NYC);
    double* Rf = alloc_zeroed(NXC * NYC);
    double* Pg = alloc_zeroed(NXF * NYF);

    for (size_t j = 1; j < NYF - 1; j++) {
        for (size_t i = 1; i < NXF - 1; i++) {
            f[IDX_2D(i, j, NXF)] = lcg_next();
        }
    }
    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            g[IDX_2D(I, J, NXC)] = lcg_next();
        }
    }

    mg_restrict_2d(f, Rf, NXF, NYF, NXC, NYC, 0);
    mg_prolongate_add_2d(g, Pg, NXC, NYC, NXF, NYF);

    double lhs = 0.0, rhs = 0.0;
    for (size_t n = 0; n < NXC * NYC; n++) {
        lhs += Rf[n] * g[n];
    }
    for (size_t n = 0; n < NXF * NYF; n++) {
        rhs += f[n] * Pg[n];
    }
    rhs *= 0.25;

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(rhs) + 1e-14, rhs, lhs);

    free(f);
    free(g);
    free(Rf);
    free(Pg);
}

void test_adjointness_3d(void) {
    const size_t NF = 9, NC = 5;
    double* f = alloc_zeroed(NF * NF * NF);
    double* g = alloc_zeroed(NC * NC * NC);
    double* Rf = alloc_zeroed(NC * NC * NC);
    double* Pg = alloc_zeroed(NF * NF * NF);

    for (size_t k = 1; k < NF - 1; k++) {
        for (size_t j = 1; j < NF - 1; j++) {
            for (size_t i = 1; i < NF - 1; i++) {
                f[IDX_3D(i, j, k, NF, NF)] = lcg_next();
            }
        }
    }
    for (size_t K = 1; K < NC - 1; K++) {
        for (size_t J = 1; J < NC - 1; J++) {
            for (size_t I = 1; I < NC - 1; I++) {
                g[IDX_3D(I, J, K, NC, NC)] = lcg_next();
            }
        }
    }

    mg_restrict_3d(f, Rf, NF, NF, NF, NC, NC, NC, 0);
    mg_prolongate_add_3d(g, Pg, NC, NC, NC, NF, NF, NF);

    double lhs = 0.0, rhs = 0.0;
    for (size_t n = 0; n < NC * NC * NC; n++) {
        lhs += Rf[n] * g[n];
    }
    for (size_t n = 0; n < NF * NF * NF; n++) {
        rhs += f[n] * Pg[n];
    }
    rhs *= 0.125;

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(rhs) + 1e-14, rhs, lhs);

    free(f);
    free(g);
    free(Rf);
    free(Pg);
}

/* ============================================================================
 * CONSERVATION: sum(Rf) = c * sum(f)  (c = 1/4 in 2D, 1/8 in 3D)
 *
 * Holds when f is supported away from the first interior ring, so every
 * nonzero fine point has its full restriction-stencil coverage (total
 * weight exactly 1/4 per fine point in 2D, 1/8 in 3D).
 * ============================================================================ */

void test_conservation_2d(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* f = alloc_zeroed(NXF * NYF);
    double* Rf = alloc_zeroed(NXC * NYC);

    double sum_f = 0.0;
    for (size_t j = 2; j < NYF - 2; j++) {
        for (size_t i = 2; i < NXF - 2; i++) {
            double v = lcg_next();
            f[IDX_2D(i, j, NXF)] = v;
            sum_f += v;
        }
    }

    mg_restrict_2d(f, Rf, NXF, NYF, NXC, NYC, 0);

    double sum_Rf = 0.0;
    for (size_t n = 0; n < NXC * NYC; n++) {
        sum_Rf += Rf[n];
    }

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(sum_f) + 1e-14,
                              0.25 * sum_f, sum_Rf);

    free(f);
    free(Rf);
}

void test_conservation_3d(void) {
    const size_t NF = 9, NC = 5;
    double* f = alloc_zeroed(NF * NF * NF);
    double* Rf = alloc_zeroed(NC * NC * NC);

    double sum_f = 0.0;
    for (size_t k = 2; k < NF - 2; k++) {
        for (size_t j = 2; j < NF - 2; j++) {
            for (size_t i = 2; i < NF - 2; i++) {
                double v = lcg_next();
                f[IDX_3D(i, j, k, NF, NF)] = v;
                sum_f += v;
            }
        }
    }

    mg_restrict_3d(f, Rf, NF, NF, NF, NC, NC, NC, 0);

    double sum_Rf = 0.0;
    for (size_t n = 0; n < NC * NC * NC; n++) {
        sum_Rf += Rf[n];
    }

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(sum_f) + 1e-14,
                              0.125 * sum_f, sum_Rf);

    free(f);
    free(Rf);
}

/* ============================================================================
 * NEUMANN-FOLDED RESTRICTION (fold_neumann = 1)
 *
 * With zero-gradient BCs, prolongation reads mirrored ghost values (boundary
 * = adjacent interior; corners = diagonal interior). The folded restriction
 * must be the exact adjoint of that folded prolongation, and constants must
 * be reproduced by the folded prolongation, which makes the conservation
 * identity hold for ARBITRARY interior fields.
 * ============================================================================ */

/** Mirror ghost fill matching bc_apply_scalar Neumann: x-faces then y-faces
 * (corners end up as diagonal interior copies) */
static void mirror_fill_2d(double* f, size_t nx, size_t ny) {
    for (size_t j = 0; j < ny; j++) {
        f[IDX_2D(0, j, nx)] = f[IDX_2D(1, j, nx)];
        f[IDX_2D(nx - 1, j, nx)] = f[IDX_2D(nx - 2, j, nx)];
    }
    for (size_t i = 0; i < nx; i++) {
        f[IDX_2D(i, 0, nx)] = f[IDX_2D(i, 1, nx)];
        f[IDX_2D(i, ny - 1, nx)] = f[IDX_2D(i, ny - 2, nx)];
    }
}

/** 3D mirror fill: x-faces, y-faces per plane, then z-plane copies */
static void mirror_fill_3d(double* f, size_t nx, size_t ny, size_t nz) {
    for (size_t k = 0; k < nz; k++) {
        mirror_fill_2d(f + k * nx * ny, nx, ny);
    }
    for (size_t n = 0; n < nx * ny; n++) {
        f[n] = f[nx * ny + n];
        f[(nz - 1) * nx * ny + n] = f[(nz - 2) * nx * ny + n];
    }
}

void test_adjointness_2d_folded(void) {
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* f = alloc_zeroed(NXF * NYF);
    double* g = alloc_zeroed(NXC * NYC);
    double* Rf = alloc_zeroed(NXC * NYC);
    double* Pg = alloc_zeroed(NXF * NYF);

    for (size_t j = 1; j < NYF - 1; j++) {
        for (size_t i = 1; i < NXF - 1; i++) {
            f[IDX_2D(i, j, NXF)] = lcg_next();
        }
    }
    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            g[IDX_2D(I, J, NXC)] = lcg_next();
        }
    }

    mg_restrict_2d(f, Rf, NXF, NYF, NXC, NYC, 1);
    mirror_fill_2d(g, NXC, NYC);
    mg_prolongate_add_2d(g, Pg, NXC, NYC, NXF, NYF);

    double lhs = 0.0, rhs = 0.0;
    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            lhs += Rf[IDX_2D(I, J, NXC)] * g[IDX_2D(I, J, NXC)];
        }
    }
    for (size_t j = 1; j < NYF - 1; j++) {
        for (size_t i = 1; i < NXF - 1; i++) {
            rhs += f[IDX_2D(i, j, NXF)] * Pg[IDX_2D(i, j, NXF)];
        }
    }
    rhs *= 0.25;

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(rhs) + 1e-14, rhs, lhs);

    free(f);
    free(g);
    free(Rf);
    free(Pg);
}

void test_adjointness_3d_folded(void) {
    const size_t NF = 9, NC = 5;
    double* f = alloc_zeroed(NF * NF * NF);
    double* g = alloc_zeroed(NC * NC * NC);
    double* Rf = alloc_zeroed(NC * NC * NC);
    double* Pg = alloc_zeroed(NF * NF * NF);

    for (size_t k = 1; k < NF - 1; k++) {
        for (size_t j = 1; j < NF - 1; j++) {
            for (size_t i = 1; i < NF - 1; i++) {
                f[IDX_3D(i, j, k, NF, NF)] = lcg_next();
            }
        }
    }
    for (size_t K = 1; K < NC - 1; K++) {
        for (size_t J = 1; J < NC - 1; J++) {
            for (size_t I = 1; I < NC - 1; I++) {
                g[IDX_3D(I, J, K, NC, NC)] = lcg_next();
            }
        }
    }

    mg_restrict_3d(f, Rf, NF, NF, NF, NC, NC, NC, 1);
    mirror_fill_3d(g, NC, NC, NC);
    mg_prolongate_add_3d(g, Pg, NC, NC, NC, NF, NF, NF);

    double lhs = 0.0, rhs = 0.0;
    for (size_t K = 1; K < NC - 1; K++) {
        for (size_t J = 1; J < NC - 1; J++) {
            for (size_t I = 1; I < NC - 1; I++) {
                lhs += Rf[IDX_3D(I, J, K, NC, NC)]
                     * g[IDX_3D(I, J, K, NC, NC)];
            }
        }
    }
    for (size_t k = 1; k < NF - 1; k++) {
        for (size_t j = 1; j < NF - 1; j++) {
            for (size_t i = 1; i < NF - 1; i++) {
                rhs += f[IDX_3D(i, j, k, NF, NF)]
                     * Pg[IDX_3D(i, j, k, NF, NF)];
            }
        }
    }
    rhs *= 0.125;

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(rhs) + 1e-14, rhs, lhs);

    free(f);
    free(g);
    free(Rf);
    free(Pg);
}

void test_conservation_2d_folded(void) {
    /* Folded prolongation reproduces constants exactly, so conservation
     * holds for ANY interior field (no support restriction) */
    const size_t NXF = 17, NYF = 17, NXC = 9, NYC = 9;
    double* f = alloc_zeroed(NXF * NYF);
    double* Rf = alloc_zeroed(NXC * NYC);

    double sum_f = 0.0;
    for (size_t j = 1; j < NYF - 1; j++) {
        for (size_t i = 1; i < NXF - 1; i++) {
            double v = lcg_next();
            f[IDX_2D(i, j, NXF)] = v;
            sum_f += v;
        }
    }

    mg_restrict_2d(f, Rf, NXF, NYF, NXC, NYC, 1);

    double sum_Rf = 0.0;
    for (size_t J = 1; J < NYC - 1; J++) {
        for (size_t I = 1; I < NXC - 1; I++) {
            sum_Rf += Rf[IDX_2D(I, J, NXC)];
        }
    }

    TEST_ASSERT_DOUBLE_WITHIN(1e-12 * fabs(sum_f) + 1e-14,
                              0.25 * sum_f, sum_Rf);

    free(f);
    free(Rf);
}

/* ============================================================================
 * INTERIOR MEAN
 * ============================================================================ */

void test_subtract_interior_mean(void) {
    const size_t NX = 17, NY = 9;
    double* f = alloc_zeroed(NX * NY);

    for (size_t n = 0; n < NX * NY; n++) {
        f[n] = 5.0 + lcg_next();
    }
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            if (is_boundary_2d(i, j, NX, NY)) {
                f[IDX_2D(i, j, NX)] = SENTINEL;
            }
        }
    }

    mg_subtract_interior_mean(f, NX, NY, 1);

    TEST_ASSERT_DOUBLE_WITHIN(1e-14, 0.0, mg_interior_mean(f, NX, NY, 1));
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            if (is_boundary_2d(i, j, NX, NY)) {
                TEST_ASSERT_EQUAL_DOUBLE(SENTINEL, f[IDX_2D(i, j, NX)]);
            }
        }
    }

    free(f);
}

/* ============================================================================
 * DIMENSION PREDICATE
 * ============================================================================ */

void test_is_pow2_plus1(void) {
    static const size_t valid[] = {3, 5, 9, 17, 33, 65, 129, 257};
    static const size_t invalid[] = {0, 1, 2, 4, 6, 7, 8, 10, 16, 32, 34};

    for (size_t n = 0; n < sizeof(valid) / sizeof(valid[0]); n++) {
        TEST_ASSERT_TRUE_MESSAGE(mg_is_pow2_plus1(valid[n]),
                                 "expected 2^k+1 dimension to be accepted");
    }
    for (size_t n = 0; n < sizeof(invalid) / sizeof(invalid[0]); n++) {
        TEST_ASSERT_FALSE_MESSAGE(mg_is_pow2_plus1(invalid[n]),
                                  "expected non-2^k+1 dimension to be rejected");
    }
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_restrict_2d_constant_exact);
    RUN_TEST(test_restrict_2d_linear_exact);
    RUN_TEST(test_restrict_2d_boundary_contract);
    RUN_TEST(test_restrict_3d_constant_and_linear_exact);

    RUN_TEST(test_prolongate_2d_constant_exact);
    RUN_TEST(test_prolongate_2d_linear_exact);
    RUN_TEST(test_prolongate_3d_linear_exact);

    RUN_TEST(test_adjointness_2d);
    RUN_TEST(test_adjointness_3d);
    RUN_TEST(test_conservation_2d);
    RUN_TEST(test_conservation_3d);

    RUN_TEST(test_adjointness_2d_folded);
    RUN_TEST(test_adjointness_3d_folded);
    RUN_TEST(test_conservation_2d_folded);

    RUN_TEST(test_subtract_interior_mean);
    RUN_TEST(test_is_pow2_plus1);

    return UNITY_END();
}
