/**
 * Unit Tests for 3D Boundary Conditions
 *
 * Tests the bc_apply_*_3d functions for all supported BC types:
 * - Neumann (zero-gradient) on all 6 faces
 * - Periodic wrap-around on all 6 faces
 * - Dirichlet (fixed value) on all 6 faces
 * - No-slip wall (zero velocity) on all 6 faces
 * - Symmetry on z-planes (FRONT/BACK)
 * - Inlet on FRONT z-face
 * - Outlet (zero-gradient) on BACK z-face
 * - Backward compatibility: nz=1, stride_z=0 matches 2D API
 *
 * Grid layout: field[k * stride_z + j * nx + i], stride_z = nx * ny.
 */

#include "unity.h"
#include "cfd/core/cfd_init.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/core/indexing.h"

#include <stdlib.h>
#include <string.h>

/* Test grid dimensions */
#define NX 5
#define NY 5
#define NZ 5
#define STRIDE_Z (NX * NY)
#define FIELD_SIZE (NX * NY * NZ)

/* Floating-point comparison tolerance */
#define TOL 1e-12

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    cfd_finalize();
}

/* ============================================================================
 * Test: Neumann (zero-gradient) 3D
 * ============================================================================ */

void test_neumann_3d(void) {
    double* field = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "Failed to allocate field");

    /* Fill interior with a known pattern */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t i = 0; i < NX; i++) {
                field[k * STRIDE_Z + IDX_2D(i, j, NX)] =
                    (double)(i + j * 10 + k * 100);
            }
        }
    }

    cfd_status_t status = bc_apply_scalar_3d(field, NX, NY, NZ, STRIDE_Z, BC_TYPE_NEUMANN);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify left/right faces for each z-plane */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            /* Left face: i=0 copies from i=1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + j * NX + 1],
                field[k * STRIDE_Z + j * NX + 0]);
            /* Right face: i=NX-1 copies from i=NX-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + j * NX + NX - 2],
                field[k * STRIDE_Z + j * NX + NX - 1]);
        }
    }

    /* Verify bottom/top faces for each z-plane */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t i = 0; i < NX; i++) {
            /* Bottom face: j=0 copies from j=1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + 1 * NX + i],
                field[k * STRIDE_Z + 0 * NX + i]);
            /* Top face: j=NY-1 copies from j=NY-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + (NY - 2) * NX + i],
                field[k * STRIDE_Z + (NY - 1) * NX + i]);
        }
    }

    /* Verify back/front z-faces */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            /* Back face (k=0) copies from k=1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[1 * STRIDE_Z + j * NX + i],
                field[0 * STRIDE_Z + j * NX + i]);
            /* Front face (k=NZ-1) copies from k=NZ-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[(NZ - 2) * STRIDE_Z + j * NX + i],
                field[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    free(field);
}

/* ============================================================================
 * Test: Periodic 3D
 * ============================================================================ */

void test_periodic_3d(void) {
    double* field = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "Failed to allocate field");

    /* Fill with a known pattern */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t i = 0; i < NX; i++) {
                field[k * STRIDE_Z + IDX_2D(i, j, NX)] =
                    (double)(i + j * 10 + k * 100);
            }
        }
    }

    cfd_status_t status = bc_apply_scalar_3d(field, NX, NY, NZ, STRIDE_Z, BC_TYPE_PERIODIC);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify left/right wrap-around for each z-plane */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            /* Left (i=0) wraps to interior right: nx-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + j * NX + NX - 2],
                field[k * STRIDE_Z + j * NX + 0]);
            /* Right (i=NX-1) wraps to interior left: 1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + j * NX + 1],
                field[k * STRIDE_Z + j * NX + NX - 1]);
        }
    }

    /* Verify bottom/top wrap-around for each z-plane */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t i = 0; i < NX; i++) {
            /* Bottom (j=0) wraps to interior top: ny-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + (NY - 2) * NX + i],
                field[k * STRIDE_Z + 0 * NX + i]);
            /* Top (j=NY-1) wraps to interior bottom: 1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[k * STRIDE_Z + 1 * NX + i],
                field[k * STRIDE_Z + (NY - 1) * NX + i]);
        }
    }

    /* Verify back/front z-face wrap-around */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            /* Back (k=0) wraps to interior front: nz-2 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[(NZ - 2) * STRIDE_Z + j * NX + i],
                field[0 * STRIDE_Z + j * NX + i]);
            /* Front (k=NZ-1) wraps to interior back: 1 */
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[1 * STRIDE_Z + j * NX + i],
                field[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    free(field);
}

/* ============================================================================
 * Test: Dirichlet 3D
 * ============================================================================ */

void test_dirichlet_3d(void) {
    double* field = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "Failed to allocate field");

    bc_dirichlet_values_t values = {
        .left   = 1.0,
        .right  = 2.0,
        .bottom = 3.0,
        .top    = 4.0,
        .front  = 5.0,
        .back   = 6.0
    };

    cfd_status_t status = bc_apply_dirichlet_scalar_3d(
        field, NX, NY, NZ, STRIDE_Z, &values);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Application order: x-faces, then y-faces (overwrite x corners), then z-faces
     * (overwrite entire k=0 and k=NZ-1 planes).
     *
     * Ownership per point:
     *   k=0 or k=NZ-1:      z-face (back=6 / front=5)
     *   j=0 or j=NY-1, interior k: y-face (bottom=3 / top=4), including corners with x
     *   i=0 or i=NX-1, interior k, interior j: x-face (left=1 / right=2)
     */

    /* Verify left face (i=0) on interior z-planes AND interior y-rows */
    for (size_t k = 1; k < NZ - 1; k++) {
        for (size_t j = 1; j < NY - 1; j++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 1.0,
                field[k * STRIDE_Z + j * NX + 0]);
        }
    }

    /* Verify right face (i=NX-1) on interior z-planes AND interior y-rows */
    for (size_t k = 1; k < NZ - 1; k++) {
        for (size_t j = 1; j < NY - 1; j++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 2.0,
                field[k * STRIDE_Z + j * NX + NX - 1]);
        }
    }

    /* Verify bottom face (j=0) on interior z-planes, all i */
    for (size_t k = 1; k < NZ - 1; k++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 3.0,
                field[k * STRIDE_Z + 0 * NX + i]);
        }
    }

    /* Verify top face (j=NY-1) on interior z-planes, all i */
    for (size_t k = 1; k < NZ - 1; k++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 4.0,
                field[k * STRIDE_Z + (NY - 1) * NX + i]);
        }
    }

    /* Verify front face (k=NZ-1): entire plane is val_front=5 */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 5.0,
                field[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    /* Verify back face (k=0): entire plane is val_back=6 */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 6.0,
                field[0 * STRIDE_Z + j * NX + i]);
        }
    }

    free(field);
}

/* ============================================================================
 * Test: No-slip 3D
 * ============================================================================ */

void test_noslip_3d(void) {
    double* u = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* v = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* w = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(u, "Failed to allocate u");
    TEST_ASSERT_NOT_NULL_MESSAGE(v, "Failed to allocate v");
    TEST_ASSERT_NOT_NULL_MESSAGE(w, "Failed to allocate w");

    /* Fill all points with a sentinel value */
    for (size_t n = 0; n < FIELD_SIZE; n++) {
        u[n] = 99.0;
        v[n] = 99.0;
        w[n] = 99.0;
    }

    cfd_status_t status = bc_apply_noslip_3d(u, v, w, NX, NY, NZ, STRIDE_Z);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify left face (i=0): all components zero */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[k * STRIDE_Z + j * NX + 0]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[k * STRIDE_Z + j * NX + 0]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[k * STRIDE_Z + j * NX + 0]);
        }
    }

    /* Verify right face (i=NX-1): all components zero */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[k * STRIDE_Z + j * NX + NX - 1]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[k * STRIDE_Z + j * NX + NX - 1]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[k * STRIDE_Z + j * NX + NX - 1]);
        }
    }

    /* Verify bottom face (j=0): all components zero */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[k * STRIDE_Z + 0 * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[k * STRIDE_Z + 0 * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[k * STRIDE_Z + 0 * NX + i]);
        }
    }

    /* Verify top face (j=NY-1): all components zero */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[k * STRIDE_Z + (NY - 1) * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[k * STRIDE_Z + (NY - 1) * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[k * STRIDE_Z + (NY - 1) * NX + i]);
        }
    }

    /* Verify back face (k=0): all components zero */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[0 * STRIDE_Z + j * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[0 * STRIDE_Z + j * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[0 * STRIDE_Z + j * NX + i]);
        }
    }

    /* Verify front face (k=NZ-1): all components zero */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, u[(NZ - 1) * STRIDE_Z + j * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[(NZ - 1) * STRIDE_Z + j * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    /* Verify a true interior point is unchanged */
    size_t interior_idx = 2 * STRIDE_Z + IDX_2D(2, 2, NX);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, 99.0, u[interior_idx]);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, 99.0, v[interior_idx]);
    TEST_ASSERT_DOUBLE_WITHIN(TOL, 99.0, w[interior_idx]);

    free(u);
    free(v);
    free(w);
}

/* ============================================================================
 * Test: Symmetry on z-planes (FRONT and BACK)
 * ============================================================================ */

void test_symmetry_z_planes(void) {
    double* u = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* v = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* w = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(u, "Failed to allocate u");
    TEST_ASSERT_NOT_NULL_MESSAGE(v, "Failed to allocate v");
    TEST_ASSERT_NOT_NULL_MESSAGE(w, "Failed to allocate w");

    /* Fill u and v with a 2D pattern repeated across all z-planes */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t i = 0; i < NX; i++) {
                u[k * STRIDE_Z + IDX_2D(i, j, NX)] = (double)(i + j * 10);
                v[k * STRIDE_Z + IDX_2D(i, j, NX)] = (double)(i + j * 10);
            }
        }
    }

    /* Fill w with a non-zero value */
    for (size_t n = 0; n < FIELD_SIZE; n++) {
        w[n] = 5.0;
    }

    bc_symmetry_config_t config = { .edges = BC_EDGE_FRONT | BC_EDGE_BACK };

    cfd_status_t status = bc_apply_symmetry_3d(u, v, w, NX, NY, NZ, STRIDE_Z, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* w must be zero on the back face (k=0) and front face (k=NZ-1) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[0 * STRIDE_Z + j * NX + i]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    /* u on back face (k=0) copies from k=1 (zero gradient) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                u[1 * STRIDE_Z + j * NX + i],
                u[0 * STRIDE_Z + j * NX + i]);
        }
    }

    /* v on back face (k=0) copies from k=1 (zero gradient) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                v[1 * STRIDE_Z + j * NX + i],
                v[0 * STRIDE_Z + j * NX + i]);
        }
    }

    /* u on front face (k=NZ-1) copies from k=NZ-2 (zero gradient) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                u[(NZ - 2) * STRIDE_Z + j * NX + i],
                u[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    /* v on front face (k=NZ-1) copies from k=NZ-2 (zero gradient) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                v[(NZ - 2) * STRIDE_Z + j * NX + i],
                v[(NZ - 1) * STRIDE_Z + j * NX + i]);
        }
    }

    free(u);
    free(v);
    free(w);
}

/* ============================================================================
 * Test: Inlet on FRONT z-face
 * ============================================================================ */

void test_inlet_z_face(void) {
    double* u = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* v = (double*)calloc(FIELD_SIZE, sizeof(double));
    double* w = (double*)malloc(FIELD_SIZE * sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(u, "Failed to allocate u");
    TEST_ASSERT_NOT_NULL_MESSAGE(v, "Failed to allocate v");
    TEST_ASSERT_NOT_NULL_MESSAGE(w, "Failed to allocate w");

    /* Initialize w to sentinel so we can detect modification */
    for (size_t n = 0; n < FIELD_SIZE; n++) {
        w[n] = -999.0;
    }

    bc_inlet_config_t config = bc_inlet_config_uniform(1.0, 0.0);
    bc_inlet_set_edge(&config, BC_EDGE_FRONT);

    cfd_status_t status = bc_apply_inlet_3d(u, v, w, NX, NY, NZ, STRIDE_Z, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Verify u=1.0, v=0.0, w=0.0 on front face (k=NZ-1) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            size_t idx = (NZ - 1) * STRIDE_Z + j * NX + i;
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 1.0, u[idx]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, v[idx]);
            TEST_ASSERT_DOUBLE_WITHIN(TOL, 0.0, w[idx]);
        }
    }

    /* Verify w on interior planes was NOT modified */
    for (size_t k = 0; k < NZ - 1; k++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t i = 0; i < NX; i++) {
                TEST_ASSERT_DOUBLE_WITHIN(TOL, -999.0,
                    w[k * STRIDE_Z + j * NX + i]);
            }
        }
    }

    free(u);
    free(v);
    free(w);
}

/* ============================================================================
 * Test: Outlet zero-gradient on BACK z-face
 * ============================================================================ */

void test_outlet_z_face(void) {
    double* field = (double*)calloc(FIELD_SIZE, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field, "Failed to allocate field");

    /* Fill with a known pattern */
    for (size_t k = 0; k < NZ; k++) {
        for (size_t j = 0; j < NY; j++) {
            for (size_t i = 0; i < NX; i++) {
                field[k * STRIDE_Z + IDX_2D(i, j, NX)] =
                    (double)(i + j * 10 + k * 100);
            }
        }
    }

    bc_outlet_config_t config = bc_outlet_config_zero_gradient();
    bc_outlet_set_edge(&config, BC_EDGE_BACK);

    cfd_status_t status = bc_apply_outlet_scalar_3d(
        field, NX, NY, NZ, STRIDE_Z, &config);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status);

    /* Back face (k=0) must equal adjacent interior (k=1) */
    for (size_t j = 0; j < NY; j++) {
        for (size_t i = 0; i < NX; i++) {
            TEST_ASSERT_DOUBLE_WITHIN(TOL,
                field[1 * STRIDE_Z + j * NX + i],
                field[0 * STRIDE_Z + j * NX + i]);
        }
    }

    free(field);
}

/* ============================================================================
 * Test: Backward compatibility — nz=1, stride_z=0 matches 2D API
 * ============================================================================ */

void test_backward_compat_2d_vs_3d(void) {
    const size_t nx = NX;
    const size_t ny = NY;
    const size_t n2d = nx * ny;

    double* field1 = (double*)calloc(n2d, sizeof(double));
    double* field2 = (double*)calloc(n2d, sizeof(double));
    TEST_ASSERT_NOT_NULL_MESSAGE(field1, "Failed to allocate field1");
    TEST_ASSERT_NOT_NULL_MESSAGE(field2, "Failed to allocate field2");

    /* Fill both fields with identical values */
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double val = (double)(i * 3 + j * 7 + 13);
            field1[IDX_2D(i, j, nx)] = val;
            field2[IDX_2D(i, j, nx)] = val;
        }
    }

    /* Apply 2D Neumann to field1 */
    cfd_status_t status1 = bc_apply_scalar(field1, nx, ny, BC_TYPE_NEUMANN);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status1);

    /* Apply 3D Neumann with nz=1, stride_z=0 to field2 */
    cfd_status_t status2 = bc_apply_scalar_3d(field2, nx, ny, 1, 0, BC_TYPE_NEUMANN);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, status2);

    /* Both fields must be identical */
    for (size_t n = 0; n < n2d; n++) {
        TEST_ASSERT_DOUBLE_WITHIN(TOL, field1[n], field2[n]);
    }

    free(field1);
    free(field2);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_neumann_3d);
    RUN_TEST(test_periodic_3d);
    RUN_TEST(test_dirichlet_3d);
    RUN_TEST(test_noslip_3d);
    RUN_TEST(test_symmetry_z_planes);
    RUN_TEST(test_inlet_z_face);
    RUN_TEST(test_outlet_z_face);
    RUN_TEST(test_backward_compat_2d_vs_3d);
    return UNITY_END();
}
