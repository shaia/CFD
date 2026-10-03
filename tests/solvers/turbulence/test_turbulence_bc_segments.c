/**
 * @file test_turbulence_bc_segments.c
 * @brief Unit tests for turbulence BCs on part of a face (ns_turbulence_bc_segment_t)
 *
 * Tests:
 * 1. Zero-initialized and default params carry no segments.
 * 2. A segment covering a whole face gives bit-identical fields to the same type
 *    set on the face itself, for DIRICHLET and for NOSLIP: the segment path and
 *    the face path are one code path.
 * 3. The backward-facing-step layout: a wall-function left face with a DIRICHLET
 *    inflow segment over its upper half. Nodes in the segment get exactly the
 *    fixed k, eps and nu_t = C_mu k^2/eps; nodes below get the wall function.
 *    The node sitting exactly on the bound belongs to the segment.
 * 4. A profile is called once per node with the position rescaled to span the
 *    segment, as a velocity inlet under bc_inlet_set_range() is.
 * 5. Overlapping segments: the last one wins.
 * 6. turbulence_bc_add_segment refuses every malformed segment and leaves the
 *    config unchanged; turbulence_apply_bcs refuses the same in a hand-filled
 *    config before touching any field, plus the model-dependent cases.
 * 7. Wall distance (Spalart-Allmaras) sees only the wall part of a face.
 * ============================================================================ */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/boundary/boundary_conditions.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/turbulence_solver.h"
#include "unity.h"

#include <math.h>
#include <string.h>

/* Internal (lib/src/solvers/turbulence/turbulence_solver_internal.h) */
extern double turb_wall_distance(const grid* grid, const ns_turbulence_bc_config_t* tbc,
                                 size_t i, size_t j, int* has_wall);

#define SEG_C_MU 0.09

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

/* 9 x 9 on the unit square: node 4 sits exactly at position 0.5 */
#define NX 9
#define NY 9

static grid* make_grid(void) {
    grid* g = grid_create(NX, NY, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);
    return g;
}

/* A field with distinct, non-trivial values everywhere, so a NEUMANN or
 * PERIODIC copy is distinguishable from an untouched node. */
static flow_field* make_field(void) {
    flow_field* field = flow_field_create(NX, NY, 1);
    TEST_ASSERT_NOT_NULL(field);
    for (size_t n = 0; n < (size_t)NX * NY; n++) {
        field->rho[n] = 1.0;
        field->u[n] = 2.0 + 0.01 * (double)n;
        field->v[n] = 0.5 + 0.003 * (double)n;
        field->turb_k[n] = 1e-3 * (1.0 + 0.1 * (double)n);
        field->turb_eps[n] = 1e-4 * (1.0 + 0.05 * (double)n);
        field->turb_nu_tilde[n] = 1e-4 * (1.0 + 0.02 * (double)n);
        field->nu_t[n] = 1e-5 * (1.0 + 0.07 * (double)n);
    }
    return field;
}

static ns_solver_params_t make_params(turbulence_model_t model) {
    ns_solver_params_t params = ns_solver_params_default();
    params.mu = 1e-4; /* rho = 1, first node in the log layer at u ~ 2 */
    params.turb_model = model;
    params.turb_bc.left = BC_TYPE_NEUMANN;
    params.turb_bc.right = BC_TYPE_NEUMANN;
    params.turb_bc.bottom = BC_TYPE_NEUMANN;
    params.turb_bc.top = BC_TYPE_NEUMANN;
    return params;
}

static void assert_fields_identical(const flow_field* a, const flow_field* b) {
    const size_t bytes = (size_t)NX * NY * sizeof(double);
    TEST_ASSERT_EQUAL_MEMORY(a->turb_k, b->turb_k, bytes);
    TEST_ASSERT_EQUAL_MEMORY(a->turb_eps, b->turb_eps, bytes);
    TEST_ASSERT_EQUAL_MEMORY(a->turb_nu_tilde, b->turb_nu_tilde, bytes);
    TEST_ASSERT_EQUAL_MEMORY(a->nu_t, b->nu_t, bytes);
}

/* ============================================================================
 * TEST 1: no segments by default
 * ============================================================================ */

static void test_default_has_no_segments(void) {
    ns_solver_params_t zero;
    memset(&zero, 0, sizeof(zero));
    TEST_ASSERT_EQUAL_size_t(0, zero.turb_bc.n_segments);
    ns_solver_params_t params = ns_solver_params_default();
    TEST_ASSERT_EQUAL_size_t(0, params.turb_bc.n_segments);
}

/* ============================================================================
 * TEST 2: a whole-face segment is the face type
 * ============================================================================ */

static bc_type_t* face_type_of(ns_turbulence_bc_config_t* bc, bc_edge_t edge) {
    switch (edge) {
        case BC_EDGE_LEFT: return &bc->left;
        case BC_EDGE_RIGHT: return &bc->right;
        case BC_EDGE_BOTTOM: return &bc->bottom;
        default: return &bc->top;
    }
}

static void check_whole_face_segment(turbulence_model_t model, bc_edge_t edge,
                                     bc_type_t type) {
    grid* g = make_grid();
    flow_field* by_face = make_field();
    flow_field* by_segment = make_field();

    ns_solver_params_t face = make_params(model);
    *face_type_of(&face.turb_bc, edge) = type;
    face.turb_bc.k_values = (bc_dirichlet_values_t){.left = 0.02, .right = 0.02,
                                                    .bottom = 0.02, .top = 0.02};
    face.turb_bc.eps_values = (bc_dirichlet_values_t){.left = 0.003, .right = 0.003,
                                                      .bottom = 0.003, .top = 0.003};
    face.turb_bc.nu_tilde_values = (bc_dirichlet_values_t){.left = 4e-4, .right = 4e-4,
                                                           .bottom = 4e-4, .top = 4e-4};

    ns_solver_params_t seg = make_params(model);
    ns_turbulence_bc_segment_t s = {.edge = edge, .start = 0.0, .end = 1.0, .type = type,
                                    .k = 0.02, .eps = 0.003, .nu_tilde = 4e-4};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&seg.turb_bc, &s));

    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(by_face, g, &face));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(by_segment, g, &seg));
    assert_fields_identical(by_face, by_segment);

    flow_field_destroy(by_segment);
    flow_field_destroy(by_face);
    grid_destroy(g);
}

static void test_whole_face_segment_matches_face(void) {
    const bc_edge_t edges[] = {BC_EDGE_LEFT, BC_EDGE_RIGHT, BC_EDGE_BOTTOM, BC_EDGE_TOP};
    const bc_type_t types[] = {BC_TYPE_DIRICHLET, BC_TYPE_NOSLIP, BC_TYPE_NEUMANN};
    const turbulence_model_t models[] = {TURB_MODEL_K_EPSILON, TURB_MODEL_SPALART_ALLMARAS};
    for (size_t m = 0; m < 2; m++) {
        for (size_t e = 0; e < 4; e++) {
            for (size_t t = 0; t < 3; t++) {
                check_whole_face_segment(models[m], edges[e], types[t]);
            }
        }
    }
}

/* ============================================================================
 * TEST 3: step face plus inflow on one edge
 * ============================================================================ */

static void test_wall_with_inflow_segment(void) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_bc.left = BC_TYPE_NOSLIP;

    const double k_in = 0.01;
    const double eps_in = 0.002;
    ns_turbulence_bc_segment_t inflow = {.edge = BC_EDGE_LEFT, .start = 0.5, .end = 1.0,
                                         .type = BC_TYPE_DIRICHLET, .k = k_in, .eps = eps_in};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &inflow));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    const double y_p = g->x[1] - g->x[0];
    const double nu = params.mu;
    /* j = 0 and j = NY-1 are corners, rewritten by the bottom and top faces */
    for (size_t j = 1; j < NY - 1; j++) {
        const size_t idx_w = j * NX;
        const size_t idx_p = idx_w + 1;
        if (j >= 4) { /* position j/8 >= 0.5: the inflow, j = 4 on the bound */
            TEST_ASSERT_EQUAL_DOUBLE(k_in, field->turb_k[idx_w]);
            TEST_ASSERT_EQUAL_DOUBLE(eps_in, field->turb_eps[idx_w]);
            TEST_ASSERT_DOUBLE_WITHIN(1e-15, SEG_C_MU * k_in * k_in / eps_in,
                                      field->nu_t[idx_w]);
        } else { /* the step face: wall function on the wall-parallel v */
            const double ut = turbulence_wall_u_tau(NS_WALL_LAW_LOG, fabs(field->v[idx_p]),
                                                    y_p, nu);
            TEST_ASSERT_TRUE(ut > 0.0);
            TEST_ASSERT_EQUAL_DOUBLE(0.0, field->turb_k[idx_w]);
            TEST_ASSERT_DOUBLE_WITHIN(1e-12, ut * ut / sqrt(SEG_C_MU), field->turb_k[idx_p]);
            TEST_ASSERT_EQUAL_DOUBLE(0.0, field->nu_t[idx_w]);
        }
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 4: profile callback
 * ============================================================================ */

typedef struct {
    int calls;
    double positions[NX];
} profile_log_t;

static void linear_profile(double position, double* k, double* eps, double* nu_tilde,
                           void* user_data) {
    profile_log_t* log = (profile_log_t*)user_data;
    if (log->calls < NX) {
        log->positions[log->calls] = position;
    }
    log->calls++;
    *k = 1e-3 * (1.0 + position);
    *eps = 2e-4;
    *nu_tilde = 3e-4 * (1.0 + position);
}

static void test_profile_callback(void) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);

    profile_log_t log = {0};
    ns_turbulence_bc_segment_t seg = {.edge = BC_EDGE_BOTTOM, .start = 0.25, .end = 0.75,
                                      .type = BC_TYPE_DIRICHLET, .profile = linear_profile,
                                      .profile_user_data = &log};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &seg));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    /* Nodes i = 2..6 lie in [0.25, 0.75]; the profile sees 0, 1/4, ..., 1 */
    TEST_ASSERT_EQUAL_INT(5, log.calls);
    for (int n = 0; n < 5; n++) {
        TEST_ASSERT_DOUBLE_WITHIN(1e-12, 0.25 * n, log.positions[n]);
        TEST_ASSERT_DOUBLE_WITHIN(1e-15, 1e-3 * (1.0 + 0.25 * n), field->turb_k[2 + n]);
        TEST_ASSERT_EQUAL_DOUBLE(2e-4, field->turb_eps[2 + n]);
    }
    /* Outside the segment the face's NEUMANN copy applies */
    TEST_ASSERT_EQUAL_DOUBLE(field->turb_k[1 + NX], field->turb_k[1]);
    TEST_ASSERT_EQUAL_DOUBLE(field->turb_k[7 + NX], field->turb_k[7]);

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 5: the last overlapping segment wins
 * ============================================================================ */

static void test_last_segment_wins(void) {
    grid* g = make_grid();
    flow_field* field = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);

    ns_turbulence_bc_segment_t wide = {.edge = BC_EDGE_TOP, .start = 0.0, .end = 1.0,
                                       .type = BC_TYPE_DIRICHLET, .k = 0.1, .eps = 0.01};
    ns_turbulence_bc_segment_t narrow = {.edge = BC_EDGE_TOP, .start = 0.5, .end = 0.75,
                                         .type = BC_TYPE_DIRICHLET, .k = 0.2, .eps = 0.02};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &wide));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &narrow));
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    const size_t row = (NY - 1) * NX;
    for (size_t i = 0; i < NX; i++) {
        const double expected = (i >= 4 && i <= 6) ? 0.2 : 0.1;
        TEST_ASSERT_EQUAL_DOUBLE(expected, field->turb_k[row + i]);
    }

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 6: refusals
 * ============================================================================ */

static void expect_add_refused(ns_turbulence_bc_segment_t seg) {
    ns_turbulence_bc_config_t bc;
    memset(&bc, 0, sizeof(bc));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_bc_add_segment(&bc, &seg));
    TEST_ASSERT_EQUAL_size_t(0, bc.n_segments);

    /* The same segment hand-filled is refused by apply, fields untouched */
    grid* g = make_grid();
    flow_field* field = make_field();
    flow_field* before = make_field();
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_bc.segments[0] = seg;
    params.turb_bc.n_segments = 1;
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_apply_bcs(field, g, &params));
    assert_fields_identical(before, field);
    flow_field_destroy(before);
    flow_field_destroy(field);
    grid_destroy(g);
}

static void test_add_segment_refusals(void) {
    const ns_turbulence_bc_segment_t good = {.edge = BC_EDGE_LEFT, .start = 0.25,
                                             .end = 0.75, .type = BC_TYPE_DIRICHLET,
                                             .k = 1e-3, .eps = 1e-4};
    ns_turbulence_bc_segment_t s;

    s = good; s.edge = BC_EDGE_FRONT; expect_add_refused(s);
    s = good; s.edge = (bc_edge_t)(BC_EDGE_LEFT | BC_EDGE_RIGHT); expect_add_refused(s);
    s = good; s.start = 0.75; expect_add_refused(s);            /* start == end */
    s = good; s.start = 0.8; expect_add_refused(s);             /* start > end */
    s = good; s.start = -0.1; expect_add_refused(s);
    s = good; s.end = 1.1; expect_add_refused(s);
    s = good; s.start = NAN; expect_add_refused(s);
    s = good; s.type = BC_TYPE_PERIODIC; expect_add_refused(s);
    s = good; s.type = BC_TYPE_INLET; expect_add_refused(s);
    s = good; s.k = -1e-3; expect_add_refused(s);
    s = good; s.eps = NAN; expect_add_refused(s);
    s = good; s.nu_tilde = INFINITY; expect_add_refused(s);

    /* NULLs, and a full config */
    ns_turbulence_bc_config_t bc;
    memset(&bc, 0, sizeof(bc));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_bc_add_segment(NULL, &good));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_bc_add_segment(&bc, NULL));
    for (size_t n = 0; n < NS_TURB_BC_MAX_SEGMENTS; n++) {
        TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&bc, &good));
    }
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_bc_add_segment(&bc, &good));
    TEST_ASSERT_EQUAL_size_t(NS_TURB_BC_MAX_SEGMENTS, bc.n_segments);
}

static void negative_k_profile(double position, double* k, double* eps, double* nu_tilde,
                               void* user_data) {
    (void)position;
    (void)user_data;
    *k = -1.0;
    *eps = 1e-4;
    *nu_tilde = 1e-4;
}

static void zero_eps_profile(double position, double* k, double* eps, double* nu_tilde,
                             void* user_data) {
    (void)position;
    (void)user_data;
    *k = 1e-3;
    *eps = 0.0;
    *nu_tilde = 1e-4;
}

static void test_apply_refusals(void) {
    grid* g = make_grid();
    flow_field* field = make_field();

    /* A count past the array */
    ns_solver_params_t params = make_params(TURB_MODEL_K_EPSILON);
    params.turb_bc.n_segments = NS_TURB_BC_MAX_SEGMENTS + 1;
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_apply_bcs(field, g, &params));
    /* ...and by the transport step, whose SA wall distance reads the segments */
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
                      turbulence_step_explicit(field, g, &params, 1e-3, 0.0));

    /* k-epsilon DIRICHLET with eps = 0 and no profile: refused, not run with
     * zero inflow dissipation. Spalart-Allmaras reads no eps and accepts it. */
    ns_turbulence_bc_segment_t s = {.edge = BC_EDGE_LEFT, .start = 0.5, .end = 1.0,
                                    .type = BC_TYPE_DIRICHLET};
    params = make_params(TURB_MODEL_K_EPSILON);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &s));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_apply_bcs(field, g, &params));
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    /* A profile returning a value no model can use */
    s.profile = negative_k_profile;
    params = make_params(TURB_MODEL_K_EPSILON);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &s));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_apply_bcs(field, g, &params));

    s.profile = zero_eps_profile;
    params = make_params(TURB_MODEL_K_EPSILON);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&params.turb_bc, &s));
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID, turbulence_apply_bcs(field, g, &params));
    params.turb_model = TURB_MODEL_SPALART_ALLMARAS; /* eps unread: accepted */
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_apply_bcs(field, g, &params));

    flow_field_destroy(field);
    grid_destroy(g);
}

/* ============================================================================
 * TEST 7: wall distance sees only the wall part of a face
 * ============================================================================ */

static void test_wall_distance_with_segments(void) {
    grid* g = make_grid(); /* spacing 1/8 */
    const double h = 0.125;
    int has_wall = 0;

    /* Left face a wall only over [0, 0.5]: a step face. No other walls. */
    ns_turbulence_bc_config_t tbc;
    memset(&tbc, 0, sizeof(tbc));
    tbc.left = BC_TYPE_NOSLIP;
    ns_turbulence_bc_segment_t inflow = {.edge = BC_EDGE_LEFT, .start = 0.5, .end = 1.0,
                                         .type = BC_TYPE_DIRICHLET, .k = 1e-3, .eps = 1e-4};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&tbc, &inflow));

    /* Beside the wall part: the plain normal distance, exactly */
    double d = turb_wall_distance(g, &tbc, 3, 2, &has_wall);
    TEST_ASSERT_EQUAL_INT(1, has_wall);
    TEST_ASSERT_EQUAL_DOUBLE(3 * h, d);

    /* Beside the inflow part: the distance to the wall's top end, (0, 0.5) */
    d = turb_wall_distance(g, &tbc, 3, 7, &has_wall);
    TEST_ASSERT_EQUAL_INT(1, has_wall);
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, hypot(3 * h, 3 * h), d);

    /* A wall segment on a non-wall face is the only wall */
    memset(&tbc, 0, sizeof(tbc));
    ns_turbulence_bc_segment_t wall = {.edge = BC_EDGE_BOTTOM, .start = 0.25, .end = 0.5,
                                       .type = BC_TYPE_NOSLIP};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&tbc, &wall));
    d = turb_wall_distance(g, &tbc, 3, 4, &has_wall); /* above it */
    TEST_ASSERT_EQUAL_INT(1, has_wall);
    TEST_ASSERT_EQUAL_DOUBLE(4 * h, d);
    d = turb_wall_distance(g, &tbc, 8, 0, &has_wall); /* on the bottom, past its end */
    TEST_ASSERT_DOUBLE_WITHIN(1e-15, 4 * h, d);

    /* Every face cut back to non-wall: no wall anywhere */
    memset(&tbc, 0, sizeof(tbc));
    tbc.bottom = BC_TYPE_NOSLIP;
    ns_turbulence_bc_segment_t open = {.edge = BC_EDGE_BOTTOM, .start = 0.0, .end = 1.0,
                                       .type = BC_TYPE_NEUMANN};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, turbulence_bc_add_segment(&tbc, &open));
    (void)turb_wall_distance(g, &tbc, 4, 4, &has_wall);
    TEST_ASSERT_EQUAL_INT(0, has_wall);

    grid_destroy(g);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_default_has_no_segments);
    RUN_TEST(test_whole_face_segment_matches_face);
    RUN_TEST(test_wall_with_inflow_segment);
    RUN_TEST(test_profile_callback);
    RUN_TEST(test_last_segment_wins);
    RUN_TEST(test_add_segment_refusals);
    RUN_TEST(test_apply_refusals);
    RUN_TEST(test_wall_distance_with_segments);
    return UNITY_END();
}
