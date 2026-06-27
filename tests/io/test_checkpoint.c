/**
 * @file test_checkpoint.c
 * @brief Tests for binary save/restore of complete simulation state.
 *
 * Covers: low-level round-trip (2D uniform + 3D stretched, bit-exact),
 * high-level save/load/restore, version & magic rejection, truncation & CRC
 * corruption, restart continuity (the key correctness property) across scalar
 * and SIMD solvers, and the custom-callback contract.
 */

#include "cfd/api/simulation_api.h"
#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/io/checkpoint.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "unity.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const char* CK_PATH = "test_checkpoint_tmp.cfdchk";
static const char* CK_PATH2 = "test_checkpoint_tmp2.cfdchk";

void setUp(void) {
    cfd_init();
}

void tearDown(void) {
    remove(CK_PATH);
    remove(CK_PATH2);
}

/* ------------------------------------------------------------------ helpers */

static int bits_equal(double a, double b) {
    return memcmp(&a, &b, sizeof(double)) == 0;
}

static void fill_field_known(flow_field* f, double seed) {
    size_t n = f->nx * f->ny * f->nz;
    for (size_t i = 0; i < n; i++) {
        double t = (double)i + seed;
        f->u[i] = sin(0.7 * t) * 1.3;
        f->v[i] = cos(0.4 * t) * 0.9;
        f->w[i] = sin(0.2 * t) * 0.5;
        f->p[i] = 100000.0 + 11.0 * t;
        f->rho[i] = 1.2 + 0.001 * t;
        f->T[i] = 300.0 + 0.05 * t;
    }
}

static ns_solver_params_t make_nondefault_params(void) {
    ns_solver_params_t p = ns_solver_params_default();
    p.dt = 0.0023;
    p.cfl = 0.37;
    p.gamma = 1.41;
    p.mu = 0.013;
    p.k = 0.029;
    p.max_iter = 7;
    p.tolerance = 1e-7;
    p.source_amplitude_u = 0.11;
    p.source_amplitude_v = 0.06;
    p.source_decay_rate = 0.12;
    p.pressure_coupling = 0.13;
    p.alpha = 0.0042;
    p.beta = 0.0033;
    p.T_ref = 295.5;
    p.gravity[0] = 0.1;
    p.gravity[1] = -9.81;
    p.gravity[2] = 0.2;
    p.thermal_bc.left = BC_TYPE_DIRICHLET;
    p.thermal_bc.right = BC_TYPE_NEUMANN;
    p.thermal_bc.bottom = BC_TYPE_PERIODIC;
    p.thermal_bc.top = BC_TYPE_DIRICHLET;
    p.thermal_bc.front = BC_TYPE_NEUMANN;
    p.thermal_bc.back = BC_TYPE_PERIODIC;
    p.thermal_bc.dirichlet_values.left = 310.0;
    p.thermal_bc.dirichlet_values.right = 290.0;
    p.thermal_bc.dirichlet_values.top = 305.0;
    p.thermal_bc.dirichlet_values.bottom = 300.0;
    p.thermal_bc.dirichlet_values.front = 301.0;
    p.thermal_bc.dirichlet_values.back = 302.0;
    return p;
}

static void assert_grid_equal(const grid* a, const grid* b) {
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->nx, (uint64_t)b->nx);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->ny, (uint64_t)b->ny);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->nz, (uint64_t)b->nz);
    TEST_ASSERT_TRUE(bits_equal(a->xmin, b->xmin));
    TEST_ASSERT_TRUE(bits_equal(a->xmax, b->xmax));
    TEST_ASSERT_TRUE(bits_equal(a->ymin, b->ymin));
    TEST_ASSERT_TRUE(bits_equal(a->ymax, b->ymax));
    TEST_ASSERT_TRUE(bits_equal(a->zmin, b->zmin));
    TEST_ASSERT_TRUE(bits_equal(a->zmax, b->zmax));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->x, b->x, a->nx * sizeof(double)));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->y, b->y, a->ny * sizeof(double)));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->dx, b->dx, (a->nx - 1) * sizeof(double)));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->dy, b->dy, (a->ny - 1) * sizeof(double)));
    if (a->nz > 1) {
        TEST_ASSERT_EQUAL_INT(0, memcmp(a->z, b->z, a->nz * sizeof(double)));
        TEST_ASSERT_EQUAL_INT(0, memcmp(a->dz, b->dz, (a->nz - 1) * sizeof(double)));
        TEST_ASSERT_TRUE(bits_equal(a->inv_dz2, b->inv_dz2));
    }
}

static void assert_field_equal(const flow_field* a, const flow_field* b) {
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->nx, (uint64_t)b->nx);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->ny, (uint64_t)b->ny);
    TEST_ASSERT_EQUAL_UINT64((uint64_t)a->nz, (uint64_t)b->nz);
    size_t bytes = a->nx * a->ny * a->nz * sizeof(double);
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->u, b->u, bytes));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->v, b->v, bytes));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->w, b->w, bytes));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->p, b->p, bytes));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->rho, b->rho, bytes));
    TEST_ASSERT_EQUAL_INT(0, memcmp(a->T, b->T, bytes));
}

static void assert_params_equal(const ns_solver_params_t* a, const ns_solver_params_t* b) {
    TEST_ASSERT_TRUE(bits_equal(a->dt, b->dt));
    TEST_ASSERT_TRUE(bits_equal(a->cfl, b->cfl));
    TEST_ASSERT_TRUE(bits_equal(a->gamma, b->gamma));
    TEST_ASSERT_TRUE(bits_equal(a->mu, b->mu));
    TEST_ASSERT_TRUE(bits_equal(a->k, b->k));
    TEST_ASSERT_EQUAL_INT(a->max_iter, b->max_iter);
    TEST_ASSERT_TRUE(bits_equal(a->tolerance, b->tolerance));
    TEST_ASSERT_TRUE(bits_equal(a->source_amplitude_u, b->source_amplitude_u));
    TEST_ASSERT_TRUE(bits_equal(a->source_amplitude_v, b->source_amplitude_v));
    TEST_ASSERT_TRUE(bits_equal(a->source_decay_rate, b->source_decay_rate));
    TEST_ASSERT_TRUE(bits_equal(a->pressure_coupling, b->pressure_coupling));
    TEST_ASSERT_TRUE(bits_equal(a->alpha, b->alpha));
    TEST_ASSERT_TRUE(bits_equal(a->beta, b->beta));
    TEST_ASSERT_TRUE(bits_equal(a->T_ref, b->T_ref));
    TEST_ASSERT_TRUE(bits_equal(a->gravity[0], b->gravity[0]));
    TEST_ASSERT_TRUE(bits_equal(a->gravity[1], b->gravity[1]));
    TEST_ASSERT_TRUE(bits_equal(a->gravity[2], b->gravity[2]));
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.left, b->thermal_bc.left);
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.right, b->thermal_bc.right);
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.bottom, b->thermal_bc.bottom);
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.top, b->thermal_bc.top);
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.front, b->thermal_bc.front);
    TEST_ASSERT_EQUAL_INT(a->thermal_bc.back, b->thermal_bc.back);
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.left, b->thermal_bc.dirichlet_values.left));
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.right, b->thermal_bc.dirichlet_values.right));
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.top, b->thermal_bc.dirichlet_values.top));
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.bottom, b->thermal_bc.dirichlet_values.bottom));
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.front, b->thermal_bc.dirichlet_values.front));
    TEST_ASSERT_TRUE(bits_equal(a->thermal_bc.dirichlet_values.back, b->thermal_bc.dirichlet_values.back));
}

/* File-corruption helpers for the rejection tests. */
static long file_size_of(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    return sz;
}

static void patch_byte(const char* path, long offset, unsigned char value) {
    FILE* fp = fopen(path, "r+b");
    TEST_ASSERT_NOT_NULL(fp);
    fseek(fp, offset, SEEK_SET);
    fputc(value, fp);
    fclose(fp);
}

static void truncate_to(const char* src, const char* dst, long newsize) {
    FILE* in = fopen(src, "rb");
    FILE* out = fopen(dst, "wb");
    TEST_ASSERT_NOT_NULL(in);
    TEST_ASSERT_NOT_NULL(out);
    for (long i = 0; i < newsize; i++) {
        int c = fgetc(in);
        if (c == EOF) {
            break;
        }
        fputc(c, out);
    }
    fclose(in);
    fclose(out);
}

/* ----------------------------------------------------------- 1. round-trip */

void test_lowlevel_roundtrip_2d_uniform(void) {
    grid* g = grid_create(12, 8, 1, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0);
    grid_initialize_uniform(g);
    flow_field* f = flow_field_create(12, 8, 1);
    fill_field_known(f, 3.0);
    ns_solver_params_t p = make_nondefault_params();

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_checkpoint_write(CK_PATH, g, f, &p, 1.25, "rk2", "myrun", "/base/dir"));

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    double t2 = 0.0;
    char name[128] = {0}, prefix[256] = {0}, base[512] = {0};
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_checkpoint_read(CK_PATH, &g2, &f2, &p2, &t2, name, sizeof(name), prefix,
                                          sizeof(prefix), base, sizeof(base)));

    assert_grid_equal(g, g2);
    assert_field_equal(f, f2);
    assert_params_equal(&p, &p2);
    TEST_ASSERT_TRUE(bits_equal(1.25, t2));
    TEST_ASSERT_EQUAL_STRING("rk2", name);
    TEST_ASSERT_EQUAL_STRING("myrun", prefix);
    TEST_ASSERT_EQUAL_STRING("/base/dir", base);
    /* function-pointer params must come back NULL */
    TEST_ASSERT_NULL(p2.source_func);
    TEST_ASSERT_NULL(p2.heat_source_func);

    grid_destroy(g);
    grid_destroy(g2);
    flow_field_destroy(f);
    flow_field_destroy(f2);
}

void test_lowlevel_roundtrip_3d_stretched(void) {
    grid* g = grid_create(6, 5, 4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    grid_initialize_stretched(g, 2.0);
    flow_field* f = flow_field_create(6, 5, 4);
    fill_field_known(f, 9.0);
    ns_solver_params_t p = make_nondefault_params();

    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_checkpoint_write(CK_PATH, g, f, &p, 0.5, "rk4", NULL, NULL));

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    double t2 = 0.0;
    char name[128] = {0};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_read(CK_PATH, &g2, &f2, &p2, &t2, name,
                                                       sizeof(name), NULL, 0, NULL, 0));

    assert_grid_equal(g, g2); /* exercises z/dz/inv_dz2 */
    assert_field_equal(f, f2);
    TEST_ASSERT_EQUAL_STRING("rk4", name);

    grid_destroy(g);
    grid_destroy(g2);
    flow_field_destroy(f);
    flow_field_destroy(f2);
}

/* ---------------------------------------------------- 2. high-level API */

void test_highlevel_save_load_roundtrip(void) {
    simulation_data* sim = init_simulation_with_solver(10, 10, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_RK2);
    TEST_ASSERT_NOT_NULL(sim);
    fill_field_known(sim->field, 2.0);
    sim->current_time = 0.875;

    TEST_ASSERT_EQUAL(CFD_SUCCESS, save_simulation_checkpoint(sim, CK_PATH));

    simulation_data* loaded = load_simulation_from_checkpoint(CK_PATH);
    TEST_ASSERT_NOT_NULL(loaded);
    assert_grid_equal(sim->grid, loaded->grid);
    assert_field_equal(sim->field, loaded->field);
    TEST_ASSERT_TRUE(bits_equal(sim->current_time, loaded->current_time));
    TEST_ASSERT_NOT_NULL(loaded->solver);
    TEST_ASSERT_EQUAL_STRING(NS_SOLVER_TYPE_RK2, loaded->solver->name);

    free_simulation(sim);
    free_simulation(loaded);
}

void test_highlevel_restore_into_different_dims(void) {
    /* Source checkpoint is 14x9; restore into an existing 6x6 sim. */
    simulation_data* src = init_simulation_with_solver(14, 9, 1, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_RK2);
    TEST_ASSERT_NOT_NULL(src);
    fill_field_known(src->field, 5.0);
    src->current_time = 4.2;
    TEST_ASSERT_EQUAL(CFD_SUCCESS, save_simulation_checkpoint(src, CK_PATH));

    simulation_data* dst = init_simulation_with_solver(6, 6, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_EXPLICIT_EULER);
    TEST_ASSERT_NOT_NULL(dst);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, restore_simulation_checkpoint(dst, CK_PATH));

    TEST_ASSERT_EQUAL_UINT64(14, (uint64_t)dst->grid->nx);
    TEST_ASSERT_EQUAL_UINT64(9, (uint64_t)dst->grid->ny);
    assert_field_equal(src->field, dst->field);
    TEST_ASSERT_TRUE(bits_equal(4.2, dst->current_time));
    TEST_ASSERT_EQUAL_STRING(NS_SOLVER_TYPE_RK2, dst->solver->name);

    free_simulation(src);
    free_simulation(dst);
}

/* ------------------------------------------------- 3. version / magic */

void test_reject_bad_version(void) {
    grid* g = grid_create(8, 8, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    grid_initialize_uniform(g);
    flow_field* f = flow_field_create(8, 8, 1);
    fill_field_known(f, 1.0);
    ns_solver_params_t p = ns_solver_params_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_write(CK_PATH, g, f, &p, 0.0, "rk2", NULL, NULL));

    /* format_version is the u32 immediately after the 8-byte magic. */
    patch_byte(CK_PATH, 8, 99);

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    char name[64] = {0};
    TEST_ASSERT_EQUAL(CFD_ERROR_UNSUPPORTED,
                      cfd_checkpoint_read(CK_PATH, &g2, &f2, &p2, NULL, name, sizeof(name), NULL, 0,
                                          NULL, 0));
    TEST_ASSERT_NULL(g2);
    TEST_ASSERT_NULL(f2);

    grid_destroy(g);
    flow_field_destroy(f);
}

void test_reject_bad_magic(void) {
    grid* g = grid_create(8, 8, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    grid_initialize_uniform(g);
    flow_field* f = flow_field_create(8, 8, 1);
    fill_field_known(f, 1.0);
    ns_solver_params_t p = ns_solver_params_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_write(CK_PATH, g, f, &p, 0.0, "rk2", NULL, NULL));

    patch_byte(CK_PATH, 0, 'X');

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    char name[64] = {0};
    TEST_ASSERT_EQUAL(CFD_ERROR_INVALID,
                      cfd_checkpoint_read(CK_PATH, &g2, &f2, &p2, NULL, name, sizeof(name), NULL, 0,
                                          NULL, 0));

    grid_destroy(g);
    flow_field_destroy(f);
}

/* ----------------------------------------------- 4. truncation / CRC */

void test_reject_truncated(void) {
    grid* g = grid_create(16, 16, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    grid_initialize_uniform(g);
    flow_field* f = flow_field_create(16, 16, 1);
    fill_field_known(f, 1.0);
    ns_solver_params_t p = ns_solver_params_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_write(CK_PATH, g, f, &p, 0.0, "rk2", NULL, NULL));

    long sz = file_size_of(CK_PATH);
    TEST_ASSERT_TRUE(sz > 64);
    truncate_to(CK_PATH, CK_PATH2, sz / 2); /* cut mid-arrays */

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    char name[64] = {0};
    TEST_ASSERT_EQUAL(CFD_ERROR_IO, cfd_checkpoint_read(CK_PATH2, &g2, &f2, &p2, NULL, name,
                                                        sizeof(name), NULL, 0, NULL, 0));
    TEST_ASSERT_NULL(g2);
    TEST_ASSERT_NULL(f2);

    grid_destroy(g);
    flow_field_destroy(f);
}

void test_reject_crc_corruption(void) {
    grid* g = grid_create(16, 16, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    grid_initialize_uniform(g);
    flow_field* f = flow_field_create(16, 16, 1);
    fill_field_known(f, 1.0);
    ns_solver_params_t p = ns_solver_params_default();
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_write(CK_PATH, g, f, &p, 0.0, "rk2", NULL, NULL));

    /* Flip one byte deep in the body (a field value), leaving structure intact. */
    long sz = file_size_of(CK_PATH);
    long off = sz / 2;
    FILE* fp = fopen(CK_PATH, "rb");
    fseek(fp, off, SEEK_SET);
    int orig = fgetc(fp);
    fclose(fp);
    patch_byte(CK_PATH, off, (unsigned char)(orig ^ 0xFF));

    grid* g2 = NULL;
    flow_field* f2 = NULL;
    ns_solver_params_t p2;
    char name[64] = {0};
    TEST_ASSERT_EQUAL(CFD_ERROR_IO, cfd_checkpoint_read(CK_PATH, &g2, &f2, &p2, NULL, name,
                                                        sizeof(name), NULL, 0, NULL, 0));

    grid_destroy(g);
    flow_field_destroy(f);
}

/* ------------------------------------------ 5/6. restart continuity */

/* Run `name` for `steps` solver_step calls on a fresh field; returns 0 on
 * success, -1 if the solver/backend is unavailable. The final field is written
 * into `out`. */
static int run_steps(const char* name, const grid* g, const ns_solver_params_t* params,
                     const flow_field* init, int steps, flow_field* out) {
    ns_solver_registry_t* reg = cfd_registry_create();
    cfd_registry_register_defaults(reg);
    ns_solver_t* slv = cfd_solver_create(reg, name);
    if (!slv) {
        cfd_registry_destroy(reg);
        return -1;
    }
    if (solver_init(slv, g, params) == CFD_ERROR_UNSUPPORTED) {
        solver_destroy(slv);
        cfd_registry_destroy(reg);
        return -1;
    }

    size_t bytes = init->nx * init->ny * init->nz * sizeof(double);
    memcpy(out->u, init->u, bytes);
    memcpy(out->v, init->v, bytes);
    memcpy(out->w, init->w, bytes);
    memcpy(out->p, init->p, bytes);
    memcpy(out->rho, init->rho, bytes);
    memcpy(out->T, init->T, bytes);

    ns_solver_params_t pp = *params;
    ns_solver_stats_t stats = ns_solver_stats_default();
    for (int s = 0; s < steps; s++) {
        solver_step(slv, out, g, &pp, &stats);
    }

    solver_destroy(slv);
    cfd_registry_destroy(reg);
    return 0;
}

/* N steps + checkpoint + M steps must equal N+M continuous, bit-for-bit. */
static void check_continuity_for(const char* name) {
    const size_t nx = 16, ny = 16;
    const int N = 5, M = 5;
    grid* g = grid_create(nx, ny, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    grid_initialize_uniform(g);

    ns_solver_params_t p = ns_solver_params_default();
    p.dt = 1e-4;
    p.mu = 0.01;
    p.max_iter = 1; /* one time step per solver_step call */

    flow_field* init = flow_field_create(nx, ny, 1);
    fill_field_known(init, 1.0);

    /* Continuous run of N+M steps. */
    flow_field* cont = flow_field_create(nx, ny, 1);
    if (run_steps(name, g, &p, init, N + M, cont) != 0) {
        /* TEST_IGNORE_MESSAGE longjmps out, skipping the cleanup below, so free
         * everything allocated so far first to avoid a LeakSanitizer report. */
        grid_destroy(g);
        flow_field_destroy(init);
        flow_field_destroy(cont);
        TEST_IGNORE_MESSAGE("solver/backend unavailable");
    }

    /* Run N, checkpoint, then run M from the restored state. */
    flow_field* partial = flow_field_create(nx, ny, 1);
    TEST_ASSERT_EQUAL(0, run_steps(name, g, &p, init, N, partial));
    TEST_ASSERT_EQUAL(CFD_SUCCESS,
                      cfd_checkpoint_write(CK_PATH, g, partial, &p, (double)N * p.dt, name, NULL,
                                           NULL));

    grid* gr = NULL;
    flow_field* fr = NULL;
    ns_solver_params_t pr;
    double tr = 0.0;
    char rname[64] = {0};
    TEST_ASSERT_EQUAL(CFD_SUCCESS, cfd_checkpoint_read(CK_PATH, &gr, &fr, &pr, &tr, rname,
                                                       sizeof(rname), NULL, 0, NULL, 0));

    flow_field* restored = flow_field_create(nx, ny, 1);
    TEST_ASSERT_EQUAL(0, run_steps(rname, gr, &pr, fr, M, restored));

    assert_field_equal(cont, restored); /* bit-exact */

    grid_destroy(g);
    grid_destroy(gr);
    flow_field_destroy(init);
    flow_field_destroy(cont);
    flow_field_destroy(partial);
    flow_field_destroy(fr);
    flow_field_destroy(restored);
}

void test_restart_continuity_scalar(void) {
    check_continuity_for(NS_SOLVER_TYPE_RK2);
}

void test_restart_continuity_simd(void) {
    if (!cfd_backend_is_available(NS_SOLVER_BACKEND_SIMD)) {
        TEST_IGNORE_MESSAGE("SIMD backend unavailable");
    }
    check_continuity_for(NS_SOLVER_TYPE_RK2_OPTIMIZED);
}

/* --------------------------------------------- 7. callback contract */

static void dummy_source(double x, double y, double z, double t, void* ctx, double* su, double* sv,
                         double* sw) {
    (void)x; (void)y; (void)z; (void)t; (void)ctx;
    *su = 0.0; *sv = 0.0; *sw = 0.0;
}

void test_callback_contract(void) {
    int ctx_marker = 42;
    simulation_data* sim = init_simulation_with_solver(8, 8, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                                                       NS_SOLVER_TYPE_RK2);
    TEST_ASSERT_NOT_NULL(sim);
    sim->params.source_func = dummy_source;
    sim->params.source_context = &ctx_marker;
    fill_field_known(sim->field, 1.0);
    TEST_ASSERT_EQUAL(CFD_SUCCESS, save_simulation_checkpoint(sim, CK_PATH));

    /* In-place restore preserves the existing callback + context. */
    TEST_ASSERT_EQUAL(CFD_SUCCESS, restore_simulation_checkpoint(sim, CK_PATH));
    TEST_ASSERT_EQUAL_PTR(dummy_source, sim->params.source_func);
    TEST_ASSERT_EQUAL_PTR(&ctx_marker, sim->params.source_context);

    /* Constructor path leaves callbacks NULL (documented). */
    simulation_data* loaded = load_simulation_from_checkpoint(CK_PATH);
    TEST_ASSERT_NOT_NULL(loaded);
    TEST_ASSERT_NULL(loaded->params.source_func);
    TEST_ASSERT_NULL(loaded->params.source_context);

    free_simulation(sim);
    free_simulation(loaded);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_lowlevel_roundtrip_2d_uniform);
    RUN_TEST(test_lowlevel_roundtrip_3d_stretched);
    RUN_TEST(test_highlevel_save_load_roundtrip);
    RUN_TEST(test_highlevel_restore_into_different_dims);
    RUN_TEST(test_reject_bad_version);
    RUN_TEST(test_reject_bad_magic);
    RUN_TEST(test_reject_truncated);
    RUN_TEST(test_reject_crc_corruption);
    RUN_TEST(test_restart_continuity_scalar);
    RUN_TEST(test_restart_continuity_simd);
    RUN_TEST(test_callback_contract);
    return UNITY_END();
}
