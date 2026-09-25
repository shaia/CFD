/**
 * @file test_pressure_bc_config.c
 * @brief ns_solver_params_t.pressure_bc is validated at solver init.
 *
 * Two statuses, and they are not interchangeable:
 *
 *   - A face outside poisson_wall_t is CFD_ERROR_INVALID, on every solver. No
 *     backend can honour a value the enum does not define, so reporting it as
 *     UNSUPPORTED sends the caller to try another backend that refuses the same
 *     garbage. cfd_checkpoint_read() casts these faces straight out of the file,
 *     so the value is not always one the compiler chose.
 *
 *   - A legal prescribed face on a solver that cannot honour it is
 *     CFD_ERROR_UNSUPPORTED, because another backend genuinely can. The project's
 *     optional-backend policy has tests SKIP on UNSUPPORTED and FAIL on INVALID,
 *     so collapsing the two would turn a caller error into a silent skip.
 *
 * The distinction was changed once already without a test to hold it; this file
 * is that test.
 */

#include "cfd/core/cfd_init.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include "cfd/solvers/poisson_solver.h"
#include "unity.h"

#include <stdio.h>
#include <string.h>

void setUp(void) { cfd_init(); }
void tearDown(void) { cfd_finalize(); }

#define PBC_N 17

/* Every CPU solver. The time integrators solve no Poisson equation at all, which
 * is exactly why a pressure BC configured on one must not be accepted. */
static const char* const CPU_SOLVERS[] = {
    NS_SOLVER_TYPE_EXPLICIT_EULER, NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
    NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
    NS_SOLVER_TYPE_PROJECTION, NS_SOLVER_TYPE_PROJECTION_OMP,
    NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    NS_SOLVER_TYPE_RK2, NS_SOLVER_TYPE_RK2_OMP, NS_SOLVER_TYPE_RK2_OPTIMIZED,
    NS_SOLVER_TYPE_RK4, NS_SOLVER_TYPE_RK4_OMP, NS_SOLVER_TYPE_RK4_OPTIMIZED,
};
#define NUM_CPU_SOLVERS (sizeof(CPU_SOLVERS) / sizeof(CPU_SOLVERS[0]))

/* The solvers that honour a prescribed face; the rest must refuse one. */
static int honours_per_face(const char* type) {
    return strcmp(type, NS_SOLVER_TYPE_PROJECTION) == 0
        || strcmp(type, NS_SOLVER_TYPE_PROJECTION_OMP) == 0
        || strcmp(type, NS_SOLVER_TYPE_PROJECTION_OPTIMIZED) == 0;
}

static ns_solver_registry_t* create_registry(void) {
    ns_solver_registry_t* registry = cfd_registry_create();
    TEST_ASSERT_NOT_NULL_MESSAGE(registry, "Failed to create registry");
    cfd_registry_register_defaults(registry);
    return registry;
}

static ns_solver_params_t make_params(void) {
    ns_solver_params_t params = ns_solver_params_default();
    params.dt = 5e-4;
    params.mu = 0.01;
    params.max_iter = 1;
    return params;
}

/* Run one params struct past every registered CPU solver's init, asserting the
 * status each one reports. Returns how many were actually reached. */
static int expect_init_status(const ns_solver_params_t* params,
                              cfd_status_t on_per_face, cfd_status_t on_others) {
    ns_solver_registry_t* registry = create_registry();
    grid* g = grid_create(PBC_N, PBC_N, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    TEST_ASSERT_NOT_NULL(g);
    grid_initialize_uniform(g);

    int checked = 0;
    for (size_t s = 0; s < NUM_CPU_SOLVERS; s++) {
        ns_solver_t* slv = cfd_solver_create(registry, CPU_SOLVERS[s]);
        if (!slv) {
            printf("  %s not registered (skipping)\n", CPU_SOLVERS[s]);
            continue;
        }
        cfd_status_t status = solver_init(slv, g, params);
        solver_destroy(slv);
        cfd_status_t want = honours_per_face(CPU_SOLVERS[s]) ? on_per_face : on_others;
        TEST_ASSERT_EQUAL_INT_MESSAGE((int)want, (int)status, CPU_SOLVERS[s]);
        checked++;
    }
    grid_destroy(g);
    cfd_registry_destroy(registry);
    return checked;
}

/** Default (all zero-gradient) walls are what every solver has always run. */
void test_default_walls_accepted_everywhere(void) {
    ns_solver_params_t params = make_params();
    int checked = expect_init_status(&params, CFD_SUCCESS, CFD_SUCCESS);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No CPU solver was checked");
}

/** A face outside the enum is the caller's error, on every solver alike. */
void test_off_enum_face_is_invalid_on_every_solver(void) {
    ns_solver_params_t params = make_params();
    params.pressure_bc.top = (poisson_wall_t)99;
    int checked = expect_init_status(&params, CFD_ERROR_INVALID, CFD_ERROR_INVALID);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No CPU solver was checked");
}

/** A legal prescribed face is UNSUPPORTED where it cannot be honoured. */
void test_prescribed_face_is_unsupported_off_projection(void) {
    ns_solver_params_t params = make_params();
    params.pressure_bc.left = POISSON_WALL_DIRICHLET;
    params.pressure_bc.right = POISSON_WALL_DIRICHLET;
    params.pressure_bc.values.left = 0.0;
    params.pressure_bc.values.right = 1.0;
    int checked = expect_init_status(&params, CFD_SUCCESS, CFD_ERROR_UNSUPPORTED);
    TEST_ASSERT_TRUE_MESSAGE(checked > 0, "No CPU solver was checked");
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_default_walls_accepted_everywhere);
    RUN_TEST(test_off_enum_face_is_invalid_on_every_solver);
    RUN_TEST(test_prescribed_face_is_unsupported_off_projection);
    return UNITY_END();
}
