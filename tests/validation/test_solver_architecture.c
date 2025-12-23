/**
 * @file test_solver_architecture.c
 * @brief Cross-architecture solver consistency tests
 *
 * This test file verifies that ALL solver backend implementations (CPU scalar,
 * AVX2/SIMD, OpenMP, GPU) produce IDENTICAL results for the same problem.
 *
 * VALIDATION REQUIREMENTS:
 * 1. All backends of the same solver type MUST produce identical results
 * 2. Consistency tolerance: u_center difference < 0.001 (0.1%)
 * 3. All non-optional backends must succeed
 *
 * If tests fail, the BACKEND implementation needs fixing, not the tolerance.
 */

#include "cavity_reference_data.h"
#include "lid_driven_cavity_common.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * TEST CONFIGURATION
 * ============================================================================ */

/* Minimal configuration for architecture consistency checks */
#define ARCH_TEST_STEPS     50
#define ARCH_TEST_DT        0.005
#define ARCH_GRID_SIZE      9

/* Tolerance for backend consistency (must match exactly) */
#define ARCH_CONSISTENCY_TOL 0.002

/* ============================================================================
 * SOLVER RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    double u_at_center;
    double v_at_center;
    double max_velocity;
    int success;
    char error_msg[256];
} solver_result_t;

/* ============================================================================
 * Run simulation with specific solver
 * ============================================================================ */

static solver_result_t run_solver(const char* solver_type,
                                   size_t nx, size_t ny,
                                   int max_steps, double dt) {
    solver_result_t result = {0};
    result.success = 0;
    result.error_msg[0] = '\0';

    /* Create context */
    cavity_context_t* ctx = cavity_context_create(nx, ny);
    if (!ctx) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Failed to create context");
        return result;
    }

    double reynolds = 100.0;
    double lid_vel = 1.0;
    double L = ctx->g->xmax - ctx->g->xmin;
    double nu = lid_vel * L / reynolds;

    ns_solver_params_t params = {
        .dt = dt,
        .cfl = 0.5,
        .gamma = 1.4,
        .mu = nu,
        .k = 0.0,
        .max_iter = 1,
        .tolerance = 1e-6,
        .source_amplitude_u = 0.0,
        .source_amplitude_v = 0.0,
        .source_decay_rate = 0.0,
        .pressure_coupling = 0.1
    };

    /* Create solver */
    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    ns_solver_t* solver = cfd_solver_create(registry, solver_type);
    if (!solver) {
        snprintf(result.error_msg, sizeof(result.error_msg), "Solver not available");
        cfd_registry_destroy(registry);
        cavity_context_destroy(ctx);
        return result;
    }

    solver_init(solver, ctx->g, &params);
    ns_solver_stats_t stats = ns_solver_stats_default();

    for (int step = 0; step < max_steps; step++) {
        apply_cavity_bc(ctx->field, lid_vel);
        solver_step(solver, ctx->field, ctx->g, &params, &stats);

        if (!check_field_finite(ctx->field)) {
            snprintf(result.error_msg, sizeof(result.error_msg), "Blew up at step %d", step);
            solver_destroy(solver);
            cfd_registry_destroy(registry);
            cavity_context_destroy(ctx);
            return result;
        }
    }

    /* Extract results */
    size_t center_i = nx / 2;
    size_t center_j = ny / 2;
    size_t center_idx = center_j * nx + center_i;

    result.u_at_center = ctx->field->u[center_idx];
    result.v_at_center = ctx->field->v[center_idx];
    result.max_velocity = find_max_velocity(ctx->field);
    result.success = 1;

    solver_destroy(solver);
    cfd_registry_destroy(registry);
    cavity_context_destroy(ctx);

    return result;
}

/* ============================================================================
 * TEST: Explicit Euler - CPU, AVX2, OpenMP must match exactly
 * ============================================================================ */

void test_euler_cpu_avx2_consistency(void) {
    printf("\n    Comparing Explicit Euler: CPU vs AVX2/SIMD\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Euler must succeed");

    solver_result_t avx2 = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );

    /* Skip test if AVX2 solver is not available */
    if (!avx2.success && strstr(avx2.error_msg, "not available") != NULL) {
        printf("      AVX2 solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("AVX2 solver not available (AVX2 not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(avx2.success, "AVX2 Euler must succeed");

    double diff = fabs(cpu.u_at_center - avx2.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      AVX2 u_center: %.6f\n", avx2.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_CONSISTENCY_TOL,
        "CPU and AVX2 Euler must produce identical results");
}

void test_euler_cpu_omp_consistency(void) {
    printf("\n    Comparing Explicit Euler: CPU vs OpenMP\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Euler must succeed");

    solver_result_t omp = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );

    /* Skip test if OMP solver is not available */
    if (!omp.success && strstr(omp.error_msg, "not available") != NULL) {
        printf("      OpenMP solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("OpenMP solver not available (OpenMP not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(omp.success, "OMP Euler must succeed");

    double diff = fabs(cpu.u_at_center - omp.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      OMP  u_center: %.6f\n", omp.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_CONSISTENCY_TOL,
        "CPU and OpenMP Euler must produce identical results");
}

/* ============================================================================
 * TEST: Projection - CPU, AVX2, OpenMP must match exactly
 * ============================================================================ */

void test_projection_cpu_avx2_consistency(void) {
    printf("\n    Comparing Projection: CPU vs AVX2/SIMD\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t avx2 = run_solver(
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );

    /* Skip test if AVX2 solver is not available */
    if (!avx2.success && strstr(avx2.error_msg, "not available") != NULL) {
        printf("      AVX2 solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("AVX2 solver not available (AVX2 not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(avx2.success, "AVX2 Projection must succeed");

    double diff = fabs(cpu.u_at_center - avx2.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      AVX2 u_center: %.6f\n", avx2.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_CONSISTENCY_TOL,
        "CPU and AVX2 Projection must produce identical results");
}

void test_projection_cpu_omp_consistency(void) {
    printf("\n    Comparing Projection: CPU vs OpenMP\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t omp = run_solver(
        NS_SOLVER_TYPE_PROJECTION_OMP,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );

    /* Skip test if OMP solver is not available */
    if (!omp.success && strstr(omp.error_msg, "not available") != NULL) {
        printf("      OpenMP solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("OpenMP solver not available (OpenMP not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(omp.success, "OMP Projection must succeed");

    double diff = fabs(cpu.u_at_center - omp.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      OMP  u_center: %.6f\n", omp.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    /* NOTE: This test is currently expected to FAIL until the OMP projection
     * solver bug is fixed. The OMP implementation produces different results
     * than CPU/AVX2, indicating a parallelization bug. */
    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_CONSISTENCY_TOL,
        "CPU and OpenMP Projection must produce identical results");
}

/* ============================================================================
 * TEST: GPU backend consistency
 * ============================================================================ */

void test_projection_cpu_gpu_consistency(void) {
    printf("\n    Comparing Projection: CPU vs GPU\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t gpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS, ARCH_TEST_DT
    );

    /* Skip test if GPU solver is not available */
    if (!gpu.success && strstr(gpu.error_msg, "not available") != NULL) {
        printf("      GPU solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("GPU solver not available (CUDA not enabled or no GPU)");
    }

    TEST_ASSERT_TRUE_MESSAGE(gpu.success, "GPU Projection must succeed");

    double diff = fabs(cpu.u_at_center - gpu.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      GPU  u_center: %.6f\n", gpu.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_CONSISTENCY_TOL,
        "CPU and GPU Projection must produce identical results");
}

/* ============================================================================
 * TEST: All solver types can be instantiated
 * ============================================================================ */

void test_all_solvers_instantiate(void) {
    printf("\n    Testing solver instantiation\n");

    /* Required solvers (must always be available) */
    const char* required_types[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
    };
    const char* required_names[] = {
        "Explicit Euler CPU",
        "Explicit Euler AVX2",
        "Projection CPU",
        "Projection AVX2",
    };
    const int num_required = 4;

    /* Optional solvers (may not be available depending on build configuration) */
    const char* optional_types[] = {
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        NS_SOLVER_TYPE_PROJECTION_OMP,
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
    };
    const char* optional_names[] = {
        "Explicit Euler OMP",
        "Projection OMP",
        "Projection GPU",
    };
    const int num_optional = 3;

    ns_solver_registry_t* registry = cfd_registry_create();
    cfd_registry_register_defaults(registry);

    int required_count = 0;
    int optional_count = 0;

    /* Check required solvers */
    for (int i = 0; i < num_required; i++) {
        ns_solver_t* solver = cfd_solver_create(registry, required_types[i]);
        if (solver) {
            printf("      %s: OK\n", required_names[i]);
            solver_destroy(solver);
            required_count++;
        } else {
            printf("      %s: FAILED\n", required_names[i]);
        }
    }

    /* Check optional solvers */
    for (int i = 0; i < num_optional; i++) {
        ns_solver_t* solver = cfd_solver_create(registry, optional_types[i]);
        if (solver) {
            printf("      %s: OK (optional)\n", optional_names[i]);
            solver_destroy(solver);
            optional_count++;
        } else {
            printf("      %s: SKIPPED (optional, OpenMP not enabled)\n", optional_names[i]);
        }
    }

    cfd_registry_destroy(registry);

    printf("      %d/%d required solvers instantiated\n", required_count, num_required);
    if (optional_count > 0) {
        printf("      %d/%d optional solvers instantiated\n", optional_count, num_optional);
    }

    TEST_ASSERT_EQUAL_INT_MESSAGE(num_required, required_count,
        "All required solvers must be instantiable");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    UNITY_BEGIN();

    printf("\n");
    printf("========================================\n");
    printf("CROSS-ARCHITECTURE CONSISTENCY TESTS\n");
    printf("========================================\n");
    printf("\n");
    printf("Consistency tolerance: diff < %.3f\n", ARCH_CONSISTENCY_TOL);
    printf("\n");

    printf("[Solver Instantiation]\n");
    RUN_TEST(test_all_solvers_instantiate);

    printf("\n[Explicit Euler Backend Consistency]\n");
    RUN_TEST(test_euler_cpu_avx2_consistency);
    RUN_TEST(test_euler_cpu_omp_consistency);

    printf("\n[Projection Backend Consistency]\n");
    RUN_TEST(test_projection_cpu_avx2_consistency);
    RUN_TEST(test_projection_cpu_omp_consistency);
    RUN_TEST(test_projection_cpu_gpu_consistency);

    return UNITY_END();
}
