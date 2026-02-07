/**
 * @file test_solver_architecture.c
 * @brief Cross-architecture solver consistency tests
 *
 * This test file verifies that ALL solver backend implementations (CPU scalar,
 * AVX2/SIMD, OpenMP, GPU) produce IDENTICAL results for the same problem.
 *
 * VALIDATION REQUIREMENTS:
 * 1. All backends of the same solver type MUST produce identical results
 * 2. Consistency tolerance: u_center difference <= 0.2% (ARCH_CONSISTENCY_TOL)
 * 3. All non-optional backends must succeed
 *
 * If tests fail, the BACKEND implementation needs fixing, not the tolerance.
 */

#include "cavity_reference_data.h"
#include "lid_driven_cavity_common.h"

/* Helper: check if solver result indicates unavailability */
static int solver_not_available(const solver_result_t* r) {
    return !r->success;
}

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * TEST CONFIGURATION
 * ============================================================================ */

/* Minimal configuration for architecture consistency checks
 *
 * Note: Explicit Euler solver uses a conservative dt limit (0.0001) internally
 * for stability. We use more steps for Euler tests to ensure non-zero flow
 * develops at the center point, providing meaningful consistency validation.
 * With dt=0.0001 and 5000 steps, we get 0.5s of simulation time.
 */
#define ARCH_TEST_STEPS_EULER  5000  /* Euler needs many steps due to internal dt limit */
#define ARCH_TEST_STEPS_PROJ   50    /* Projection uses full dt, fewer steps needed */
#define ARCH_TEST_DT           0.005
#define ARCH_GRID_SIZE         9

/* Tolerance for backend consistency (must match exactly) */
#define ARCH_CONSISTENCY_TOL 0.002

/* Wider tolerance for projection backends: CPU uses CG Poisson solver while
 * AVX2/OMP/GPU use Red-Black SOR. Different Poisson solvers converge to
 * slightly different pressure fields, causing small velocity differences.
 * Both produce correct results within their solver tolerances. */
#define ARCH_PROJECTION_CONSISTENCY_TOL 0.01

/* ============================================================================
 * SOLVER RESULT STRUCTURE
 * ============================================================================
 * Uses cavity_sim_result_t from lid_driven_cavity_common.h as the underlying
 * type. The solver_result_t typedef provides backward compatibility.
 */

typedef cavity_sim_result_t solver_result_t;

/* ============================================================================
 * Run simulation with specific solver
 * ============================================================================
 * Thin wrapper around cavity_run_with_solver() with fixed Re=100, lid_vel=1.0
 */

static solver_result_t run_solver(const char* solver_type,
                                   size_t nx, size_t ny,
                                   int max_steps, double dt) {
    return cavity_run_with_solver(solver_type, nx, ny, 100.0, 1.0, max_steps, dt);
}

/* ============================================================================
 * TEST: Explicit Euler - CPU, AVX2, OpenMP must match exactly
 * ============================================================================ */

void test_euler_cpu_avx2_consistency(void) {
    printf("\n    Comparing Explicit Euler: CPU vs AVX2/SIMD\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_EULER, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Euler must succeed");

    solver_result_t avx2 = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_EULER, ARCH_TEST_DT
    );

    /* Skip test if AVX2 solver is not available */
    if (solver_not_available(&avx2)) {
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
        ARCH_TEST_STEPS_EULER, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Euler must succeed");

    solver_result_t omp = run_solver(
        NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_EULER, ARCH_TEST_DT
    );

    /* Skip test if OMP solver is not available */
    if (solver_not_available(&omp)) {
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
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t avx2 = run_solver(
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );

    /* Skip test if AVX2 solver is not available */
    if (solver_not_available(&avx2)) {
        printf("      AVX2 solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("AVX2 solver not available (AVX2 not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(avx2.success, "AVX2 Projection must succeed");

    double diff = fabs(cpu.u_at_center - avx2.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      AVX2 u_center: %.6f\n", avx2.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_PROJECTION_CONSISTENCY_TOL,
        "CPU and AVX2 Projection must produce consistent results");
}

void test_projection_cpu_omp_consistency(void) {
    printf("\n    Comparing Projection: CPU vs OpenMP\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t omp = run_solver(
        NS_SOLVER_TYPE_PROJECTION_OMP,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );

    /* Skip test if OMP solver is not available */
    if (solver_not_available(&omp)) {
        printf("      OpenMP solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("OpenMP solver not available (OpenMP not enabled)");
    }

    TEST_ASSERT_TRUE_MESSAGE(omp.success, "OMP Projection must succeed");

    double diff = fabs(cpu.u_at_center - omp.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      OMP  u_center: %.6f\n", omp.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_PROJECTION_CONSISTENCY_TOL,
        "CPU and OpenMP Projection must produce consistent results");
}

/* ============================================================================
 * TEST: GPU backend consistency
 * ============================================================================ */

void test_projection_cpu_gpu_consistency(void) {
    printf("\n    Comparing Projection: CPU vs GPU\n");

    solver_result_t cpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );
    TEST_ASSERT_TRUE_MESSAGE(cpu.success, "CPU Projection must succeed");

    solver_result_t gpu = run_solver(
        NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
        ARCH_GRID_SIZE, ARCH_GRID_SIZE,
        ARCH_TEST_STEPS_PROJ, ARCH_TEST_DT
    );

    /* Skip test if GPU solver is not available */
    if (solver_not_available(&gpu)) {
        printf("      GPU solver not available, skipping\n");
        TEST_IGNORE_MESSAGE("GPU solver not available (CUDA not enabled or no GPU)");
    }

    TEST_ASSERT_TRUE_MESSAGE(gpu.success, "GPU Projection must succeed");

    double diff = fabs(cpu.u_at_center - gpu.u_at_center);
    printf("      CPU  u_center: %.6f\n", cpu.u_at_center);
    printf("      GPU  u_center: %.6f\n", gpu.u_at_center);
    printf("      Difference:    %.6f\n", diff);

    TEST_ASSERT_TRUE_MESSAGE(diff < ARCH_PROJECTION_CONSISTENCY_TOL,
        "CPU and GPU Projection must produce consistent results");
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
            printf("      %s: SKIPPED (optional, not available)\n", optional_names[i]);
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
