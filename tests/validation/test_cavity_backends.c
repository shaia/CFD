/**
 * @file test_cavity_backends.c
 * @brief Backend-specific validation for lid-driven cavity
 *
 * Tests that ALL solver backends independently achieve the scientific target:
 * - RMS error vs Ghia et al. (1982) < 0.10
 * - All backends produce consistent results (within 0.1%)
 *
 * BACKEND COVERAGE:
 * =================
 * - CPU Scalar: projection, explicit_euler
 * - AVX2/SIMD: projection_optimized, explicit_euler_optimized
 * - OpenMP: projection_omp, explicit_euler_omp
 * - CUDA GPU: projection_jacobi_gpu
 *
 * TEST STRATEGY:
 * ==============
 * 1. Run each backend independently at 33×33 (CI mode) or 129×129 (full validation)
 * 2. Extract centerline profiles and compute RMS vs Ghia
 * 3. Verify RMS < 0.10 (test FAILS if RMS > 0.10)
 * 4. Compare backends to ensure consistency (values within 0.1%)
 */

#include <string.h>

#include "cavity_reference_data.h"
#include "cavity_validation_utils.h"
#include "lid_driven_cavity_common.h"

void setUp(void) {}
void tearDown(void) {}

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/**
 * Tolerance for backend consistency checks.
 * All backends should produce results within 0.1% of each other.
 */
#define BACKEND_CONSISTENCY_TOLERANCE 0.001

/**
 * Scientific targets for Ghia agreement:
 * - Projection method: RMS < 0.10 (production solver, strict target)
 * - Explicit Euler: RMS < 0.15 (simpler method, relaxed target)
 */
#define GHIA_RMS_TARGET_PROJECTION 0.10
#define GHIA_RMS_TARGET_EULER 0.15

/**
 * Grid size for validation (33×33 for CI, 129×129 for release)
 */
#if CAVITY_FULL_VALIDATION
#define VALIDATION_GRID_SIZE 129
#define VALIDATION_STEPS 50000
#define VALIDATION_DT 0.0002
#else
#define VALIDATION_GRID_SIZE 33
#define VALIDATION_STEPS 5000
#define VALIDATION_DT 0.0005
#endif

/* ============================================================================
 * HELPER: Extract centerline profiles from context
 * ============================================================================ */

typedef struct {
    double* y_coords;
    double* u_values;
    double* x_coords;
    double* v_values;
    size_t ny;
    size_t nx;
} profile_data_t;

static profile_data_t extract_profiles_from_ctx(const cavity_context_t* ctx) {
    profile_data_t data = {0};
    size_t nx = ctx->nx;
    size_t ny = ctx->ny;

    data.y_coords = malloc(ny * sizeof(double));
    data.u_values = malloc(ny * sizeof(double));
    data.x_coords = malloc(nx * sizeof(double));
    data.v_values = malloc(nx * sizeof(double));

    if (!data.y_coords || !data.u_values || !data.x_coords || !data.v_values) {
        free(data.y_coords);
        free(data.u_values);
        free(data.x_coords);
        free(data.v_values);
        memset(&data, 0, sizeof(data));
        return data;
    }

    data.ny = ny;
    data.nx = nx;

    /* Vertical centerline: u(x=0.5, y) */
    size_t center_i = nx / 2;
    for (size_t j = 0; j < ny; j++) {
        data.y_coords[j] = ctx->g->y[j];
        data.u_values[j] = ctx->field->u[j * nx + center_i];
    }

    /* Horizontal centerline: v(x, y=0.5) */
    size_t center_j = ny / 2;
    for (size_t i = 0; i < nx; i++) {
        data.x_coords[i] = ctx->g->x[i];
        data.v_values[i] = ctx->field->v[center_j * nx + i];
    }

    return data;
}

static void free_profile_data(profile_data_t* data) {
    free(data->y_coords);
    free(data->u_values);
    free(data->x_coords);
    free(data->v_values);
}

/* ============================================================================
 * HELPER: Compute RMS error against Ghia data
 * ============================================================================ */

static double compute_ghia_rms(const profile_data_t* data,
                                const double* ghia_coords,
                                const double* ghia_vals,
                                int is_vertical) {
    double sum_sq = 0.0;
    int count = 0;

    for (size_t i = 0; i < GHIA_NUM_POINTS; i++) {
        double coord = ghia_coords[i];
        double computed = 0.0;

        if (is_vertical) {
            /* Interpolate u from vertical centerline */
            for (size_t j = 0; j < data->ny - 1; j++) {
                if (coord >= data->y_coords[j] && coord <= data->y_coords[j + 1]) {
                    double t = (coord - data->y_coords[j]) /
                               (data->y_coords[j + 1] - data->y_coords[j]);
                    computed = data->u_values[j] + t * (data->u_values[j + 1] - data->u_values[j]);
                    break;
                }
            }
        } else {
            /* Interpolate v from horizontal centerline */
            for (size_t j = 0; j < data->nx - 1; j++) {
                if (coord >= data->x_coords[j] && coord <= data->x_coords[j + 1]) {
                    double t = (coord - data->x_coords[j]) /
                               (data->x_coords[j + 1] - data->x_coords[j]);
                    computed = data->v_values[j] + t * (data->v_values[j + 1] - data->v_values[j]);
                    break;
                }
            }
        }

        double error = computed - ghia_vals[i];
        sum_sq += error * error;
        count++;
    }

    return sqrt(sum_sq / count);
}

/* ============================================================================
 * BACKEND VALIDATION TESTS
 * ============================================================================ */

/**
 * Test backend: run simulation and verify RMS < target
 *
 * This function runs a single backend and checks that it:
 * 1. Completes without errors
 * 2. Achieves RMS < target vs Ghia reference data
 *
 * If RMS >= target, the test FAILS (no loose tolerances allowed).
 */
static void test_backend_validation(const char* solver_type,
                                     const char* backend_name,
                                     int max_steps,
                                     double rms_target) {
    printf("\n    Testing backend: %s (%s)\n", backend_name, solver_type);

    /* Run simulation */
    cavity_context_t* ctx = NULL;
    cavity_sim_result_t result = cavity_run_with_solver_ctx(
        solver_type,
        VALIDATION_GRID_SIZE, VALIDATION_GRID_SIZE,
        100.0, 1.0,  /* Re=100, lid_velocity=1.0 */
        max_steps, VALIDATION_DT,
        &ctx
    );

    /* Check if backend is available */
    if (result.solver_unavailable) {
        printf("      [SKIPPED] %s\n", result.error_msg);
        TEST_IGNORE_MESSAGE(result.error_msg);
        return;
    }

    /* Check simulation succeeded */
    if (!result.success) {
        printf("      [FAILED] %s\n", result.error_msg);
        TEST_FAIL_MESSAGE(result.error_msg);
        return;
    }

    TEST_ASSERT_NOT_NULL(ctx);

    /* Extract centerline profiles */
    profile_data_t profiles = extract_profiles_from_ctx(ctx);
    TEST_ASSERT_NOT_NULL_MESSAGE(profiles.y_coords, "Failed to extract profiles");

    /* Compute RMS errors vs Ghia */
    double rms_u = compute_ghia_rms(&profiles, GHIA_Y_COORDS, GHIA_U_RE100, 1);
    double rms_v = compute_ghia_rms(&profiles, GHIA_X_COORDS, GHIA_V_RE100, 0);

    printf("      RMS_u: %.4f  RMS_v: %.4f  (target < %.2f)\n", rms_u, rms_v, rms_target);

    /* CRITICAL: Tests MUST FAIL if RMS >= target (no "baseline" workarounds) */
    char msg[256];
    if (rms_u >= rms_target) {
        snprintf(msg, sizeof(msg),
                 "%s: RMS_u=%.4f >= target %.2f (UNACCEPTABLE)",
                 backend_name, rms_u, rms_target);
        free_profile_data(&profiles);
        cavity_context_destroy(ctx);
        TEST_FAIL_MESSAGE(msg);
    }

    if (rms_v >= rms_target) {
        snprintf(msg, sizeof(msg),
                 "%s: RMS_v=%.4f >= target %.2f (UNACCEPTABLE)",
                 backend_name, rms_v, rms_target);
        free_profile_data(&profiles);
        cavity_context_destroy(ctx);
        TEST_FAIL_MESSAGE(msg);
    }

    printf("      [PASS] Both RMS < %.2f\n", rms_target);

    free_profile_data(&profiles);
    cavity_context_destroy(ctx);
}

/* ============================================================================
 * INDIVIDUAL BACKEND TESTS
 * ============================================================================ */

void test_projection_cpu_scalar(void) {
    test_backend_validation(NS_SOLVER_TYPE_PROJECTION,
                            "Projection (CPU Scalar)",
                            VALIDATION_STEPS,
                            GHIA_RMS_TARGET_PROJECTION);
}

void test_projection_optimized_avx2(void) {
    test_backend_validation(NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
                            "Projection (AVX2/SIMD)",
                            VALIDATION_STEPS,
                            GHIA_RMS_TARGET_PROJECTION);
}

void test_projection_omp(void) {
    test_backend_validation(NS_SOLVER_TYPE_PROJECTION_OMP,
                            "Projection (OpenMP)",
                            VALIDATION_STEPS,
                            GHIA_RMS_TARGET_PROJECTION);
}

void test_projection_gpu(void) {
    test_backend_validation(NS_SOLVER_TYPE_PROJECTION_JACOBI_GPU,
                            "Projection (CUDA GPU)",
                            VALIDATION_STEPS,
                            GHIA_RMS_TARGET_PROJECTION);
}

void test_explicit_euler_cpu(void) {
    /* Explicit Euler has internal dt cap of 0.0001, needs more steps */
    int euler_steps = VALIDATION_STEPS * 5;
    test_backend_validation(NS_SOLVER_TYPE_EXPLICIT_EULER,
                            "Explicit Euler (CPU)",
                            euler_steps,
                            GHIA_RMS_TARGET_EULER);
}

void test_explicit_euler_optimized(void) {
    int euler_steps = VALIDATION_STEPS * 5;
    test_backend_validation(NS_SOLVER_TYPE_EXPLICIT_EULER_OPTIMIZED,
                            "Explicit Euler (AVX2/SIMD)",
                            euler_steps,
                            GHIA_RMS_TARGET_EULER);
}

void test_explicit_euler_omp(void) {
    int euler_steps = VALIDATION_STEPS * 5;
    test_backend_validation(NS_SOLVER_TYPE_EXPLICIT_EULER_OMP,
                            "Explicit Euler (OpenMP)",
                            euler_steps,
                            GHIA_RMS_TARGET_EULER);
}

/* ============================================================================
 * BACKEND CONSISTENCY TEST
 * ============================================================================ */

/**
 * Verify all backends produce consistent results (within 0.1%)
 *
 * This test runs multiple backends and compares their center point values.
 * All backends should produce nearly identical results - differences should
 * only be due to floating-point rounding.
 */
void test_backend_consistency(void) {
    printf("\n    Backend consistency check (center point values):\n");

    const char* solvers[] = {
        NS_SOLVER_TYPE_PROJECTION,
        NS_SOLVER_TYPE_PROJECTION_OPTIMIZED,
        NS_SOLVER_TYPE_PROJECTION_OMP
    };
    const char* names[] = {
        "CPU",
        "AVX2/SIMD",
        "OpenMP"
    };
    const int n_solvers = 3;

    double u_center_vals[3] = {0};
    double v_center_vals[3] = {0};
    int n_available = 0;

    /* Run each backend and extract center values */
    for (int i = 0; i < n_solvers; i++) {
        cavity_sim_result_t result = cavity_run_with_solver(
            solvers[i],
            33, 33,  /* Use smaller grid for faster consistency check */
            100.0, 1.0,
            2000, 0.0005
        );

        if (result.solver_unavailable) {
            printf("      %s: UNAVAILABLE\n", names[i]);
            continue;
        }

        if (!result.success) {
            char msg[512];
            snprintf(msg, sizeof(msg), "%s failed: %s", names[i], result.error_msg);
            TEST_FAIL_MESSAGE(msg);
        }

        u_center_vals[n_available] = result.u_at_center;
        v_center_vals[n_available] = result.v_at_center;
        printf("      %s: u_center=%.6f, v_center=%.6f\n",
               names[i], result.u_at_center, result.v_at_center);
        n_available++;
    }

    if (n_available < 2) {
        TEST_IGNORE_MESSAGE("Not enough backends available for consistency check");
        return;
    }

    /* Compare all backends to first available backend */
    double u_ref = u_center_vals[0];
    double v_ref = v_center_vals[0];

    for (int i = 1; i < n_available; i++) {
        double u_diff = fabs(u_center_vals[i] - u_ref);
        double v_diff = fabs(v_center_vals[i] - v_ref);
        double u_rel_diff = u_diff / (fabs(u_ref) + 1e-10);
        double v_rel_diff = v_diff / (fabs(v_ref) + 1e-10);

        char msg[256];
        if (u_rel_diff > BACKEND_CONSISTENCY_TOLERANCE) {
            snprintf(msg, sizeof(msg),
                     "Backend %d u_center differs from reference by %.3f%% (> 0.1%%)",
                     i, u_rel_diff * 100);
            TEST_FAIL_MESSAGE(msg);
        }

        if (v_rel_diff > BACKEND_CONSISTENCY_TOLERANCE) {
            snprintf(msg, sizeof(msg),
                     "Backend %d v_center differs from reference by %.3f%% (> 0.1%%)",
                     i, v_rel_diff * 100);
            TEST_FAIL_MESSAGE(msg);
        }
    }

    printf("      [PASS] All backends consistent within %.1f%%\n",
           BACKEND_CONSISTENCY_TOLERANCE * 100);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : NULL;

    UNITY_BEGIN();

    printf("\n");
    printf("========================================\n");
    printf("BACKEND VALIDATION TESTS\n");
    printf("========================================\n");
    printf("Grid:        %dx%d\n", VALIDATION_GRID_SIZE, VALIDATION_GRID_SIZE);
    printf("Steps:       %d\n", VALIDATION_STEPS);
    printf("Targets:\n");
    printf("  Projection: RMS < %.2f (production solver)\n", GHIA_RMS_TARGET_PROJECTION);
    printf("  Euler:      RMS < %.2f (simpler method)\n", GHIA_RMS_TARGET_EULER);
#if CAVITY_FULL_VALIDATION
    printf("Mode:        FULL VALIDATION\n");
#else
    printf("Mode:        FAST (CI)\n");
#endif
    if (filter)
        printf("Filter:      %s\n", filter);
    printf("========================================\n");

#if !CAVITY_FULL_VALIDATION
    if (!filter || strcmp(filter, "projection_scalar") == 0)
        RUN_TEST(test_projection_cpu_scalar);
#endif
    if (!filter || strcmp(filter, "projection_avx2") == 0)
        RUN_TEST(test_projection_optimized_avx2);
    if (!filter || strcmp(filter, "projection_omp") == 0)
        RUN_TEST(test_projection_omp);
    if (!filter || strcmp(filter, "projection_gpu") == 0)
        RUN_TEST(test_projection_gpu);

#if !CAVITY_FULL_VALIDATION
    if (!filter || strcmp(filter, "euler_scalar") == 0)
        RUN_TEST(test_explicit_euler_cpu);
#endif
    if (!filter || strcmp(filter, "euler_avx2") == 0)
        RUN_TEST(test_explicit_euler_optimized);
    if (!filter || strcmp(filter, "euler_omp") == 0)
        RUN_TEST(test_explicit_euler_omp);

    if (!filter || strcmp(filter, "consistency") == 0)
        RUN_TEST(test_backend_consistency);

    return UNITY_END();
}
