/**
 * Poisson Solver Tuning Example
 *
 * Compares Poisson solver methods, backends, and preconditioners
 * for solving the pressure equation. Uses a manufactured solution
 * with known analytical answer for accuracy verification.
 *
 * This example demonstrates:
 *   - poisson_solver_create() factory API with method/backend selection
 *   - Configuring solver parameters (tolerance, max_iterations, omega, preconditioner)
 *   - Reading convergence statistics (iterations, residual, timing)
 *   - poisson_solve() convenience API
 *   - Error handling for unavailable solvers
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/poisson_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Set up a Poisson problem for method/backend comparison.
 *
 * Reference field: p(x,y) = sin(pi*x) * sin(pi*y)
 * RHS:             nabla^2 p = -2*pi^2 * sin(pi*x) * sin(pi*y)
 *
 * Note: p_exact satisfies Dirichlet p=0 on all boundaries, but the Poisson
 * solvers in this library apply homogeneous Neumann BCs by default. The
 * reported L2 error therefore measures relative agreement between solver
 * methods/backends, not absolute accuracy against this analytical field.
 */
static void setup_poisson_problem(double* rhs, double* p_exact,
                                  size_t nx, size_t ny,
                                  double dx, double dy) {
    for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
            double x = (double)i * dx;
            double y = (double)j * dy;
            size_t idx = j * nx + i;

            p_exact[idx] = sin(M_PI * x) * sin(M_PI * y);
            rhs[idx] = -2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

static double compute_l2_error(const double* p, const double* p_exact,
                               size_t nx, size_t ny) {
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t j = 1; j < ny - 1; j++) {
        for (size_t i = 1; i < nx - 1; i++) {
            double diff = p[j * nx + i] - p_exact[j * nx + i];
            sum_sq += diff * diff;
            count++;
        }
    }
    return sqrt(sum_sq / (double)count);
}

static void benchmark_method(const char* label,
                             poisson_solver_method_t method,
                             poisson_solver_backend_t backend,
                             poisson_precond_type_t precond,
                             size_t nx, size_t ny, double dx, double dy,
                             const double* rhs, double* p, double* p_temp,
                             const double* p_exact) {
    poisson_solver_t* solver = poisson_solver_create(method, backend);
    if (!solver) {
        printf("  %-20s  (not available)\n", label);
        return;
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-8;
    params.max_iterations = 10000;
    params.preconditioner = precond;
    if (method == POISSON_METHOD_SOR || method == POISSON_METHOD_REDBLACK_SOR) {
        params.omega = 1.5;
    }

    cfd_status_t status = poisson_solver_init(solver, nx, ny, 1, dx, dy, 0.0, &params);
    if (status != CFD_SUCCESS) {
        printf("  %-20s  init failed (%d)\n", label, status);
        poisson_solver_destroy(solver);
        return;
    }

    /* Zero-initialize solution */
    memset(p, 0, nx * ny * sizeof(double));

    poisson_solver_stats_t stats = poisson_solver_stats_default();
    cfd_status_t solve_status = poisson_solver_solve(solver, p, p_temp, rhs, &stats);

    double l2_err = (solve_status == CFD_SUCCESS) ? compute_l2_error(p, p_exact, nx, ny) : -1.0;
    const char* status_str = (stats.status == POISSON_CONVERGED) ? "converged" :
                             (stats.status == POISSON_MAX_ITER)  ? "max_iter" :
                             (stats.status == POISSON_DIVERGED)  ? "DIVERGED" : "error";

    printf("  %-20s  %5d iters  res=%.2e  L2=%.2e  %6.1f ms  %s\n",
           label, stats.iterations, stats.final_residual,
           l2_err, stats.elapsed_time_ms, status_str);

    poisson_solver_destroy(solver);
}

int main(void) {
    printf("Poisson Solver Tuning Example\n");
    printf("=============================\n");

    size_t nx = 64, ny = 64;
    double dx = 1.0 / (double)(nx - 1);
    double dy = 1.0 / (double)(ny - 1);

    printf("Grid: %zu x %zu, dx=%.4f, dy=%.4f\n", nx, ny, dx, dy);
    printf("Problem: sin(pi*x)*sin(pi*y)\n\n");

    /* Allocate arrays */
    size_t n = nx * ny;
    double* rhs = (double*)calloc(n, sizeof(double));
    double* p = (double*)calloc(n, sizeof(double));
    double* p_temp = (double*)calloc(n, sizeof(double));
    double* p_exact = (double*)calloc(n, sizeof(double));
    if (!rhs || !p || !p_temp || !p_exact) {
        fprintf(stderr, "Memory allocation failed\n");
        free(rhs);
        free(p);
        free(p_temp);
        free(p_exact);
        return 1;
    }

    setup_poisson_problem(rhs, p_exact, nx, ny, dx, dy);

    /* Section 1: Method comparison (scalar backend) */
    printf("--- Method Comparison (Scalar Backend) ---\n");
    printf("  %-20s  %5s %5s  %10s  %10s  %8s  %s\n",
           "Method", "Iters", "", "Residual", "L2 Error", "Time", "Status");

    benchmark_method("Jacobi", POISSON_METHOD_JACOBI, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("SOR (w=1.5)", POISSON_METHOD_SOR, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("Red-Black SOR", POISSON_METHOD_REDBLACK_SOR, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("CG", POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("CG + Jacobi PC", POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_JACOBI, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("BiCGSTAB", POISSON_METHOD_BICGSTAB, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);

    /* Section 2: Backend comparison (CG method) */
    printf("\n--- Backend Comparison (CG Method) ---\n");
    benchmark_method("CG Scalar", POISSON_METHOD_CG, POISSON_BACKEND_SCALAR,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("CG SIMD", POISSON_METHOD_CG, POISSON_BACKEND_SIMD,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);
    benchmark_method("CG OMP", POISSON_METHOD_CG, POISSON_BACKEND_OMP,
                     POISSON_PRECOND_NONE, nx, ny, dx, dy, rhs, p, p_temp, p_exact);

    /* Section 3: Convenience API */
    printf("\n--- Convenience API ---\n");
    memset(p, 0, n * sizeof(double));
    int iters = poisson_solve(p, p_temp, rhs, nx, ny, dx, dy, POISSON_SOLVER_CG_SCALAR);
    if (iters < 0) {
        printf("  poisson_solve(CG_SCALAR): FAILED (non-converged or error)\n");
        printf("    Status: \"%s\"\n", cfd_get_error_string(cfd_get_last_status()));
        cfd_clear_error();
    } else {
        double err = compute_l2_error(p, p_exact, nx, ny);
        printf("  poisson_solve(CG_SCALAR): %d iterations, L2 error = %.2e\n", iters, err);
    }

    /* Section 4: Error handling for unavailable solver */
    printf("\n--- Error Handling ---\n");
    poisson_solver_t* solver = poisson_solver_create(POISSON_METHOD_MULTIGRID, POISSON_BACKEND_SCALAR);
    if (!solver) {
        printf("  Multigrid solver: not available (returns NULL)\n");
        printf("  Status: \"%s\"\n", cfd_get_error_string(cfd_get_last_status()));
        cfd_clear_error();
    } else {
        printf("  Multigrid solver: available\n");
        poisson_solver_destroy(solver);
    }

    free(rhs);
    free(p);
    free(p_temp);
    free(p_exact);

    printf("\nDone.\n");
    return 0;
}
