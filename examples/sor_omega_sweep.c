/**
 * SOR Omega Sweep Example
 *
 * Counts the sweeps SOR and Red-Black SOR take to reduce the residual a million
 * times for every relaxation factor omega across a range, on one grid. Plotted,
 * the counts form a U whose lowest point is the best omega for that grid; the
 * automatic omega (params.omega = 0) is solved last for comparison.
 *
 * The right-hand side is a fixed-seed pseudo-random field with zero interior
 * mean, so every mode of the error is present and the zero-gradient problem has
 * a solution. --walls dirichlet installs an apply_bc that leaves the boundary at
 * zero instead of copying the interior onto it; the CUDA solvers reject it.
 *
 * This example demonstrates:
 *   - Setting params.omega explicitly, and leaving it at 0 for the automatic value
 *   - Replacing the default boundary condition through solver->apply_bc
 *   - Reading iterations and convergence status from poisson_solver_stats_t
 *
 * Usage:
 *   sor_omega_sweep [--grid N] [--method sor|redblack] [--backend scalar|simd|omp|gpu]
 *                   [--walls neumann|dirichlet] [--from W] [--to W] [--step W]
 *                   [--seed S]
 *
 * Output is CSV, one row per omega and a final "auto" row:
 *   omega,sweeps,status,final_residual
 */

#include "cfd/core/cfd_status.h"
#include "cfd/core/indexing.h"
#include "cfd/solvers/poisson_solver.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t grid;
    poisson_solver_method_t method;
    poisson_solver_backend_t backend;
    int dirichlet;
    double from, to, step;
    uint64_t seed;
} sweep_options_t;

/* splitmix64, mapped to [-1, 1) */
static double next_uniform(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return 2.0 * ((double)(z >> 11) / 9007199254740992.0) - 1.0;
}

/* Fill the interior with seeded noise, then remove its mean so the problem with
 * zero-gradient walls is solvable. */
static void fill_rhs(double* rhs, size_t n, uint64_t seed) {
    uint64_t state = seed;
    double sum = 0.0;
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            rhs[IDX_2D(i, j, n)] = next_uniform(&state);
            sum += rhs[IDX_2D(i, j, n)];
        }
    }
    double shift = -sum / (double)((n - 2) * (n - 2));
    for (size_t j = 1; j < n - 1; j++) {
        for (size_t i = 1; i < n - 1; i++) {
            rhs[IDX_2D(i, j, n)] += shift;
        }
    }
}

/* The boundary stays at the zero it started with: a homogeneous Dirichlet wall */
static void hold_walls_at_zero(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

/* Solve once from a zero start; returns the sweep count, or -1 if unavailable */
static int solve_once(const sweep_options_t* opt, double omega, double* x,
                      const double* rhs, poisson_solver_stats_t* stats) {
    poisson_solver_t* solver = poisson_solver_create(opt->method, opt->backend);
    if (!solver) {
        return -1;
    }
    if (opt->dirichlet) {
        solver->apply_bc = hold_walls_at_zero;  /* before init, which reads it */
    }

    poisson_solver_params_t params = poisson_solver_params_default();
    params.tolerance = 1e-6;
    params.max_iterations = 100000;
    params.omega = omega;

    double h = 1.0 / (double)(opt->grid - 1);
    if (poisson_solver_init(solver, opt->grid, opt->grid, 1, h, h, 0.0, &params) != CFD_SUCCESS) {
        poisson_solver_destroy(solver);
        return -1;
    }

    memset(x, 0, opt->grid * opt->grid * sizeof(double));
    *stats = poisson_solver_stats_default();
    poisson_solver_solve(solver, x, NULL, rhs, stats);
    poisson_solver_destroy(solver);
    return stats->iterations;
}

static void print_row(const char* omega_label, int sweeps, const poisson_solver_stats_t* stats) {
    const char* status = (stats->status == POISSON_CONVERGED) ? "converged" :
                         (stats->status == POISSON_MAX_ITER)  ? "max_iter" :
                         (stats->status == POISSON_DIVERGED)  ? "diverged" : "error";
    printf("%s,%d,%s,%.6e\n", omega_label, sweeps, status, stats->final_residual);
}

static int parse_options(int argc, char** argv, sweep_options_t* opt) {
    opt->grid = 33;
    opt->method = POISSON_METHOD_REDBLACK_SOR;
    opt->backend = POISSON_BACKEND_SCALAR;
    opt->dirichlet = 0;
    opt->from = 1.0;
    opt->to = 1.99;
    opt->step = 0.01;
    opt->seed = 20260913ULL;

    for (int a = 1; a < argc; a++) {
        const char* key = argv[a];
        const char* value = (a + 1 < argc) ? argv[a + 1] : NULL;
        if (!value) {
            return 0;
        }
        if (strcmp(key, "--grid") == 0) {
            opt->grid = (size_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "--method") == 0) {
            opt->method = (strcmp(value, "sor") == 0) ? POISSON_METHOD_SOR : POISSON_METHOD_REDBLACK_SOR;
        } else if (strcmp(key, "--backend") == 0) {
            opt->backend = (strcmp(value, "simd") == 0) ? POISSON_BACKEND_SIMD :
                           (strcmp(value, "omp") == 0)  ? POISSON_BACKEND_OMP :
                           (strcmp(value, "gpu") == 0)  ? POISSON_BACKEND_GPU : POISSON_BACKEND_SCALAR;
        } else if (strcmp(key, "--walls") == 0) {
            opt->dirichlet = (strcmp(value, "dirichlet") == 0);
        } else if (strcmp(key, "--from") == 0) {
            opt->from = strtod(value, NULL);
        } else if (strcmp(key, "--to") == 0) {
            opt->to = strtod(value, NULL);
        } else if (strcmp(key, "--step") == 0) {
            opt->step = strtod(value, NULL);
        } else if (strcmp(key, "--seed") == 0) {
            opt->seed = (uint64_t)strtoull(value, NULL, 10);
        } else {
            return 0;
        }
        a++;
    }
    return opt->grid >= 3 && opt->step > 0.0 && opt->from > 0.0 && opt->to >= opt->from && opt->to < 2.0;
}

int main(int argc, char** argv) {
    sweep_options_t opt;
    if (!parse_options(argc, argv, &opt)) {
        fprintf(stderr, "usage: sor_omega_sweep [--grid N] [--method sor|redblack] "
                        "[--backend scalar|simd|omp|gpu] [--walls neumann|dirichlet] "
                        "[--from W] [--to W] [--step W] [--seed S]\n");
        return 2;
    }

    size_t n = opt.grid * opt.grid;
    double* x = (double*)calloc(n, sizeof(double));
    double* rhs = (double*)calloc(n, sizeof(double));
    if (!x || !rhs) {
        fprintf(stderr, "Memory allocation failed\n");
        free(x);
        free(rhs);
        return 1;
    }
    fill_rhs(rhs, opt.grid, opt.seed);

    printf("omega,sweeps,status,final_residual\n");
    poisson_solver_stats_t stats;
    int count = (int)((opt.to - opt.from) / opt.step + 0.5) + 1;
    for (int s = 0; s < count; s++) {
        double omega = opt.from + s * opt.step;
        int sweeps = solve_once(&opt, omega, x, rhs, &stats);
        if (sweeps < 0) {
            fprintf(stderr, "Solver not available for this method, backend and walls\n");
            free(x);
            free(rhs);
            return 1;
        }
        char label[32];
        snprintf(label, sizeof(label), "%.4f", omega);
        print_row(label, sweeps, &stats);
    }
    int sweeps = solve_once(&opt, 0.0, x, rhs, &stats);
    print_row("auto", sweeps, &stats);

    free(x);
    free(rhs);
    return 0;
}
