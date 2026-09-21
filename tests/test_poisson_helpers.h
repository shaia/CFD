/**
 * @file test_poisson_helpers.h
 * @brief Helpers shared by the Poisson solver tests under tests/math and
 *        tests/solvers.
 */
#ifndef CFD_TEST_POISSON_HELPERS_H
#define CFD_TEST_POISSON_HELPERS_H

#include "cfd/solvers/poisson_solver.h"

#include <stddef.h>

/**
 * Hold every wall at zero, making the operator Dirichlet and so nonsingular.
 *
 * Install it on solver->apply_bc before poisson_solver_init(), which reads the
 * slot when it resolves omega.
 *
 * Why a test wants it: the default zero-gradient walls give a singular Neumann
 * system whose nullspace is the constants, so a right-hand side with a nonzero
 * interior mean -- a uniform one, or the strictly negative RHS of a
 * manufactured solution -- describes a system with no solution, and
 * poisson_solver_solve() refuses it. Prescribing every face pins the level and
 * admits any RHS. It is also the problem a solution vanishing on the boundary
 * actually poses.
 *
 * It writes the zeros itself rather than relying on the caller. Only
 * poisson_solver_krylov_apply_bc() zeroes the halo before invoking this hook;
 * poisson_solver_apply_bc(), which is the path the stationary SOR and
 * Red-Black SOR solvers take, calls it directly. An empty body therefore held
 * nothing at zero there and left whatever the iterate already carried on the
 * walls, while its mere presence still flipped poisson_solver_resolve_omega()
 * to the Dirichlet formula -- correct only for as long as the buffer happened
 * to arrive from calloc.
 *
 * params.walls = poisson_walls_uniform(POISSON_WALL_DIRICHLET, 0.0) expresses
 * the same thing declaratively and is the better choice in new tests; this hook
 * is what the tests predating that field use, what the stationary solvers have
 * no other way to ask for, and it exercises the hook path itself, which still
 * has to keep working.
 */
static void hold_walls_at_zero(poisson_solver_t* solver, double* x) {
    size_t nx = solver->nx;
    size_t ny = solver->ny;
    size_t nz = solver->nz;
    size_t plane = nx * ny;

    for (size_t k = 0; k < nz; k++) {
        double* p = x + k * plane;
        /* z faces: the whole plane, on a 3D grid. */
        if (nz > 1 && (k == 0 || k == nz - 1)) {
            for (size_t idx = 0; idx < plane; idx++) {
                p[idx] = 0.0;
            }
            continue;
        }
        for (size_t i = 0; i < nx; i++) {
            p[i] = 0.0;                       /* bottom */
            p[(ny - 1) * nx + i] = 0.0;       /* top */
        }
        for (size_t j = 0; j < ny; j++) {
            p[j * nx] = 0.0;                  /* left */
            p[j * nx + (nx - 1)] = 0.0;       /* right */
        }
    }
}

#endif /* CFD_TEST_POISSON_HELPERS_H */
