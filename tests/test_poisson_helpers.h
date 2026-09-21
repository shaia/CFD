/**
 * @file test_poisson_helpers.h
 * @brief Helpers shared by the Poisson solver tests under tests/math and
 *        tests/solvers.
 */
#ifndef CFD_TEST_POISSON_HELPERS_H
#define CFD_TEST_POISSON_HELPERS_H

#include "cfd/solvers/poisson_solver.h"

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
 * Why the body is empty: the Krylov solvers zero the halo before calling this
 * hook, so zero walls are already in place by the time it runs. That is a
 * contract between this function and poisson_solver_krylov_apply_bc, which is
 * the reason it lives in one place rather than three -- it has changed once
 * already.
 *
 * params.walls = poisson_walls_uniform(POISSON_WALL_DIRICHLET, 0.0) expresses
 * the same thing declaratively and is the better choice in new tests; this hook
 * is what the tests predating that field use, and it exercises the hook path
 * itself, which still has to keep working.
 */
static void hold_walls_at_zero(poisson_solver_t* solver, double* x) {
    (void)solver;
    (void)x;
}

#endif /* CFD_TEST_POISSON_HELPERS_H */
