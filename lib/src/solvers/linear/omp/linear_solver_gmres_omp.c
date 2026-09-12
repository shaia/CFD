/**
 * @file linear_solver_gmres_omp.c
 * @brief Restarted GMRES(m) solver - OpenMP parallelized implementation
 *
 * Supplies the OpenMP-parallelized O(n) primitives (linear_solver_primitives_omp.h,
 * shared with CG OMP) to the shared GMRES(m) algorithm template
 * (gmres_template/linear_solver_gmres_template.h). The dense Givens/Hessenberg and
 * back-substitution work in the template stays serial. dot_product_omp's
 * reduction regroups partial sums per thread, so results agree with the scalar
 * backend to rounding rather than bit-for-bit (2D on one thread is exact).
 *
 * Operator/sign convention matches CG and scalar GMRES: A = -nabla^2, b = -rhs.
 */

#include "../linear_solver_internal.h"

#ifdef CFD_ENABLE_OPENMP

#include "linear_solver_primitives_omp.h"

#define GMRES_SUFFIX            omp
#define GMRES_SOLVER_NAME       POISSON_SOLVER_TYPE_GMRES_OMP
#define GMRES_DESCRIPTION       "Restarted GMRES(m) (OpenMP)"
#define GMRES_BACKEND           POISSON_BACKEND_OMP
#define GMRES_LOG_TAG           "GMRES-OMP"
#define GMRES_VEC_CALLOC(count) cfd_calloc((count), sizeof(double))
#define GMRES_VEC_FREE(ptr)     cfd_free(ptr)

#define GMRES_DOT               dot_product_omp
#define GMRES_AXPY              axpy_omp
#define GMRES_SCALE             scale_vector_omp
#define GMRES_COPY              copy_vector_omp
#define GMRES_APPLY_A           apply_laplacian_omp
#define GMRES_RESIDUAL          compute_residual_omp
#define GMRES_PRECOND           apply_jacobi_precond_omp

#include "../gmres_template/linear_solver_gmres_template.h"

#endif /* CFD_ENABLE_OPENMP */
