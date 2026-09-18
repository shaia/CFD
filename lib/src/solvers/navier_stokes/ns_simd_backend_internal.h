/**
 * @file ns_simd_backend_internal.h
 * @brief Availability of the SIMD Navier-Stokes backend.
 *
 * A SIMD NS solver is usable only when BOTH hold:
 *   1. the SIMD kernels were compiled in (CFD_HAS_AVX2, set by CMake from
 *      -DCFD_ENABLE_AVX2=ON), and
 *   2. the CPU actually supports those instructions at runtime.
 *
 * Checking only the runtime half reports a backend that the build does not
 * contain; checking only the compile-time half faults on an older CPU.
 *
 * Note this is narrower than cfd_has_simd(), which answers "does this CPU
 * support any SIMD" and is the right question for a CPU capability report but
 * the wrong one for backend selection. It is also narrower than the linear
 * solvers' poisson_solver_simd_backend_available(), which additionally
 * requires OpenMP because the SIMD Poisson kernels are threaded. The NS
 * kernels are not, so they carry no such requirement.
 *
 * There are no NEON Navier-Stokes kernels, so NEON CPUs report unavailable
 * even though cfd_has_simd() reports true for them.
 */
#ifndef CFD_NS_SIMD_BACKEND_INTERNAL_H
#define CFD_NS_SIMD_BACKEND_INTERNAL_H

#include "cfd/core/cfd_status.h"

#include <stdbool.h>

/**
 * Whether a SIMD Navier-Stokes solver can run in this build on this CPU.
 *
 * @return true when the SIMD kernels are compiled in and the CPU supports them
 */
bool ns_simd_backend_available(void);

/**
 * Init-time guard for the SIMD Navier-Stokes solvers.
 *
 * @return CFD_SUCCESS, or CFD_ERROR_UNSUPPORTED with the reason set
 */
cfd_status_t ns_check_simd_backend(void);

#endif /* CFD_NS_SIMD_BACKEND_INTERNAL_H */
