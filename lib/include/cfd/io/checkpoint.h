#ifndef CFD_CHECKPOINT_H
#define CFD_CHECKPOINT_H

#include "cfd/cfd_export.h"

#include "cfd/core/cfd_status.h"
#include "cfd/core/grid.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file checkpoint.h
 * @brief Binary save/restore of complete simulation state (restart/checkpoint).
 *
 * The checkpoint format is a portable, versioned binary file (`.cfdchk`) that
 * captures everything needed to resume a simulation bit-exactly from a step
 * boundary: the grid geometry, the flow field, the scalar solver parameters,
 * the accumulated simulation time, and the active solver's registry name.
 *
 * Solver context buffers (e.g. Runge-Kutta stage arrays) are deliberately NOT
 * serialized: they are pure per-step scratch, recomputed from the field at the
 * start of each step. On restore the solver is recreated by name and re-init'd,
 * which reconstructs all scratch. The field at a step boundary is therefore the
 * complete dynamical state.
 *
 * Function-pointer parameters (`source_func`, `heat_source_func`, and their
 * contexts) cannot be serialized and are excluded; callers that use custom
 * callbacks must re-supply them after a restore.
 *
 * Portability: all multi-byte values are written little-endian with fixed-width
 * types and IEEE-754 doubles (no raw struct dumps). A header endianness marker
 * lets the reader reject files authored on a different-endian platform. A
 * trailing CRC32 detects truncation or bit-rot.
 */

/** Current on-disk checkpoint format version. Bumped on any layout change. */
#define CFD_CHECKPOINT_FORMAT_VERSION 1u

/** Recommended file extension for checkpoint files. */
#define CFD_CHECKPOINT_EXTENSION ".cfdchk"

/**
 * Write a checkpoint to @p path.
 *
 * @param path             Destination file path (overwritten if it exists).
 * @param g                Grid geometry to serialize.
 * @param field            Flow field to serialize (dimensions must match @p g).
 * @param params           Solver parameters (scalar fields only are stored).
 * @param current_time     Accumulated simulation time to record.
 * @param solver_name      Registry name of the active solver (e.g. "rk2_optimized").
 * @param run_prefix       Optional run-directory prefix to record (may be NULL).
 * @param output_base_dir  Optional base output directory to record (may be NULL).
 * @return CFD_SUCCESS, or CFD_ERROR_INVALID on bad arguments, CFD_ERROR_IO on a
 *         file/write failure.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_checkpoint_write(const char* path,
                                                     const grid* g,
                                                     const flow_field* field,
                                                     const ns_solver_params_t* params,
                                                     double current_time,
                                                     const char* solver_name,
                                                     const char* run_prefix,
                                                     const char* output_base_dir);

/**
 * Read a checkpoint from @p path.
 *
 * On success the grid and flow field are newly allocated and ownership passes to
 * the caller (free with grid_destroy() / flow_field_destroy()). On any failure
 * nothing is leaked and @p out_grid / @p out_field are set to NULL.
 *
 * @param out_grid            Receives a newly allocated grid (caller frees).
 * @param out_field           Receives a newly allocated flow field (caller frees).
 * @param out_params          Caller-provided; zero-initialized then scalar fields
 *                            are filled. Function-pointer fields are left NULL.
 * @param out_current_time    Receives the recorded simulation time (may be NULL).
 * @param out_solver_name     Buffer to receive the solver name (may be NULL).
 * @param solver_name_cap     Capacity of @p out_solver_name in bytes.
 * @param out_run_prefix      Buffer to receive the run prefix (may be NULL).
 * @param run_prefix_cap      Capacity of @p out_run_prefix in bytes.
 * @param out_output_base_dir Buffer to receive the base output dir (may be NULL).
 * @param output_base_dir_cap Capacity of @p out_output_base_dir in bytes.
 * @return CFD_SUCCESS, or:
 *         - CFD_ERROR_INVALID     bad arguments, bad magic, corruption, or a
 *                                 stored string that exceeds the caller buffer;
 *         - CFD_ERROR_UNSUPPORTED unknown format version or foreign endianness;
 *         - CFD_ERROR_IO          open/read failure, truncation, or CRC mismatch;
 *         - CFD_ERROR_NOMEM       allocation failure.
 */
CFD_LIBRARY_EXPORT cfd_status_t cfd_checkpoint_read(const char* path,
                                                    grid** out_grid,
                                                    flow_field** out_field,
                                                    ns_solver_params_t* out_params,
                                                    double* out_current_time,
                                                    char* out_solver_name,
                                                    size_t solver_name_cap,
                                                    char* out_run_prefix,
                                                    size_t run_prefix_cap,
                                                    char* out_output_base_dir,
                                                    size_t output_base_dir_cap);

#ifdef __cplusplus
}
#endif

#endif  // CFD_CHECKPOINT_H
