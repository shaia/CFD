#ifndef CSV_OUTPUT_H
#define CSV_OUTPUT_H

#include "cfd/cfd_export.h"

#include "cfd/core/derived_fields.h"
#include "cfd/solvers/navier_stokes_solver.h"
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

/**
 * CSV Output API - Low-Level Data Export Functions
 *
 * WHEN TO USE THIS HEADER:
 * - You need custom CSV output formats not provided by the registry
 * - You are writing analysis tools that process solver data
 * - You want to export data at specific points in your code
 * - You are doing post-processing or custom data logging
 *
 * WHEN NOT TO USE THIS HEADER:
 * - You are using simulation_api.h for simulations
 *   → Use output_registry via simulation_register_output() instead
 * - You want regular CSV exports (timeseries, statistics, profiles)
 *   → Use OUTPUT_CSV_TIMESERIES, OUTPUT_CSV_CENTERLINE, OUTPUT_CSV_STATISTICS
 *
 * The output registry provides:
 * - Automatic CSV file management (headers, appending)
 * - Consistent timestamping and step numbering
 * - Multiple CSV outputs configured once
 * - Proper file handling (create_new flag management)
 *
 * VELOCITY MAGNITUDE:
 * - All CSV functions accept a DerivedFields* parameter (can be NULL)
 * - When derived fields contain velocity_magnitude, it's included in output
 * - If NULL or velocity_magnitude not computed, those columns are omitted
 */

//=============================================================================
// CSV TIMESERIES OUTPUT
//=============================================================================

// Write or append timeseries data to CSV file
// Creates: step,time,dt,max_u,max_v,max_p,avg_u,avg_v,avg_p,iterations,...
// If derived fields available: adds max_vel_mag,avg_vel_mag columns
// On first write, creates file with header. Subsequent writes append data.
CFD_LIBRARY_EXPORT void write_csv_timeseries(const char* filename, int step, double time,
                                             const flow_field* field, const derived_fields* derived,
                                             const ns_solver_params_t* params, const ns_solver_stats_t* stats,
                                             size_t nx, size_t ny, int create_new);

//=============================================================================
// CSV CENTERLINE PROFILE OUTPUT
//=============================================================================

// Write centerline profile to CSV file
// Horizontal centerline: x, u, v, p, rho, T[, vel_mag] along y = ymid
// Vertical centerline: y, u, v, p, rho, T[, vel_mag] along x = xmid
// vel_mag column included if derived fields contain velocity_magnitude
typedef enum {
    PROFILE_HORIZONTAL,  // Along x-axis at y = domain_height/2
    PROFILE_VERTICAL     // Along y-axis at x = domain_width/2
} profile_direction;

CFD_LIBRARY_EXPORT void write_csv_centerline(const char* filename, const flow_field* field,
                                             const derived_fields* derived, const double* x_coords,
                                             const double* y_coords, size_t nx, size_t ny,
                                             profile_direction direction);

//=============================================================================
// CSV STATISTICS OUTPUT
//=============================================================================

// Write or append global statistics to CSV file
// Creates: step,time,min_u,max_u,avg_u,min_v,max_v,avg_v,min_p,max_p,avg_p,...
// If derived fields available: adds min_vel_mag,max_vel_mag,avg_vel_mag columns
// On first write, creates file with header. Subsequent writes append data.
CFD_LIBRARY_EXPORT void write_csv_statistics(const char* filename, int step, double time,
                                             const flow_field* field, const derived_fields* derived,
                                             size_t nx, size_t ny, int create_new);

#ifdef __cplusplus
}
#endif

#endif  // CSV_OUTPUT_H
