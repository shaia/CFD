#ifndef DERIVED_FIELDS_H
#define DERIVED_FIELDS_H

#include "cfd/cfd_export.h"

#include "cfd/core/grid.h"
#include "cfd/solvers/solver_interface.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// FIELD STATISTICS
//=============================================================================

// Statistical measures for field data (min, max, avg, sum)
typedef struct {
    double min_val;
    double max_val;
    double avg_val;
    double sum_val;
} FieldStats;

// Calculate statistics for a field array (OpenMP parallelized for large arrays)
// Returns FieldStats with min, max, avg, sum computed
CFD_LIBRARY_EXPORT FieldStats calculate_field_statistics(const double* data, size_t count);

//=============================================================================
// DERIVED FIELDS
//=============================================================================

// Container for pre-computed derived fields and statistics
// These are computed from the primary FlowField data (u, v, p, rho, T)
// before being passed to the output system
typedef struct {
    // Derived field arrays
    double* velocity_magnitude;  // sqrt(u^2 + v^2), NULL if not computed
    size_t nx;
    size_t ny;

    // Pre-computed statistics (for CSV output)
    FieldStats u_stats;
    FieldStats v_stats;
    FieldStats p_stats;
    FieldStats rho_stats;
    FieldStats T_stats;
    FieldStats vel_mag_stats;
    int stats_computed;  // Flag indicating if statistics are valid
} DerivedFields;

// Create derived fields container (all fields initially NULL, stats zeroed)
CFD_LIBRARY_EXPORT DerivedFields* derived_fields_create(size_t nx, size_t ny);

// Destroy derived fields and free all computed arrays
CFD_LIBRARY_EXPORT void derived_fields_destroy(DerivedFields* derived);

// Compute velocity magnitude from flow field
// Stores result in derived->velocity_magnitude
CFD_LIBRARY_EXPORT void derived_fields_compute_velocity_magnitude(DerivedFields* derived,
                                                                  const FlowField* field);

// Compute statistics for all fields (including velocity magnitude if computed)
// Stores results in derived->*_stats fields
CFD_LIBRARY_EXPORT void derived_fields_compute_statistics(DerivedFields* derived,
                                                          const FlowField* field);

// Clear all computed fields (frees memory, sets pointers to NULL, resets stats)
CFD_LIBRARY_EXPORT void derived_fields_clear(DerivedFields* derived);

#ifdef __cplusplus
}
#endif

#endif  // DERIVED_FIELDS_H
