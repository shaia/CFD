#ifndef DERIVED_FIELDS_H
#define DERIVED_FIELDS_H

#include "grid.h"
#include "solver_interface.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// DERIVED FIELDS
//=============================================================================

// Container for pre-computed derived fields
// These are computed from the primary FlowField data (u, v, p, rho, T)
// before being passed to the output system
typedef struct {
    double* velocity_magnitude;  // sqrt(u^2 + v^2), NULL if not computed
    size_t nx;
    size_t ny;
} DerivedFields;

// Create derived fields container (all fields initially NULL)
DerivedFields* derived_fields_create(size_t nx, size_t ny);

// Destroy derived fields and free all computed arrays
void derived_fields_destroy(DerivedFields* derived);

// Compute velocity magnitude from flow field
// Stores result in derived->velocity_magnitude
void derived_fields_compute_velocity_magnitude(DerivedFields* derived, const FlowField* field);

// Clear all computed fields (frees memory, sets pointers to NULL)
void derived_fields_clear(DerivedFields* derived);

#ifdef __cplusplus
}
#endif

#endif  // DERIVED_FIELDS_H
