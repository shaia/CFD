#include "derived_fields.h"
#include "utils.h"
#include <math.h>

//=============================================================================
// DERIVED FIELDS LIFECYCLE
//=============================================================================

DerivedFields* derived_fields_create(size_t nx, size_t ny) {
    DerivedFields* derived = (DerivedFields*)cfd_calloc(1, sizeof(DerivedFields));
    if (derived) {
        derived->nx = nx;
        derived->ny = ny;
        derived->velocity_magnitude = NULL;
    }
    return derived;
}

void derived_fields_destroy(DerivedFields* derived) {
    if (!derived)
        return;

    derived_fields_clear(derived);
    cfd_free(derived);
}

void derived_fields_clear(DerivedFields* derived) {
    if (!derived)
        return;

    if (derived->velocity_magnitude) {
        cfd_free(derived->velocity_magnitude);
        derived->velocity_magnitude = NULL;
    }

    // Reset statistics
    derived->stats_computed = 0;
}

//=============================================================================
// DERIVED FIELD COMPUTATION
//=============================================================================

void derived_fields_compute_velocity_magnitude(DerivedFields* derived, const FlowField* field) {
    if (!derived || !field || !field->u || !field->v)
        return;

    size_t n = derived->nx * derived->ny;

    // Allocate if needed
    if (!derived->velocity_magnitude) {
        derived->velocity_magnitude = (double*)cfd_malloc(n * sizeof(double));
        if (!derived->velocity_magnitude)
            return;
    }

    // Compute velocity magnitude
    for (size_t i = 0; i < n; i++) {
        derived->velocity_magnitude[i] =
            sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }
}

void derived_fields_compute_statistics(DerivedFields* derived, const FlowField* field) {
    if (!derived || !field)
        return;

    size_t count = derived->nx * derived->ny;

    // Compute statistics for primary fields
    if (field->u) {
        derived->u_stats = calculate_field_statistics(field->u, count);
    }
    if (field->v) {
        derived->v_stats = calculate_field_statistics(field->v, count);
    }
    if (field->p) {
        derived->p_stats = calculate_field_statistics(field->p, count);
    }
    if (field->rho) {
        derived->rho_stats = calculate_field_statistics(field->rho, count);
    }
    if (field->T) {
        derived->T_stats = calculate_field_statistics(field->T, count);
    }

    // Compute velocity magnitude statistics if available
    if (derived->velocity_magnitude) {
        derived->vel_mag_stats = calculate_field_statistics(derived->velocity_magnitude, count);
    }

    derived->stats_computed = 1;
}
