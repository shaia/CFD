#include "cfd/core/derived_fields.h"
#include "cfd/core/utils.h"
#include <math.h>

#ifdef CFD_ENABLE_OPENMP
#include <omp.h>
#endif

// Minimum grid size to benefit from parallelization
// Below this threshold, thread overhead exceeds benefit
#define OMP_THRESHOLD 1000

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
// FIELD STATISTICS
//=============================================================================

FieldStats calculate_field_statistics(const double* data, size_t count) {
    FieldStats stats = {0};

    if (!data || count == 0) {
        return stats;
    }

#ifdef CFD_ENABLE_OPENMP
    if (count >= OMP_THRESHOLD) {
        // Parallel reduction for min, max, and sum
        double min_val = data[0];
        double max_val = data[0];
        double sum_val = 0.0;

        // Use signed type for OpenMP compatibility with MSVC
        long long nn = (long long)count;
        long long i;
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val) reduction(+ : sum_val)
        for (i = 0; i < nn; i++) {
            double val = data[i];
            if (val < min_val)
                min_val = val;
            if (val > max_val)
                max_val = val;
            sum_val += val;
        }

        stats.min_val = min_val;
        stats.max_val = max_val;
        stats.sum_val = sum_val;
    } else {
        // Sequential for small arrays
        stats.min_val = data[0];
        stats.max_val = data[0];
        stats.sum_val = 0.0;

        for (size_t i = 0; i < count; i++) {
            double val = data[i];
            if (val < stats.min_val)
                stats.min_val = val;
            if (val > stats.max_val)
                stats.max_val = val;
            stats.sum_val += val;
        }
    }
#else
    // Sequential implementation when OpenMP not available
    stats.min_val = data[0];
    stats.max_val = data[0];
    stats.sum_val = 0.0;

    for (size_t i = 0; i < count; i++) {
        double val = data[i];
        if (val < stats.min_val)
            stats.min_val = val;
        if (val > stats.max_val)
            stats.max_val = val;
        stats.sum_val += val;
    }
#endif

    stats.avg_val = stats.sum_val / (double)count;
    return stats;
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

    // Compute velocity magnitude (embarrassingly parallel)
    const double* u = field->u;
    const double* v = field->v;
    double* vel_mag = derived->velocity_magnitude;

#ifdef CFD_ENABLE_OPENMP
    if (n >= OMP_THRESHOLD) {
        // Use signed type for OpenMP compatibility with MSVC
        long long nn = (long long)n;
        long long i;
#pragma omp parallel for
        for (i = 0; i < nn; i++) {
            vel_mag[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
        }
    } else {
        for (size_t i = 0; i < n; i++) {
            vel_mag[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
        }
    }
#else
    for (size_t i = 0; i < n; i++) {
        vel_mag[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
    }
#endif
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
