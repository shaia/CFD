#include "cfd/core/field_ops.h"
#include "cfd/core/memory.h"
#include "cfd/solvers/navier_stokes_solver.h"

#include <math.h>
#include <stdlib.h>

double* calculate_velocity_magnitude(const flow_field* field, size_t nx, size_t ny) {
    if (!field || !field->u || !field->v) {
        return NULL;
    }

    double* velocity_magnitude = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude[i] = sqrt((field->u[i] * field->u[i]) + (field->v[i] * field->v[i]));
    }

    return velocity_magnitude;
}

double* calculate_velocity_magnitude_squared(const flow_field* field, size_t nx, size_t ny) {
    if (!field || !field->u || !field->v) {
        return NULL;
    }

    double* velocity_magnitude_sq = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude_sq[i] = (field->u[i] * field->u[i]) + (field->v[i] * field->v[i]);
    }

    return velocity_magnitude_sq;
}
