#include "cfd/core/field_ops.h"
#include "cfd/core/cfd_status.h"
#include "cfd/core/memory.h"
#include "cfd/core/logging.h"
#include "cfd/core/filesystem.h"
#include "cfd/core/math_utils.h"

#include <math.h>
#include <stdlib.h>

double* calculate_velocity_magnitude(const FlowField* field, size_t nx, size_t ny) {
    if (!field || !field->u || !field->v)
        return NULL;

    double* velocity_magnitude = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude[i] = sqrt(field->u[i] * field->u[i] + field->v[i] * field->v[i]);
    }

    return velocity_magnitude;
}

double* calculate_velocity_magnitude_squared(const FlowField* field, size_t nx, size_t ny) {
    if (!field || !field->u || !field->v)
        return NULL;

    double* velocity_magnitude_sq = (double*)cfd_malloc(nx * ny * sizeof(double));

    for (size_t i = 0; i < nx * ny; i++) {
        velocity_magnitude_sq[i] = field->u[i] * field->u[i] + field->v[i] * field->v[i];
    }

    return velocity_magnitude_sq;
}
