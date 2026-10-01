/**
 * @file steady_state.h
 * @brief Steady-state detection shared by the validation tests
 *
 * Steady means the fields have stopped changing, so the tests measure the
 * largest pointwise change per unit time. An integral quantity such as the
 * kinetic energy is not enough: it has turning points where its rate passes
 * through zero while the flow is still evolving, and the lid-driven cavity's
 * overshoot stopped runs there (see lid_driven_cavity_common.h).
 */

#ifndef STEADY_STATE_H
#define STEADY_STATE_H

#include <math.h>
#include <stddef.h>

/* Largest |cur[k] - prev[k]| over n points; INFINITY if any difference is not
 * finite, since a NaN fails every comparison and would otherwise be skipped. */
static inline double steady_max_change(const double* cur, const double* prev, size_t n) {
    double change = 0.0;
    for (size_t k = 0; k < n; k++) {
        double d = fabs(cur[k] - prev[k]);
        if (!isfinite(d)) {
            return INFINITY;
        }
        if (d > change) {
            change = d;
        }
    }
    return change;
}

#endif /* STEADY_STATE_H */
