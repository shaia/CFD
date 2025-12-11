#include "cfd/core/math_utils.h"

//=============================================================================
// MATH UTILITIES
//=============================================================================

double min_double(double a, double b) {
    return (a < b) ? a : b;
}

double max_double(double a, double b) {
    return (a > b) ? a : b;
}

double sign(double x) {
    return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
}
