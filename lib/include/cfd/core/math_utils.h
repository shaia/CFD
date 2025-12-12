#ifndef CFD_MATH_UTILS_H
#define CFD_MATH_UTILS_H

#include "cfd/cfd_export.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// MATH UTILITIES
//=============================================================================

CFD_LIBRARY_EXPORT double min_double(double a, double b);
CFD_LIBRARY_EXPORT double max_double(double a, double b);
CFD_LIBRARY_EXPORT double sign(double x);

#ifdef __cplusplus
}
#endif

#endif  // CFD_MATH_UTILS_H
