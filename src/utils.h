#ifndef CFD_UTILS_H
#define CFD_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error handling
void cfd_error(const char* message);
void cfd_warning(const char* message);

// Memory management
void* cfd_malloc(size_t size);
void* cfd_calloc(size_t count, size_t size);
void cfd_free(void* ptr);

// Mathematical utilities
double min_double(double a, double b);
double max_double(double a, double b);
double sign(double x);

#endif // CFD_UTILS_H 