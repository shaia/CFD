/**
 * Sub-range of a domain edge - shared by every BC that applies to part of an edge
 *
 * A position along an x/y edge is node index based: node i of `count` sits at
 * t = i / (count - 1), so 0 is the low-coordinate end and 1 the high one. On the
 * uniform grids these BCs run on, that is also the normalized coordinate.
 *
 * Used by the velocity inlet range (bc_inlet_set_range) and by the turbulence
 * BC segments (turbulence_bc_add_segment), so the two agree on which nodes a
 * range such as [7/64, 1] covers. The CUDA inlet kernels mirror this in
 * inlet_node_position_gpu().
 *
 * This header is internal.
 */

#ifndef CFD_BC_EDGE_RANGE_H
#define CFD_BC_EDGE_RANGE_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

/* Slack on the range test, so a bound placed exactly on a node (0.5 on an odd
 * node count) keeps that node despite the rounding in i/(n-1). */
#define BC_EDGE_RANGE_EPS 1e-9

/** Normalized position of node i of `count` along an edge. */
static inline double bc_edge_node_t(size_t i, size_t count) {
    return (count > 1) ? (double)i / (double)(count - 1) : 0.5;
}

/**
 * Whether normalized edge position t lies in [start, end], and if so its
 * position rescaled to span that range, so a profile covers the range rather
 * than the whole edge.
 */
static inline bool bc_edge_range_position(double t, double start, double end,
                                          double* position) {
    if (t < start - BC_EDGE_RANGE_EPS || t > end + BC_EDGE_RANGE_EPS) {
        return false;
    }
    double s = (t - start) / (end - start);
    *position = fmin(1.0, fmax(0.0, s));
    return true;
}

#endif /* CFD_BC_EDGE_RANGE_H */
