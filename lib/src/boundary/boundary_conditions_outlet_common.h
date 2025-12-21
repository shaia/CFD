/**
 * Outlet Boundary Conditions - Common Definitions
 *
 * Shared helper functions and data structures used by all outlet BC
 * backend implementations (scalar, OMP, AVX2+OMP, NEON+OMP).
 *
 * This header is internal and should only be included by outlet BC
 * implementation files.
 */

#ifndef CFD_BOUNDARY_CONDITIONS_OUTLET_COMMON_H
#define CFD_BOUNDARY_CONDITIONS_OUTLET_COMMON_H

#include "boundary_conditions_internal.h"

/* ============================================================================
 * Edge validation
 *
 * bc_edge_t uses bit flags (0x01, 0x02, 0x04, 0x08) for potential combining.
 * We convert these to sequential indices (0, 1, 2, 3) for array lookup.
 * ============================================================================ */

/**
 * Check if edge value is valid (exactly one of the four valid edges).
 */
static inline bool bc_outlet_is_valid_edge(bc_edge_t edge) {
    return edge == BC_EDGE_LEFT || edge == BC_EDGE_RIGHT ||
           edge == BC_EDGE_BOTTOM || edge == BC_EDGE_TOP;
}

/**
 * Check if outlet type is valid.
 */
static inline bool bc_outlet_is_valid_type(bc_outlet_type_t type) {
    return type == BC_OUTLET_ZERO_GRADIENT || type == BC_OUTLET_CONVECTIVE;
}

/* ============================================================================
 * Index computation functions for each edge
 *
 * For zero-gradient outlet: boundary = adjacent interior value
 * Left:   field[j*nx + 0] = field[j*nx + 1]
 * Right:  field[j*nx + nx-1] = field[j*nx + nx-2]
 * Bottom: field[i] = field[nx + i]
 * Top:    field[(ny-1)*nx + i] = field[(ny-2)*nx + i]
 * ============================================================================ */

typedef size_t (*bc_outlet_idx_func_t)(size_t i, size_t nx, size_t ny);

/* Destination index functions (boundary) */
static inline size_t bc_outlet_dst_left(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx;
}

static inline size_t bc_outlet_dst_right(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx + (nx - 1);
}

static inline size_t bc_outlet_dst_bottom(size_t i, size_t nx, size_t ny) {
    (void)nx; (void)ny;
    return i;
}

static inline size_t bc_outlet_dst_top(size_t i, size_t nx, size_t ny) {
    return (ny - 1) * nx + i;
}

/* Source index functions (adjacent interior) */
static inline size_t bc_outlet_src_left(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx + 1;
}

static inline size_t bc_outlet_src_right(size_t j, size_t nx, size_t ny) {
    (void)ny;
    return j * nx + (nx - 2);
}

static inline size_t bc_outlet_src_bottom(size_t i, size_t nx, size_t ny) {
    (void)ny;
    return nx + i;
}

static inline size_t bc_outlet_src_top(size_t i, size_t nx, size_t ny) {
    return (ny - 2) * nx + i;
}

/**
 * Edge loop configuration for table-driven boundary application.
 */
typedef struct {
    bc_outlet_idx_func_t dst_fn;        /* Boundary index computation */
    bc_outlet_idx_func_t src_fn;        /* Interior index computation */
    int use_ny_for_count;               /* 1 if loop count is ny, 0 if nx */
} bc_outlet_edge_loop_t;

/**
 * Edge loop configuration indexed 0-3 (left, right, bottom, top).
 */
static const bc_outlet_edge_loop_t bc_outlet_edge_loops[4] = {
    { .dst_fn = bc_outlet_dst_left,   .src_fn = bc_outlet_src_left,   .use_ny_for_count = 1 },  /* LEFT */
    { .dst_fn = bc_outlet_dst_right,  .src_fn = bc_outlet_src_right,  .use_ny_for_count = 1 },  /* RIGHT */
    { .dst_fn = bc_outlet_dst_bottom, .src_fn = bc_outlet_src_bottom, .use_ny_for_count = 0 },  /* BOTTOM */
    { .dst_fn = bc_outlet_dst_top,    .src_fn = bc_outlet_src_top,    .use_ny_for_count = 0 },  /* TOP */
};

/**
 * Convert bc_edge_t bit flag to sequential array index (0-3).
 * BC_EDGE_LEFT (0x01) -> 0, BC_EDGE_RIGHT (0x02) -> 1,
 * BC_EDGE_BOTTOM (0x04) -> 2, BC_EDGE_TOP (0x08) -> 3
 *
 * Uses bit manipulation to find the position of the least significant set bit
 * by counting trailing zeros in the bit flag (clamped to the range 0-3).
 * The behavior is only defined for valid, single-bit edges; callers must
 * ensure validity via bc_outlet_is_valid_edge() before calling.
 */
static inline int bc_outlet_edge_to_index(bc_edge_t edge) {
    int idx = 0;
    unsigned int e = (unsigned int)edge;
    while ((e & 1) == 0 && idx < 3) {
        e >>= 1;
        idx++;
    }
    return idx;
}

#endif /* CFD_BOUNDARY_CONDITIONS_OUTLET_COMMON_H */
